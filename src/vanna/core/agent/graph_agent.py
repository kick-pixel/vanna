"""
GraphAgent implementation for the Vanna Agents framework using LangGraph.

This module provides the GraphAgent class that orchestrates the interaction
between LLM services, tools, and conversation storage using a state graph.
"""

import asyncio
import logging
import traceback
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, TypedDict, Union

from langgraph.graph import END, StateGraph

from vanna.components import (
    ChatInputUpdateComponent,
    RichTextComponent,
    SimpleTextComponent,
    StatusBarUpdateComponent,
    StatusCardComponent,
    Task,
    TaskTrackerUpdateComponent,
    UiComponent,
)
from vanna.capabilities.agent_memory import AgentMemory
from vanna.core.agent.config import AgentConfig, UiFeature
from vanna.core.audit import AuditLogger
from vanna.core.enricher import ToolContextEnricher
from vanna.core.enhancer import LlmContextEnhancer, DefaultLlmContextEnhancer
from vanna.core.filter import ConversationFilter
from vanna.core.lifecycle import LifecycleHook
from vanna.core.llm import LlmMessage, LlmRequest, LlmResponse, LlmService
from vanna.core.middleware import LlmMiddleware
from vanna.core.observability import ObservabilityProvider
from vanna.core.recovery import ErrorRecoveryStrategy
from vanna.core.registry import ToolRegistry
from vanna.core.storage import Conversation, ConversationStore, Message
from vanna.core.system_prompt import DefaultSystemPromptBuilder, SystemPromptBuilder
from vanna.core.tool import ToolContext, ToolSchema
from vanna.core.user import User
from vanna.core.user.request_context import RequestContext
from vanna.core.user.resolver import UserResolver
from vanna.core.workflow import DefaultWorkflowHandler, WorkflowHandler

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the Agent execution graph."""

    # Context
    request_context: RequestContext
    user: User
    conversation_id: str
    request_id: str
    conversation: Conversation
    agent_memory: AgentMemory
    observability_provider: Optional[ObservabilityProvider]

    # Inputs
    message: str

    # Workflow Data
    ui_queue: asyncio.Queue  # Queue for streaming UI components
    is_starter_request: bool
    should_stop: bool

    # LLM Interaction
    tool_schemas: List[ToolSchema]
    system_prompt: Optional[str]
    messages: List[LlmMessage]
    llm_request: Optional[LlmRequest]
    llm_response: Optional[LlmResponse]

    # Execution Control
    tool_iterations: int
    tool_context: Optional[ToolContext]


# Helper type for partial state updates
PartialAgentState = Dict[str, Any]


class GraphAgent:
    """
    Agent implementation using LangGraph for macro-orchestration.

    This class maintains API compatibility with the standard Agent class
    while using a directed cyclic graph for the internal decision loop.
    """

    def __init__(
        self,
        llm_service: LlmService,
        tool_registry: ToolRegistry,
        user_resolver: UserResolver,
        agent_memory: AgentMemory,
        conversation_store: Optional[ConversationStore] = None,
        config: AgentConfig = AgentConfig(),
        system_prompt_builder: SystemPromptBuilder = DefaultSystemPromptBuilder(),
        lifecycle_hooks: List[LifecycleHook] = [],
        llm_middlewares: List[LlmMiddleware] = [],
        workflow_handler: Optional[WorkflowHandler] = None,
        error_recovery_strategy: Optional[ErrorRecoveryStrategy] = None,
        context_enrichers: List[ToolContextEnricher] = [],
        llm_context_enhancer: Optional[LlmContextEnhancer] = None,
        conversation_filters: List[ConversationFilter] = [],
        observability_provider: Optional[ObservabilityProvider] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.user_resolver = user_resolver
        self.agent_memory = agent_memory

        if conversation_store is None:
            from vanna.integrations.local import MemoryConversationStore
            conversation_store = MemoryConversationStore()

        self.conversation_store = conversation_store
        self.config = config
        self.system_prompt_builder = system_prompt_builder
        self.lifecycle_hooks = lifecycle_hooks
        self.llm_middlewares = llm_middlewares

        if workflow_handler is None:
            workflow_handler = DefaultWorkflowHandler()
        self.workflow_handler = workflow_handler

        self.error_recovery_strategy = error_recovery_strategy
        self.context_enrichers = context_enrichers

        if llm_context_enhancer is None:
            llm_context_enhancer = DefaultLlmContextEnhancer(agent_memory)
        self.llm_context_enhancer = llm_context_enhancer

        self.conversation_filters = conversation_filters
        self.observability_provider = observability_provider
        self.audit_logger = audit_logger

        if self.audit_logger and self.config.audit_config.enabled:
            self.tool_registry.audit_logger = self.audit_logger
            self.tool_registry.audit_config = self.config.audit_config

        # Initialize the graph
        self.graph = self._build_graph()
        logger.info("Initialized GraphAgent")

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        workflow.add_node("initialize", self._node_initialize)
        workflow.add_node("prepare_context", self._node_prepare_context)
        workflow.add_node("think", self._node_think)
        workflow.add_node("execute_tools", self._node_execute_tools)
        workflow.add_node("finalize", self._node_finalize)

        workflow.set_entry_point("initialize")

        workflow.add_edge("initialize", "prepare_context")

        # Conditional edge from prepare_context (handling starter UI short-circuit)
        workflow.add_conditional_edges(
            "prepare_context",
            self._router_check_stop,
            {
                "stop": "finalize",
                "continue": "think"
            }
        )

        # Conditional edge from think (Tool vs Done)
        workflow.add_conditional_edges(
            "think",
            self._router_analyze_response,
            {
                "tools": "execute_tools",
                "done": "finalize"
            }
        )

        # Conditional edge from execute_tools (Loop vs Stop)
        workflow.add_conditional_edges(
            "execute_tools",
            self._router_check_limit,
            {
                "continue": "think",
                "stop": "finalize"
            }
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def send_message(
        self,
        request_context: RequestContext,
        message: str,
        *,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[UiComponent, None]:
        """
        Process a user message using the graph and yield UI components.
        """
        ui_queue = asyncio.Queue()

        # Initial state
        initial_state: AgentState = {
            "request_context": request_context,
            "user": None,  # Resolved in initialize
            "conversation_id": conversation_id,
            "request_id": str(uuid.uuid4()),
            "conversation": None,  # Loaded in initialize
            "agent_memory": self.agent_memory,
            "observability_provider": self.observability_provider,
            "message": message,
            "ui_queue": ui_queue,
            "is_starter_request": False,
            "should_stop": False,
            "tool_schemas": [],
            "system_prompt": None,
            "messages": [],
            "llm_request": None,
            "llm_response": None,
            "tool_iterations": 0,
            "tool_context": None,
        }

        # Run the graph in a background task
        graph_task = asyncio.create_task(self.graph.ainvoke(initial_state))

        try:
            while True:
                # Wait for either a UI component or the graph task to finish
                # We use a small timeout for the queue get to check task status frequently
                try:
                    # Check if task is done and raised exception
                    if graph_task.done():
                        exc = graph_task.exception()
                        if exc:
                            raise exc

                        # Process remaining items in queue
                        while not ui_queue.empty():
                            yield await ui_queue.get()
                        break

                    # Wait for next item
                    item = await asyncio.wait_for(ui_queue.get(), timeout=0.1)

                    if item is None:  # Sentinel
                        break

                    yield item

                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            # Handle errors similarly to Agent class
            logger.error(f"Error in GraphAgent: {e}", exc_info=True)
            yield UiComponent(
                rich_component=StatusCardComponent(
                    title="Error Processing Message",
                    status="error",
                    description="An unexpected error occurred.",
                    icon="⚠️",
                ),
                simple_component=SimpleTextComponent(text=f"Error: {str(e)}")
            )
        finally:
            if not graph_task.done():
                graph_task.cancel()

    # --- Node Implementations ---

    async def _node_initialize(self, state: AgentState) -> PartialAgentState:
        """Resolve user, load conversation, handle workflow/hooks."""
        request_context = state["request_context"]
        message = state["message"]
        conversation_id = state["conversation_id"]
        ui_queue = state["ui_queue"]

        # Resolve User
        user = await self.user_resolver.resolve_user(request_context)

        # Check starter request
        is_starter_request = (not message.strip()) or request_context.metadata.get(
            "starter_ui_request", False
        )

        if is_starter_request and self.workflow_handler:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())

            conversation = await self.conversation_store.get_conversation(conversation_id, user)
            if not conversation:
                conversation = Conversation(id=conversation_id, user=user, messages=[])

            components = await self.workflow_handler.get_starter_ui(self, user, conversation)
            if components:
                for comp in components:
                    await ui_queue.put(comp)

                # Ready status
                await ui_queue.put(UiComponent(
                    rich_component=StatusBarUpdateComponent(
                        status="idle", message="Ready", detail="Choose an option or type a message"
                    )
                ))
                await ui_queue.put(UiComponent(
                    rich_component=ChatInputUpdateComponent(
                        placeholder="Ask a question...", disabled=False
                    )
                ))

                if self.config.auto_save_conversations:
                    await self.conversation_store.update_conversation(conversation)

                return {
                    "user": user,
                    "conversation": conversation,
                    "conversation_id": conversation_id,
                    "is_starter_request": True,
                    "should_stop": True
                }

        if not message.strip():
            return {"user": user, "should_stop": True}

        # Lifecycle Hooks: before_message
        for hook in self.lifecycle_hooks:
            result = await hook.before_message(user, message)
            if result is not None:
                message = result

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        await ui_queue.put(UiComponent(
            rich_component=StatusBarUpdateComponent(
                status="working", message="Processing...", detail="Analyzing query"
            )
        ))

        conversation = await self.conversation_store.get_conversation(conversation_id, user)
        is_new = False
        if not conversation:
            conversation = Conversation(id=conversation_id, user=user, messages=[])
            is_new = True

        # Workflow Handler Check
        if self.workflow_handler:
            wf_result = await self.workflow_handler.try_handle(self, user, conversation, message)
            if wf_result.should_skip_llm:
                if wf_result.conversation_mutation:
                    await wf_result.conversation_mutation(conversation)

                if wf_result.components:
                    iterable = wf_result.components
                    if not isinstance(iterable, list):
                        async for comp in iterable:
                            await ui_queue.put(comp)
                    else:
                        for comp in iterable:
                            await ui_queue.put(comp)

                await ui_queue.put(UiComponent(
                    rich_component=StatusBarUpdateComponent(
                        status="idle", message="Workflow complete", detail="Ready"
                    )
                ))
                await ui_queue.put(UiComponent(
                    rich_component=ChatInputUpdateComponent(disabled=False)
                ))

                if self.config.auto_save_conversations:
                    await self.conversation_store.update_conversation(conversation)

                return {
                    "user": user,
                    "conversation": conversation,
                    "should_stop": True
                }

        if is_new:
            await self.conversation_store.update_conversation(conversation)

        conversation.add_message(Message(role="user", content=message))

        return {
            "user": user,
            "conversation": conversation,
            "conversation_id": conversation_id,
            "message": message,
            "should_stop": False
        }

    async def _node_prepare_context(self, state: AgentState) -> PartialAgentState:
        """Enrich context, fetch schemas, build prompt."""
        if state.get("should_stop"):
            return {}

        user = state["user"]
        conversation = state["conversation"]
        ui_queue = state["ui_queue"]

        context_task = Task(title="Load context", status="pending")
        await ui_queue.put(UiComponent(
            rich_component=TaskTrackerUpdateComponent.add_task(context_task)
        ))

        # Build Tool Context
        ui_features_available = []
        for feature_name in self.config.ui_features.feature_group_access.keys():
            if self.config.ui_features.can_user_access_feature(feature_name, user):
                ui_features_available.append(feature_name)

        tool_context = ToolContext(
            user=user,
            conversation_id=conversation.id,
            request_id=state["request_id"],
            agent_memory=self.agent_memory,
            observability_provider=self.observability_provider,
            metadata={"ui_features_available": ui_features_available},
        )

        for enricher in self.context_enrichers:
            tool_context = await enricher.enrich_context(tool_context)

        # Get Tools
        tool_schemas = await self.tool_registry.get_schemas(user)

        await ui_queue.put(UiComponent(
            rich_component=TaskTrackerUpdateComponent.update_task(
                context_task.id, status="completed"
            )
        ))

        # Build System Prompt
        system_prompt = await self.system_prompt_builder.build_system_prompt(
            user, tool_schemas
        )
        if self.llm_context_enhancer and system_prompt:
            system_prompt = await self.llm_context_enhancer.enhance_system_prompt(
                system_prompt, state["message"], user
            )

        # Filter Messages
        filtered_messages = conversation.messages
        for filter in self.conversation_filters:
            filtered_messages = await filter.filter_messages(filtered_messages)

        # Convert to LlmMessage
        messages = [
            LlmMessage(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                tool_call_id=msg.tool_call_id
            )
            for msg in filtered_messages
        ]

        if self.llm_context_enhancer:
            messages = await self.llm_context_enhancer.enhance_user_messages(messages, user)

        return {
            "tool_context": tool_context,
            "tool_schemas": tool_schemas,
            "system_prompt": system_prompt,
            "messages": messages
        }

    async def _node_think(self, state: AgentState) -> PartialAgentState:
        """Execute LLM request."""
        request = LlmRequest(
            messages=state["messages"],
            tools=state["tool_schemas"] if state["tool_schemas"] else None,
            user=state["user"],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=self.config.stream_responses,
            system_prompt=state["system_prompt"],
        )

        # Middlewares: before_llm_request
        for mw in self.llm_middlewares:
            request = await mw.before_llm_request(request)

        response: LlmResponse
        if self.config.stream_responses:
            # Handle Streaming
            accumulated_content = ""
            accumulated_tool_calls = []

            async for chunk in self.llm_service.stream_request(request):
                if chunk.content:
                    accumulated_content += chunk.content
                if chunk.tool_calls:
                    accumulated_tool_calls.extend(chunk.tool_calls)

            response = LlmResponse(
                content=accumulated_content if accumulated_content else None,
                tool_calls=accumulated_tool_calls if accumulated_tool_calls else None
            )
        else:
            response = await self.llm_service.send_request(request)

        # Middlewares: after_llm_response
        for mw in self.llm_middlewares:
            response = await mw.after_llm_response(request, response)

        return {"llm_response": response}

    async def _node_execute_tools(self, state: AgentState) -> PartialAgentState:
        """Execute tools from LLM response."""
        response = state["llm_response"]
        conversation = state["conversation"]
        ui_queue = state["ui_queue"]
        user = state["user"]
        context = state["tool_context"]

        # Add Assistant Message
        assistant_msg = Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        )
        conversation.add_message(assistant_msg)

        # Yield content
        if response.content:
            await ui_queue.put(UiComponent(
                rich_component=RichTextComponent(content=response.content, markdown=True),
                simple_component=SimpleTextComponent(text=response.content)
            ))

        await ui_queue.put(UiComponent(
            rich_component=StatusBarUpdateComponent(
                status="working",
                message="Executing tools...",
                detail=f"Running {len(response.tool_calls or [])} tools"
            )
        ))

        tool_results_data = []
        for tool_call in (response.tool_calls or []):
            # Task UI
            tool_task = Task(
                title=f"Execute {tool_call.name}",
                description="Running tool...",
                status="in_progress"
            )
            await ui_queue.put(UiComponent(
                rich_component=TaskTrackerUpdateComponent.add_task(tool_task)
            ))

            # Status Card UI
            card = StatusCardComponent(
                title=f"Executing {tool_call.name}",
                status="running",
                icon="⚙️",
                metadata=tool_call.arguments
            )
            await ui_queue.put(UiComponent(rich_component=card))

            # Hooks: before_tool
            tool = await self.tool_registry.get_tool(tool_call.name)
            if tool:
                for hook in self.lifecycle_hooks:
                    await hook.before_tool(tool, context)

            # Execution
            result = await self.tool_registry.execute(tool_call, context)

            # Hooks: after_tool
            for hook in self.lifecycle_hooks:
                modified = await hook.after_tool(result)
                if modified:
                    result = modified

            # Update UI with result
            status = "success" if result.success else "error"
            await ui_queue.put(UiComponent(
                rich_component=card.set_status(status, result.result_for_llm)
            ))
            await ui_queue.put(UiComponent(
                rich_component=TaskTrackerUpdateComponent.update_task(
                    tool_task.id, status="completed"
                )
            ))

            if result.ui_component:
                await ui_queue.put(result.ui_component)

            tool_results_data.append({
                "tool_call_id": tool_call.id,
                "content": result.result_for_llm if result.success else (result.error or "Failed")
            })

        # Add Tool Messages
        for res in tool_results_data:
            conversation.add_message(Message(
                role="tool",
                content=res["content"],
                tool_call_id=res["tool_call_id"]
            ))

        # Rebuild messages for next LLM turn
        # In a real implementation, we would just append to state["messages"]
        # but here we rely on conversation.messages being the source of truth
        # so we might need to re-convert them or append them.
        # For simplicity, we assume prepare_context logic or incremental append logic.

        # Incremental append to LlmMessages
        new_messages = state["messages"][:]
        new_messages.append(LlmMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        ))
        for res in tool_results_data:
            new_messages.append(LlmMessage(
                role="tool",
                content=res["content"],
                tool_call_id=res["tool_call_id"]
            ))

        return {
            "tool_iterations": state["tool_iterations"] + 1,
            "messages": new_messages
        }

    async def _node_finalize(self, state: AgentState) -> PartialAgentState:
        """Save conversation, hooks, clean up."""
        if state.get("should_stop"):
            return {}

        conversation = state["conversation"]
        ui_queue = state["ui_queue"]

        # If we came here from "done" state of LLM (no tools)
        response = state.get("llm_response")
        if response and not response.is_tool_call():
            # Add final assistant message if not already added
            # (In execute_tools we add it, but if we skipped tools, we need to add it here)
            conversation.add_message(Message(role="assistant", content=response.content))

            if response.content:
                await ui_queue.put(UiComponent(
                    rich_component=RichTextComponent(content=response.content, markdown=True),
                    simple_component=SimpleTextComponent(text=response.content)
                ))

        await ui_queue.put(UiComponent(
            rich_component=StatusBarUpdateComponent(
                status="idle", message="Response complete", detail="Ready"
            )
        ))
        await ui_queue.put(UiComponent(
            rich_component=ChatInputUpdateComponent(
                placeholder="Ask a follow-up...", disabled=False
            )
        ))

        if self.config.auto_save_conversations:
            await self.conversation_store.update_conversation(conversation)

        for hook in self.lifecycle_hooks:
            await hook.after_message(conversation)

        return {"is_complete": True}

    # --- Routers ---

    def _router_check_stop(self, state: AgentState) -> Literal["stop", "continue"]:
        return "stop" if state.get("should_stop") else "continue"

    def _router_analyze_response(self, state: AgentState) -> Literal["tools", "done"]:
        response = state["llm_response"]
        if response and response.is_tool_call():
            return "tools"
        return "done"

    def _router_check_limit(self, state: AgentState) -> Literal["continue", "stop"]:
        if state["tool_iterations"] < self.config.max_tool_iterations:
            return "continue"

        # Limit reached logic could be added here (logging, warning UI)
        return "stop"
