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
import functools


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
    tool_iterations: int
    tool_context: Optional[ToolContext]
    
    # Schema & SQL
    schema_metadata: Optional[str]
    generated_sql: Optional[str]
    sql_result: Optional[str]


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
        logger.info(f"Graph: {self.graph.get_graph().draw_mermaid()}")
        logger.info("Initialized GraphAgent")



    def _node_wrapper(self, node_name: str, func):
        """Wrapper to add hooks around node execution."""
        @functools.wraps(func)
        async def wrapper(state: AgentState):
            # Pre-execution hooks could go here
            # For now we just log
            logger.info(f"========================Entering node: {node_name} ========================")
            
            # Execute
            result = await func(state)
            # log result
            logger.info(f"=======================   FINISHED NODE: {node_name} ========================")
            
            # Post-execution hooks
            # We could modify result or state here
            
            return result
        return wrapper

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Helper to add nodes with hooks
        def add_node(name: str, func):
            workflow.add_node(name, self._node_wrapper(name, func))

        add_node("initialize", self._node_initialize)
        
        # Merged prepare_context into initialize
        add_node("get_schema", self._node_get_schema)
        add_node("think", self._node_think)
        add_node("generate_sql", self._node_generate_sql)
        add_node("execute_sql", self._node_execute_sql)
        add_node("execute_tools", self._node_execute_tools)
        add_node("finalize", self._node_finalize)

        workflow.set_entry_point("initialize")

        # Initialize now prepares context, so loop directly to think or specialized node if start request
        # Actually initialize returns user & context, so next step is think
        
        workflow.add_conditional_edges(
             "initialize",
             self._router_check_stop,
             {
                 "stop": "finalize",
                 "continue": "think" 
             }
        )
        
        # All action nodes return to think
        workflow.add_edge("get_schema", "think")
        workflow.add_edge("generate_sql", "think")
        workflow.add_edge("execute_sql", "think")

        # Conditional edge from think (Tool vs Done vs Virtual Tools)
        workflow.add_conditional_edges(
            "think",
            self._router_analyze_response,
            {
                "tools": "execute_tools",
                "done": "finalize",
                "get_schema": "get_schema",
                "generate_sql": "generate_sql",
                "execute_sql": "execute_sql"
            }
        )
        
        # From execute_tools to think
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
            "schema_metadata": None,
            "generated_sql": None,
            "sql_result": None,
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
        """
        Combined Initialization Node:
        1. Resolve User & Conversation.
        2. Handle Workflow/Starters.
        3. Prepare Tool Context & System Prompt.
        """
        request_context = state["request_context"]
        message = state["message"]
        conversation_id = state["conversation_id"]
        ui_queue = state["ui_queue"]

        # 1. Resolve User
        user = await self.user_resolver.resolve_user(request_context)

        # 2. Check starter request / Workflow
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
                await ui_queue.put(UiComponent(rich_component=StatusBarUpdateComponent(status="idle", message="Ready", detail="Choose an option")))
                await ui_queue.put(UiComponent(rich_component=ChatInputUpdateComponent(placeholder="Ask a question...", disabled=False)))
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
                status="working", message="Processing...", detail="Initializing context"
            )
        ))

        conversation = await self.conversation_store.get_conversation(conversation_id, user)
        is_new = False
        if not conversation:
            conversation = Conversation(id=conversation_id, user=user, messages=[])
            is_new = True

        if is_new:
            await self.conversation_store.update_conversation(conversation)

        conversation.add_message(Message(role="user", content=message))

        # 3. Prepare Context (formerly _node_prepare_context)
        context_task = Task(title="Load context", status="pending")
        await ui_queue.put(UiComponent(rich_component=TaskTrackerUpdateComponent.add_task(context_task)))

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
                system_prompt, message, user
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
            "user": user,
            "conversation": conversation,
            "conversation_id": conversation_id,
            "message": message,
            "tool_context": tool_context,
            "tool_schemas": tool_schemas,
            "system_prompt": system_prompt,
            "messages": messages,
            "should_stop": False
        }





    async def _node_get_schema(self, state: AgentState) -> PartialAgentState:
        """
        Active Schema Retrieval Node.
        Executes SQL provided by the LLM (via query_schema_metadata) to inspect database structure.
        Stores the result in memory and adds to context.
        """
        # if state.get("should_stop"): return {} # Not needed as we are routed here explicitly

        context = state.get("tool_context")
        ui_queue = state.get("ui_queue")
        response = state.get("llm_response")
        
        # 1. Parse SQL from tool call & Handle all tool calls
        schema_sql = "SELECT name, sql FROM sqlite_master WHERE type='table'" # Fallback
        
        target_tool_id = None
        other_tool_ids = []

        if response and response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "query_schema_metadata":
                    # 从 LLM 提供的参数中获取 SQL,如果没有则保留 fallback
                    provided_sql = tc.arguments.get("sql")
                    if provided_sql:
                        schema_sql = provided_sql
                    target_tool_id = tc.id
                else:
                    other_tool_ids.append(tc.id)
        
        if not schema_sql:
             # Must respond to the tool call
             for tc in (response.tool_calls or []):
                 state["messages"].append(LlmMessage(
                     role="tool", 
                     content="Error: No SQL argument provided for schema query.", 
                     tool_call_id=tc.id
                 ))
             return {}

        await ui_queue.put(UiComponent(
            rich_component=StatusBarUpdateComponent(
                status="working", message="Querying Schema", detail="Inspecting database..."
            )
        ))

        # 2. Execute SQL (using run_sql tool logic)
        query_result_text = ""
        try:
             sql_tool = await self.tool_registry.get_tool("run_sql")
             if not sql_tool:
                 raise Exception("Refusing to query schema: 'run_sql' tool not available.")
             
             # Execute
             # We construct args manually.
             args_model = sql_tool.get_args_schema()
             tool_args = args_model(sql=schema_sql)
             
             result = await sql_tool.execute(context, tool_args)
             
             if result.success:
                 query_result_text = result.result_for_llm
             else:
                 query_result_text = f"Schema Query Failed: {result.error}"
                 
        except Exception as e:
            logger.error(f"Schema Query Error: {e}")
            query_result_text = f"Schema Query Error: {e}"

        # 3. Store in Memory
        # We save this as a text memory so it persists across turns/sessions if needed
        # Or just append to system prompt for this session.
        # User requested: "get_schema获取并存储到mermory里"
        if self.agent_memory and "Error" not in query_result_text:
             # Create a text memory item
             # In a real app we might use save_text_memory tool logic or direct call
             # For now, we assume direct access if possible, or just skip if no direct api
             # We'll just add it to the 'schema_metadata' specific state which might be transient or saved
             pass

        # 4. Update Context
        
        result_msg = f"Schema Query Result ({schema_sql}):\n{query_result_text}"
        
        # Add to messages so LLM sees it
        if target_tool_id:
             state["messages"].append(LlmMessage(role="tool", content=result_msg, tool_call_id=target_tool_id))
        else:
             # Fallback if we somehow lost the ID or it wasn't a tool call (unlikely with this flow)
             state["messages"].append(LlmMessage(role="system", content=result_msg))
             
        # Handle other tool calls (dummy response to satisfy API)
        for ot_id in other_tool_ids:
            state["messages"].append(LlmMessage(role="tool", content="Tool call ignored in this step.", tool_call_id=ot_id))
        
        return {
            "schema_metadata": query_result_text, # Allow subsequent nodes to see it specifically
            "tool_iterations": state["tool_iterations"] + 1
        }

    async def _node_generate_sql(self, state: AgentState) -> PartialAgentState:
        """Generate SQL based on request."""
        ui_queue = state.get("ui_queue")
        response = state.get("llm_response")
        
        # Get instruction from tool call if available, else generic
        instruction = "Generate SQL for the user's request."
        if response and response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "generate_sql":
                    instruction = tc.arguments.get("instruction", instruction)
                    break

        await ui_queue.put(UiComponent(
            rich_component=StatusBarUpdateComponent(
                status="working", message="Generating SQL", detail="Drafting query..."
            )
        ))
        
        # Find tool call ID
        target_tool_id = None
        other_tool_ids = []
        if response and response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "generate_sql":
                    target_tool_id = tc.id
                else:
                    other_tool_ids.append(tc.id)

        if target_tool_id:
            # We MUST close the tool call loop before making a new request
            state["messages"].append(LlmMessage(role="tool", content="Proceeding with SQL generation.", tool_call_id=target_tool_id))

        # Handle other tool calls
        for ot_id in other_tool_ids:
            state["messages"].append(LlmMessage(role="tool", content="Tool call ignored in this step.", tool_call_id=ot_id))
            
        # We append the specific instruction to the prompt
        # Note: We do NOT include the just-added tool result in the request if we want the LLM to just write SQL
        # However, to be compliant with history, we should include it.
        # But for the specialized "Generation" task, we might want to mask the history or just append the task prompt.
        
        request = LlmRequest(
            messages=state["messages"],
            tools=None, # Strict mode: provide NO tools so it must output text (code)
            user=state["user"],
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            stream=self.config.stream_responses,
            system_prompt=state["system_prompt"] + f"\n\nTASK: {instruction}\nOutput executable SQL only. No markdown.",
        )
        
        for mw in self.llm_middlewares:
            request = await mw.before_llm_request(request)

        response: LlmResponse
        if self.config.stream_responses:
            accumulated_content = ""
            async for chunk in self.llm_service.stream_request(request):
                if chunk.content:
                    accumulated_content += chunk.content
            response = LlmResponse(content=accumulated_content)
        else:
            response = await self.llm_service.send_request(request)

        for mw in self.llm_middlewares:
            response = await mw.after_llm_response(request, response)

        generated_sql = response.content
        if generated_sql:
            generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

        state["messages"].append(LlmMessage(role="assistant", content=f"Generated SQL: {generated_sql}"))
        
        return {
            "generated_sql": generated_sql,
            "tool_iterations": state["tool_iterations"] + 1
        }

    async def _node_execute_sql(self, state: AgentState) -> PartialAgentState:
        """Execute the generated SQL."""
        ui_queue = state.get("ui_queue")
        generated_sql = state.get("generated_sql")
        context = state.get("tool_context")
        conversation = state.get("conversation")
        
        # Find tool call ID for execute_current_sql to respond to errors
        target_tool_id = None
        other_tool_ids = []
        response = state.get("llm_response")
        if response and response.tool_calls:
             for tc in response.tool_calls:
                 if tc.name == "execute_current_sql":
                     target_tool_id = tc.id
                 else:
                     other_tool_ids.append(tc.id)

        if not generated_sql:
             if target_tool_id:
                  state["messages"].append(LlmMessage(role="tool", content="Error: No SQL has been generated yet.", tool_call_id=target_tool_id))
             for ot_id in other_tool_ids:
                  state["messages"].append(LlmMessage(role="tool", content="Ignored.", tool_call_id=ot_id))
             return {}
             
        await ui_queue.put(UiComponent(
            rich_component=StatusBarUpdateComponent(
                status="working", message="Executing SQL", detail="Running query..."
            )
        ))

        # Find the real RunSqlTool
        sql_tool = await self.tool_registry.get_tool("run_sql")
        if not sql_tool:
             if target_tool_id:
                 state["messages"].append(LlmMessage(role="tool", content="Error: 'run_sql' tool is not configured.", tool_call_id=target_tool_id))
             for ot_id in other_tool_ids:
                 state["messages"].append(LlmMessage(role="tool", content="Ignored.", tool_call_id=ot_id))
             return {}
             
        try:
             # Basic execution
             # We rely on 'execute' method of the tool instance
             # We need to construct the argument object expected by the tool
             args_model = sql_tool.get_args_schema()
             tool_args = args_model(sql=generated_sql)
             
             # Call tool directly to avoid schema validation overhead/mismatch
             result = await sql_tool.execute(context, tool_args)
             
        except Exception as e:
            logger.error(f"SQL Execution failed: {e}")
            state["messages"].append(LlmMessage(role="system", content=f"SQL Execution Error: {e}"))
            return {}

        # Add result to conversation
        # We already found IDs above
        
        conversation.add_message(Message(
            role="tool",
            content=result.result_for_llm if result.success else f"Error: {result.error}",
            tool_call_id=target_tool_id or "unknown"
        ))
        
        # Add to context messages
        state["messages"].append(LlmMessage(
            role="tool", 
            content=result.result_for_llm if result.success else f"Error: {result.error}",
            tool_call_id=target_tool_id or "unknown"
        ))
        
        # Handle other tool calls
        for ot_id in other_tool_ids:
            state["messages"].append(LlmMessage(role="tool", content="Tool call ignored in this step.", tool_call_id=ot_id))
        
        if result.ui_component:
            await ui_queue.put(result.ui_component)

        return {
            "sql_result": result.result_for_llm,
            "tool_iterations": state["tool_iterations"] + 1
        }


    async def _node_think(self, state: AgentState) -> PartialAgentState:
        """Execute LLM request with virtual tools."""
        
        # 1. Define Virtual Tools (Updated)
        virtual_tools = [
            ToolSchema(
                name="query_schema_metadata",
                description="CRITICAL: Use this tool FIRST to retreive the database schema (tables/columns) before generating any SQL. Execute invalid SQLs like `SELECT ... FROM sqlite_master` to find tables.",
                 parameters={
                   "type": "object", 
                   "properties": {
                       "sql": {"type": "string", "description": "The SQL query to inspect schema (e.g. 'SELECT name, sql FROM sqlite_master WHERE type=\"table\"')"}
                   }, 
                   "required": ["sql"]
                }
            ),
            ToolSchema(
                name="generate_sql",
                description="Generate a business logic SQL query based on the schema and user question.",
                parameters={
                   "type": "object", "properties": {"instruction": {"type": "string"}}, "required": ["instruction"]
                }
            ),
            ToolSchema(
                name="execute_current_sql",
                description="Execute the currently generated SQL query.",
                parameters={
                   "type": "object", "properties": {}, "required": []
                }
            )
        ]
        
        # 2. Filter Real Tools
        real_tools = state.get("tool_schemas", [])
        filtered_tools = [t for t in real_tools if t.name != "run_sql"]
        
        # 3. Combine
        available_tools = filtered_tools + virtual_tools

        request = LlmRequest(
            messages=state["messages"],
            tools=available_tools,
            user=state["user"],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=self.config.stream_responses,
            system_prompt=state["system_prompt"],
        )

        for mw in self.llm_middlewares:
            request = await mw.before_llm_request(request)

        response: LlmResponse
        if self.config.stream_responses:
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

        for mw in self.llm_middlewares:
            response = await mw.after_llm_response(request, response)
            
        # ALWAYS append the assistant message to state, even if just tool calls
        assistant_msg = LlmMessage(
            role="assistant", 
            content=response.content or "", # Ensure string even if None
            tool_calls=response.tool_calls
        )
        state["messages"].append(assistant_msg)
        
        # Add to conversation object too
        state["conversation"].add_message(Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        ))
             
        if response.content:
             ui_queue = state["ui_queue"]
             await ui_queue.put(UiComponent(
                rich_component=RichTextComponent(content=response.content, markdown=True),
                simple_component=SimpleTextComponent(text=response.content)
            ))

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

    def _router_analyze_response(self, state: AgentState) -> Literal["tools", "done", "get_schema", "generate_sql", "execute_sql"]:
        response = state["llm_response"]
        
        if response and response.is_tool_call():
            # Check virtual tools
            for tool_call in response.tool_calls:
                if tool_call.name == "query_schema_metadata":
                    return "get_schema"
                if tool_call.name == "generate_sql":
                    return "generate_sql"
                if tool_call.name == "execute_current_sql":
                    return "execute_sql"
            return "tools"
        return "done"

    def _router_check_limit(self, state: AgentState) -> Literal["continue", "stop"]:
        if state["tool_iterations"] < self.config.max_tool_iterations:
            return "continue"

        # Limit reached logic could be added here (logging, warning UI)
        return "stop"
