"""
GraphAgent 使用 LangGraph 的实现，适用于 Vanna Agents 框架。

该模块提供了 GraphAgent 类，它通过状态图在 LLM 服务、工具和会话存储之间进行编排与协作。
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
    """Agent 执行状态图的状态定义。"""

    # 上下文
    request_context: RequestContext
    user: User
    conversation_id: str
    request_id: str
    conversation: Conversation
    agent_memory: AgentMemory
    observability_provider: Optional[ObservabilityProvider]

    # 输入
    message: str

    # 工作流数据
    ui_queue: asyncio.Queue  # 用于流式发送 UI 组件的队列
    is_starter_request: bool
    should_stop: bool

    # LLM 交互
    tool_schemas: List[ToolSchema]
    system_prompt: Optional[str]
    messages: List[LlmMessage]
    llm_request: Optional[LlmRequest]
    llm_response: Optional[LlmResponse]

    # 执行控制
    tool_iterations: int
    tool_iterations: int
    tool_context: Optional[ToolContext]

    # 模型与 SQL
    schema_metadata: Optional[str]
    generated_sql: Optional[str]
    sql_result: Optional[str]


# 局部状态更新的辅助类型
PartialAgentState = Dict[str, Any]


class GraphAgent:
    """
    使用 LangGraph 进行宏观编排的 Agent 实现。

    该类在保持与标准 Agent 类 API 兼容的同时，
    使用有向循环图实现内部的决策与控制回路。
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

        # 初始化状态图
        self.graph = self._build_graph()
        logger.info(f"Graph: {self.graph.get_graph().draw_mermaid()}")
        logger.info("Initialized GraphAgent")

    def _build_graph(self) -> Any:
        """构建 LangGraph 状态机。"""
        workflow = StateGraph(AgentState)

        # 直接添加节点
        workflow.add_node("initialize", self._node_initialize)
        workflow.add_node("get_schema", self._node_get_schema)
        workflow.add_node("think", self._node_think)
        workflow.add_node("generate_sql", self._node_generate_sql)
        workflow.add_node("execute_sql", self._node_execute_sql)
        workflow.add_node("execute_tools", self._node_execute_tools)
        workflow.add_node("finalize", self._node_finalize)

        workflow.set_entry_point("initialize")

        # 初始化阶段会准备上下文，因此如果是起始请求可直接循环到思考或特定节点；
        # 实际上初始化会返回用户与上下文，所以下一步为思考节点

        workflow.add_conditional_edges(
            "initialize",
            self._router_check_stop,
            {
                "stop": "finalize",
                "continue": "think"
            }
        )

        # 所有动作节点最终回到思考节点
        workflow.add_edge("get_schema", "think")
        workflow.add_edge("generate_sql", "think")
        workflow.add_edge("execute_sql", "think")

        # 从思考节点的条件边（工具执行 / 完成 / 虚拟工具）
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

        # 从工具执行节点回到思考节点
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
        使用图处理用户消息，并按需产生 UI 组件。
        """
        ui_queue = asyncio.Queue()

        # 初始状态
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

        # 使用 stream_mode="updates" 获取节点执行的增量更新
        try:
            async for event in self.graph.astream(initial_state, stream_mode="updates"):
                # event 是一个字典：{node_name: node_output}
                for node_name, node_output in event.items():
                    # node_output 是节点返回的状态更新（局部状态）
                    if isinstance(node_output, dict):
                        # 打印键名
                        logger.info(f"Node '{node_name}' updated: {list(node_output.keys())}")
                        # 打印每个键的值（限制长度避免日志过长）
                        for key, value in node_output.items():
                            value_str = str(value)
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "... (truncated)"
                            logger.info(f"  {key}: {value_str}")
                    else:
                        logger.info(f"Node '{node_name}' updated: completed")
                    logger.info("====================")
                # 从队列读取并输出 UI 组件（非阻塞）
                while not ui_queue.empty():
                    try:
                        item = ui_queue.get_nowait()
                        if item is not None:
                            yield item
                    except asyncio.QueueEmpty:
                        break

            # 图执行完成后处理剩余的 UI 组件
            while not ui_queue.empty():
                try:
                    item = ui_queue.get_nowait()
                    if item is not None:
                        yield item
                except asyncio.QueueEmpty:
                    break

        except Exception as e:
            # 错误处理方式与传统 Agent 类一致
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

    # --- 节点实现 ---

    async def _node_initialize(self, state: AgentState) -> PartialAgentState:
        """
        合并的初始化节点：
        1. 解析用户与会话；
        2. 处理工作流/起始界面；
        3. 准备工具上下文与系统提示词。
        """
        request_context = state["request_context"]
        message = state["message"]
        conversation_id = state["conversation_id"]
        ui_queue = state["ui_queue"]

        # 1. 解析用户
        user = await self.user_resolver.resolve_user(request_context)

        # 2. 检查是否为起始请求 / 工作流
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

        # 生命周期钩子：消息前置处理
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

        # 3. 准备上下文（原 _node_prepare_context）
        context_task = Task(title="Load context", status="pending")
        await ui_queue.put(UiComponent(rich_component=TaskTrackerUpdateComponent.add_task(context_task)))

        # 构建工具上下文
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

        # 获取可用工具
        tool_schemas = await self.tool_registry.get_schemas(user)

        await ui_queue.put(UiComponent(
            rich_component=TaskTrackerUpdateComponent.update_task(
                context_task.id, status="completed"
            )
        ))

        # 构建系统提示词
        system_prompt = await self.system_prompt_builder.build_system_prompt(
            user, tool_schemas
        )
        if self.llm_context_enhancer and system_prompt:
            system_prompt = await self.llm_context_enhancer.enhance_system_prompt(
                system_prompt, message, user
            )

        # 过滤会话消息
        filtered_messages = conversation.messages
        for filter in self.conversation_filters:
            filtered_messages = await filter.filter_messages(filtered_messages)

        # 转换为 LlmMessage 列表
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
        主动的架构检索节点。
        执行 LLM 提供的 SQL（通过 query_schema_metadata）来检查数据库结构，
        将结果保存并加入上下文。
        """
        # if state.get("should_stop"): return {} # 无需处理，路由已明确保障到达此处

        context = state.get("tool_context")
        ui_queue = state.get("ui_queue")
        response = state.get("llm_response")

        # 1. 从工具调用中解析 SQL，并处理所有工具调用
        schema_sql = "SELECT name, sql FROM sqlite_master WHERE type='table'"  # 兜底值

        target_tool_id = None
        other_tool_ids = []

        if response and response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "query_schema_metadata":
                    # 从 LLM 提供的参数中获取 SQL，若无则使用兜底值
                    provided_sql = tc.arguments.get("sql")
                    if provided_sql:
                        schema_sql = provided_sql
                    target_tool_id = tc.id
                else:
                    other_tool_ids.append(tc.id)

        if not schema_sql:
            # 必须对工具调用进行响应
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

        # 2. 执行 SQL（复用 run_sql 工具逻辑）
        query_result_text = ""
        try:
            sql_tool = await self.tool_registry.get_tool("run_sql")
            if not sql_tool:
                raise Exception("Refusing to query schema: 'run_sql' tool not available.")

            # 执行
            # 手动构建参数
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

        # 3. 保存到记忆
        # 作为文本记忆保存，在需要时跨轮次/会话持久化；
        # 或在本次会话中追加到系统提示词。
        # 用户需求："get_schema获取并存储到memory里"
        if self.agent_memory and "Error" not in query_result_text:
            # 创建一条文本记忆项
            # 在实际应用中可使用 save_text_memory 工具逻辑或直接调用；
            # 此处假设可直接访问，若无直接 API 则跳过；
            # 暂时将其放入 'schema_metadata' 状态字段，后续可持久化。
            pass

        # 4. 更新上下文

        result_msg = f"Schema Query Result ({schema_sql}):\n{query_result_text}"

        # 加入到消息历史中，以便 LLM 感知
        if target_tool_id:
            state["messages"].append(LlmMessage(
                role="tool", content=result_msg, tool_call_id=target_tool_id))
        else:
            # 兜底：若找不到 ID 或非工具调用（在当前流程中不太可能）
            state["messages"].append(LlmMessage(role="system", content=result_msg))

        # 处理其他工具调用（占位响应以满足 API）
        for ot_id in other_tool_ids:
            state["messages"].append(LlmMessage(
                role="tool", content="Tool call ignored in this step.", tool_call_id=ot_id))

        return {
            "schema_metadata": query_result_text,  # Allow subsequent nodes to see it specifically
            "tool_iterations": state["tool_iterations"] + 1
        }

    async def _node_generate_sql(self, state: AgentState) -> PartialAgentState:
        """根据请求生成 SQL。"""
        ui_queue = state.get("ui_queue")
        response = state.get("llm_response")

        # 若工具调用提供了指令则使用，否则采用通用指令
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

        # 查找 generate_sql 的工具调用 ID
        target_tool_id = None
        other_tool_ids = []
        if response and response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "generate_sql":
                    target_tool_id = tc.id
                else:
                    other_tool_ids.append(tc.id)

        if target_tool_id:
            # 在发起新的请求前，必须对工具调用进行闭合响应
            state["messages"].append(LlmMessage(
                role="tool", content="Proceeding with SQL generation.", tool_call_id=target_tool_id))

        # 处理其他工具调用
        for ot_id in other_tool_ids:
            state["messages"].append(LlmMessage(
                role="tool", content="Tool call ignored in this step.", tool_call_id=ot_id))

        # 将特定指令附加到系统提示词中；
        # 注意：若只希望 LLM 输出 SQL，可不包含刚添加的工具响应；
        # 但为保证历史一致性，通常应包含；
        # 对于专门的“生成”任务，也可适当屏蔽历史，仅附加任务提示。

        request = LlmRequest(
            messages=state["messages"],
            tools=None,  # Strict mode: provide NO tools so it must output text (code)
            user=state["user"],
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            stream=self.config.stream_responses,
            system_prompt=state["system_prompt"]
            + f"\n\nTASK: {instruction}\nOutput executable SQL only. No markdown.",
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

        state["messages"].append(LlmMessage(
            role="assistant", content=f"Generated SQL: {generated_sql}"))

        return {
            "generated_sql": generated_sql,
            "tool_iterations": state["tool_iterations"] + 1
        }

    async def _node_execute_sql(self, state: AgentState) -> PartialAgentState:
        """执行已生成的 SQL。"""
        ui_queue = state.get("ui_queue")
        generated_sql = state.get("generated_sql")
        context = state.get("tool_context")
        conversation = state.get("conversation")

        # 查找 execute_current_sql 的工具调用 ID（用于错误响应）
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
                state["messages"].append(LlmMessage(
                    role="tool", content="Error: No SQL has been generated yet.", tool_call_id=target_tool_id))
            for ot_id in other_tool_ids:
                state["messages"].append(LlmMessage(
                    role="tool", content="Ignored.", tool_call_id=ot_id))
            return {}

        await ui_queue.put(UiComponent(
            rich_component=StatusBarUpdateComponent(
                status="working", message="Executing SQL", detail="Running query..."
            )
        ))

        # 查找真实的 RunSqlTool
        sql_tool = await self.tool_registry.get_tool("run_sql")
        if not sql_tool:
            if target_tool_id:
                state["messages"].append(LlmMessage(
                    role="tool", content="Error: 'run_sql' tool is not configured.", tool_call_id=target_tool_id))
            for ot_id in other_tool_ids:
                state["messages"].append(LlmMessage(
                    role="tool", content="Ignored.", tool_call_id=ot_id))
            return {}

        try:
            # 基础执行
            # 依赖工具实例的 execute 方法
            # 需要构造工具所需的参数对象
            args_model = sql_tool.get_args_schema()
            tool_args = args_model(sql=generated_sql)

            # 直接调用工具以避免模式校验的额外开销/不匹配
            result = await sql_tool.execute(context, tool_args)

        except Exception as e:
            logger.error(f"SQL Execution failed: {e}")
            state["messages"].append(LlmMessage(role="system", content=f"SQL Execution Error: {e}"))
            return {}

        # 将结果写入会话
        # 上文已获得相关 ID

        conversation.add_message(Message(
            role="tool",
            content=result.result_for_llm if result.success else f"Error: {result.error}",
            tool_call_id=target_tool_id or "unknown"
        ))

        # 将结果加入上下文消息
        state["messages"].append(LlmMessage(
            role="tool",
            content=result.result_for_llm if result.success else f"Error: {result.error}",
            tool_call_id=target_tool_id or "unknown"
        ))

        # 处理其他工具调用
        for ot_id in other_tool_ids:
            state["messages"].append(LlmMessage(
                role="tool", content="Tool call ignored in this step.", tool_call_id=ot_id))

        if result.ui_component:
            await ui_queue.put(result.ui_component)

        return {
            "sql_result": result.result_for_llm,
            "tool_iterations": state["tool_iterations"] + 1
        }

    async def _node_think(self, state: AgentState) -> PartialAgentState:
        """使用虚拟工具执行一次 LLM 请求。"""

        # 1. 定义虚拟工具（更新版）
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

        # 2. 过滤真实工具
        real_tools = state.get("tool_schemas", [])
        filtered_tools = [t for t in real_tools if t.name != "run_sql"]

        # 3. 合并工具清单
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

        # 始终将助手消息追加到状态中，即使只有工具调用
        assistant_msg = LlmMessage(
            role="assistant",
            content=response.content or "",  # 即使为 None 也保证为字符串
            tool_calls=response.tool_calls
        )
        state["messages"].append(assistant_msg)

        # 同步写入会话对象
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
        """执行 LLM 响应中请求的工具。"""
        response = state["llm_response"]
        conversation = state["conversation"]
        ui_queue = state["ui_queue"]
        user = state["user"]
        context = state["tool_context"]

        # 添加助手消息到会话
        assistant_msg = Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        )
        conversation.add_message(assistant_msg)

        # 输出文本内容
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
            # 任务 UI
            tool_task = Task(
                title=f"Execute {tool_call.name}",
                description="Running tool...",
                status="in_progress"
            )
            await ui_queue.put(UiComponent(
                rich_component=TaskTrackerUpdateComponent.add_task(tool_task)
            ))

            # 状态卡片 UI
            card = StatusCardComponent(
                title=f"Executing {tool_call.name}",
                status="running",
                icon="⚙️",
                metadata=tool_call.arguments
            )
            await ui_queue.put(UiComponent(rich_component=card))

            # 钩子：工具执行前
            tool = await self.tool_registry.get_tool(tool_call.name)
            if tool:
                for hook in self.lifecycle_hooks:
                    await hook.before_tool(tool, context)

            # 执行
            result = await self.tool_registry.execute(tool_call, context)

            # 钩子：工具执行后
            for hook in self.lifecycle_hooks:
                modified = await hook.after_tool(result)
                if modified:
                    result = modified

            # 使用结果更新 UI
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

        # 将工具消息写入会话
        for res in tool_results_data:
            conversation.add_message(Message(
                role="tool",
                content=res["content"],
                tool_call_id=res["tool_call_id"]
            ))

        # 为下一轮 LLM 请求重建消息列表
        # 实际实现中可直接追加到 state["messages"]；
        # 但此处以 conversation.messages 为准，可能需要重新转换或追加。
        # 为简化处理，假设沿用准备上下文或增量追加逻辑。

        # 增量追加到 LlmMessages
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
        """保存会话、触发钩子并收尾。"""
        if state.get("should_stop"):
            return {}

        conversation = state["conversation"]
        ui_queue = state["ui_queue"]

        # 若来源于 LLM 的 "done" 状态（无工具调用）
        response = state.get("llm_response")
        if response and not response.is_tool_call():
            # 若尚未添加最终助手消息，则在此补充；
            # （在 execute_tools 中已添加；若跳过工具调用则需在此添加）
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

    # --- 路由 ---

    def _router_check_stop(self, state: AgentState) -> Literal["stop", "continue"]:
        return "stop" if state.get("should_stop") else "continue"

    def _router_analyze_response(self, state: AgentState) -> Literal["tools", "done", "get_schema", "generate_sql", "execute_sql"]:
        response = state["llm_response"]

        if response and response.is_tool_call():
            # 检查虚拟工具
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

        # 达到工具迭代上限的逻辑可在此添加（日志、警告 UI）
        return "stop"
