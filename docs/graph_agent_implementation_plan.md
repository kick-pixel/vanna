# GraphAgent Implementation Plan (New Module)

This document outlines the plan to create a **new** `src/vanna/core/agent/graph_agent.py` module. This module will contain the `GraphAgent` class, which implements the same public interface as `Agent` but uses **LangGraph** for internal macro-orchestration.

## Phase 1: Preparation & Dependencies

1.  **Dependency Check**: Ensure `langgraph` and `langchain-core` are installed.
2.  **File Creation**: Create `src/vanna/core/agent/graph_agent.py`.
3.  **Imports**: Import necessary types and components from `vanna.core` to ensure API compatibility.

## Phase 2: State Definition (`AgentState`)

4.  **Define `AgentState` TypedDict**:

    - This state will hold all context required for the graph execution.
    - **Fields**:
      - `request_context`: `RequestContext`
      - `user`: `User`
      - `conversation`: `Conversation`
      - `message`: `str` (User's input)
      - `messages`: `List[LlmMessage]` (History formatted for LLM)
      - `tool_schemas`: `List[ToolSchema]`
      - `system_prompt`: `str`
      - `llm_request`: `LlmRequest`
      - `llm_response`: `Optional[LlmResponse]`
      - `tool_iterations`: `int`
      - `max_tool_iterations`: `int`
      - `is_complete`: `bool`

5.  **Define UI Callback Protocol**:
    - Define an `AgentGraphConfig` (inherits from `RunnableConfig`) or simply pass a callback function in the `config` to allow nodes to emit `UiComponent`s to the `send_message` generator.

## Phase 3: `GraphAgent` Class Structure

6.  **Initialization (`__init__`)**:

    - Copy the `__init__` signature and logic from `Agent`.
    - Initialize the same components: `llm_service`, `tool_registry`, `conversation_store`, `lifecycle_hooks`, etc.
    - **New**: Compile the LangGraph in `__init__` (or lazily) and store it as `self.graph`.

7.  **`send_message` Implementation**:
    - Signature: `async def send_message(self, request_context, message, conversation_id=None) -> AsyncGenerator[UiComponent, None]`
    - **Logic**:
      - Create an `asyncio.Queue`.
      - Define a `ui_emitter` callback that puts components into the Queue.
      - Launch the graph execution (`self.graph.ainvoke`) as a background task.
      - Pass the `ui_emitter` to the graph via `config`.
      - Loop: `await queue.get()` and `yield component` until a "DONE" sentinel is received or the task completes.

## Phase 4: Graph Nodes (Method Implementation)

Implement the graph nodes as private methods within `GraphAgent`.

8.  **`_node_initialize`**:

    - Resolve user, load conversation, handle `workflow_handler`.
    - Run `before_message` hooks.

9.  **`_node_prepare_context`**:

    - Run `context_enrichers`.
    - Fetch tool schemas (`self.tool_registry.get_schemas`).
    - Build system prompt (`self.system_prompt_builder`).
    - Enhance context (`self.llm_context_enhancer`).

10. **`_node_think` (LLM Call)**:

    - Build `LlmRequest`.
    - Run `before_llm_request` middleware.
    - Call `self.llm_service.send_request` (or stream).
    - Run `after_llm_response` middleware.

11. **`_node_execute_tools`**:

    - Iterate `llm_response.tool_calls`.
    - **Core Logic**: Call `await self.tool_registry.execute(tool_call, context)`.
    - Run `before_tool` / `after_tool` hooks.
    - Update conversation history.

12. **`_node_finalize`**:
    - Run `after_message` hooks.
    - Save conversation.

## Phase 5: Graph Wiring

13. **`_build_graph`**:
    - Create `StateGraph(AgentState)`.
    - Add nodes: `initialize`, `prepare`, `think`, `tools`, `finalize`.
    - **Edges**:
      - `initialize` -> `prepare` -> `think`.
      - `think` -> **conditional** (`router`):
        - If tool calls & limit not reached -> `tools`.
        - Else -> `finalize`.
      - `tools` -> `think` (Loop back).
      - `finalize` -> `END`.

## Phase 6: Verification

14. **New Example**: Create `src/vanna/examples/graph_agent_example.py`.
    - Import `GraphAgent` instead of `Agent`.
    - Configure it identically to `openai_sqlite_example.py`.
    - Run it to verify parity.
