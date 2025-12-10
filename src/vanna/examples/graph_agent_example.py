import importlib
import os
import sys
import asyncio
from vanna.core.agent.graph_agent import GraphAgent
from vanna.core.registry import ToolRegistry
from vanna.core.user.models import User
from vanna.core.user.resolver import UserResolver
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.sqlite import SqliteRunner
from vanna.integrations.local.agent_memory import DemoAgentMemory
from vanna.core.user import RequestContext

from vanna.tools import (
    RunSqlTool,
    VisualizeDataTool,
    LocalFileSystem,
)


def ensure_env() -> None:
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv

        # Load from local .env without overriding existing env
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
    else:
        print(
            "[warn] python-dotenv not installed; skipping .env load. Install with: pip install python-dotenv"
        )

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "[error] OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )
        sys.exit(1)


async def main() -> None:
    ensure_env()

    # Configure your LLM
    llm = OpenAILlmService(
        model=os.getenv("OPENAI_MODEL", "qwen-plus"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )

    # Get the path to the Chinook database
    database_path = os.path.join(
        os.path.dirname(__file__), "Chinook.sqlite"
    )
    database_path = os.path.abspath(database_path)

    if not os.path.exists(database_path):
        print(f"[error] Chinook database not found at {database_path}")
        print(
            "Please download it with: curl -o Chinook.sqlite https://vanna.ai/Chinook.sqlite"
        )
        sys.exit(1)

    # Configure your database

    sqlite_runner = SqliteRunner(database_path=database_path)
    file_system = LocalFileSystem(working_directory="./data")
    sql_tool = RunSqlTool(sql_runner=sqlite_runner, file_system=file_system)
    tool_registry = ToolRegistry()
    # Configure your agent memory
    agent_memory = DemoAgentMemory(max_items=1000)

    # Register visualization tool if available
    try:
        viz_tool = VisualizeDataTool(file_system=file_system, )
        tool_registry.register_local_tool(viz_tool, access_groups=['admin', 'user'])
    except ImportError:
        pass  # Visualization tool not available

    # Configure user authentication

    class SimpleUserResolver(UserResolver):
        def __init__(self, cookie_name: str = "vanna_email"):
            self.cookie_name = cookie_name

        async def resolve_user(self, request_context: RequestContext) -> User:
            user_email = request_context.get_cookie(self.cookie_name) or 'guest@example.com'
            group = 'admin' if user_email == 'admin@example.com' else 'user'
            return User(id=user_email, email=user_email, group_memberships=[group])

    user_resolver = SimpleUserResolver()

    tool_registry.register_local_tool(sql_tool, access_groups=['admin', 'user'])
    tool_registry.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin', 'user'])
    tool_registry.register_local_tool(
        SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
    tool_registry.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])

    # Use GraphAgent instead of Agent
    agent = GraphAgent(
        llm_service=llm,
        tool_registry=tool_registry,
        user_resolver=user_resolver,
        agent_memory=agent_memory
    )

    # Simulate a logged-in demo user via cookie-based resolver
    request_context = RequestContext(
        cookies={user_resolver.cookie_name: "demo-user@example.com"},
        metadata={"demo": True},
        remote_addr="127.0.0.1",
    )
    conversation_id = "sqlite-graph-demo"

    # Sample queries to demonstrate different capabilities
    sample_questions = [
        "What tables are in this database?",
        # "Show me the first 5 customers with their names"
    ]

    print("\n" + "=" * 60)
    print("GraphAgent SQLite Database Assistant Demo")
    print("=" * 60)

    for i, question in enumerate(sample_questions, 1):
        print(f"\n--- Question {i}: {question} ---")

        async for component in agent.send_message(
            request_context=request_context,
            message=question,
            conversation_id=conversation_id,
        ):
            # Handle different component types
            if hasattr(component, "simple_component") and component.simple_component:
                if hasattr(component.simple_component, "text"):
                    print("Assistant (Simple):", component.simple_component.text)
            elif hasattr(component, "rich_component") and component.rich_component:
                if hasattr(component.rich_component, "content") and component.rich_component.content:
                    print("Assistant (Rich):", component.rich_component.content)
                elif hasattr(component.rich_component, "title"):
                    print(
                        f"Assistant (Card): [{component.rich_component.status}] {component.rich_component.title}")

        print()  # Add spacing between questions

    print("\n" + "=" * 60)
    print("Demo complete! successfully queried the database.")
    print("=" * 60)

    # Run the server
    # server = VannaFastAPIServer(agent)
    # server.run()


if __name__ == "__main__":
    asyncio.run(main())
