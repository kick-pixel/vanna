"""
Default system prompt builder implementation with memory workflow support.

This module provides a default implementation of the SystemPromptBuilder interface
that automatically includes memory workflow instructions when memory tools are available.
"""

from typing import TYPE_CHECKING, List, Optional
from datetime import datetime

from .base import SystemPromptBuilder

if TYPE_CHECKING:
    from ..tool.models import ToolSchema
    from ..user.models import User


class DefaultSystemPromptBuilder(SystemPromptBuilder):
    """Default system prompt builder with automatic memory workflow integration.

    Dynamically generates system prompts that include memory workflow
    instructions when memory tools (search_saved_correct_tool_uses and
    save_question_tool_args) are available.
    """

    def __init__(self, base_prompt: Optional[str] = None):
        """Initialize with an optional base prompt.

        Args:
            base_prompt: Optional base system prompt. If not provided, uses a default.
        """
        self.base_prompt = base_prompt

    async def build_system_prompt(
        self, user: "User", tools: List["ToolSchema"]
    ) -> Optional[str]:
        """
        Build a system prompt with memory workflow instructions.

        Args:
            user: The user making the request
            tools: List of tools available to the user

        Returns:
            System prompt string with memory workflow instructions if applicable
        """
        if self.base_prompt is not None:
            return self.base_prompt

        # Check which memory tools are available
        tool_names = [tool.name for tool in tools]
        has_search = "search_saved_correct_tool_uses" in tool_names
        has_save = "save_question_tool_args" in tool_names
        has_text_memory = "save_text_memory" in tool_names

        # Get today's date
        today_date = datetime.now().strftime("%Y-%m-%d")

        # Base system prompt
        prompt_parts = [
            f"You are Vanna, an AI data analyst assistant created to help users with data analysis tasks. Today's date is {today_date}.",
            "",
            "Response Guidelines:",
            "- Any summary of what you did or observations should be the final step.",
            "- Use the available tools to help the user accomplish their goals.",
            "- When you execute a query, that raw result is shown to the user outside of your response so YOU DO NOT need to include it in your response. Focus on summarizing and interpreting the results.",
        ]

        if tools:
            prompt_parts.append(
                f"\nYou have access to the following tools: {', '.join(tool_names)}"
            )

        # Add memory workflow instructions based on available tools (Legacy support + Agentic Context)
        if has_search or has_save or has_text_memory:
            prompt_parts.append("\n" + "=" * 60)
            prompt_parts.append("MEMORY & SQL WORKFLOW:")
            prompt_parts.append("=" * 60)
            
            prompt_parts.append("\nYou are an Agentic Data Analyst. You have access to a set of tools to query data.")
            prompt_parts.append("When asked a question that requires database access, follow this strict workflow:")
            prompt_parts.append("")
            prompt_parts.append("1. **Lookup Schema**: Unless you already have the schema in your context, you MUST first inspect the database structure. Use `query_schema_metadata(sql=...)` (e.g., `SELECT * FROM sqlite_master` or `SELECT table_name FROM information_schema.tables`).")
            prompt_parts.append("2. **Generate SQL**: Once you have the schema context, use `generate_sql(instruction=...)` to write a valid SQL query.")
            prompt_parts.append("3. **Execute SQL**: After generating the SQL, use `execute_current_sql()` to run it and get results.")
            prompt_parts.append("4. **Analyze**: Finally, analyze the results and answer the user's question.")
            prompt_parts.append("")
            prompt_parts.append("CRITICAL: Do NOT guess table names. You MUST verify the schema first.")
            prompt_parts.append("Do NOT try to execute SQL directly with generic tools. Use the specific `execute_current_sql` action.")

        if has_search or has_save:
            prompt_parts.append("\nTOOL USAGE MEMORY (Optimization):")
            prompt_parts.append("-" * 50)
            prompt_parts.append("• You can also use `search_saved_correct_tool_uses` to find past successful queries.")
            prompt_parts.append("• After a successful analysis, use `save_question_tool_args` to remember it.")

        if has_text_memory:
            prompt_parts.extend(
                [
                    "",
                    "TEXT MEMORY (Domain Knowledge):",
                    "-" * 50,
                    "• save_text_memory: Save important context about the database, schema, or domain",
                ]
            )
            
        # Add instructions for using injected schema context
        prompt_parts.append("")
        prompt_parts.append("SCHEMA CONTEXT:")
        prompt_parts.append("If schema information is provided in the context below (labeled 'SCHEMA CONTEXT'), usage it to guide your SQL generation and tool selection. Trust the provided schema over general assumptions.")

        return "\n".join(prompt_parts)
