"""SQLite implementation of SqlRunner interface."""

import sqlite3
import pandas as pd

from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs
from vanna.core.tool import ToolContext


class SqliteRunner(SqlRunner):
    """SQLite implementation of the SqlRunner interface."""

    def __init__(self, database_path: str):
        """Initialize with a SQLite database path.

        Args:
            database_path: Path to the SQLite database file
        """
        self.database_path = database_path

    @property
    def dialect(self) -> str:
        return "SQLite"

    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        """Execute SQL query against SQLite database and return results as DataFrame.

        Args:
            args: SQL query arguments
            context: Tool execution context

        Returns:
            DataFrame with query results

        Raises:
            sqlite3.Error: If query execution fails
        """
        # Connect to the database
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()

        try:
            # Execute the query
            cursor.execute(args.sql)

            # If cursor.description is not None, it's a query that returns rows (SELECT, PRAGMA, etc.)
            if cursor.description is not None:
                # Fetch results
                rows = cursor.fetchall()
                if not rows:
                    return pd.DataFrame()
                
                # Convert rows to list of dictionaries
                results_data = [dict(row) for row in rows]
                return pd.DataFrame(results_data)
            else:
                # For non-returning queries (INSERT, UPDATE, DELETE, etc.)
                conn.commit()
                rows_affected = cursor.rowcount
                return pd.DataFrame({"rows_affected": [rows_affected]})

        finally:
            cursor.close()
            conn.close()
