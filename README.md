# SQL-Generator-using-AI-Agents
This Streamlit application uses a LangChain-powered AI agent (Claude 3 Sonnet) to inspect an SQLite database schema, generate valid SQL queries from natural language questions, execute them, and display results as tables or interactive charts.

🚀 **Features**
- **Natural Language to SQL via AI Agent**: Uses Anthropic Claude through LangChain agents to translate plain English into fully-formed SQLite `SELECT` queries.
- **Schema-Aware Querying**: Agent inspects database schema before generating queries to ensure correct table and column references.
- **SQLite Execution**: Runs generated SQL queries directly on a local SQLite database.
- **Dynamic Charting**: Supports bar, line, area, and scatter plots with user-selectable axes.
- **One-Time Initialization**: AI agent and tools are loaded once per session for faster performance.
- **Error Handling**: Graceful handling of invalid SQL, missing numeric columns for charts, or empty result sets.
- **Interactive UI**: Built entirely with Streamlit for an intuitive, browser-based interface.

🛠 **How It Works**
1. User enters a query in plain English (e.g., _"What’s the average total compensation by department?"_).
2. The app uses the AI agent with two tools:
   - **inspect_schema**: Retrieves tables and columns from the SQLite database.
   - **execute_sql**: Executes read-only SQL queries and returns results.
3. The agent outputs a single SQL `SELECT` statement (no explanations, only SQL).
4. The app runs the query and displays:
   - A results table (with CSV download option)
   - An optional chart with selectable type and axes.
5. Users can switch between table view and different chart types dynamically.

📊 **Chart Options**
- **Bar / Line / Area**: Supports multiple numeric series.
- **Scatter**: Requires a numeric Y-axis column.
- Fully interactive charts powered by Altair.

📂 **Configuration**
- **Database path**: Set via `EMP_DB_PATH` environment variable (defaults to `/mnt/data/employees.db`).
- **API key file**: Set via `ENV_FILE` (defaults to `key_claude.env`).

## Contact
For questions or feedback, contact **Rishi Kora** at **[korarishi@gmail.com](mailto:korarishi@gmail.com)**.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

