import os
import re
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
import altair as alt  # NEW: charting

DB_PATH = os.environ.get("EMP_DB_PATH", "/mnt/data/employees.db")
ENV_FILE = os.environ.get("ENV_FILE", "key_claude.env")
load_dotenv(ENV_FILE)

def get_db_schema() -> str:
    if not os.path.exists(DB_PATH):
        return f"Database not found at {DB_PATH}"
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # List tables
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [r[0] for r in cur.fetchall()]
        lines = []
        for t in tables:
            # Columns
            cur.execute(f"PRAGMA table_info('{t}');")
            cols = cur.fetchall()  # cid, name, type, notnull, dflt_value, pk
            col_desc = ", ".join([f"{c[1]} {c[2]}" for c in cols])
            lines.append(f"TABLE {t}: {col_desc}")
        conn.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading schema: {e}"

def run_sql_query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()

# =====================================================
# 3) Build Tools
# =====================================================
def make_tools():
    return [
        Tool(
            name="inspect_schema",
            func=lambda _: get_db_schema(),
            description=(
                "Return the SQLite database schema: tables and columns. "
                "Use this BEFORE writing SQL so you reference correct names."
            )
        ),
        Tool(
            name="execute_sql",
            func=lambda sql: run_sql_query(sql).to_csv(index=False),
            description=(
                "Execute a READ-ONLY SQL query against the database and return CSV. "
                "Only use for SELECT queries. If non-SELECT is needed, explain why and stop."
            )
        )
    ]

# =====================================================
# 4) Initialize LLM + Agent ONCE
#    - spinner appears only on first load
#    - heavy objects are cached
# =====================================================
@st.cache_resource(show_spinner=False)
def init_agent_once():
    # This inner function does the heavy work
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        st.warning("No ANTHROPIC_API_KEY found in environment. The app will still render, but the agent won't run.")

    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
    tools = make_tools()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )
    return agent

def get_agent_with_one_time_spinner():
    # Ensure we show the spinner JUST ONCE per session
    if "_did_init_spinner" not in st.session_state:
        with st.spinner("Booting AI agent and tools... (one-time setup)"):
            agent = init_agent_once()
        st.session_state["_did_init_spinner"] = True
        return agent
    else:
        # No spinner after the first time
        return init_agent_once()

# =====================================================
# 5) SQL extraction helper
# =====================================================
SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

def extract_sql(text: str) -> str:
    if not text:
        raise ValueError("Empty response while extracting SQL.")
    m = SQL_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip(" ;\n\r\t")
    # fallback: naive attempt to find SELECT ... line
    sel = re.search(r"(?is)(select\s+.+)", text)
    if sel:
        return sel.group(1).strip(" ;\n\r\t")
    raise ValueError("No SQL block found in the model's response.")

# =====================================================
# 6) Streamlit UI
# =====================================================
st.set_page_config(page_title="SQL Generator Agent", layout="wide")
st.title("SQL Generator using AI Agents")
st.caption("Ask a question about the SQLite database; the agent will inspect schema, write SQL, and run it.")

# Show schema for transparency
with st.expander("🔎 Show database schema"):
    st.code(get_db_schema())

question = st.text_area(
    "Your question",
    placeholder="e.g., What's the average total compensation by department?",
    height=100
)
col1, col2 = st.columns([1,1])
run_btn = col1.button("Generate SQL & Run", use_container_width=True)
clear_btn = col2.button("Clear", use_container_width=True)

if clear_btn:
    st.experimental_rerun()

if run_btn and question.strip():
    agent = get_agent_with_one_time_spinner()  # spinner only first time

    st.status("Generating SQL and running the query...")
    try:
        prompt = (
            "You are a helpful data analyst. You have two tools: 'inspect_schema' and 'execute_sql'. "
            "First, call inspect_schema to see tables/columns. Then write a single, runnable SQLite SELECT query. "
            "Return your final answer as a fenced SQL block only. Do not explain.\n\n"
            f"User question: {question}"
        )

        full_response = agent.run(prompt)

        # Extract SQL
        try:
            sql_query = extract_sql(full_response)
        except Exception as e:
            st.error(f"Error extracting SQL: {e}")
            st.text_area("Model Response (for debugging)", value=str(full_response), height=200)
            st.stop()

        # Execute safely (read-only expectation)
        try:
            df = run_sql_query(sql_query)
        except Exception as e:
            st.error(f"SQL execution failed: {e}")
            st.code(sql_query, language="sql")
            st.stop()

        st.success("Done! Here's your result:")
        st.code(sql_query, language="sql")

        # -----------------------------
        # TABLE: Show results
        # -----------------------------
        st.subheader("Results (Table)")
        st.dataframe(df, use_container_width=True)

        # Optional: download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="result.csv", mime="text/csv")

        # -----------------------------
        # CHART: User-selectable options
        # -----------------------------
        st.subheader("Visualize Results")
        if df.empty:
            st.info("No rows returned, so there’s nothing to chart.")
        else:
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area", "Scatter"], index=0)
            x_axis = st.selectbox("X-axis", all_cols, index=0)

            if chart_type == "Scatter":
                # single numeric Y for scatter
                if not numeric_cols:
                    st.warning("No numeric columns available for Y-axis.")
                else:
                    y_axis = st.selectbox("Y-axis (numeric)", numeric_cols, index=0)

                    chart = alt.Chart(df).mark_circle(size=60).encode(
                        x=alt.X(x_axis, sort=None),
                        y=alt.Y(y_axis, type="quantitative"),
                        tooltip=all_cols
                    ).interactive()

                    st.altair_chart(chart, use_container_width=True)

            else:
                # multi-select Y for bar/line/area
                if not numeric_cols:
                    st.warning("No numeric columns available for Y-axis.")
                else:
                    default_y = [numeric_cols[0]] if numeric_cols else []
                    y_axes = st.multiselect("Y-axis column(s) (numeric)", numeric_cols, default=default_y)

                    if not y_axes:
                        st.info("Pick at least one numeric Y-axis column to draw the chart.")
                    else:
                        # Fold to long format for multiple series
                        base = alt.Chart(df).transform_fold(
                            y_axes, as_=["Series", "Value"]
                        )

                        mark_map = {"Bar": "bar", "Line": "line", "Area": "area"}
                        mark = mark_map[chart_type]

                        if mark == "bar":
                            chart = base.mark_bar().encode(
                                x=alt.X(x_axis, sort=None),
                                y=alt.Y("Value:Q"),
                                color=alt.Color("Series:N"),
                                tooltip=[x_axis, "Series:N", "Value:Q"]
                            ).interactive()
                        elif mark == "line":
                            chart = base.mark_line(point=True).encode(
                                x=alt.X(x_axis, sort=None),
                                y=alt.Y("Value:Q"),
                                color=alt.Color("Series:N"),
                                tooltip=[x_axis, "Series:N", "Value:Q"]
                            ).interactive()
                        else:  # area
                            chart = base.mark_area(opacity=0.6).encode(
                                x=alt.X(x_axis, sort=None),
                                y=alt.Y("Value:Q"),
                                color=alt.Color("Series:N"),
                                tooltip=[x_axis, "Series:N", "Value:Q"]
                            ).interactive()

                        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
