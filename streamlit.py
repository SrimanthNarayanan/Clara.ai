# app.py

import streamlit as st
import pandas as pd
import re
import google.generativeai as genai
import time
from collections import defaultdict

# --- Imports for HTML & PDF Generation ---
import markdown
from xhtml2pdf import pisa
from io import BytesIO

# --- Standard Imports ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import warnings
import pyodbc
import mysql.connector
import psycopg2
import snowflake.connector
import matplotlib.pyplot as plt
import seaborn as sns

# HIDE THE PANDAS USERWARNING
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --- Page Configuration ---
st.set_page_config(
    page_title="Analytixhub.ai",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 🤫 Secrets and Configuration ---
GEMINI_API_KEY = "AIzaSyDjQB5Nc0nNRz1bA7uEtMwhg0Z9OAfQp4c" # Replace with your key
SENDER_EMAIL = "narayanannarayanan15644@gmail.com"
SENDER_PASSWORD = "cwqe byra cbnl kazs"

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please check your API key. Error: {e}")

# --------------------------------------------------------------------------------
# --- Helper Functions ---
# --------------------------------------------------------------------------------

def create_html_pdf(html_content: str, question: str) -> bytes:
    """Generates a PDF from an HTML string using xhtml2pdf."""
    css_style = """
    <style>
        @page {
            size: a4 portrait;
            margin: 1.5cm;
        }
        h1, h2, h3, strong { color: #1E3A8A; font-family: "Helvetica", sans-serif; }
        h1 { font-size: 20pt; }
        h2 { font-size: 16pt; }
        body { font-family: "Helvetica", sans-serif; font-size: 11pt; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #DBEAFE;
            font-weight: bold;
        }
    </style>
    """
    
    full_html = f"""
    <html>
    <head>
    {css_style}
    </head>
    <body>
        <h1>Analytix Hub Report</h1>
        <h2>Query: "{question}"</h2>
        <hr/>
        {html_content}
    </body>
    </html>
    """
    
    result = BytesIO()
    pdf = pisa.CreatePDF(BytesIO(full_html.encode("UTF-8")), dest=result)
    
    if not pdf.err:
        return result.getvalue()
    else:
        st.error(f"PDF creation error: {pdf.err}")
        return b""

# --------------------------------------------------------------------------------
# --- Core Logic Functions ---
# --------------------------------------------------------------------------------

def get_db_schema(conn, db_type):
    try:
        if db_type == "Snowflake": return get_snowflake_schema(conn)
        elif db_type == "SQL Server": return get_sql_server_schema(conn)
        elif db_type == "MySQL": return get_mysql_schema(conn)
        elif db_type == "PostgreSQL": return get_postgresql_schema(conn)
        else:
            st.error(f"Schema retrieval not supported for {db_type}")
            return [], []
    except Exception as e:
        st.error(f"Failed to retrieve schema for {db_type}: {e}")
        return [], []

def get_snowflake_schema(conn):
    cursor = conn.cursor()
    db_name = conn.database
    cols_sql = f"SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM {db_name}.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA' ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION;"
    cursor.execute(cols_sql)
    cols = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    return cols, []

def get_sql_server_schema(conn):
    tables_query = "SELECT s.name AS TABLE_SCHEMA, t.name AS TABLE_NAME, c.name AS COLUMN_NAME, ty.name AS DATA_TYPE FROM sys.tables t JOIN sys.schemas s ON t.schema_id = s.schema_id JOIN sys.columns c ON t.object_id = c.object_id JOIN sys.types ty ON c.user_type_id = ty.user_type_id ORDER BY s.name, t.name, c.column_id"
    cols_df = pd.read_sql(tables_query, conn)
    fk_query = "SELECT OBJECT_SCHEMA_NAME(fk.parent_object_id) AS TABLE_SCHEMA, OBJECT_NAME(fk.parent_object_id) AS TABLE_NAME, COL_NAME(fk.parent_object_id, fkc.parent_column_id) AS COLUMN_NAME, OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS REF_SCHEMA, OBJECT_NAME(fk.referenced_object_id) AS REF_TABLE, COL_NAME(fk.referenced_object_id, fkc.referenced_column_id) AS REF_COLUMN FROM sys.foreign_keys fk JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id"
    fk_df = pd.read_sql(fk_query, conn)
    return cols_df.to_dict('records'), fk_df.to_dict('records')

def get_mysql_schema(conn):
    db_name = conn.database
    cols_sql = f"SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{db_name}' ORDER BY TABLE_NAME, ORDINAL_POSITION;"
    cols_df = pd.read_sql(cols_sql, conn)
    fk_sql = f"SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_SCHEMA AS REF_SCHEMA, REFERENCED_TABLE_NAME AS REF_TABLE, REFERENCED_COLUMN_NAME AS REF_COLUMN FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = '{db_name}' AND REFERENCED_TABLE_NAME IS NOT NULL;"
    fk_df = pd.read_sql(fk_sql, conn)
    return cols_df.to_dict('records'), fk_df.to_dict('records')

def get_postgresql_schema(conn):
    cols_sql = "SELECT table_schema as TABLE_SCHEMA, table_name as TABLE_NAME, column_name as COLUMN_NAME, data_type as DATA_TYPE FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;"
    cols_df = pd.read_sql(cols_sql, conn)
    fk_sql = "SELECT tc.table_schema AS TABLE_SCHEMA, tc.table_name AS TABLE_NAME, kcu.column_name AS COLUMN_NAME, ccu.table_schema AS REF_SCHEMA, ccu.table_name AS REF_TABLE, ccu.column_name AS REF_COLUMN FROM information_schema.table_constraints AS tc JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema WHERE tc.constraint_type = 'FOREIGN KEY';"
    fk_df = pd.read_sql(fk_sql, conn)
    return cols_df.to_dict('records'), fk_df.to_dict('records')

def pick_relevant_tables(columns: list[dict], prompt: str, max_tables: int = 10) -> set[str]:
    if not prompt: return set()
    words = set(re.findall(r"[A-Za-z0-9_]+", prompt.lower()))
    scores = defaultdict(float)
    for c in columns:
        schema, table = c.get('TABLE_SCHEMA', 'PUBLIC'), c.get('TABLE_NAME', '')
        if not table: continue
        full_table = f"{schema}.{table}"
        if table.lower() in words: scores[full_table] += 10.0
        for word in words:
            if word in table.lower(): scores[full_table] += 5.0
        col_name = c.get('COLUMN_NAME', '').lower()
        if col_name in words: scores[full_table] += 3.0
        else:
            for word in words:
                if word in col_name: scores[full_table] += 1.0
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tables = {t for t, s in ranked if s > 0}
    return set(list(top_tables)[:max_tables])

def build_schema_card(columns: list[dict], fks: list[dict], tables_filter: set[str] | None = None) -> str:
    table_cols = defaultdict(list)
    for c in columns:
        full_table_name = f"{c.get('TABLE_SCHEMA', 'PUBLIC')}.{c['TABLE_NAME']}"
        if tables_filter and full_table_name not in tables_filter: continue
        table_cols[full_table_name].append(f"{c['COLUMN_NAME']}:{c['DATA_TYPE']}")
    fk_lines = []
    for fk in fks:
        parent_full = f"{fk.get('TABLE_SCHEMA', 'PUBLIC')}.{fk['TABLE_NAME']}"
        ref_full = f"{fk.get('REF_SCHEMA', 'PUBLIC')}.{fk['REF_TABLE']}"
        if tables_filter and (parent_full not in tables_filter or ref_full not in tables_filter): continue
        fk_lines.append(f"- {parent_full}.{fk['COLUMN_NAME']} -> {ref_full}.{fk['REF_COLUMN']}")
    lines = ["Schema (relevant tables and columns):"]
    for table, cols in sorted(table_cols.items()):
        lines.append(f"* {table}: {', '.join(cols)}")
    if fk_lines:
        lines.append("\nRelationships:")
        lines.extend(fk_lines)
    return "\n".join(lines)

def generate_sql_with_gemini(question: str, schema_card: str, db_type: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""You are an expert {db_type} database analyst. Generate a precise SELECT query. STRICT REQUIREMENTS: - Return EXACTLY one SELECT statement. - Use ONLY SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT clauses. - NO DML or DDL statements. - Use proper JOINs based on the relationships shown. AVAILABLE SCHEMA: {schema_card} BUSINESS QUESTION: "{question}" Generate only the SQL SELECT statement (no explanations, no markdown backticks):"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error during SQL generation: {e}"); return ""

def generate_insight_with_gemini(df: pd.DataFrame, question: str) -> str:
    if df.empty: return "The query returned no data."
    model = genai.GenerativeModel("gemini-2.5-flash")
    data_summary = df.head(20).to_csv(index=False)
    if len(data_summary) > 4000: data_summary = data_summary[:4000] + "\n... (data truncated)"
    prompt = f"""You are a business analyst. Analyze the following data which was generated to answer the question: "{question}" Analyze the data and provide concise, business-ready insights in markdown format. Focus on key trends, outliers, and actionable recommendations. Respond in bullet points and use tables if appropriate. DATA: {data_summary}"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating insights: {e}"); return "Failed to generate insights."

def generate_visualization_code(df: pd.DataFrame, question: str) -> str:
    """Acts as a data analyst to generate code for two meaningful, side-by-side charts."""
    if df.empty or len(df.columns) < 2: return "# Not enough data to generate meaningful visualizations."
    model = genai.GenerativeModel("gemini-2.5-flash")
    data_summary = df.head(20).to_csv(index=False)
    
    prompt = f"""You are a senior data analyst. Your goal is to create two distinct, insightful visualizations from a given dataset.
    
    USER QUESTION: "{question}"
    
    DATA SUMMARY (CSV format):
    {data_summary}
    
    ---
    YOUR TASK:
    1.  **Analyze**: Look at the data columns, types, and the user's question. Decide on the two most insightful and different chart types that tell a clear story. Examples: a bar chart for rankings, a line chart for trends, a pie chart for distribution (use only for < 6 categories), a scatter plot for correlations.
    2.  **Generate Code**: Write Python code to create these two visualizations side-by-side on a single figure.
    
    STRICT CODE REQUIREMENTS:
    -   Create a figure with two subplots: `fig, axes = plt.subplots(1, 2, figsize=(14, 6))`.
    -   Use the pre-defined DataFrame named `df`.
    -   Plot the first chart on `axes[0]` and the second on `axes[1]`.
    -   Each subplot MUST have a clear, descriptive title using `axes[i].set_title(...)` and labels using `axes[i].set_xlabel(...)` etc.
    -   If a plot has overlapping x-axis labels, rotate and align them. Use this exact pattern: `plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")`. Do NOT use `tick_params` for rotation.
    -   After all plotting, call `plt.tight_layout()` to prevent overlap.
    -   CRITICAL: DO NOT call `plt.show()`.
    -   Return ONLY the raw, executable Python code. No explanations or markdown.
    """
    try:
        response = model.generate_content(prompt)
        raw_code = response.text.strip()
        return re.sub(r"```(?:python)?\n(.*?)```", r"\1", raw_code, flags=re.DOTALL).strip()
    except Exception as e:
        return f"# Visualization code generation failed: {e}"

def is_safe_select(sql: str) -> bool:
    if not isinstance(sql, str) or not sql.strip(): return False
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'GRANT', 'REVOKE', 'EXECUTE', 'DECLARE', 'BEGIN']
    sql_no_comments = re.sub(r'--.*?\n|/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    sql_normalized = ' '.join(sql_no_comments.upper().split())
    if not sql_normalized.startswith("SELECT"): return False
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', sql_normalized): return False
    return True

# --------------------------------------------------------------------------------
# --- Streamlit UI and Application Flow ---
# --------------------------------------------------------------------------------

def connect_to_database(db_type, credentials):
    try:
        if db_type == "SQL Server":
            conn = pyodbc.connect(f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={credentials['host']},{credentials['port']};DATABASE={credentials['dbname']};UID={credentials['user']};PWD={credentials['password']}")
        elif db_type == "MySQL":
            creds_mysql = credentials.copy(); creds_mysql['database'] = creds_mysql.pop('dbname')
            conn = mysql.connector.connect(**creds_mysql)
        elif db_type == "PostgreSQL":
            conn = psycopg2.connect(**credentials)
        elif db_type == "Snowflake":
            conn = snowflake.connector.connect(**credentials)
        else: return None
        return conn
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        return None

def init_session_state():
    defaults = {"logged_in": False, "db_conn": None, "db_type": "", "sql_query": "", "query_result": pd.DataFrame(), "insight": "", "viz_code": "", "last_question": ""}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def display_login_page():
    st.title("🔐 Database Login")
    db_type = st.selectbox("Database Type", ["Snowflake", "SQL Server", "MySQL", "PostgreSQL"])
    creds = {}
    if db_type == "Snowflake":
        creds['account'], creds['warehouse'], creds['database'], creds['user'], creds['password'] = st.text_input("Account"), st.text_input("Warehouse"), st.text_input("Database Name"), st.text_input("Username"), st.text_input("Password", type="password")
    else:
        creds['host'], port_map = st.text_input("Server Host", value="localhost"), {"SQL Server": "1433", "MySQL": "3306", "PostgreSQL": "5432"}
        creds['port'], creds['dbname'], creds['user'], creds['password'] = st.text_input("Port", value=port_map[db_type]), st.text_input("Database Name"), st.text_input("Username"), st.text_input("Password", type="password")
    if st.button("Connect"):
        if any(not v for k, v in creds.items() if k != 'password'):
            st.warning("Please fill in all connection details.")
        else:
            with st.spinner(f"Connecting to {db_type}..."):
                conn = connect_to_database(db_type, creds)
                if conn:
                    st.session_state.logged_in, st.session_state.db_conn, st.session_state.db_type = True, conn, db_type
                    st.success("✅ Connection successful!"); time.sleep(1); st.rerun()

def display_main_app():
    st.title("🧠 Analytixhub.ai")
    st.markdown("Ask a question in plain English. I'll generate the SQL, run it, and provide insights.")

    with st.form("ask_form"):
        question = st.text_input("Your Question:", value=st.session_state.last_question, placeholder="e.g., 'What were the top 5 selling products last quarter?'")
        submit_button = st.form_submit_button("💡 Get Insights")

    if submit_button and question:
        st.session_state.last_question = question
        st.session_state.sql_query, st.session_state.query_result, st.session_state.insight, st.session_state.viz_code = "", pd.DataFrame(), "", ""

        with st.status("Analytixhub.ai is thinking...", expanded=True) as status:
            status.update(label="Analyzing schema...", state="running")
            cols, fks = get_db_schema(st.session_state.db_conn, st.session_state.db_type)
            if not cols: st.error("Could not retrieve database schema."); return
            status.update(label="Generating SQL query...", state="running")
            relevant_tables = pick_relevant_tables(cols, question)
            schema_card = build_schema_card(cols, fks, relevant_tables)
            st.session_state.sql_query = generate_sql_with_gemini(question, schema_card, st.session_state.db_type)
            if not st.session_state.sql_query or not is_safe_select(st.session_state.sql_query):
                st.error("Failed to generate a safe SQL query."); return
            status.update(label="Executing query...", state="running")
            try:
                st.session_state.query_result = pd.read_sql(st.session_state.sql_query, st.session_state.db_conn)
            except Exception as e:
                st.error(f"❌ Query execution failed: {e}"); return
            if not st.session_state.query_result.empty:
                status.update(label="Generating insights & visualizations...", state="running")
                st.session_state.insight = generate_insight_with_gemini(st.session_state.query_result, question)
                st.session_state.viz_code = generate_visualization_code(st.session_state.query_result, question)
            status.update(label="Analysis complete!", state="complete", expanded=False)

    if not st.session_state.query_result.empty:
        st.header("Results & Insights", divider="rainbow")
        
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💡 Business Insights")
                st.markdown(st.session_state.insight or "No insights generated.")
            with col2:
                st.subheader("📈 Visualizations")
                if st.session_state.viz_code and not st.session_state.viz_code.startswith("#"):
                    
                    # --- FIX APPLIED HERE ---
                    try:
                        df = st.session_state.query_result
                        # Define a dictionary to serve as the scope for the exec call
                        # This ensures the 'fig' variable is captured correctly
                        exec_scope = {
                            "st": st,
                            "df": df,
                            "plt": plt,
                            "sns": sns,
                            "pd": pd
                        }
                        # Execute the generated code within the defined scope
                        exec(st.session_state.viz_code, exec_scope)
                        
                        # Retrieve the 'fig' object from the scope after execution
                        fig = exec_scope.get("fig")

                        if fig:
                            st.pyplot(fig)
                        else:
                            st.warning("The generated visualization code did not produce a 'fig' object.")
                            st.code(st.session_state.viz_code, language="python")
                            
                    except Exception as e:
                        st.error(f"Failed to execute visualization code: {e}")
                        st.code(st.session_state.viz_code, language="python")
                else:
                    st.info("No visualization was generated.")
        
        st.subheader("Export Options", divider="blue")
        insight_html = markdown.markdown(st.session_state.insight, extensions=['tables'])
        pdf_bytes = create_html_pdf(insight_html, st.session_state.last_question)
        csv_bytes = st.session_state.query_result.to_csv(index=False).encode('utf-8')

        b_col1, b_col2 = st.columns(2)
        b_col1.download_button("📄 Download Report as PDF", data=pdf_bytes, file_name="insights_report.pdf", mime="application/pdf", use_container_width=True)
        b_col2.download_button("⬇️ Download Raw Data as CSV", data=csv_bytes, file_name="query_results.csv", mime="text/csv", use_container_width=True)
        
        with st.expander("📧 Email Report"):
            if SENDER_EMAIL == "your_email@gmail.com" or SENDER_PASSWORD == "your_gmail_app_password":
                st.warning("Email feature not configured. Set credentials in the script.")
            else:
                with st.form("email_form"):
                    recipient = st.text_input("Recipient Email")
                    send_button = st.form_submit_button("Send Email")
                    if send_button:
                        if not recipient:
                            st.warning("Please enter a recipient email.")
                        else:
                            try:
                                msg = MIMEMultipart()
                                msg['From'] = SENDER_EMAIL
                                msg['To'] = recipient
                                msg['Subject'] = f"Analytix Hub Report: {st.session_state.last_question}"
                                msg.attach(MIMEText(insight_html, 'html'))
                                server = smtplib.SMTP('smtp.gmail.com', 587)
                                server.starttls(); server.login(SENDER_EMAIL, SENDER_PASSWORD)
                                server.send_message(msg); server.quit()
                                st.success("Email sent successfully!")
                            except Exception as e:
                                st.error(f"Failed to send email: {e}")

        with st.expander("Developer Details"):
            st.subheader("💾 Generated SQL Query"); st.code(st.session_state.sql_query, language="sql")
            st.subheader("📊 Raw Query Results"); st.dataframe(st.session_state.query_result)
            st.subheader("🐍 Generated Plotting Code"); st.code(st.session_state.viz_code, language="python")

if __name__ == "__main__":
    init_session_state()
    if not st.session_state.logged_in:
        display_login_page()
    else:
        display_main_app()