import time
import os
import streamlit as st
from google import genai
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool
import psycopg2 
from psycopg2 import sql 
from dotenv import load_dotenv
import json

# --- Configuration ---
load_dotenv()

# Set Google Cloud project information
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

if not PROJECT_ID or not LOCATION:
    st.error("Error: GOOGLE_CLOUD_PROJECT and VERTEX_AI_LOCATION environment variables must be set.")
    st.stop()

# PostgreSQL Connection Details 
db_host = os.getenv("db_host")
db_port = os.getenv("db_port")
db_name = os.getenv("db_name")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
table_name = os.getenv("table_name")

# Model
MODEL_ID = "gemini-2.0-flash-001" 

# --- PostgreSQL Helper Functions ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        # Log error to console/logs for backend visibility
        print(f"Database Connection Error: {e}")
        # Provide a user-friendly error
        st.error(f"Database Connection Error: Could not connect to the database. Please check the configuration.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during DB connection: {e}")
        st.error(f"An unexpected error occurred during DB connection.")
        return None

# --- Tool Function Definitions (Adapted for PostgreSQL) ---
list_tables_func = FunctionDeclaration(
    name="list_tables",
    description="List tables in the PostgreSQL database that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {}, # No dataset_id needed for single PostgreSQL database
    },
)

get_table_func = FunctionDeclaration(
    name="get_table",
    description="Get information about a table, including the description, schema (column names), and sample rows that will help answer the user's question.",
    parameters={
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to get information about",
            }
        },
        "required": [
            "table_name",
        ],
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Get information from data in PostgreSQL using SQL queries",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query on a single line that will help give quantitative answers to the user's question when run on the PostgreSQL database and table. In the SQL query, always use the table names directly.",
            }
        },
        "required": [
            "query",
        ],
    },
)

# --- Tool Definition ---
postgres_tool = Tool(
    function_declarations=[
        list_tables_func,
        get_table_func, 
        sql_query_func, 
    ],
)

# --- Gemini Client Initialization (Using Vertex AI) ---
try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
except Exception as e:
    st.error(f"Failed to initialize Vertex AI Client: {e}")
    st.stop()

# --- Streamlit App UI ---
st.set_page_config(
    page_title="SQL Talk to Data",
    page_icon="🤖", # Simple icon
    layout="wide",
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("SQL Talk to Data")
with col2:
    st.image("gemini-logo.png")

st.subheader("Powered by Gemini")

st.markdown(
    "Ask questions in natural language about your database!"
)

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - What tables are in this database?
        - What kind of data is in the transactions table?
        - How many transactions are there?
        - What is the average transaction amount?
        - Show me some sample transactions.
        """
    )

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  
        try:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])
        except KeyError:
            pass

# Get user input
if prompt := st.chat_input("Ask me about information in the database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send message to Gemini and handle response/function calls
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        chat = client.chats.create(
            model=MODEL_ID,
            config=GenerateContentConfig(temperature=0, tools=[postgres_tool]),
        )

        prompt += """
            Please give a concise, high-level summary followed by detail in
            plain language about where the information in your response is
            coming from in the database. Only use information that you learn
            from PostgreSQL, do not make up information.
            """

        try:
            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]

            print(response)

            api_requests_and_responses = []
            backend_details = ""

            function_calling_in_process = True
            while function_calling_in_process:
                try:
                    params = {}
                    for key, value in response.function_call.args.items():
                        params[key] = value

                    print(response.function_call.name)
                    print(params)


                    if response.function_call.name == "list_tables":
                        try:
                            conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
                            cur = conn.cursor()
                            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
                            tables = cur.fetchall()
                            api_response = str([table[0] for table in tables])
                        except Exception as e:
                            api_response = f"Error listing tables: {e}"
                        finally:
                            if conn:
                                cur.close()
                                conn.close()
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )


                    if response.function_call.name == "get_table":
                        table_name_to_get = params.get("table_name")
                        if not table_name_to_get:
                            api_response = "Error: table_name parameter is missing for get_table function."
                        else:
                            try:
                                conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
                                cur = conn.cursor()

                                # Get table description (PostgreSQL doesn't have table descriptions by default, might need extensions or custom fields)
                                table_description = "No description available in this example." # Placeholder

                                # Get column names
                                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name_to_get}'")
                                column_names = [row[0] for row in cur.fetchall()]

                                # Get sample rows (first 5 rows)
                                cur.execute(f"SELECT * FROM {table_name_to_get} LIMIT 5")
                                sample_rows = cur.fetchall()

                                api_response = [
                                    str(table_description),
                                    str(column_names),
                                    str(sample_rows)
                                ]

                            except Exception as e:
                                api_response = f"Error getting table details: {e}"
                            finally:
                                if conn:
                                    cur.close()
                                    conn.close()

                        api_requests_and_responses.append(
                            [
                                response.function_call.name,
                                params,
                                api_response,
                            ]
                        )
                        api_response = str(api_response) # Ensure api_response is string for Gemini


                    if response.function_call.name == "sql_query":
                        try:
                            conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
                            cur = conn.cursor()
                            cleaned_query = (
                                params["query"]
                                .replace("\\n", " ")
                                .replace("\n", "")
                                .replace("\\", "")
                            )
                            cur.execute(cleaned_query)
                            column_names_query = [desc[0] for desc in cur.description] # Get column names
                            query_results = cur.fetchall()
                            api_response = str([dict(zip(column_names_query, row)) for row in query_results]) # Return results as list of dictionaries
                            api_response = api_response.replace("\\", "").replace(
                                "\n", ""
                            )
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )

                        except Exception as e:
                            error_message = f"""
                            We're having trouble running this SQL query against PostgreSQL. This
                            could be due to an invalid query or the structure of
                            the data. Try rephrasing your question to help the
                            model generate a valid query. Details:

                            {str(e)}"""
                            st.error(error_message)
                            api_response = error_message
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": error_message,
                                }
                            )
                        finally:
                            if conn:
                                cur.close()
                                conn.close()


                    print(api_response)

                    response = chat.send_message(
                        Part.from_function_response(
                            name=response.function_call.name,
                            response={
                                "content": api_response,
                            },
                        ),
                    )
                    response = response.candidates[0].content.parts[0]

                    backend_details += "- Function call:\n"
                    backend_details += (
                        "   - Function name: ```"
                        + str(api_requests_and_responses[-1][0])
                        + "```"
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - Function parameters: ```"
                        + str(api_requests_and_responses[-1][1])
                        + "```"
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - API response: ```"
                        + str(api_requests_and_responses[-1][2])
                        + "```"
                    )
                    backend_details += "\n\n"
                    with message_placeholder.container():
                        st.markdown(backend_details)

                except AttributeError:
                    function_calling_in_process = False

            time.sleep(3)

            full_response = response.text
            with message_placeholder.container():
                st.markdown(full_response.replace("$", r"\$"))  # noqa: W605
                with st.expander("Function calls, parameters, and responses:"):
                    st.markdown(backend_details)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "backend_details": backend_details,
                }
            )
        except Exception as e:
            print(e)
            error_message = f"""
                Something went wrong! We encountered an unexpected error while
                trying to process your request. Please try rephrasing your
                question. Details:

                {str(e)}"""
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )