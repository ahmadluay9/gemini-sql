import os
from dotenv import load_dotenv
import logging

from google.cloud import aiplatform

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import START, StateGraph


load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# Initialize Vertex AI SDK
if PROJECT_ID and LOCATION:
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        logging.info(f"Vertex AI initialized for project '{PROJECT_ID}' in location '{LOCATION}'")
    except Exception as e:
        logging.error(f"Error initializing Vertex AI: {e}")
        raise  
else:
    error_message = "PROJECT_ID or LOCATION environment variables not set. Vertex AI SDK cannot be initialized."
    logging.error(error_message)
    raise ValueError(error_message)

# mySQL Database Connection Details
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_DB = os.getenv("MYSQL_DB")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

# Construct the database URI
DATABASE_URI = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# Create a SQLDatabase instance
try:
    db = SQLDatabase.from_uri(DATABASE_URI)
    # Test connection by getting dialect or tables
    logging.info(f"Database Dialect: {db.dialect}")
    usable_tables = db.get_usable_table_names()
    logging.info(f"Tables found: {usable_tables}")
    if not usable_tables:
        logging.warning("No usable tables found in the database. Ensure tables exist and are accessible.")
except Exception as e:
    error_message = f"Error connecting to database: {e}"
    logging.error(error_message)
    raise Exception(error_message) 

# --- Initialize LLM ---
try:
    llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
    logging.info(f"LLM Initialized: {llm}")
except Exception as e:
    logging.error(f"Error initializing LLM: {e}")
    raise Exception(f"Failed to initialize LLM: {e}")

user_input = input("Please ask a question:")

class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())

system_prompt_table_selection = f"""Return the names of any SQL tables that are relevant to the user question.
The tables are:

{table_names}

"""

prompt_table_selection  = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_table_selection),
        ("human", "{input}"),
    ]
)

llm_with_tools = llm.bind_tools([Table])
output_parser = PydanticToolsParser(tools=[Table])

table_chain = prompt_table_selection  | llm_with_tools | output_parser

relevant_table_objects: List[Table] = []

try:
    relevant_table_objects = table_chain.invoke({"input": f"{user_input}"})
    # Extract just the names
    relevant_table_names = [t.name for t in relevant_table_objects]
    if not relevant_table_names:
         logging.warning("No relevant tables identified by the initial chain. SQL generation might fail or be inaccurate.")
         # Decide how to handle this: maybe use all tables as fallback?
         # relevant_table_names = usable_tables # Optional: Fallback
    logging.info(f"Relevant table names identified: {relevant_table_names}")
except Exception as e:
    logging.error(f"Error in table selection chain: {e}")
    # Decide on fallback if table selection fails
    # relevant_table_names = usable_tables # Optional: Fallback
    relevant_table_names = []

# --- Few-Shot Examples ---
examples = [
    # Example 1: Simple lookup with join
    {
        "input": "What is the email address for customer 'Januja Verma'?",
        "query": "SELECT email FROM customers WHERE customer_name = 'Januja Verma';"
    },
    # Example 2: Aggregation and Join
    {
        "input": "What is the total quantity of 'Potatoes' sold across all orders?",
        "query": "SELECT SUM(oi.quantity) AS total_quantity_sold FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE p.product_name = 'Potatoes';"
    },
    # Example 3: Aggregation, Join, Group By, Order By
    {
        "input": "Show the total order value for each customer, highest first.",
        "query": "SELECT c.customer_name, SUM(o.order_total) AS total_spent FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.customer_name ORDER BY total_spent DESC;"
    },
    # Example 4: Filtering based on status and Join
    {
        "input": "List the order IDs and reasons for orders that were Significantly Delayed.",
        "query": "SELECT order_id, reasons_if_delayed FROM delivery_performance WHERE delivery_status = 'Significantly Delayed';"

    },
    # Example 5: Filtering by rating and Join
    {
        "input": "Find the names and feedback text for customers who gave a rating of 1 or 2.",
        "query": "SELECT c.customer_name, f.rating, f.feedback_text FROM customer_feedback f JOIN customers c ON f.customer_id = c.customer_id WHERE f.rating <= 2;"
    },
    # Example 6: MAX aggregation and Join
    {
        "input": "Which product has the highest price?",
        "query": "SELECT product_name, price FROM products ORDER BY price DESC LIMIT 1;"
    },
    # Example 7: Counting with filtering and Join
    {
        "input": "How many orders were placed using 'Credit Card'?",
        "query": "SELECT COUNT(order_id) AS number_of_orders FROM orders WHERE payment_method = 'Card';"
    },
    # Example 8: Date-based filtering and Join
    {
        "input": "List products for which stock was received after '2024-12-01'.",
        "query": "SELECT DISTINCT p.product_name FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE i.date > '2024-12-01';"
    },
    # Example 9: Finding minimum with filtering
    {
        "input": "What was the lowest ROAS achieved in any marketing campaign?",
        "query": "SELECT MIN(roas) AS lowest_roas FROM marketing_performance;"
    },
    # Example 10: Multiple Joins
     {
        "input": "Show the product name, quantity, and customer name for order ID 'ORD123'.",
        "query": "SELECT p.product_name, oi.quantity, c.customer_name FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_id = 'ORD123';"
    }
]

# --- Embeddings and Example Selector ---
try:
    embeddings = VertexAIEmbeddings(model="text-embedding-005")
    logging.info("Embeddings model initialized.")
except Exception as e:
    logging.error(f"Error initializing embeddings model: {e}")
    raise Exception(f"Failed to initialize embeddings model: {e}")

# Check if examples list is not empty
if not examples:
    error_message = "Error: Examples list is empty. Cannot create example selector."
    logging.error(error_message)
    raise ValueError(error_message)

# Example Selector
try:
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        FAISS, # Using FAISS for in-memory vector storage
        k=3,
        input_keys=["input"], # Base similarity on the 'input' field
    )
    logging.info(f"Example selector created with k={example_selector.k}")
except Exception as e:
    logging.error(f"Error creating example selector: {e}")
    raise Exception(f"Failed to create example selector: {e}")

selected_examples = example_selector.select_examples({"input": user_input})
example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
formatted_examples = "\n".join([example_prompt.format(**example) for example in selected_examples])

# ---SQL Generation---
table_info = db.get_table_info(relevant_table_names)

system_prompt_sql_gen  = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Only use the following tables:
{relevant_table_names}

Here are the schema:
{table_info}

Here are some examples of user inputs and their corresponding SQL queries:
{examples}

Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

Use format:

First draft: <<FIRST_DRAFT_QUERY>>
Final answer: <<FINAL_ANSWER_QUERY>>
"""
prompt_sql_gen = ChatPromptTemplate.from_messages(
    [("system", system_prompt_sql_gen), ("human", "{input}")]
).partial(dialect=db.dialect, top_k=5, relevant_table_names=relevant_table_names, table_info=table_info, examples=formatted_examples)

def parse_final_answer(output: str) -> str:
    """Parses the LLM output to extract the final SQL query."""
    # 1. Split by "Final answer: " and get the SQL query part
    sql_query_part = output.split("Final answer: ")[1]
    # 2. Strip leading/trailing <<>> and whitespace
    sql_query_part = sql_query_part.strip("<> ").strip()
    # 3. Remove double quotes "
    sql_query_part = sql_query_part.replace('"', '')
    return sql_query_part

chain = create_sql_query_chain(llm, db, prompt=prompt_sql_gen) | parse_final_answer

# --- LangGraph Integration ---
class State(BaseModel):
    """LangGraph state for SQL query and answer generation."""
    question: str = Field(description="The user's question.")
    query: str = Field(description="Generated SQL query.", default=None)
    result: str = Field(description="Result from executing the SQL query.", default=None)
    answer: str = Field(description="Final answer to the user's question.", default=None)
    table_info: str = Field(description="Database table information", default=table_info) # Pass table_info to state
    relevant_table_names: str = Field(description="Relevant table name", default=str(relevant_table_names)) # Pass relevant_table

def write_query(state: State):
    """Generate SQL query based on user question."""
    logging.info("Entering `write_query` node")
    try:
        sql_query = chain.invoke({"question": state.question}) # Use "question" key as expected by chain
        logging.info(f"Generated SQL Query: {sql_query}")
        return {"query": sql_query} # Return as a dictionary to update state
    except Exception as e:
        error_message_chain = f"Error in SQL query generation chain within LangGraph: {e}"
        logging.error(error_message_chain)
        return {"query": None, "error": error_message_chain} # Indicate error in state

def execute_query(state: State):
    """Execute SQL query and get result from the database."""
    logging.info("Entering `execute_query` node")
    if not state.query:
        logging.warning("No SQL query to execute. Skipping database execution.")
        return {"result": "No SQL query generated."}

    try:
        db_response = db.run(state.query)
        logging.info(f"Database Query Executed Successfully. Response: {db_response}")
        return {"result": db_response} # Return result to update state
    except Exception as db_error:
        error_message_db = f"Error executing SQL query against the database within LangGraph: {db_error}"
        logging.error(error_message_db)
        return {"result": None, "error": error_message_db} # Indicate error in state
    
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    logging.info("Entering `generate_answer` node")
    if not state.result:
        logging.warning("No SQL result to generate answer. Cannot provide a meaningful answer.")
        return {"answer": "Could not retrieve database information to answer the question."}

    prompt_answer = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question concisely.\n\n"
        f'Question: {state.question}\n'
        f'SQL Query: {state.query}\n'
        f'SQL Result: {state.result}'
    )
    try:
        response = llm.invoke(prompt_answer)
        answer_text = response.content
        logging.info(f"Generated Answer: {answer_text}")
        return {"answer": answer_text}
    except Exception as e:
        error_message_answer = f"Error generating final answer within LangGraph: {e}"
        logging.error(error_message_answer)
        return {"answer": "Error generating answer.", "error": error_message_answer}

# --- Graph Definition ---
graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

# --- Edges ---
graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")
graph_builder.set_entry_point("write_query") # Set the starting node explicitly

graph = graph_builder.compile()
for step in graph.stream(
    {"question": user_input}, stream_mode="updates"
):
    print(step)