import os
import sqlite3
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "https://box-mcp-server-373951769736.us-central1.run.app/mcp"
)
DB_PATH = os.getenv("SQLITE_DB_PATH", "/tmp/invoices.db")

def save_invoice_record(file_name: str, client_name: str, invoice_amount: float, product_name: str) -> dict:
    """Persist an extracted invoice record into the local SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                file TEXT PRIMARY KEY UNIQUE,
                client TEXT,
                amount REAL,
                product TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            """
            INSERT INTO invoices (file, client, amount, product)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file) DO UPDATE SET
                client=excluded.client,
                amount=excluded.amount,
                product=excluded.product,
                updated_at=CURRENT_TIMESTAMP
            """,
            (file_name, client_name, invoice_amount, product_name),
        )
        conn.commit()
    return {"status": "saved", "file": file_name}

box_mcp_tools = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=MCP_SERVER_URL,
        timeout=30.0,
    ),
    tool_filter=[
        "box_ai_extract_freeform_tool",
        "box_folder_items_list_tool",
        "box_folder_info_tool",
    ],
)

root_agent = LlmAgent(
    name="extractor_agent",
    model=os.getenv("MODEL_NAME", "gemini-3.8-flash"),
    description="Extracts structured invoice fields from files inside a specified Box folder.",
    instruction=(
        "You are an automated invoice extraction agent. "
        "When given a Box folder_id:\n"
        "1. List all items in that folder using box_folder_items_list_tool.\n"
        "2. For each invoice file, extract client_name, invoice_amount (as a float), and product_name "
        "using box_ai_extract_freeform_tool.\n"
        "3. Save each extracted invoice using save_invoice_record.\n"
        "4. Summarize the extracted invoices and total amount."
    ),
    tools=[box_mcp_tools, save_invoice_record],
)