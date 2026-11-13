import json
import os
import sqlite3
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams
from google.genai import types

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "https://box-mcp-server-373951769736.us-central1.run.app/mcp",
)
DB_PATH = os.getenv("SQLITE_DB_PATH", "/tmp/invoices.db")


def _emit_log(event: str, **fields) -> None:
    """Emit single-line structured JSON logs that Cloud Logging parses into jsonPayload."""
    payload = {"severity": "INFO", "event": event, **fields}
    print(json.dumps(payload, default=str), flush=True)


def log_model_response(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Log every model turn (tool calls + generated text) to Cloud Logging."""
    text_parts = []
    tool_calls = []

    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                text_parts.append(part.text)
            if part.function_call:
                tool_calls.append({
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args) if part.function_call.args else {},
                })

    _emit_log(
        "adk_model_response",
        agent=callback_context.agent_name,
        invocation_id=callback_context.invocation_id,
        text="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
    )
    return None  # Preserve original LlmResponse unmodified


def log_agent_final_response(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """Log completion of the full agent invocation."""
    print("----------------Agent Call Completing--------------")
    _emit_log(
        "adk_agent_turn_complete",
        agent=callback_context.agent_name,
        invocation_id=callback_context.invocation_id,
    )
    return None


def save_invoice_record(
    file_name: str, client_name: str, invoice_amount: float, product_name: str
) -> dict:
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

    _emit_log(
        "invoice_record_saved",
        file=file_name,
        client=client_name,
        amount=invoice_amount,
        product=product_name,
    )
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
        "box_file_move_tool",
        "box_folder_move_tool",
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
    after_model_callback=log_model_response,
    after_agent_callback=log_agent_final_response,
)