import datetime
import json
import os
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams
from google.cloud import datastore
from google.genai import types

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "https://box-mcp-server-373951769736.us-central1.run.app/mcp",
)
DATASTORE_KIND = os.getenv("DATASTORE_KIND", "Invoice")
DATASTORE_NAMESPACE = os.getenv("DATASTORE_NAMESPACE") or None

_datastore_client: Optional[datastore.Client] = None


def _get_datastore_client() -> datastore.Client:
    """Lazy-initialize a reusable Datastore client using Application Default Credentials."""
    global _datastore_client
    if _datastore_client is None:
        _datastore_client = datastore.Client(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            namespace=DATASTORE_NAMESPACE,
        )
    return _datastore_client


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
    return None


def log_agent_final_response(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
    """Log completion of the full agent invocation."""
    _emit_log(
        "adk_agent_turn_complete",
        agent=callback_context.agent_name,
        invocation_id=callback_context.invocation_id,
    )
    return None


def save_invoice_record(
    file_name: str, client_name: str, invoice_amount: float, product_name: str
) -> dict:
    """Persist or upsert an extracted invoice record in GCP Cloud Datastore."""
    client = _get_datastore_client()
    key = client.key(DATASTORE_KIND, file_name)

    entity = datastore.Entity(key=key)
    entity.update({
        "file": file_name,
        "client": client_name,
        "amount": float(invoice_amount),
        "product": product_name,
        "updated_at": datetime.datetime.now(datetime.timezone.utc),
    })
    client.put(entity)

    _emit_log(
        "invoice_record_saved",
        backend="datastore",
        kind=DATASTORE_KIND,
        file=file_name,
        client=client_name,
        amount=float(invoice_amount),
        product=product_name,
    )
    return {
        "status": "saved",
        "backend": "datastore",
        "kind": DATASTORE_KIND,
        "file": file_name,
    }


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
    description="Extracts structured invoice fields from files inside a specified Box folder and moves processed items.",
    instruction=(
        "You are an automated invoice extraction agent. "
        "When given a Box folder_id:\n"
        "1. List all items in that folder using box_folder_items_list_tool.\n"
        "2. For each invoice file, extract client_name, invoice_amount (as a float), and product_name "
        "using box_ai_extract_freeform_tool.\n"
        "3. Save each extracted invoice using save_invoice_record.\n"
        "4. If a destination folder_id is provided, move processed files using box_file_move_tool "
        "(or move an entire folder using box_folder_move_tool when explicitly requested).\n"
        "5. Summarize the extracted invoices, total amount, and any moved items."
    ),
    tools=[box_mcp_tools, save_invoice_record],
    after_model_callback=log_model_response,
    after_agent_callback=log_agent_final_response,
)