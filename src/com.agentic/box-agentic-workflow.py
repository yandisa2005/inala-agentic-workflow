#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import sqlite3
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, status
from pydantic import BaseModel, Field
import httpx
from google import genai
from google.genai import types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("box-extractor-service")

# ---------------------------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
MCP_SERVER_SSE_URL = os.getenv("MCP_SERVER_SSE_URL", "http://10.128.0.7:8000/sse")
BOX_DEVELOPER_TOKEN = os.getenv("BOX_DEVELOPER_TOKEN", "")
DB_PATH = os.getenv("SQLITE_DB_PATH", "invoices.db")
SERVICE_API_KEY = os.getenv("EXTRACTION_API_KEY", "")

BOX_MCP_TOOLS = [
    "box_ai_extract_freeform_tool",
    "box_folder_items_list_tool",
    "box_folder_info_tool",
]

# Initialize Gemini Client lazily or at bootstrap
llm_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(
    title="Box MCP Agentic Invoice Extraction Service",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class ExtractionRequest(BaseModel):
    folder_id: str = Field(..., description="Box target folder ID to process")

class ExtractionResponse(BaseModel):
    status: str
    message: str
    folder_id: str

# ---------------------------------------------------------------------------
# Database Utilities
# ---------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH):
    """Ensure the target SQLite schema exists."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                file TEXT PRIMARY KEY UNIQUE,
                client TEXT,
                amount REAL,
                product TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

# ---------------------------------------------------------------------------
# MCP & Gemini Agent Engine
# ---------------------------------------------------------------------------
async def get_mcp_tools(session: ClientSession) -> List[types.Tool]:
    """Inspect the remote MCP server and adapt selected tools for Gemini."""
    mcp_tools = await session.list_tools()
    formatted_tools = []

    for tool in mcp_tools.tools:
        if tool.name in BOX_MCP_TOOLS:
            logger.info(f"Binding MCP Tool: {tool.name}")
            schema_obj = getattr(tool, "input_schema", getattr(tool, "inputSchema", {}))

            if hasattr(schema_obj, "model_dump"):
                schema_dict = schema_obj.model_dump()
            elif isinstance(schema_obj, dict):
                schema_dict = schema_obj
            else:
                schema_dict = dict(schema_obj)

            formatted_tools.append(
                types.Tool(
                    function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": schema_dict,
                        }
                    ]
                )
            )

    logger.info(f"Configured {len(formatted_tools)} MCP tools for Gemini function calling.")
    return formatted_tools


def parse_json_payload(content: str) -> Any:
    """Resiliently parse JSON strings or extract nested JSON blocks."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass

    import re
    json_match = re.search(r"\{.*\}", str(content), re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return content


async def generate(
    prompt: str,
    client: genai.Client,
    session: Optional[ClientSession] = None,
    tools: Optional[List[types.Tool]] = None,
) -> Any:
    """Orchestrate Gemini generation and handle MCP function calling."""
    config = (
        types.GenerateContentConfig(temperature=0, tools=tools)
        if session is not None and tools is not None
        else None
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    if not response.candidates or not response.candidates[0].content.parts:
        return ""

    first_part = response.candidates[0].content.parts[0]

    # Handle Function Call from Gemini -> invoke MCP server
    if getattr(first_part, "function_call", None):
        fn_call = first_part.function_call
        logger.info(f"Executing MCP Tool Call: {fn_call.name}")
        tool_result = await session.call_tool(
            fn_call.name,
            arguments=dict(fn_call.args)
        )

        results = []
        for item in tool_result.content:
            text = getattr(item, "text", str(item))
            results.append(parse_json_payload(text))
        return results

    return first_part.text


async def extract_invoice_fields(
    invoice: Dict[str, Any],
    client: genai.Client,
    session: ClientSession,
    tools: List[types.Tool],
) -> Optional[Dict[str, Any]]:
    """Extract structured schema from an individual file."""
    logger.info(f"Extracting fields from invoice: {invoice.get('name')}")
    prompt = (
        f"Extract the following fields from the invoice with file_id {invoice.get('id')}: "
        "client_name, invoice_amount, product_name. "
        "Return the invoice_amount as a float."
    )
    response = await generate(prompt, client, session, tools)
    first_item = response[0] if isinstance(response, list) and response else response

    if isinstance(first_item, dict) and "error" in first_item:
        logger.error(f"Box API Error on file '{invoice.get('name')}': {first_item['error']}")
        return None

    if isinstance(first_item, dict) and "AI_response" in first_item:
        result = parse_json_payload(first_item["AI_response"].get("answer", "{}"))
    elif isinstance(first_item, str):
        result = parse_json_payload(first_item)
    else:
        result = first_item

    if isinstance(result, dict):
        result["file"] = invoice.get("name")
        return result
    return None


async def run_extraction_flow(folder_id: str):
    """Core async ETL task executed in background."""
    headers = {}
    if BOX_DEVELOPER_TOKEN:
        headers["Authorization"] = f"Bearer {BOX_DEVELOPER_TOKEN}"

    logger.info(f"Starting extraction flow for folder: {folder_id} against {MCP_SERVER_SSE_URL}")

    try:
        async with sse_client(MCP_SERVER_SSE_URL, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await get_mcp_tools(session)

                # Fetch listing of folder items
                listing_result = await generate(
                    f"List the content of the folder with id {folder_id}",
                    llm_client,
                    session,
                    tools,
                )

                folder_items = []
                if isinstance(listing_result, list) and listing_result:
                    folder_items = listing_result[0].get("folder_items", [])
                elif isinstance(listing_result, dict):
                    folder_items = listing_result.get("folder_items", [])

                logger.info(f"Found {len(folder_items)} items in folder {folder_id}")

                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    for invoice in folder_items:
                        data = await extract_invoice_fields(invoice, llm_client, session, tools)
                        if data:
                            cursor.execute(
                                """
                                INSERT INTO invoices (file, client, amount, product)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT(file) DO UPDATE SET
                                    client=excluded.client,
                                    amount=excluded.amount,
                                    product=excluded.product,
                                    updated_at=CURRENT_TIMESTAMP
                                """,
                                (
                                    data.get("file"),
                                    data.get("client_name"),
                                    data.get("invoice_amount"),
                                    data.get("product_name"),
                                ),
                            )
                            logger.info(f"Saved invoice: {data.get('file')}")
                        else:
                            logger.warning(f"Skipping {invoice.get('name')}; extraction returned empty.")
                    conn.commit()

                logger.info(f"Batch completed successfully for folder {folder_id}.")

    except Exception as e:
        logger.error(f"Extraction flow failed: {str(e)}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/healthz", status_code=status.HTTP_200_OK)
def healthz():
    """Liveness & Readiness probe."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractionResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_extraction(
    payload: ExtractionRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None)
):
    """
    Accept an extraction job for a dynamic Box folder ID and dispatch execution to background.
    """
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header"
        )

    if not llm_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured on the server."
        )

    folder_id = payload.folder_id.strip()
    if not folder_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="folder_id cannot be empty."
        )

    # Dispatch to background task to allow Cloud Run to return 202 immediately
    background_tasks.add_task(run_extraction_flow, folder_id)

    return ExtractionResponse(
        status="ACCEPTED",
        message="Extraction pipeline scheduled.",
        folder_id=folder_id
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")