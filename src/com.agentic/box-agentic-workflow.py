#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import sqlite3
import traceback
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, status
from pydantic import BaseModel, Field
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
DB_PATH = os.getenv("SQLITE_DB_PATH", "/tmp/invoices.db")  # /tmp is writable tmpfs in Cloud Run
SERVICE_API_KEY = os.getenv("EXTRACTION_API_KEY", "")

BOX_MCP_TOOLS = [
    "box_ai_extract_freeform_tool",
    "box_folder_items_list_tool",
    "box_folder_info_tool",
]

# Initialize Gemini Client
llm_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ---------------------------------------------------------------------------
# Database Utilities & Lifespan
# ---------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH):
    """Ensure directory and SQLite schema exist."""
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                file TEXT PRIMARY KEY UNIQUE,
                client TEXT,
                amount REAL,
                product TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SQLite database at %s", DB_PATH)
    init_db()
    yield
    logger.info("Shutting down Box Extractor Service.")


app = FastAPI(
    title="Box MCP Agentic Invoice Extraction Service",
    version="1.0.0",
    lifespan=lifespan
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
# MCP & Gemini Agent Engine
# ---------------------------------------------------------------------------
async def get_mcp_tools(session: ClientSession) -> List[types.Tool]:
    """Inspect the remote MCP server and adapt selected tools for Gemini."""
    mcp_tools = await session.list_tools()
    formatted_tools = []

    for tool in mcp_tools.tools:
        if tool.name in BOX_MCP_TOOLS:
            logger.info("Registering MCP Tool: %s", tool.name)
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

    logger.info("Successfully bound %d MCP tools to Gemini.", len(formatted_tools))
    return formatted_tools


def parse_json_payload(content: Any) -> Any:
    """Resiliently parse JSON strings or extract nested JSON objects."""
    if isinstance(content, (dict, list)):
        return content

    text_content = str(content)
    try:
        return json.loads(text_content)
    except (json.JSONDecodeError, TypeError):
        pass

    import re
    json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return text_content


async def generate(
    prompt: str,
    client: genai.Client,
    session: Optional[ClientSession] = None,
    tools: Optional[List[types.Tool]] = None,
) -> Any:
    """Non-blocking Gemini generation with MCP tool dispatch."""
    config = (
        types.GenerateContentConfig(temperature=0, tools=tools)
        if session is not None and tools is not None
        else None
    )

    # Use the non-blocking async client
    response = await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    if not response.candidates or not response.candidates[0].content.parts:
        return ""

    first_part = response.candidates[0].content.parts[0]

    # Handle Function Call request from the model
    if getattr(first_part, "function_call", None):
        fn_call = first_part.function_call
        logger.info("Gemini invoked tool: %s", fn_call.name)
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
    """Extract and validate fields from an invoice file via Gemini."""
    file_name = invoice.get("name", "unknown")
    file_id = invoice.get("id")

    logger.info("Extracting fields for: %s (id: %s)", file_name, file_id)
    prompt = (
        f"Extract the following fields from the invoice with file_id {file_id}: "
        "client_name, invoice_amount, product_name. "
        "Return the invoice_amount as a float."
    )

    raw_response = await generate(prompt, client, session, tools)
    first_item = raw_response[0] if isinstance(raw_response, list) and raw_response else raw_response

    if isinstance(first_item, dict) and "error" in first_item:
        logger.error("Box API Error on %s: %s", file_name, first_item["error"])
        return None

    if isinstance(first_item, dict) and "AI_response" in first_item:
        result = parse_json_payload(first_item["AI_response"].get("answer", "{}"))
    else:
        result = parse_json_payload(first_item)

    if isinstance(result, dict):
        result["file"] = file_name
        return result

    logger.warning("Could not structure extraction for: %s. Output: %s", file_name, raw_response)
    return None


async def run_extraction_flow(folder_id: str):
    """Background task orchestrating Box discovery, extraction, and persistence."""
    headers = {}
    if BOX_DEVELOPER_TOKEN:
        headers["Authorization"] = f"Bearer {BOX_DEVELOPER_TOKEN}"

    logger.info("Starting background ETL for folder: %s", folder_id)

    try:
        async with sse_client(MCP_SERVER_SSE_URL, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await get_mcp_tools(session)

                listing = await generate(
                    f"List the content of the folder with id {folder_id}",
                    llm_client,
                    session,
                    tools,
                )

                folder_items = []
                if isinstance(listing, list) and listing:
                    folder_items = listing[0].get("folder_items", [])
                elif isinstance(listing, dict):
                    folder_items = listing.get("folder_items", [])

                logger.info("Found %d items in folder %s", len(folder_items), folder_id)

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
                            logger.info("Persisted invoice: %s", data.get("file"))
                    conn.commit()

                logger.info("Batch extraction successfully finished for folder %s.", folder_id)

    except Exception as e:
        logger.error("Failed during folder extraction: %s\n%s", str(e), traceback.format_exc())

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz", status_code=status.HTTP_200_OK)
def healthz():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractionResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_extraction(
    payload: ExtractionRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None)
):
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header"
        )

    if not llm_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured on the service."
        )

    folder_id = payload.folder_id.strip()
    if not folder_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="folder_id cannot be blank."
        )

    # Cloud Run will return 202 immediately, while extraction continues in background
    background_tasks.add_task(run_extraction_flow, folder_id)

    return ExtractionResponse(
        status="ACCEPTED",
        message="Extraction pipeline dispatched.",
        folder_id=folder_id
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)