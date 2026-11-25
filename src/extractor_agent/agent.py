import datetime
import json
import os
import secrets
import smtplib
import ssl
import time
from email.message import EmailMessage
from typing import Generator, Optional

import google.auth.transport.requests
import google.oauth2.id_token
import httpx
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
# DATASTORE_KIND = os.getenv("DATASTORE_KIND", "Invoice")
APPROVAL_KIND = os.getenv("APPROVAL_DATASTORE_KIND", "box-move-request")
DATASTORE_NAMESPACE = os.getenv("DATASTORE_NAMESPACE") or None
DATASTORE_DATABASE = os.getenv("DATASTORE_DATABASE", "approval-db")
APPROVAL_BASE_URL = os.getenv(
    "APPROVAL_BASE_URL",
    "https://box-adk-extractor-service-373951769736.us-central1.run.app",
)
DEFAULT_APPROVER_EMAIL = os.getenv("BOX_MOVE_APPROVER_EMAIL", "finance-approvals@yourdomain.com")

_datastore_client: Optional[datastore.Client] = None


def _get_datastore_client() -> datastore.Client:
    """Lazy-initialize a reusable Datastore client using Application Default Credentials."""
    global _datastore_client
    if _datastore_client is None:
        _datastore_client = datastore.Client(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            namespace=DATASTORE_NAMESPACE,
            database=DATASTORE_DATABASE,
        )
    return _datastore_client


def _emit_log(event: str, severity: str = "INFO", **fields) -> None:
    """Emit single-line structured JSON logs that Cloud Logging parses into jsonPayload."""
    payload = {"severity": severity, "event": event, **fields}
    print(json.dumps(payload, default=str), flush=True)


class CloudRunOIDCAuth(httpx.Auth):
    """Automatically attaches and refreshes Google Cloud Run OIDC ID tokens."""

    def __init__(self, audience: str):
        self.audience = audience.removesuffix("/mcp").rstrip("/")
        self._token: Optional[str] = None
        self._expiry: float = 0.0

    def _get_valid_token(self) -> str:
        if not self._token or time.time() >= (self._expiry - 300):
            auth_req = google.auth.transport.requests.Request()
            self._token = google.oauth2.id_token.fetch_id_token(auth_req, self.audience)
            self._expiry = time.time() + 3600
        return self._token

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._get_valid_token()}"
        response = yield request
        if response.status_code in (401, 403):
            self._token = None
            request.headers["Authorization"] = f"Bearer {self._get_valid_token()}"
            yield request


def _mcp_httpx_client_factory(
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[httpx.Timeout] = None,
    auth: Optional[httpx.Auth] = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(30.0),
        auth=CloudRunOIDCAuth(MCP_SERVER_URL),
    )


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


# --- Temporarily disabled: Invoice Extraction Persistence ---
# def save_invoice_record(
#     file_name: str, client_name: str, invoice_amount: float, product_name: str
# ) -> dict:
#     """Persist or upsert an extracted invoice record in GCP Cloud Datastore."""
#     client = _get_datastore_client()
#     key = client.key(DATASTORE_KIND, file_name)
#
#     entity = datastore.Entity(key=key)
#     entity.update({
#         "file": file_name,
#         "client": client_name,
#         "amount": float(invoice_amount),
#         "product": product_name,
#         "updated_at": datetime.datetime.now(datetime.timezone.utc),
#     })
#     client.put(entity)
#
#     _emit_log(
#         "invoice_record_saved",
#         backend="datastore",
#         kind=DATASTORE_KIND,
#         file=file_name,
#         client=client_name,
#         amount=float(invoice_amount),
#         product=product_name,
#     )
#     return {
#         "status": "saved",
#         "backend": "datastore",
#         "kind": DATASTORE_KIND,
#         "file": file_name,
#     }


def _send_email_notification(to_email: str, subject: str, html_body: str) -> None:
    """Dispatch approval notification email via SMTP (credentials injected from Secret Manager)."""
    raw_host = (os.getenv("SMTP_HOST") or os.getenv("SMTP_Host") or "").strip()
    # Strip protocol schemes and port numbers if present
    smtp_host = raw_host.removeprefix("http://").removeprefix("https://").removeprefix("smtp://").split(":")[0]
    smtp_port = int(os.getenv("SMTP_PORT", os.getenv("SMTP_Port", "587")))
    smtp_user = (os.getenv("SMTP_USER") or os.getenv("SMTP_User") or "").strip()
    smtp_password = (os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_Password") or "").strip()

    missing_vars = [
        var_name
        for var_name, val in [("SMTP_HOST", smtp_host), ("SMTP_USER", smtp_user), ("SMTP_PASSWORD", smtp_password)]
        if not val
    ]
    if missing_vars:
        _emit_log("smtp_config_missing", severity="ERROR", missing=missing_vars)
        raise ValueError(f"Missing required SMTP environment variable(s): {', '.join(missing_vars)}")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content("Open this message in an HTML-capable email client to approve or reject the Box move request.")
    msg.add_alternative(html_body, subtype="html")

    tls_context = ssl.create_default_context()

    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=tls_context, timeout=15) as server:
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=tls_context)
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)


def request_box_move_approval(
    items: list[dict[str, str]],
    destination_parent_id: str,
    destination_parent_name: str,
    requested_by: str,
    reason: str = "",
    approver_email: Optional[str] = None,
) -> dict:
    """Queue one or more Box file/folder moves in Datastore and email the approver.

    Args:
        items: List of dicts, each containing:
            - "item_id": Box file ID or folder ID
            - "item_name": Human-readable file or folder name
            - "item_type": Must be "file" or "folder"
        destination_parent_id: Target Box parent folder ID
        destination_parent_name: Human-readable target parent folder name
        requested_by: Name or email of the requester
        reason: Business justification for moving the items
        approver_email: Optional override email for the approver
    """
    if not items:
        return {"status": "error", "message": "At least one file or folder item is required."}

    normalized_items = []
    for entry in items:
        item_type = str(entry.get("item_type", "")).strip().lower()
        if item_type not in ("file", "folder"):
            return {
                "status": "error",
                "message": f"Invalid item_type '{item_type}'. Must be 'file' or 'folder'.",
            }
        normalized_items.append({
            "item_id": str(entry["item_id"]),
            "item_name": str(entry.get("item_name", entry["item_id"])),
            "item_type": item_type,
        })

    client = _get_datastore_client()
    approval_token = secrets.token_urlsafe(32)
    recipient = approver_email or DEFAULT_APPROVER_EMAIL
    now = datetime.datetime.now(datetime.timezone.utc)

    key = client.key(APPROVAL_KIND, approval_token)
    entity = datastore.Entity(key=key)
    entity.update({
        "approval_token": approval_token,
        "status": "PENDING",
        "items": normalized_items,
        "destination_parent_id": str(destination_parent_id),
        "destination_parent_name": destination_parent_name,
        "requested_by": requested_by,
        "approver_email": recipient,
        "reason": reason,
        "created_at": now,
        "expires_at": now + datetime.timedelta(hours=24),
    })
    client.put(entity)

    approve_url = f"{APPROVAL_BASE_URL}/approvals/respond?token={approval_token}&decision=approve"
    reject_url = f"{APPROVAL_BASE_URL}/approvals/respond?token={approval_token}&decision=reject"

    item_rows = "".join(
        f"<li><strong>[{it['item_type'].upper()}]</strong> {it['item_name']} (ID: <code>{it['item_id']}</code>)</li>"
        for it in normalized_items
    )

    html_body = f"""
    <h3>Box Move Approval Required ({len(normalized_items)} item(s))</h3>
    <p><strong>Requested By:</strong> {requested_by}</p>
    <p><strong>Destination Parent Folder:</strong> {destination_parent_name} (ID: <code>{destination_parent_id}</code>)</p>
    <p><strong>Reason:</strong> {reason or "Not specified"}</p>
    <p><strong>Items Queued for Move:</strong></p>
    <ul>{item_rows}</ul>
    <p>
      <a href="{approve_url}" style="padding:10px 16px;background:#1a73e8;color:#fff;text-decoration:none;border-radius:4px;">
        Approve Move ({len(normalized_items)} item(s))
      </a>
      &nbsp;&nbsp;
      <a href="{reject_url}" style="padding:10px 16px;background:#d93025;color:#fff;text-decoration:none;border-radius:4px;">
        Reject
      </a>
    </p>
    <p><small>This approval link expires in 24 hours. No files or folders have been moved yet.</small></p>
    """

    try:
        _send_email_notification(
            to_email=recipient,
            subject=f"[Action Required] Approve Box Move ({len(normalized_items)} item(s)) -> {destination_parent_name}",
            html_body=html_body,
        )
    except Exception as exc:
        _emit_log(
            "box_move_approval_email_failed",
            severity="ERROR",
            error=str(exc),
            approver=recipient,
            approval_token=approval_token,
        )
        return {
            "status": "error",
            "message": f"Approval request recorded ({approval_token[:8]}...), but failed to dispatch email: {exc}",
            "approval_token": approval_token,
        }

    _emit_log(
        "box_move_approval_requested",
        approval_token_prefix=approval_token[:8],
        item_count=len(normalized_items),
        items=normalized_items,
        destination_parent_id=destination_parent_id,
        approver=recipient,
    )
    return {
        "status": "pending_approval",
        "approver_email": recipient,
        "item_count": len(normalized_items),
        "destination_parent_id": destination_parent_id,
        "message": (
            f"Approval email sent to {recipient} for {len(normalized_items)} item(s). "
            "Nothing will be moved until the approver clicks Approve."
        ),
    }


# Box MCP Toolset (move tools gated behind approval)
box_tools = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=MCP_SERVER_URL,
        timeout=30.0,
        httpx_client_factory=_mcp_httpx_client_factory,
    ),
    tool_filter=[
        # "box_ai_extract_freeform_tool",
        "box_folder_items_list_tool",
        "box_folder_info_tool",
        "box_file_metadata_create_tool",
        "box_file_metadata_update_tool",
    ],
)

move_approval_agent = LlmAgent(
    name="move_approval_agent",
    model=os.getenv("MODEL_NAME", "gemini-3.8-flash"),
    description="Handles governance and email approval requests whenever a user asks to move any Box file(s) or folder(s) to a different parent folder.",
    instruction=(
        "You enforce strict governance over all Box file and folder moves. You NEVER move files or folders directly.\n"
        "When asked to move one or more files or folders to a destination parent folder:\n"
        "1. Use box_folder_info_tool (and box_folder_items_list_tool if needed) to verify the exact names and IDs "
        "of the items being moved and the destination parent folder.\n"
        "2. Confirm who is requesting the move (ask if unknown, or use the user's identity if provided).\n"
        "3. Batch all requested items into a single call to request_box_move_approval with accurate "
        "item_type ('file' or 'folder'), item_id, and item_name values.\n"
        "4. Inform the user that the approval email has been dispatched and that all files/folders remain in "
        "their original locations until approved."
    ),
    tools=[box_tools, request_box_move_approval],
    after_model_callback=log_model_response,
    after_agent_callback=log_agent_final_response,
)

root_agent = LlmAgent(
    name="extractor_agent",
    model=os.getenv("MODEL_NAME", "gemini-3.8-flash"),
    description="Inspects Box folders/items and coordinates governed file and folder moves through email approval.",
    instruction=(
        "You are a Box workspace management and governance agent.\n"
        "1. When given a Box folder_id to inspect, list items using box_folder_items_list_tool "
        "and retrieve folder info using box_folder_info_tool.\n"
        "2. When requested to update or set metadata on Box files, use box_file_metadata_update_tool "
        "(or box_file_metadata_create_tool if no metadata instance exists yet).\n"
        "3. Whenever a user asks to move ANY Box file(s) OR folder(s) to a different parent folder, "
        "ALWAYS transfer to move_approval_agent.\n"
        "4. Summarize folder contents, metadata updates, and any pending move approval requests clearly."
    ),
    tools=[box_tools],
    sub_agents=[move_approval_agent],
    after_model_callback=log_model_response,
    after_agent_callback=log_agent_final_response,
)