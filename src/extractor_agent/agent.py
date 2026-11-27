import datetime
import html
import gc
import json
import os
import secrets
import smtplib
import ssl
import time
from email.message import EmailMessage
from typing import Generator, Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
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


def _get_config_var(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve configuration from environment variable, falling back to Secret Manager if available."""
    val = os.getenv(secret_name) or os.getenv(secret_name.upper()) or os.getenv(secret_name.lower())
    if val:
        return val.strip()

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "inala-development-project")
    try:
        from google.cloud import secretmanager  # type: ignore[attr-defined]

        client = secretmanager.SecretManagerServiceClient()
        resource_name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(name=resource_name)
        payload = response.payload.data.decode("UTF-8").strip()
        if payload:
            _emit_log("secret_resolved_from_manager", secret_name=secret_name, project_id=project_id)
            return payload
    except (ImportError, AttributeError):
        # SDK not installed; environment variables are the primary source
        pass
    except Exception as exc:
        _emit_log("secret_lookup_failed", severity="DEBUG", secret_name=secret_name, error=str(exc))

    return default


def _send_email_notification(to_email: str, subject: str, html_body: str) -> None:
    """Dispatch approval notification email via HTTPS APIs (Resend/SendGrid) or SMTP."""
    # 1. Check for Resend HTTPS API (bypasses Cloud Run SMTP port drops)
    resend_api_key = _get_config_var("RESEND_API_KEY")
    if resend_api_key:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {resend_api_key.strip()}"},
            json={
                "from": os.getenv("EMAIL_FROM", "onboarding@resend.dev"),
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
            timeout=15.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Resend HTTP error ({resp.status_code}): {resp.text}")
        _emit_log("email_sent_via_https", provider="resend", to=to_email)
        return

    # 2. Check for SendGrid HTTPS API
    sendgrid_api_key = _get_config_var("SENDGRID_API_KEY")
    if sendgrid_api_key:
        resp = httpx.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {sendgrid_api_key.strip()}"},
            json={
                "personalizations": [{"to": [{"email": to_email}]}],
                "from": {"email": os.getenv("EMAIL_FROM", "no-reply@yourdomain.com")},
                "subject": subject,
                "content": [{"type": "text/html", "value": html_body}],
            },
            timeout=15.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"SendGrid HTTP error ({resp.status_code}): {resp.text}")
        _emit_log("email_sent_via_https", provider="sendgrid", to=to_email)
        return

    # 3. SMTP Transport
    raw_host = (os.getenv("SMTP_HOST") or os.getenv("SMTP_Host") or "").strip()
    # Strip protocol schemes and port numbers if present
    smtp_host = raw_host.removeprefix("http://").removeprefix("https://").removeprefix("smtp://").split(":")[0]
    smtp_port = int(os.getenv("SMTP_PORT", os.getenv("SMTP_Port", "587")))
    smtp_user = (os.getenv("SMTP_USER") or os.getenv("SMTP_User") or "").strip()
    smtp_password = (os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_Password") or "").strip().replace(" ", "")

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

    def _attempt_send(port: int, use_ssl: bool) -> None:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, port, context=tls_context, timeout=10) as server:
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, port, timeout=10) as server:
                server.ehlo()
                server.starttls(context=tls_context)
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

    try:
        _attempt_send(smtp_port, use_ssl=(smtp_port == 465))
    except (smtplib.SMTPServerDisconnected, ConnectionResetError, OSError) as exc:
        # If port 587 dropped connection, retry with implicit SSL on port 465
        if smtp_port != 465:
            _emit_log(
                "smtp_port_fallback",
                severity="WARNING",
                smtp_host=smtp_host,
                original_port=smtp_port,
                fallback_port=465,
                error=str(exc),
            )
            _attempt_send(465, use_ssl=True)
        else:
            raise


def request_box_move_approval(
    items: list[dict[str, str]],
    destination_parent_id: str,
    destination_parent_name: str,
    requested_by: str = "Web User",
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
    # Ensure routes are attached to the running FastAPI app
    _register_approval_routes()

    if not recipient or recipient.endswith("@yourdomain.com"):
        return {
            "status": "error",
            "message": "A valid approver email must be provided or configured via BOX_MOVE_APPROVER_EMAIL.",
        }
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

    safe_requester = html.escape(requested_by)
    safe_dest_name = html.escape(destination_parent_name)
    safe_dest_id = html.escape(str(destination_parent_id))
    safe_reason = html.escape(reason) if reason else "Not specified"

    item_rows = "".join(
        f"<li><strong>[{it['item_type'].upper()}]</strong> {html.escape(it['item_name'])} "
        f"(ID: <code>{html.escape(it['item_id'])}</code>)</li>"
        for it in normalized_items
    )

    html_body = f"""
    <h3>Box Move Approval Required ({len(normalized_items)} item(s))</h3>
    <p><strong>Requested By:</strong> {safe_requester}</p>
    <p><strong>Destination Parent Folder:</strong> {safe_dest_name} (ID: <code>{safe_dest_id}</code>)</p>
    <p><strong>Reason:</strong> {safe_reason}</p>
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
            smtp_host=os.getenv("SMTP_HOST"),
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


def check_box_move_status(approval_token: str) -> dict:
    """Check whether a Box move approval request was approved, rejected, or is still pending.

    Args:
        approval_token: The approval token returned when the move was requested.
    """
    token_clean = approval_token.strip()
    client = _get_datastore_client()
    key = client.key(APPROVAL_KIND, token_clean)
    entity = client.get(key)

    if not entity:
        return {"status": "not_found", "message": f"No move request found for token: {token_clean}"}

    return {
        "status": entity.get("status"),
        "destination_parent_name": entity.get("destination_parent_name"),
        "requested_by": entity.get("requested_by"),
        "created_at": str(entity.get("created_at")),
        "approved_at": str(entity.get("approved_at")) if entity.get("approved_at") else None,
        "rejected_at": str(entity.get("rejected_at")) if entity.get("rejected_at") else None,
        "items": entity.get("items", []),
        "execution_results": entity.get("execution_results", []),
    }


# Box MCP Toolset (move tools gated behind approval)
box_tools = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=MCP_SERVER_URL,
        timeout=30.0,
        httpx_client_factory=_mcp_httpx_client_factory,
    ),
    tool_filter=[
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
        "2. Identify the requester and any designated approver email address from the user request.\n"
        "3. Immediately batch all requested items into a call to request_box_move_approval with "
        "item_type ('file' or 'folder'), item_id, item_name, requested_by, and approver_email (if specified). Do not pause to ask for confirmation.\n"
        "4. Inform the user that the approval email has been dispatched and provide the approval token.\n"
        "5. If asked about the status of a move request, use check_box_move_status to check whether it has been approved."
    ),
    tools=[box_tools, request_box_move_approval, check_box_move_status],
    after_model_callback=log_model_response,
    after_agent_callback=log_agent_final_response,
)

# ---------------------------------------------------------------------------
# Webhook / Approval Endpoint Handling
# ---------------------------------------------------------------------------

approval_router = APIRouter()


async def _execute_box_move(item_id: str, item_type: str, dest_id: str) -> dict:
    """Invoke Box MCP server over SSE to move an item to destination_parent_id."""
    try:
        auth = CloudRunOIDCAuth(MCP_SERVER_URL)
        token = auth._get_valid_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        print("------In_execute_box_move 00 ------")

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(MCP_SERVER_URL, headers=headers, timeout=30.0) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tools = tools_resp.tools

                target_keyword = "file" if item_type == "file" else "folder"
                print("------In_execute_box_move 11 ------")
                move_tool = None
                for t in tools:
                    print("------In_execute_box_move 22 Tool : {t.name} ------")
                    name = t.name.lower()
                    if "move" in name and target_keyword in name:
                        print('----------Move Tool Found------------')
                        move_tool = t
                        break
                if not move_tool:
                    for t in tools:
                        if "move" in t.name.lower():
                            move_tool = t
                            break

                if not move_tool:
                    return {"status": "error", "message": f"No move tool available on Box MCP for {item_type}."}

                schema = getattr(move_tool, "inputSchema", {}) or {}
                props = schema.get("properties", {}) if isinstance(schema, dict) else {}

                print("------In_execute_box_move 33 ------")

                args = {}
                for k in [f"{target_keyword}_id", "item_id", "id", "file_id", "folder_id"]:
                    if k in props:
                        args[k] = item_id
                        break
                if not args and props:
                    args[list(props.keys())[0]] = item_id

                for k in ["destination_parent_id", "parent_id", "destination_folder_id", "parent"]:
                    if k in props:
                        args[k] = {"id": dest_id} if k == "parent" else dest_id
                        break

                print("------Calling move tool------")
                call_res = await session.call_tool(move_tool.name, arguments=args)
                call_res_json = (
                    call_res.model_dump_json(indent=2)
                    if hasattr(call_res, "model_dump_json")
                    else json.dumps(call_res, indent=2, default=str)
                )
                print(f"-------------Call response:\n{call_res_json}")
                return {"status": "success", "tool": move_tool.name, "result": str(call_res)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


async def process_approval_decision(token: str, decision: str) -> HTMLResponse:
    """Validate approval token in Datastore and execute or reject the move."""
    _emit_log(
        "-----approval_http_request_received-----",
        token_prefix=token[:8] if token else "",
        decision=decision,
    )

    decision_clean = decision.strip().lower()
    if decision_clean not in ("approve", "reject"):
        return HTMLResponse("<h2>Invalid decision parameter. Must be 'approve' or 'reject'.</h2>", status_code=400)

    client = _get_datastore_client()
    key = client.key(APPROVAL_KIND, token)
    entity = client.get(key)

    if not entity:
        return HTMLResponse(
            "<div style='font-family:sans-serif;padding:30px;max-width:600px;margin:auto;'>"
            "<h2 style='color:#d93025;'>Approval Request Not Found</h2>"
            "<p>This token is invalid or has expired.</p></div>",
            status_code=404,
        )

    current_status = entity.get("status", "PENDING")
    if current_status != "PENDING":
        return HTMLResponse(
            f"<div style='font-family:sans-serif;padding:30px;max-width:600px;margin:auto;'>"
            f"<h2>Request Already Processed</h2>"
            f"<p>This request has already been marked as <strong>{current_status}</strong>.</p></div>",
            status_code=200,
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    expires_at = entity.get("expires_at")
    if expires_at and now > expires_at:
        entity["status"] = "EXPIRED"
        client.put(entity)
        return HTMLResponse(
            "<div style='font-family:sans-serif;padding:30px;max-width:600px;margin:auto;'>"
            "<h2 style='color:#d93025;'>Approval Request Expired</h2>"
            "<p>This approval link has expired. A new move request must be initiated.</p></div>",
            status_code=410,
        )

    if decision_clean == "reject":
        entity["status"] = "REJECTED"
        entity["rejected_at"] = now
        client.put(entity)
        _emit_log("box_move_rejected", approval_token=token)
        return HTMLResponse(
            "<div style='font-family:sans-serif;padding:30px;max-width:600px;margin:auto;'>"
            "<h2 style='color:#d93025;'>Move Request Rejected</h2>"
            "<p>The requested file/folder moves were rejected. No items have been moved in Box.</p></div>",
            status_code=200,
        )

    # Decision is approve
    items = entity.get("items", [])
    dest_id = entity.get("destination_parent_id")
    dest_name = entity.get("destination_parent_name", dest_id)

    results = []
    for it in items:
        print(f"----------Moving the item: {it["item_id"]} to {dest_name}");
        res = await _execute_box_move(it["item_id"], it["item_type"], dest_id)
        print("------------move call completed--------");
        results.append({"item_id": it["item_id"], "item_name": it["item_name"], **res})

    entity["status"] = "APPROVED"
    entity["approved_at"] = now
    entity["execution_results"] = results
    client.put(entity)

    _emit_log("box_move_approved_and_executed", approval_token=token, results=results)

    rendered_items = "".join(
        f"<li><strong>[{it['item_type'].upper()}]</strong> {html.escape(it['item_name'])}</li>"
        for it in items
    )
    return HTMLResponse(
        f"""
        <div style="font-family:sans-serif;padding:30px;max-width:600px;margin:40px auto;border:1px solid #e0e0e0;border-radius:8px;">
            <h2 style="color:#137333;">✓ Move Request Approved & Executed</h2>
            <p>The following items have been successfully moved in Box to <strong>{html.escape(str(dest_name))}</strong>:</p>
            <ul>{rendered_items}</ul>
            <p style="color:#5f6368;font-size:12px;">Token: <code>{token[:8]}...</code> | Processed at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        """,
        status_code=200,
    )


@approval_router.get("/approvals/respond", response_class=HTMLResponse)
async def handle_approval_response(token: str = "", decision: str = ""):
    print(f"---------Decision: {decision}---------");
    return await process_approval_decision(token, decision)


def _register_approval_routes() -> None:
    """Bind approval router dynamically to any FastAPI instance in the host process."""
    try:
        from fastapi import FastAPI

        for obj in gc.get_objects():
            if isinstance(obj, FastAPI):
                if not any(getattr(r, "path", None) == "/approvals/respond" for r in obj.routes):
                    obj.include_router(approval_router)
                    _emit_log("approval_route_registered", path="/approvals/respond", app_id=id(obj))
    except Exception as exc:
        _emit_log("approval_route_registration_deferred", error=str(exc))


# Register route on startup
_register_approval_routes()

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
        "4. When asked about the status of an approval request, use check_box_move_status.\n"
        "5. Summarize folder contents, metadata updates, and any pending move approval requests clearly."
    ),
    tools=[box_tools, check_box_move_status],
    sub_agents=[move_approval_agent],
    after_model_callback=log_model_response,
    after_agent_callback=log_agent_final_response,
)