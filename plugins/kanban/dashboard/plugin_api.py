"""Kanban dashboard routes through the privileged writer boundary.

The dashboard is a client of the canonical ``bryceos`` board.  It never opens
SQLite, chooses a board, enumerates boards, or follows an attachment path.
Routes which do not map to one of the writer's finite operations fail closed
before touching the store.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
)
from pydantic import BaseModel, Field

from hermes_cli.kanban_writer import CANONICAL_SOCKET_PATH, WriterError, writer_request

log = logging.getLogger(__name__)
router = APIRouter()

_CANONICAL_BOARD = "bryceos"
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_FILENAME_RE = re.compile(r"[^\\/\x00-\x1f]{1,255}\Z")
_LINK_FIELDS = frozenset({"parent_ids", "child_ids"})
_SHOW_FIELDS = _LINK_FIELDS | {"parent_results", "worker_context"}
_PATH_FIELDS = frozenset({
    "workspace_path", "stored_path", "attachment_path", "db_path",
})

BOARD_COLUMNS = [
    "triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done",
]


def _writer_socket() -> Path:
    # Deliberately no environment or query-param override: the service owns
    # the socket location just as it owns the canonical board path.
    return CANONICAL_SOCKET_PATH


def _request_key(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail=(
                "Idempotency-Key is required for mutations and must be a "
                "stable 1-128 character identifier"
            ),
        )
    return value


def _writer(
    operation: str,
    args: Optional[dict[str, Any]] = None,
    *,
    mutation: bool = False,
    request_key: Optional[str] = None,
) -> Any:
    """Call the fixed writer and return its result; never fall back locally."""
    stable_key = _request_key(request_key) if mutation else None
    try:
        response = writer_request(
            _writer_socket(),
            operation,
            args or {},
            request_key=stable_key,
        )
    except (OSError, WriterError) as exc:
        raise HTTPException(status_code=503, detail=f"kanban writer unavailable: {exc}") from exc
    return response.get("result") if isinstance(response, dict) else response


def _decoded_attachment_limit() -> int:
    """Use the core writer's decoded limit, not a dashboard-local cap."""
    try:
        from hermes_cli.kanban_writer import MAX_ATTACHMENT_BYTES
    except ImportError as exc:  # pragma: no cover - migration guard
        raise HTTPException(
            status_code=503,
            detail=(
                "kanban writer unavailable: core writer must export "
                "MAX_ATTACHMENT_BYTES"
            ),
        ) from exc
    return int(MAX_ATTACHMENT_BYTES)


def _refuse(route: str) -> None:
    raise HTTPException(
        status_code=501,
        detail=f"kanban route refused by D0B admission boundary: {route}",
    )


def _canonical_board(board: Optional[str]) -> None:
    """Accept the frozen slug as a target only; never pass it to the writer."""
    if board not in (None, "", _CANONICAL_BOARD):
        raise HTTPException(status_code=400, detail="only the canonical bryceos board is admitted")


def _id(value: Any, name: str = "task_id") -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"{name} must be a bounded identifier")
    return value


def _task_dict(
    task: Any,
    *,
    latest_summary: Optional[str] = None,
    show: bool = False,
) -> Optional[dict[str, Any]]:
    if task is None:
        return None
    if not isinstance(task, dict):
        task = dict(task)
    result = dict(task)
    if _PATH_FIELDS.intersection(result):
        raise HTTPException(
            status_code=503,
            detail="kanban writer returned a path-bearing projection",
        )
    required = _SHOW_FIELDS if show else _LINK_FIELDS
    missing = sorted(required.difference(result))
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "kanban writer projection unavailable: missing "
                + ", ".join(missing)
            ),
        )
    if not isinstance(result.get("parent_ids"), list) or not isinstance(result.get("child_ids"), list):
        raise HTTPException(status_code=503, detail="kanban writer returned malformed link projection")
    if show and not isinstance(result.get("parent_results"), list):
        raise HTTPException(status_code=503, detail="kanban writer returned malformed parent-results projection")
    if show and not isinstance(result.get("worker_context"), str):
        raise HTTPException(status_code=503, detail="kanban writer returned malformed worker_context projection")
    result["latest_summary"] = latest_summary if latest_summary is not None else result.get("latest_summary")
    return result


def _event_dict(event: Any) -> dict[str, Any]:
    return dict(event) if isinstance(event, dict) else dict(event)


def _comment_dict(comment: Any) -> dict[str, Any]:
    return dict(comment) if isinstance(comment, dict) else dict(comment)


def _attachment_dict(attachment: Any) -> dict[str, Any]:
    result = dict(attachment) if isinstance(attachment, dict) else dict(attachment)
    if _PATH_FIELDS.intersection(result):
        raise HTTPException(status_code=503, detail="kanban writer returned a path-bearing attachment")
    return result


def _run_dict(run: Any) -> dict[str, Any]:
    return dict(run) if isinstance(run, dict) else dict(run)


def _read_list(operation: str, args: dict[str, Any]) -> list[Any]:
    """Read a writer list without turning unavailable data into ``[]``."""
    value = _writer(operation, args)
    if not isinstance(value, list):
        raise HTTPException(
            status_code=503,
            detail=f"kanban writer projection unavailable for {operation}",
        )
    return value


def _ensure_task(task_id: str) -> dict[str, Any]:
    task = _writer("show", {"task_id": _id(task_id)})
    if not task:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return task


def _task_result(result: Any) -> str:
    if isinstance(result, dict):
        task_id = result.get("task_id")
        if task_id:
            return str(task_id)
    if isinstance(result, str):
        return result
    raise HTTPException(status_code=502, detail="writer returned no task id")


def _ws_upgrade_authorized(ws: "WebSocket") -> bool:
    """Use the dashboard's existing websocket auth gate when available."""
    try:
        from hermes_cli import web_server

        return bool(web_server._ws_auth_ok(ws))
    except Exception:
        return True


@router.get("/board")
def get_board(
    tenant: Optional[str] = Query(None),
    include_archived: bool = Query(False),
    board: Optional[str] = Query(None),
    workflow_template_id: Optional[str] = Query(None),
    current_step_key: Optional[str] = Query(None),
):
    _canonical_board(board)
    tasks = _read_list("list", {
        "tenant": tenant,
        "include_archived": include_archived,
        "workflow_template_id": workflow_template_id,
        "current_step_key": current_step_key,
    })
    columns: dict[str, list[dict[str, Any]]] = {name: [] for name in BOARD_COLUMNS}
    if include_archived:
        columns["archived"] = []
    tenants: set[str] = set()
    assignees: set[str] = set()
    for raw in tasks:
        task = _task_dict(raw)
        if task is None:
            raise HTTPException(status_code=503, detail="kanban writer returned a malformed task projection")
        if task.get("tenant"):
            tenants.add(str(task["tenant"]))
        if task.get("assignee") and task.get("status") != "archived":
            assignees.add(str(task["assignee"]))
        status = task.get("status")
        task["link_counts"] = {
            "parents": len(task["parent_ids"]),
            "children": len(task["child_ids"]),
        }
        columns.setdefault(status if status in columns else "todo", []).append(task)
    return {
        "columns": [{"name": name, "tasks": values} for name, values in columns.items()],
        "tenants": sorted(tenants),
        "assignees": sorted(assignees),
        # Board-level event cursors are not admitted by the fixed writer yet.
        "latest_event_id": None,
        "now": int(time.time()),
    }


@router.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    board: Optional[str] = Query(None),
    run_state_type: Optional[str] = Query(None),
    run_state_name: Optional[str] = Query(None),
):
    _canonical_board(board)
    if (run_state_type is None) != (run_state_name is None):
        raise HTTPException(status_code=400, detail="run_state_type and run_state_name must be passed together or omitted")
    if run_state_type not in (None, "status", "outcome"):
        raise HTTPException(status_code=400, detail="run_state_type must be 'status' or 'outcome'")
    task_id = _id(task_id)
    task = _ensure_task(task_id)
    task = _task_dict(task, show=True) or {}
    task["link_counts"] = {
        "parents": len(task["parent_ids"]),
        "children": len(task["child_ids"]),
    }
    runs_args: dict[str, Any] = {"task_id": task_id}
    if run_state_type:
        runs_args["state_type"] = run_state_type
        runs_args["state_name"] = run_state_name
    payload = {
        "task": task,
        "comments": [_comment_dict(v) for v in _read_list("comments", {"task_id": task_id})],
        "events": [_event_dict(v) for v in _read_list("events", {"task_id": task_id})],
        "attachments": [_attachment_dict(v) for v in _read_list("attachments", {"task_id": task_id})],
        "links": {
            "parents": list(task["parent_ids"]),
            "children": list(task["child_ids"]),
        },
        "parent_results": list(task["parent_results"]),
        "worker_context": task["worker_context"],
        "runs": [_run_dict(v) for v in _read_list("runs", runs_args)],
    }
    # Preserve a truthful child-result projection when the writer supplies
    # one; never manufacture an empty child-results array for the UI.
    if "child_results" in task:
        if not isinstance(task["child_results"], list):
            raise HTTPException(status_code=503, detail="kanban writer returned malformed child-results projection")
        payload["child_results"] = list(task["child_results"])
    return payload


class _StrictMutationBody(BaseModel):
    class Config:
        extra = "forbid"


class CreateTaskBody(_StrictMutationBody):
    title: str
    body: Optional[str] = None
    assignee: Optional[str] = None
    tenant: Optional[str] = None
    priority: int = 0
    parents: list[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"


@router.post("/tasks")
def create_task(
    payload: CreateTaskBody,
    board: Optional[str] = Query(None),
    request_key: str = Header(..., alias="Idempotency-Key"),
):
    _canonical_board(board)
    values = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    result = _writer("create", values, mutation=True, request_key=request_key)
    task_id = _task_result(result)
    return {"task": _task_dict(_ensure_task(task_id))}


@router.get("/tasks/{task_id}/attachments")
def list_task_attachments(task_id: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    task_id = _id(task_id)
    _ensure_task(task_id)
    return {"attachments": [_attachment_dict(v) for v in _read_list("attachments", {"task_id": task_id})]}


@router.post("/tasks/{task_id}/attachments")
async def upload_task_attachment(
    task_id: str,
    file: UploadFile = File(...),
    board: Optional[str] = Query(None),
    uploaded_by: Optional[str] = Form(None),
    request_key: str = Header(..., alias="Idempotency-Key"),
):
    _canonical_board(board)
    if uploaded_by:
        raise HTTPException(status_code=400, detail="uploaded_by is derived from the writer peer")
    task_id = _id(task_id)
    _ensure_task(task_id)
    filename = file.filename or "attachment"
    if not _FILENAME_RE.fullmatch(filename) or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="attachment filename must be a basename")
    max_bytes = _decoded_attachment_limit()
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"attachment exceeds {max_bytes} decoded bytes",
        )
    result = _writer("attach", {
        "task_id": task_id,
        "filename": filename,
        "content_type": file.content_type,
        "data_b64": base64.b64encode(content).decode("ascii"),
    }, mutation=True, request_key=request_key)
    return {"attachment": result}


@router.get("/attachments/{attachment_id}")
def download_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("attachment download (writer exposes bytes, not paths, in a later packet)")


@router.delete("/attachments/{attachment_id}")
def remove_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("attachment delete")


class UpdateTaskBody(_StrictMutationBody):
    status: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[int] = None
    title: Optional[str] = None
    body: Optional[str] = None
    result: Optional[str] = None
    block_reason: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    class Config:
        extra = "forbid"


@router.patch("/tasks/{task_id}")
def update_task(task_id: str, payload: UpdateTaskBody, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("generic task update")


@router.delete("/tasks/{task_id}")
def delete_task(task_id: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("task delete")


class CommentBody(_StrictMutationBody):
    body: str
    author: Optional[str] = None


@router.post("/tasks/{task_id}/comments")
def add_comment(
    task_id: str,
    payload: CommentBody,
    board: Optional[str] = Query(None),
    request_key: str = Header(..., alias="Idempotency-Key"),
):
    _canonical_board(board)
    if payload.author is not None:
        raise HTTPException(status_code=400, detail="author is derived from the writer peer")
    task_id = _id(task_id)
    _ensure_task(task_id)
    result = _writer(
        "comment", {"task_id": task_id, "body": payload.body},
        mutation=True, request_key=request_key,
    )
    return {"comment_id": result.get("comment_id") if isinstance(result, dict) else result}


class LinkBody(_StrictMutationBody):
    parent_id: str
    child_id: str

    class Config:
        extra = "forbid"


@router.post("/links")
def add_link(
    payload: LinkBody,
    board: Optional[str] = Query(None),
    request_key: str = Header(..., alias="Idempotency-Key"),
):
    _canonical_board(board)
    args = {"parent_id": _id(payload.parent_id, "parent_id"), "child_id": _id(payload.child_id, "child_id")}
    return _writer("link", args, mutation=True, request_key=request_key)


@router.delete("/links")
def delete_link(
    parent_id: str = Query(...),
    child_id: str = Query(...),
    board: Optional[str] = Query(None),
    request_key: str = Header(..., alias="Idempotency-Key"),
):
    _canonical_board(board)
    return _writer(
        "unlink",
        {"parent_id": _id(parent_id, "parent_id"), "child_id": _id(child_id, "child_id")},
        mutation=True,
        request_key=request_key,
    )


class BulkTaskBody(_StrictMutationBody):
    ids: list[str]
    status: Optional[str] = None


@router.post("/tasks/bulk")
def bulk_update(payload: BulkTaskBody, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("bulk task update")


@router.get("/diagnostics")
def list_diagnostics(board: Optional[str] = Query(None), task_id: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("diagnostics")


@router.get("/workers/active")
def list_active_workers(board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("worker enumeration")


@router.get("/runs/{run_id}")
def get_run_endpoint(run_id: int, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("run lookup without a canonical task target")


@router.get("/runs/{run_id}/inspect")
def inspect_run_endpoint(run_id: int, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("run inspection")


@router.post("/runs/{run_id}/terminate")
def terminate_run_endpoint(run_id: int, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("run termination")


@router.post("/tasks/{task_id}/reclaim")
def reclaim_task_endpoint(task_id: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("worker recovery/reclaim")


@router.post("/tasks/{task_id}/specify")
def specify_task_endpoint(task_id: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("triage specification")


@router.post("/tasks/{task_id}/reassign")
def reassign_task_endpoint(task_id: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("worker reassignment")


@router.post("/estimate")
def estimate_text_endpoint(board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("estimate")


@router.post("/tasks/{task_id}/estimate")
def estimate_task_endpoint(task_id: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("estimate")


@router.get("/config")
def get_config():
    """Return the dashboard.kanban display settings from Hermes config."""
    try:
        from hermes_cli.config import load_config
        raw = load_config() or {}
    except Exception:
        raw = {}
    section = ((raw or {}).get("dashboard") or {}).get("kanban") or {}
    return {
        "default_tenant": section.get("default_tenant") or "",
        "lane_by_profile": bool(section.get("lane_by_profile", True)),
        "include_archived_by_default": bool(section.get("include_archived_by_default", False)),
        "render_markdown": bool(section.get("render_markdown", True)),
    }


@router.get("/home-channels")
def get_home_channels(board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("home-channel enumeration")


@router.post("/tasks/{task_id}/home-subscribe/{platform}")
def subscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("home-channel subscription")


@router.delete("/tasks/{task_id}/home-subscribe/{platform}")
def unsubscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("home-channel subscription")


@router.get("/stats")
def get_stats(board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("board stats")


@router.get("/assignees")
def get_assignees(board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("assignee enumeration")


@router.get("/tasks/{task_id}/log")
def get_task_log(task_id: str, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("worker-log path read")


@router.post("/dispatch")
def dispatch(board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("dispatcher")


@router.get("/model-options")
def model_options():
    """Return the same provider/model picker used by the rest of Hermes."""
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        payload = build_models_payload(
            load_picker_context(),
            explicit_only=True,
            canonical_order=True,
            probe_custom_providers=False,
        )
        return {
            "providers": [
                {
                    "slug": row.get("slug", ""),
                    "label": row.get("label") or row.get("slug", ""),
                    "models": list(row.get("models") or []),
                }
                for row in payload.get("providers", [])
                if row.get("models")
            ],
        }
    except Exception:
        log.exception("kanban model-options failed")
        return {"providers": []}


@router.get("/projects")
def list_kanban_projects():
    _refuse("project enumeration")


@router.get("/boards")
def list_boards(include_archived: bool = Query(False)):
    _refuse("board enumeration")


class CreateBoardBody(_StrictMutationBody):
    slug: str


class RenameBoardBody(_StrictMutationBody):
    name: Optional[str] = None


@router.post("/boards")
def create_board_endpoint(payload: CreateBoardBody):
    _refuse("board creation")


@router.patch("/boards/{slug}")
def rename_board(slug: str, payload: RenameBoardBody):
    _refuse("board rename")


@router.delete("/boards/{slug}")
def delete_board(slug: str, delete: bool = Query(False)):
    _refuse("board deletion")


@router.post("/boards/{slug}/switch")
def switch_board(slug: str):
    _refuse("board selection")


@router.get("/profiles")
def list_profile_roster():
    """Read installed profile metadata without touching the Kanban store."""
    try:
        from hermes_cli import profiles as profiles_mod
        profiles = profiles_mod.list_profiles()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to list profiles: {exc}") from exc
    return {
        "profiles": [
            {
                "name": profile.name,
                "is_default": bool(profile.is_default),
                "model": profile.model or "",
                "provider": profile.provider or "",
                "description": profile.description or "",
                "description_auto": bool(profile.description_auto),
                "skill_count": int(profile.skill_count or 0),
            }
            for profile in profiles
        ],
    }


class DescribeBody(_StrictMutationBody):
    description: Optional[str] = None


class DescribeAutoBody(_StrictMutationBody):
    overwrite: bool = False


@router.patch("/profiles/{profile_name}")
def update_profile_description(profile_name: str, payload: DescribeBody):
    """Update profile metadata; this is outside the Kanban store boundary."""
    try:
        from hermes_cli import profiles as profiles_mod
        canon = profiles_mod.normalize_profile_name(profile_name)
        if canon == "default":
            from hermes_constants import get_hermes_home
            profile_dir = Path(get_hermes_home())
        else:
            profile_dir = profiles_mod.get_profile_dir(canon)
        if not profile_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"profile '{profile_name}' not found")
        text = (payload.description or "").strip()
        profiles_mod.write_profile_meta(
            profile_dir,
            description=text,
            description_auto=False,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to update profile: {exc}") from exc
    return {"ok": True, "profile": canon, "description": text}


@router.post("/profiles/{profile_name}/describe-auto")
def auto_describe_profile(profile_name: str, payload: DescribeAutoBody):
    """Generate and persist a profile description through the profile service."""
    try:
        from hermes_cli import profile_describer
        outcome = profile_describer.describe_profile(
            profile_name,
            overwrite=bool(payload.overwrite),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"describer crashed: {exc}") from exc
    return {
        "ok": bool(outcome.ok),
        "profile": outcome.profile_name,
        "reason": outcome.reason,
        "description": outcome.description,
    }


class DecomposeBody(_StrictMutationBody):
    prompt: Optional[str] = None


@router.post("/tasks/{task_id}/decompose")
def decompose_task_endpoint(task_id: str, payload: DecomposeBody, board: Optional[str] = Query(None)):
    _canonical_board(board)
    _refuse("auto-decompose")


class OrchestrationSettingsBody(_StrictMutationBody):
    auto_decompose: Optional[bool] = None


@router.get("/orchestration")
def get_orchestration_settings():
    _refuse("orchestration settings")


@router.put("/orchestration")
def set_orchestration_settings(payload: OrchestrationSettingsBody):
    _refuse("orchestration settings")


@router.websocket("/events")
async def stream_events(ws: WebSocket):
    if not _ws_upgrade_authorized(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    await ws.close(code=1013, reason="kanban event enumeration is refused until writer IPC exposes a subscription read")


__all__ = ["router"]
