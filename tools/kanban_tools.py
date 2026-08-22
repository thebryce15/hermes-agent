"""Kanban tools — structured tool-call surface for worker + orchestrator agents.

These tools are registered into the model's schema when the agent is
running under the dispatcher (env var ``HERMES_KANBAN_TASK`` set) or when
the active profile explicitly enables the ``kanban`` toolset for
orchestrator work. A normal ``hermes chat`` session still sees **zero**
kanban tools in its schema unless configured.

Why tools instead of just shelling out to ``hermes kanban``?

1. **Backend portability.** A worker whose terminal tool points at Docker
   / Modal / Singularity / SSH still uses the local privileged writer IPC;
   it never needs a board mount or a second direct SQLite client.

2. **No shell-quoting footguns.** Passing ``--metadata '{"x": [...]}'``
   through shlex+argparse is fragile. Structured tool args skip it.

3. **Better errors.** Tool-call failures return structured JSON the
   model can reason about, not stderr strings it has to parse.

Humans continue to use the CLI (``hermes kanban …``), the dashboard
(``hermes dashboard``), and the slash command (``/kanban …``) — all
three bypass the agent entirely. The tools are for dispatcher-spawned
worker handoffs and for configured orchestrator profiles that route work
through the board.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import secrets
import threading
from typing import Any, Optional

from agent.redact import redact_sensitive_text
from hermes_cli.goals import judge_goal
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get, load_config

logger = logging.getLogger(__name__)

_WRITER_MUTATIONS = frozenset({
    "create", "link", "unlink", "comment", "block", "unblock",
    "assign", "claim", "heartbeat", "complete", "attach",
    "notify-subscribe", "notify-unsubscribe", "notify-claim",
    "notify-advance", "notify-rewind",
})
_REQUEST_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

KANBAN_LIST_DEFAULT_LIMIT = 50
KANBAN_LIST_MAX_LIMIT = 200


def _profile_has_kanban_toolset() -> bool:
    # Uses load_config() which has mtime-based caching, so this adds
    # negligible overhead. The check_fn results are further TTL-cached
    # (~30s) by the tool registry.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        toolsets = cfg.get("toolsets", [])
        return "kanban" in toolsets
    except Exception:
        return False


def _is_delegated_child_context() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


def _reject_delegated_child_mutation(tool_name: str) -> Optional[str]:
    """Deny Kanban mutations from delegate_task children.

    A delegate_task child runs in the same process as its parent, so stale or
    inherited HERMES_KANBAN_* env vars are not proof of dispatcher ownership.
    The child may summarize findings to its parent, but it must not complete,
    block, heartbeat, comment, create, link, or unblock board tasks directly.
    """
    if not _is_delegated_child_context():
        return None
    return tool_error(
        f"{tool_name} refused: delegate_task child agents are not Kanban "
        "run owners. Return findings to the parent agent; the dispatcher "
        "worker or an explicitly configured Kanban orchestrator must perform "
        "board mutations."
    )


def _check_kanban_mode() -> bool:
    """Task-lifecycle tools are available when:

    1. ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), OR
    2. The current profile has ``kanban`` in its toolsets config
       (orchestrator profiles like techlead that route work via Kanban).

    Humans running ``hermes chat`` without the kanban toolset see zero
    kanban tools. Workers spawned by the kanban dispatcher (gateway-
    embedded by default) and orchestrator profiles with the kanban
    toolset enabled see the Kanban lifecycle tool surface.
    """
    if _is_delegated_child_context():
        return False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return True
    return _profile_has_kanban_toolset()


def _check_kanban_orchestrator_mode() -> bool:
    """Board-routing tools (kanban_list, kanban_unblock) are intentionally
    hidden from task workers.

    Dispatcher-spawned workers should close their own task via the
    lifecycle tools (complete/block/heartbeat), not enumerate or unblock
    board state. Profiles that explicitly opt into the kanban toolset
    and are NOT scoped to a single task are the orchestrator surface.
    """
    if _is_delegated_child_context():
        return False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False
    return _profile_has_kanban_toolset()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_task_id(arg: Optional[str]) -> Optional[str]:
    """Resolve ``task_id`` arg or fall back to the env var the dispatcher set."""
    if arg:
        return arg
    if _is_delegated_child_context():
        return None
    env_tid = os.environ.get("HERMES_KANBAN_TASK")
    return env_tid or None


def _worker_run_id(task_id: str) -> Optional[int]:
    """Return this worker's dispatcher run id when it is scoped to task_id."""
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    raw = os.environ.get("HERMES_KANBAN_RUN_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _stamp_worker_session_metadata(
    task_id: str, metadata: Optional[dict]
) -> Optional[dict]:
    """Compatibility shim; writer attribution never comes from request data."""
    return metadata


def _enforce_worker_task_ownership(tid: str) -> Optional[str]:
    """Reject worker-driven destructive calls on foreign task IDs.

    A process spawned by the dispatcher has ``HERMES_KANBAN_TASK`` set
    to its own task id. Tools like ``kanban_complete`` / ``kanban_block``
    / ``kanban_heartbeat`` mutate run-lifecycle state, so a buggy or
    prompt-injected worker that passed an explicit ``task_id`` for some
    other task could corrupt sibling or cross-tenant runs (see #19534).

    Orchestrator profiles (kanban toolset enabled but **no**
    ``HERMES_KANBAN_TASK`` in env) aren't subject to this check — their
    job is routing, and they sometimes legitimately close out child
    tasks or reopen blocked ones. Workers are narrowly scoped to their
    one task.

    Returns ``None`` when the call is allowed, or a tool-error string
    when it must be rejected. Callers should ``return`` the error
    verbatim.
    """
    env_tid = os.environ.get("HERMES_KANBAN_TASK")
    if not env_tid:
        # Orchestrator or CLI context — no task-scope restriction.
        return None
    if tid != env_tid:
        return tool_error(
            f"worker is scoped to task {env_tid}; refusing to mutate "
            f"{tid}. Use kanban_comment to hand off information to other "
            f"tasks, or kanban_create to spawn follow-up work."
        )
    return None


def _writer_request(
    operation: str,
    args: Optional[dict[str, Any]] = None,
    *,
    request_key: Optional[str] = None,
) -> dict[str, Any]:
    """Send one fixed Kanban request to the privileged writer.

    The socket is a process-local, fixed target.  There is intentionally no
    board, database, or direct-connect fallback here: the writer owns target
    selection and derives caller identity from peer credentials.
    """
    from hermes_cli.kanban_writer import writer_request

    key = request_key
    if operation in _WRITER_MUTATIONS:
        if not isinstance(key, str) or not _REQUEST_KEY_RE.fullmatch(key):
            raise ValueError(
                f"{operation}: a stable request_key is required before writer IPC"
            )
    # ``writer_request`` owns the canonical socket target.  Adapters must not
    # reconstruct a profile/home-relative path or accept a caller-selected
    # board/socket.
    return writer_request(operation=operation, args=args or {}, request_key=key)


def _tool_request_key(args: dict[str, Any], tool_name: str) -> tuple[Optional[str], Optional[str]]:
    value = args.get("request_key")
    if not isinstance(value, str) or not _REQUEST_KEY_RE.fullmatch(value):
        return None, tool_error(
            f"{tool_name}: request_key is required and must be a stable "
            "1-128 character identifier"
        )
    return value, None


def _derived_request_key(request_key: str, purpose: str) -> str:
    digest = hashlib.sha256(f"{request_key}\0{purpose}".encode("utf-8")).hexdigest()
    return f"kanban-{digest}"


def _writer_result(response: dict[str, Any]) -> Any:
    return response.get("result") if isinstance(response, dict) else None


def _decoded_attachment_limit() -> int:
    """Return the writer's conservative decoded-byte cap.

    The native writer is the authority for the 1 MiB frame budget.  Keep this
    import local so the rest of the tool registry can still load while a
    checkout is being migrated; an attachment request fails explicitly until
    the core worker exports the decoded cap.
    """
    try:
        from hermes_cli.kanban_writer import MAX_ATTACHMENT_BYTES
    except ImportError as exc:  # pragma: no cover - migration guard
        raise RuntimeError(
            "kanban attachment adapter requires the core writer to export "
            "MAX_ATTACHMENT_BYTES"
        ) from exc
    return int(MAX_ATTACHMENT_BYTES)


_MAX_ATTACH_URL_REDIRECTS = 5


def _download_url_with_cap(url: str, max_bytes: int) -> tuple[bytes, Optional[str]]:
    """Fetch an HTTP(S) URL into bounded inline bytes.

    Redirects are followed manually so every target is checked by the shared
    URL-safety policy before the next request.  The response is streamed into
    memory and rejected as soon as it exceeds the writer's decoded-byte cap;
    no temporary file or caller-controlled path is involved.
    """
    from urllib.parse import urljoin, urlparse

    import httpx

    from tools.url_safety import is_safe_url

    current_url = str(url).strip()
    for _ in range(_MAX_ATTACH_URL_REDIRECTS + 1):
        parsed = urlparse(current_url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in {"http", "https"}:
            raise ValueError(
                f"unsupported URL scheme {scheme!r}; only http/https are allowed"
            )
        if not is_safe_url(current_url):
            raise ValueError(
                "URL blocked by SSRF protection (private/internal address)"
            )

        chunks: list[bytes] = []
        total = 0
        with httpx.stream(
            "GET",
            current_url,
            headers={"User-Agent": "hermes-kanban/attach"},
            timeout=30,
            follow_redirects=False,
        ) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("redirect response has no Location header")
                current_url = urljoin(current_url, str(location))
                continue

            response.raise_for_status()
            content_type = (
                (response.headers.get("content-type") or "")
                .split(";", 1)[0]
                .strip()
                or None
            )
            for chunk in response.iter_bytes(64 * 1024):
                data = bytes(chunk)
                total += len(data)
                if total > max_bytes:
                    raise ValueError(
                        f"attachment exceeds {max_bytes} decoded bytes"
                    )
                chunks.append(data)
        return b"".join(chunks), content_type

    raise ValueError(f"too many redirects fetching {url}")


_LINK_FIELDS = frozenset({"parent_ids", "child_ids"})
_SHOW_FIELDS = _LINK_FIELDS | {"parent_results", "worker_context"}
_PATH_FIELDS = frozenset({
    "workspace_path", "stored_path", "attachment_path", "db_path",
})


def _require_path_free_projection(
    value: Any,
    *,
    operation: str,
    fields: frozenset[str] = _SHOW_FIELDS,
) -> dict[str, Any]:
    """Require the native writer's truthful, path-free read projection."""
    if not isinstance(value, dict):
        raise ValueError(f"{operation}: writer projection unavailable")
    if _PATH_FIELDS.intersection(value):
        raise ValueError(f"{operation}: writer returned a path-bearing projection")
    missing = sorted(fields.difference(value))
    if missing:
        raise ValueError(
            f"{operation}: writer projection unavailable (missing {', '.join(missing)})"
        )
    if not isinstance(value["parent_ids"], list) or not isinstance(value["child_ids"], list):
        raise ValueError(f"{operation}: writer returned malformed link projection")
    if "parent_results" in fields and not isinstance(value["parent_results"], list):
        raise ValueError(f"{operation}: writer returned malformed parent-results projection")
    if "worker_context" in fields and not isinstance(value["worker_context"], str):
        raise ValueError(f"{operation}: writer returned malformed worker_context projection")
    return value


def _reject_board_arg(args: dict, tool_name: str) -> Optional[str]:
    """Reject caller-selected board identity before the writer is reached."""
    if "board" in args:
        return tool_error(
            f"{tool_name} refused: board selection is fixed by the privileged "
            "writer; remove the board argument"
        )
    return None


def _path_free_completion_metadata(value: Any) -> bool:
    """Allow only flat scalar completion metadata with no path-shaped data."""
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        key_text = str(key).casefold()
        if not isinstance(key, str) or any(
            token in key_text for token in ("path", "file", "artifact")
        ):
            return False
        if isinstance(item, (dict, list, tuple)):
            return False
        if item is not None and not isinstance(item, (str, int, float, bool)):
            return False
        if isinstance(item, str) and any(sep in item for sep in ("/", "\\")):
            return False
    return True


_GOAL_MODE_BLOCK_ALLOWED_KINDS = frozenset({"dependency", "needs_input"})


def _goal_judge_available() -> bool:
    """True when an auxiliary client is configured for the goal judge.

    ``judge_goal`` is fail-open at the source: when no auxiliary model can
    be reached it returns a ``"continue"`` verdict that is indistinguishable
    from a real "not done yet" judgment. The completion gate must not treat
    that as a rejection, or an unconfigured/degraded auxiliary model would
    wedge every ``goal_mode`` worker (it could never close its own task).

    So we probe availability first and only enforce the gate when a judge is
    actually reachable. This mirrors the same client lookup ``judge_goal``
    performs internally.
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return False
    return client is not None and bool(model)


# ---------------------------------------------------------------------------
# Runtime-activity → board-heartbeat bridge (#31752)
# ---------------------------------------------------------------------------
# When the agent ticks ``_touch_activity`` during normal work (between
# tool calls, mid-stream chunks, etc.), we want the kanban board's
# ``last_heartbeat_at`` columns to reflect that liveness so the dispatcher
# watchdog (which reads ``tasks.last_heartbeat_at``, not the agent's
# in-process timestamp) doesn't reclaim an actively-running worker as
# stale. The model is not required to call the explicit ``kanban_heartbeat``
# tool for this to work — that tool stays available for workers that want
# to attach a note or pre-emptively extend a claim across a known-long op.
#
# Constraints:
#   - Best-effort: never raise. The agent loop must not care if the bridge
#     fails (board missing, DB locked, etc.).
#   - Rate-limited to one DB write per 60s per-process; runtime activity
#     can tick on every chunk/tool result and we don't need that resolution.
#   - No-op outside dispatcher-spawned worker context (no ``HERMES_KANBAN_TASK``).
#   - No durable note on these auto-heartbeats; that's reserved for the
#     explicit tool which carries a model-supplied note.

_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_auto_heartbeat_last_attempt: float = 0.0
_AUTO_HEARTBEAT_MAX_PENDING = 32
_auto_heartbeat_pending_keys: dict[tuple[str, str], str] = {}
_auto_heartbeat_lock = threading.Lock()


def heartbeat_current_worker_from_env() -> bool:
    """Best-effort: extend the kanban claim + bump board heartbeat for the
    current dispatcher-spawned worker, using identity from env vars.

    Returns True if a write was attempted (whether or not it succeeded);
    False if the call was skipped (not a kanban worker, rate-limited, or
    swallowed exception). The boolean is informational — callers should
    not branch on it.

    Identity comes from:
      * ``HERMES_KANBAN_TASK`` — task id (required; absence means no-op)
      * ``HERMES_KANBAN_RUN_ID`` — pins the run row so we don't heartbeat
        a stale run that may have already been reclaimed
      * ``HERMES_KANBAN_CLAIM_LOCK`` — claim lock for ``heartbeat_claim``;
        falls back to the default ``_claimer_id()`` for locally-driven
        workers that never went through the dispatcher path

    Rate limiting and the retry key are protected by a process-local lock.
    A lost reply retains the task/run's pending key; an acknowledged writer
    response clears it so the next deliberate heartbeat receives a new key.
    """
    global _auto_heartbeat_last_attempt
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid:
        return False
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID") or ""
    identity = (tid, run_id)
    import time as _time
    now = _time.monotonic()
    with _auto_heartbeat_lock:
        if (
            now - _auto_heartbeat_last_attempt
        ) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
            return False
        _auto_heartbeat_last_attempt = now
        request_key = _auto_heartbeat_pending_keys.get(identity)
        if request_key is None:
            request_key = f"auto-heartbeat-{secrets.token_hex(16)}"
            _auto_heartbeat_pending_keys[identity] = request_key
            while len(_auto_heartbeat_pending_keys) > _AUTO_HEARTBEAT_MAX_PENDING:
                oldest = next(iter(_auto_heartbeat_pending_keys))
                _auto_heartbeat_pending_keys.pop(oldest, None)
    try:
        response = _writer_request(
            "heartbeat", {"task_id": tid}, request_key=request_key
        )
    except Exception:
        logger.debug("auto-heartbeat: bridge failed", exc_info=True)
        return False
    if not isinstance(response, dict) or response.get("ok") is not True:
        logger.debug("auto-heartbeat: writer did not acknowledge success")
        return False
    with _auto_heartbeat_lock:
        if _auto_heartbeat_pending_keys.get(identity) == request_key:
            _auto_heartbeat_pending_keys.pop(identity, None)
    return True


# Live operator-note injection: poll the worker's task for new comments and
# fold them into the running agent via the OUT-OF-BAND steer channel, so a user
# can "talk to" a running kanban task without the block → comment → unblock
# dance (or a restart). Rate-limited on its own (tighter than the 60s heartbeat
# so notes land within a few seconds), watermarked per task id.
_COMMENT_POLL_MIN_INTERVAL_SECONDS = 6.0
_comment_poll_last_attempt: float = 0.0
# task_id -> highest comment id already seen (seeded on first poll so history
# already present in build_worker_context isn't re-injected).
_comment_watermark: dict[str, int] = {}


def inject_new_comments_from_env(agent: Any) -> bool:
    """Fold new operator comments on the current worker's task into ``agent``.

    Best-effort and self-gating: no-op unless this process is a kanban worker
    (``HERMES_KANBAN_TASK`` set) and ``agent`` exposes ``steer``. Returns True
    if a steer was injected, else False. Never raises into the agent loop.

    The first poll only *seeds* the watermark to the newest existing comment —
    those are already in the worker's context — so only comments added after
    the run started are injected. The worker's own authored comments (matched
    by ``HERMES_PROFILE``) are skipped to avoid echoing itself.
    """
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid or agent is None or not hasattr(agent, "steer"):
        return False
    global _comment_poll_last_attempt
    import time as _time
    now = _time.monotonic()
    if (now - _comment_poll_last_attempt) < _COMMENT_POLL_MIN_INTERVAL_SECONDS:
        return False
    _comment_poll_last_attempt = now

    seen = _comment_watermark.get(tid)
    try:
        rows = _writer_result(
            _writer_request(
                "comments", {"task_id": tid, "after_id": seen or 0}
            )
        ) or []
    except Exception:
        logger.debug("comment-inject: bridge failed", exc_info=True)
        return False

    if seen is None:
        # First poll for this task: seed past the existing thread, inject nothing.
        _comment_watermark[tid] = max(
            (int(c.get("id", 0)) if isinstance(c, dict) else int(getattr(c, "id", 0)) for c in rows),
            default=0,
        )
        return False
    if not rows:
        return False

    # Advance the watermark past everything we just read (including our own
    # notes) so nothing is re-injected next poll.
    _comment_watermark[tid] = max(
        (int(c.get("id", 0)) if isinstance(c, dict) else int(getattr(c, "id", 0)) for c in rows),
        default=0,
    )

    own = (os.environ.get("HERMES_PROFILE") or "").strip()
    fresh = [
        c for c in rows
        if (
            (c.get("author") if isinstance(c, dict) else getattr(c, "author", "")) or ""
        ).strip() != own
        and (
            (c.get("body") if isinstance(c, dict) else getattr(c, "body", "")) or ""
        ).strip()
    ]
    if not fresh:
        return False

    lines = [
        f"- {((c.get('author') if isinstance(c, dict) else getattr(c, 'author', '')) or 'operator')}: "
        f"{((c.get('body') if isinstance(c, dict) else getattr(c, 'body', '')) or '').strip()}"
        for c in fresh
    ]
    note = (
        "New note"
        + ("s" if len(fresh) > 1 else "")
        + " on your kanban task from the operator (delivered mid-run). "
        + "Take it into account for the work you're doing right now:\n"
        + "\n".join(lines)
    )
    try:
        return bool(agent.steer(note))
    except Exception:
        logger.debug("comment-inject: steer failed", exc_info=True)
        return False


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _normalize_profile(value: Any) -> Optional[str]:
    """Normalize CLI-compatible assignee sentinels for the tool surface."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "-", "null"}:
        return None
    return text


def _parse_bool_arg(args: dict, name: str, *, default: bool = False):
    value = args.get(name)
    if value is None:
        return default, None
    if isinstance(value, bool):
        return value, None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True, None
    if text in {"false", "0", "no"}:
        return False, None
    return default, f"{name} must be a boolean or 'true'/'false'"


def _require_orchestrator_tool(tool_name: str) -> Optional[str]:
    """Belt-and-suspenders runtime guard for orchestrator-only handlers.

    The check_fn (`_check_kanban_orchestrator_mode`) keeps these tools
    out of the worker schema entirely, but in case a stale registration
    or test harness routes a worker to one of them anyway, return a
    structured tool_error so the model gets a clear refusal instead of
    silently mutating board state from a worker context.
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return tool_error(
            f"{tool_name} is orchestrator-only; dispatcher-spawned workers "
            "must use kanban_complete, kanban_block, kanban_heartbeat, or "
            "kanban_comment for their assigned task."
        )
    return None


def _task_summary_dict(task: Any) -> dict[str, Any]:
    """Compact task shape for board-listing tools."""
    if not isinstance(task, dict):
        raise ValueError("kanban_list: writer projection unavailable")
    get = task.get
    _require_path_free_projection(task, operation="kanban_list", fields=_LINK_FIELDS)
    parents = list(task["parent_ids"])
    children = list(task["child_ids"])
    return {
        "id": get("id"),
        "title": get("title"),
        "assignee": get("assignee"),
        "status": get("status"),
        "priority": get("priority"),
        "tenant": get("tenant"),
        "workspace_kind": get("workspace_kind"),
        "project_id": get("project_id"),
        "created_at": get("created_at"),
        "started_at": get("started_at"),
        "completed_at": get("completed_at"),
        "current_run_id": get("current_run_id"),
        "model_override": get("model_override"),
        "provider_override": get("provider_override"),
        "parent_ids": parents,
        "child_ids": children,
        "parent_count": len(parents),
        "child_count": len(children),
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_show(args: dict, **kw) -> str:
    """Read a task's full state: task row, parents, children, comments,
    runs (attempt history), and the last N events."""
    board_err = _reject_board_arg(args, "kanban_show")
    if board_err:
        return board_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    try:
        task = _writer_result(_writer_request("show", {"task_id": tid}))
        if task is None:
            return tool_error(f"task {tid} not found")
        task = _require_path_free_projection(task, operation="kanban_show")
        comments = _writer_result(_writer_request("comments", {"task_id": tid}))
        events = _writer_result(_writer_request("events", {"task_id": tid}))
        runs = _writer_result(_writer_request("runs", {"task_id": tid}))
        if not all(isinstance(value, list) for value in (comments, events, runs)):
            raise ValueError("kanban_show: writer read projection unavailable")
        return json.dumps({
            "task": task,
            "parent_ids": list(task["parent_ids"]),
            "child_ids": list(task["child_ids"]),
            "parent_results": list(task["parent_results"]),
            "comments": comments,
            "events": events[-50:],
            "runs": runs,
            "worker_context": task["worker_context"],
        })
    except ValueError as e:
        return tool_error(f"kanban_show: {e}")
    except Exception as e:
        logger.exception("kanban_show failed")
        return tool_error(f"kanban_show: {e}")


def _handle_list(args: dict, **kw) -> str:
    """List task summaries with the same core filters as the CLI."""
    board_err = _reject_board_arg(args, "kanban_list")
    if board_err:
        return board_err
    guard = _require_orchestrator_tool("kanban_list")
    if guard:
        return guard
    assignee = args.get("assignee")
    status = args.get("status")
    tenant = args.get("tenant")
    include_archived, bool_error = _parse_bool_arg(args, "include_archived")
    if bool_error:
        return tool_error(bool_error)
    limit = args.get("limit")
    if limit is None:
        limit = KANBAN_LIST_DEFAULT_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer")
    if limit < 1:
        return tool_error("limit must be >= 1")
    if limit > KANBAN_LIST_MAX_LIMIT:
        return tool_error(f"limit must be <= {KANBAN_LIST_MAX_LIMIT}")
    try:
        rows = _writer_result(_writer_request("list", {
            "assignee": assignee,
            "status": status,
            "tenant": tenant,
            "include_archived": include_archived,
            "limit": limit,
        }))
        if not isinstance(rows, list):
            raise ValueError("kanban_list: writer projection unavailable")
        tasks = list(rows)[:limit]
        return json.dumps({
            "tasks": [_task_summary_dict(t) for t in tasks],
            "count": len(tasks),
            "limit": limit,
            "truncated": len(rows) > limit,
        })
    except ValueError as e:
        return tool_error(f"kanban_list: {e}")
    except Exception as e:
        logger.exception("kanban_list failed")
        return tool_error(f"kanban_list: {e}")


def _handle_complete(args: dict, **kw) -> str:
    """Mark the current task done with a structured handoff."""
    board_err = _reject_board_arg(args, "kanban_complete")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_complete")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_complete")
    if key_error:
        return key_error
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    summary = args.get("summary")
    metadata = args.get("metadata")
    result = args.get("result")
    if summary:
        summary = redact_sensitive_text(str(summary), force=True)
    if result:
        result = redact_sensitive_text(str(result), force=True)
    if metadata is not None and isinstance(metadata, dict):
        meta_json = json.dumps(metadata)
        meta_json = redact_sensitive_text(meta_json, force=True)
        try:
            metadata = json.loads(meta_json)
        except json.JSONDecodeError:
            pass
    if "artifacts" in args or "created_cards" in args:
        return tool_error(
            "kanban_complete refused: artifacts and created_cards are not "
            "accepted by the path-free writer protocol; attach inline bytes "
            "with kanban_attach instead"
        )
    if not (summary or result):
        return tool_error(
            "provide at least one of: summary (preferred), result"
        )
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(
            f"metadata must be an object/dict, got {type(metadata).__name__}"
        )
    if not _path_free_completion_metadata(metadata):
        return tool_error(
            "kanban_complete refused: metadata must be flat, scalar, and path-free"
        )
    metadata = _stamp_worker_session_metadata(tid, metadata)
    try:
        response = _writer_request(
            "complete",
            {
                "task_id": tid,
                "result": result,
                "summary": summary,
                "metadata": metadata,
                "expected_run_id": _worker_run_id(tid),
            },
            request_key=request_key,
        )
        result_obj = _writer_result(response) or {}
        if not result_obj.get("ok", True):
            return tool_error(
                f"could not complete {tid} (unknown id or already terminal)"
            )
        return _ok(task_id=tid, run_id=result_obj.get("run_id"))
    except ValueError as e:
        return tool_error(f"kanban_complete: {e}")
    except Exception as e:
        logger.exception("kanban_complete failed")
        return tool_error(f"kanban_complete: {e}")


def _handle_block(args: dict, **kw) -> str:
    """Transition the task to blocked with a reason a human will read."""
    board_err = _reject_board_arg(args, "kanban_block")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_block")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_block")
    if key_error:
        return key_error
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required — explain what input you need")
    reason = redact_sensitive_text(str(reason), force=True)
    kind = args.get("kind")
    valid_kinds = {"dependency", "needs_input", "capability", "transient"}
    if kind is not None and kind not in valid_kinds:
        return tool_error(
            f"kind must be one of {sorted(valid_kinds)} (or omit it)"
        )
    try:
        response = _writer_request(
            "block",
            {
                "task_id": tid,
                "reason": reason,
                "kind": kind,
                "expected_run_id": _worker_run_id(tid),
            },
            request_key=request_key,
        )
        result_obj = _writer_result(response) or {}
        if not result_obj.get("ok", True):
            return tool_error(
                f"could not block {tid} (unknown id or not in running/ready)"
            )
        return _ok(task_id=tid, run_id=result_obj.get("run_id"),
                   status=result_obj.get("status", "blocked"), block_kind=kind)
    except ValueError as e:
        return tool_error(f"kanban_block: {e}")
    except Exception as e:
        logger.exception("kanban_block failed")
        return tool_error(f"kanban_block: {e}")


def _handle_heartbeat(args: dict, **kw) -> str:
    """Signal that the worker is still alive during a long operation.

    Extends the claim TTL via ``heartbeat_claim`` AND records a heartbeat
    event via ``heartbeat_worker``. Without the ``heartbeat_claim`` half,
    a diligent worker that loops this tool while a single tool call
    blocks the agent for >DEFAULT_CLAIM_TTL_SECONDS still gets reclaimed
    by ``release_stale_claims`` — which is exactly the trap that
    ``heartbeat_claim``'s docstring warns against.
    """
    board_err = _reject_board_arg(args, "kanban_heartbeat")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_heartbeat")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_heartbeat")
    if key_error:
        return key_error
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    try:
        response = _writer_request(
            "heartbeat", {"task_id": tid}, request_key=request_key
        )
        result_obj = _writer_result(response) or {}
        if not result_obj.get("ok", True):
            return tool_error(
                f"could not heartbeat {tid} (unknown id or not running)"
            )
        return _ok(task_id=tid)
    except ValueError as e:
        return tool_error(f"kanban_heartbeat: {e}")
    except Exception as e:
        logger.exception("kanban_heartbeat failed")
        return tool_error(f"kanban_heartbeat: {e}")


def _handle_comment(args: dict, **kw) -> str:
    """Append a comment to a task's thread."""
    board_err = _reject_board_arg(args, "kanban_comment")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_comment")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_comment")
    if key_error:
        return key_error
    tid = args.get("task_id")
    if not tid:
        return tool_error(
            "task_id is required (use the current task id if that's what "
            "you mean — pulls from env but kept explicit here)"
        )
    body = args.get("body")
    if not body or not str(body).strip():
        return tool_error("body is required")
    body = redact_sensitive_text(str(body), force=True)
    try:
        result_obj = _writer_result(
            _writer_request(
                "comment", {"task_id": tid, "body": str(body)},
                request_key=request_key,
            )
        ) or {}
        return _ok(task_id=tid, comment_id=result_obj.get("comment_id"))
    except ValueError as e:
        return tool_error(f"kanban_comment: {e}")
    except Exception as e:
        logger.exception("kanban_comment failed")
        return tool_error(f"kanban_comment: {e}")


def _handle_attach(args: dict, **kw) -> str:
    """Attach an inline (base64) file to a task.

    Mirrors the dashboard's upload endpoint for the agent surface: decode
    the payload and enforce the shared size cap; the privileged writer owns
    the attachment path and metadata row.
    """
    board_err = _reject_board_arg(args, "kanban_attach")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_attach")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_attach")
    if key_error:
        return key_error
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    filename = args.get("filename")
    if not filename or not str(filename).strip():
        return tool_error("filename is required")
    content_b64 = args.get("content_base64")
    if not content_b64 or not str(content_b64).strip():
        return tool_error("content_base64 is required")
    import base64
    import binascii
    try:
        data = base64.b64decode(str(content_b64), validate=True)
    except (binascii.Error, ValueError) as e:
        return tool_error(f"content_base64 is not valid base64: {e}")
    content_type = args.get("content_type")
    if "/" in str(filename) or "\\" in str(filename):
        return tool_error("kanban_attach refused: filename must be a basename")
    try:
        max_bytes = _decoded_attachment_limit()
    except RuntimeError as exc:
        return tool_error(f"kanban_attach: {exc}")
    if len(data) > max_bytes:
        return tool_error(
            f"kanban_attach: attachment exceeds {max_bytes} decoded bytes"
        )
    try:
        import base64
        result_obj = _writer_result(_writer_request(
            "attach",
            {
                "task_id": tid,
                "filename": str(filename),
                "content_type": content_type,
                "data_b64": base64.b64encode(data).decode("ascii"),
            },
            request_key=request_key,
        )) or {}
        return _ok(task_id=tid, attachment_id=result_obj.get("attachment_id"), size=len(data))
    except ValueError as e:
        return tool_error(f"kanban_attach: {e}")
    except Exception as e:
        logger.exception("kanban_attach failed")
        return tool_error(f"kanban_attach: {e}")


def _handle_attach_url(args: dict, **kw) -> str:
    """Attach a file fetched server-side from a URL.

    The URL is only an input-conversion source.  After the bounded, SSRF-safe
    download, the fixed inline ``attach`` adapter owns validation and sends
    only task/filename/type/bytes to the privileged writer.
    """
    board_err = _reject_board_arg(args, "kanban_attach_url")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_attach_url")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_attach_url")
    if key_error:
        return key_error
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    url = args.get("url")
    if not url or not str(url).strip():
        return tool_error("url is required")
    url = str(url).strip()

    filename = args.get("filename") or args.get("title")
    if not filename or not str(filename).strip():
        from urllib.parse import unquote, urlparse

        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1]).strip()
    filename = str(filename or "download").strip()
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        return tool_error("kanban_attach_url refused: filename must be a basename")

    content_type = args.get("content_type")
    if content_type is not None and not isinstance(content_type, str):
        return tool_error("kanban_attach_url: content_type must be a string")

    try:
        data, fetched_content_type = _download_url_with_cap(
            url, _decoded_attachment_limit()
        )
    except ValueError as exc:
        return tool_error(f"kanban_attach_url: {exc}")
    except Exception as exc:
        logger.exception("kanban_attach_url download failed")
        return tool_error(f"kanban_attach_url: failed to fetch URL: {exc}")

    import base64

    return _handle_attach(
        {
            "task_id": tid,
            "filename": filename,
            "content_type": content_type or fetched_content_type,
            "content_base64": base64.b64encode(data).decode("ascii"),
            "request_key": request_key,
        }
    )


def _handle_attachments(args: dict, **kw) -> str:
    """List a task's attachments (read-only; no ownership restriction)."""
    board_err = _reject_board_arg(args, "kanban_attachments")
    if board_err:
        return board_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    try:
        atts = _writer_result(
            _writer_request("attachments", {"task_id": tid})
        )
        if not isinstance(atts, list):
            raise ValueError("kanban_attachments: writer projection unavailable")
        return json.dumps({"ok": True, "task_id": tid, "attachments": atts})
    except ValueError as e:
        return tool_error(f"kanban_attachments: {e}")
    except Exception as e:
        logger.exception("kanban_attachments failed")
        return tool_error(f"kanban_attachments: {e}")


def _handle_create(args: dict, **kw) -> str:
    """Create a child task. Orchestrator workers use this to fan out.

    ``parents`` can be a list of task ids; dependency-gated promotion
    works as usual.
    """
    board_err = _reject_board_arg(args, "kanban_create")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_create")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_create")
    if key_error:
        return key_error
    title = args.get("title")
    if not title or not str(title).strip():
        return tool_error("title is required")
    assignee = args.get("assignee")
    if not assignee:
        return tool_error(
            "assignee is required — name the profile that should execute this "
            "task (the dispatcher will only spawn tasks with an assignee)"
        )
    supported = {
        "title", "assignee", "body", "parents", "tenant", "priority",
        "request_key",
    }
    unsupported = sorted(set(args) - supported)
    if unsupported:
        return tool_error(
            "kanban_create refused unsupported fields: " + ", ".join(unsupported)
        )
    parents = args.get("parents") or []
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, (list, tuple)):
        return tool_error(
            f"parents must be a list of task ids, got {type(parents).__name__}"
        )
    try:
        result_obj = _writer_result(_writer_request(
            "create",
            {
                "title": str(title).strip(),
                "body": args.get("body"),
                "assignee": str(assignee),
                "parents": list(parents),
                "tenant": args.get("tenant"),
                "priority": args.get("priority", 0),
            },
            request_key=request_key,
        )) or {}
        task_id = result_obj.get("task_id")
        if not task_id:
            return tool_error("kanban_create: writer returned no task id")
        subscription = _maybe_auto_subscribe(str(task_id), request_key)
        return _ok(
            task_id=task_id,
            subscribed=bool(subscription),
            subscription=subscription,
        )
    except ValueError as e:
        return tool_error(f"kanban_create: {e}")
    except Exception as e:
        logger.exception("kanban_create failed")
        return tool_error(f"kanban_create: {e}")

def _maybe_auto_subscribe(
    task_id: str, request_key: str,
) -> Optional[dict[str, Any]]:
    """Auto-subscribe the calling session to task completion / block events.

    Returns the writer's subscription projection when a row was written, or
    ``None`` when there is no delivery channel, the config gate is disabled,
    or bookkeeping fails.  The caller keeps a boolean ``subscribed`` field
    for compatibility and exposes the actual projection as ``subscription``.

    Gated by ``kanban.auto_subscribe_on_create`` in config.yaml (default
    True). Disable to mirror pre-feature behaviour, e.g. when the
    originating user/chat opted out via the per-platform notification
    toggle (see ``hermes dashboard``).

    Subscription paths:

    - **Gateway** (telegram/discord/slack/etc): ``HERMES_SESSION_PLATFORM``,
      ``HERMES_SESSION_CHAT_ID``, and ``HERMES_SESSION_CHAT_TYPE`` are set in
      ContextVars by the messaging gateway before agent dispatch. The
      notification poller already keys off these, so we just register a row.

    - **TUI** (herm desktop / herm TUI): the platform/chat_id ContextVars
      are intentionally cleared (TUI is a single-channel local UI, not
      a multi-tenant chat surface), but the agent subprocess inherits
      ``HERMES_SESSION_KEY`` from the parent session. We subscribe with
      ``platform="tui"`` and ``chat_id=<key>``; the TUI notification
      poller (``tui_gateway/server.py``) reads ``kanban_notify_subs``
      for these rows and posts the completion message into the running
      session.

    - **CLI / cron / test / unattached**: no persistent delivery channel,
      no-op.

    Failure mode: any exception inside the function is logged at WARNING
    with the offending exception + diagnostic env vars and swallowed.
    We never want a notification bookkeeping failure to fail the
    kanban_create that the agent is mid-conversation about.
    """
    try:
        cfg = load_config()
        if not cfg_get(cfg, "kanban", "auto_subscribe_on_create", default=True):
            return None
        from gateway.session_context import get_session_env
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
        if not platform or not chat_id:
            # TUI/desktop sessions intentionally have no platform/chat
            # ContextVars.  HERMES_SESSION_KEY is the fixed local delivery
            # target; ordinary CLI/cron sessions do not set it.
            session_key = (
                get_session_env("HERMES_SESSION_KEY", "")
                or os.environ.get("HERMES_SESSION_KEY", "")
            )
            if not session_key:
                return None
            platform, chat_id = "tui", session_key
        args: dict[str, Any] = {
            "task_id": task_id,
            "platform": platform,
            "chat_id": chat_id,
        }
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "")
        if thread_id:
            args["thread_id"] = thread_id
        chat_type = get_session_env("HERMES_SESSION_CHAT_TYPE", "")
        if chat_type:
            args["chat_type"] = chat_type
        response = _writer_request(
            "notify-subscribe",
            args,
            request_key=_derived_request_key(
                request_key, f"notify-subscribe:{task_id}"
            ),
        )
        return _writer_result(response) or {}
    except Exception as _exc:
        logger.warning("_maybe_auto_subscribe failed: %r", _exc)
        return None


def _handle_unblock(args: dict, **kw) -> str:
    """Transition a blocked task to ready, or todo while parents remain open."""
    board_err = _reject_board_arg(args, "kanban_unblock")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_unblock")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_unblock")
    if key_error:
        return key_error
    guard = _require_orchestrator_tool("kanban_unblock")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    ownership_err = _enforce_worker_task_ownership(str(tid))
    if ownership_err:
        return ownership_err
    try:
        result_obj = _writer_result(
            _writer_request(
                "unblock", {"task_id": str(tid)}, request_key=request_key
            )
        ) or {}
        if not result_obj.get("ok", True):
            return tool_error(f"could not unblock {tid} (not blocked or unknown)")
        return _ok(task_id=str(tid), status=result_obj.get("status"))
    except ValueError as e:
        return tool_error(f"kanban_unblock: {e}")
    except Exception as e:
        logger.exception("kanban_unblock failed")
        return tool_error(f"kanban_unblock: {e}")


def _handle_link(args: dict, **kw) -> str:
    """Add a parent→child dependency edge after the fact."""
    board_err = _reject_board_arg(args, "kanban_link")
    if board_err:
        return board_err
    delegated_err = _reject_delegated_child_mutation("kanban_link")
    if delegated_err:
        return delegated_err
    request_key, key_error = _tool_request_key(args, "kanban_link")
    if key_error:
        return key_error
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    if not parent_id or not child_id:
        return tool_error("both parent_id and child_id are required")
    try:
        _writer_request(
            "link", {"parent_id": str(parent_id), "child_id": str(child_id)},
            request_key=request_key,
        )
        return _ok(parent_id=parent_id, child_id=child_id)
    except ValueError as e:
        # Covers cycle + self-parent rejections
        return tool_error(f"kanban_link: {e}")
    except Exception as e:
        logger.exception("kanban_link failed")
        return tool_error(f"kanban_link: {e}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_DESC_TASK_ID_DEFAULT = (
    "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the env "
    "(the task the dispatcher spawned you to work on)."
)
_REQUEST_KEY_PROPERTY = {
    "type": "string",
    "description": (
        "Stable caller-owned mutation identity. Reuse the same key when "
        "retrying after an unknown/lost reply; use a new key for a deliberate "
        "new mutation."
    ),
}

KANBAN_SHOW_SCHEMA = {
    "name": "kanban_show",
    "description": (
        "Read a task's full state — title, body, assignee, parent task "
        "handoffs, your prior attempts on this task if any, comments, "
        "and recent events. Use this to (re)orient yourself before "
        "starting work, especially on retries. The response includes a "
        "pre-formatted ``worker_context`` string suitable for inclusion "
        "verbatim in your reasoning."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
        },
        "required": [],
    },
}

KANBAN_LIST_SCHEMA = {
    "name": "kanban_list",
    "description": (
        "List Kanban task summaries so an orchestrator profile can discover "
        "work to route. Supports the same core filters as the CLI: assignee, "
        "status, tenant, include_archived, and limit. Returns compact rows "
        "with ids, title, status, assignee, priority, parent/child ids, and "
        "counts. Bounded to 50 rows by default, 200 max, with truncation "
        "metadata. Also recomputes ready tasks before listing, matching the "
        "CLI. Orchestrator-only — dispatcher-spawned task workers never see "
        "this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "assignee": {
                "type": "string",
                "description": "Optional assignee/profile filter.",
            },
            "status": {
                "type": "string",
                "enum": [
                    "triage", "todo", "ready", "running",
                    "blocked", "done", "archived",
                ],
                "description": "Optional task status filter.",
            },
            "tenant": {
                "type": "string",
                "description": "Optional tenant/project namespace filter.",
            },
            "include_archived": {
                "type": "boolean",
                "description": "Include archived tasks. Defaults to false.",
            },
            "limit": {
                "type": "integer",
                "description": "Optional maximum rows to return (default 50, max 200).",
            },
        },
        "required": [],
    },
}

KANBAN_COMPLETE_SCHEMA = {
    "name": "kanban_complete",
    "description": (
        "Mark your current task done with a structured handoff for "
        "downstream workers and humans. Prefer ``summary`` for a "
        "human-readable 1-3 sentence description of what you did; put "
        "machine-readable facts in ``metadata`` (flat scalar values only). "
        "At least one of ``summary`` or ``result`` is required. Attachments "
        "are sent separately as bounded inline bytes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "summary": {
                "type": "string",
                "description": (
                    "Human-readable handoff, 1-3 sentences. Appears in "
                    "Run History on the dashboard and in downstream "
                    "workers' context."
                ),
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Flat dict of scalar, path-free facts about this "
                    "attempt. Surfaced to downstream workers alongside "
                    "``summary``."
                ),
            },
            "result": {
                "type": "string",
                "description": (
                    "Short result log line (legacy field, maps to "
                    "task.result). Use ``summary`` instead when "
                    "possible; this exists for compatibility with "
                    "callers that still set --result on the CLI."
                ),
            },
        },
        "required": ["request_key"],
    },
}

KANBAN_BLOCK_SCHEMA = {
    "name": "kanban_block",
    "description": (
        "Stop work on this task and route it according to WHY you're stuck. "
        "Set ``kind`` to say which: 'dependency' (waiting on another task — "
        "goes to todo and auto-resumes when that task finishes, no human "
        "needed), 'needs_input' (you need a human decision/answer), "
        "'capability' (a hard wall: no access, missing credentials, an action "
        "no agent can do), or 'transient' (a flaky failure that may clear). "
        "``reason`` is shown to the human on the board. If a task keeps "
        "getting unblocked and re-blocked for the same reason, it is "
        "auto-escalated to triage. Use for genuine blockers only — don't "
        "block on things you can resolve yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "reason": {
                "type": "string",
                "description": (
                    "What you need answered or what stopped you, in one or "
                    "two sentences. Don't paste the whole conversation; the "
                    "human has the board and can ask follow-ups via comments."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["dependency", "needs_input", "capability", "transient"],
                "description": (
                    "Why you're blocked. 'dependency' waits in todo and "
                    "resumes automatically; the others surface to a human. "
                    "Omit only if none apply."
                ),
            },
        },
        "required": ["reason", "request_key"],
    },
}

KANBAN_HEARTBEAT_SCHEMA = {
    "name": "kanban_heartbeat",
    "description": (
        "Signal that you're still alive during a long operation "
        "(training, encoding, large crawls). Call every few minutes so "
        "humans see liveness separately from PID checks. Pure side "
        "effect — no work changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "note": {
                "type": "string",
                "description": (
                    "Optional short note describing current progress. "
                    "Shown in the event log."
                ),
            },
        },
        "required": ["request_key"],
    },
}

KANBAN_COMMENT_SCHEMA = {
    "name": "kanban_comment",
    "description": (
        "Append a comment to a task's thread. Use for durable notes "
        "that should outlive this run (questions for the next worker, "
        "partial findings, rationale). Ephemeral reasoning doesn't "
        "belong here — use your normal response instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "task_id": {
                "type": "string",
                "description": (
                    "Task id. Required (may be your own task or "
                    "another's — comment threads are per-task)."
                ),
            },
            "body": {
                "type": "string",
                "description": "Markdown-supported comment body.",
            },
        },
        "required": ["task_id", "body", "request_key"],
    },
}

KANBAN_ATTACH_SCHEMA = {
    "name": "kanban_attach",
    "description": (
        "Attach a file to a task by passing its bytes inline (base64). "
        "Use for genuine file artifacts the next worker or a human should "
        "be able to download — generated reports, images, exports. The "
        "file is stored as a real attachment through the privileged writer, "
        "capped by the writer's decoded-byte limit within its 1 MiB frame. "
        "URL downloads are refused."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "filename": {
                "type": "string",
                "description": (
                    "File name to store it under (e.g. 'report.pdf'). "
                    "Directory components are stripped; only the leaf is kept."
                ),
            },
            "content_base64": {
                "type": "string",
                "description": (
                    "The file contents, base64-encoded. The writer enforces "
                    "the decoded-byte limit within its 1 MiB frame."
                ),
            },
            "content_type": {
                "type": "string",
                "description": "Optional MIME type (e.g. 'application/pdf').",
            },
        },
        "required": ["filename", "content_base64", "request_key"],
    },
}

KANBAN_ATTACH_URL_SCHEMA = {
    "name": "kanban_attach_url",
    "description": (
        "Fetch an HTTP(S) URL with SSRF and size guards, then pass the "
        "bounded bytes through the privileged inline-attachment writer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "url": {
                "type": "string",
                "description": "http(s) URL to fetch and store.",
            },
            "filename": {
                "type": "string",
                "description": (
                    "Optional name to store it under. Defaults to the URL "
                    "path's leaf component; directory components are rejected."
                ),
            },
            "content_type": {
                "type": "string",
                "description": (
                    "Optional MIME type override. Defaults to the "
                    "Content-Type the server returns."
                ),
            },
        },
        "required": ["url", "request_key"],
    },
}

KANBAN_ATTACHMENTS_SCHEMA = {
    "name": "kanban_attachments",
    "description": (
        "List a task's bounded inline attachments by id, filename, type, and size."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
        },
        "required": [],
    },
}

KANBAN_CREATE_SCHEMA = {
    "name": "kanban_create",
    "description": (
        "Create a new kanban task, optionally as a child of the current "
        "one (pass the current task id in ``parents``). Used by "
        "orchestrator workers to fan out — decompose work into child "
        "tasks with specific assignees, link them into a pipeline, "
        "then complete your own task. The dispatcher picks up the new "
        "tasks on its next tick and spawns the assigned profiles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "title": {
                "type": "string",
                "description": "Short task title (required).",
            },
            "assignee": {
                "type": "string",
                "description": (
                    "Profile name that should execute this task "
                    "(e.g. 'researcher-a', 'reviewer', 'writer'). "
                    "Required — tasks without an assignee are never "
                    "dispatched."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "Opening post: full spec, acceptance criteria, "
                    "links. The assigned worker reads this as part of "
                    "its context."
                ),
            },
            "parents": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Parent task ids. The new task stays in 'todo' "
                    "until every parent reaches 'done'; then it "
                    "auto-promotes to 'ready'. Typical fan-in: list "
                    "all the researcher task ids when creating a "
                    "synthesizer task."
                ),
            },
            "tenant": {
                "type": "string",
                "description": (
                    "Optional namespace for multi-project isolation. "
                    "Defaults to HERMES_TENANT env if set."
                ),
            },
            "priority": {
                "type": "integer",
                "description": (
                    "Dispatcher tiebreaker. Higher = picked sooner "
                    "when multiple ready tasks share an assignee."
                ),
            },
        },
        "required": ["title", "assignee", "request_key"],
    },
}

KANBAN_UNBLOCK_SCHEMA = {
    "name": "kanban_unblock",
    "description": (
        "Unblock a Kanban task. It moves to ready when all parents are done, "
        "or todo while any parent remains open. Orchestrator-only — only "
        "profiles with the kanban toolset can unblock routed work; "
        "dispatcher-spawned task workers never see this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "task_id": {
                "type": "string",
                "description": "Blocked task id to move to ready or parent-gated todo.",
            },
        },
        "required": ["task_id", "request_key"],
    },
}

KANBAN_LINK_SCHEMA = {
    "name": "kanban_link",
    "description": (
        "Add a parent→child dependency edge after both tasks already "
        "exist. The child won't promote to 'ready' until all parents "
        "are 'done'. Cycles and self-links are rejected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request_key": _REQUEST_KEY_PROPERTY,
            "parent_id": {"type": "string", "description": "Parent task id."},
            "child_id":  {"type": "string", "description": "Child task id."},
        },
        "required": ["parent_id", "child_id", "request_key"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="kanban_show",
    toolset="kanban",
    schema=KANBAN_SHOW_SCHEMA,
    handler=_handle_show,
    check_fn=_check_kanban_mode,
    emoji="📋",
)

registry.register(
    name="kanban_list",
    toolset="kanban",
    schema=KANBAN_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="📋",
)

registry.register(
    name="kanban_complete",
    toolset="kanban",
    schema=KANBAN_COMPLETE_SCHEMA,
    handler=_handle_complete,
    check_fn=_check_kanban_mode,
    emoji="✔",
)

registry.register(
    name="kanban_block",
    toolset="kanban",
    schema=KANBAN_BLOCK_SCHEMA,
    handler=_handle_block,
    check_fn=_check_kanban_mode,
    emoji="⏸",
)

registry.register(
    name="kanban_heartbeat",
    toolset="kanban",
    schema=KANBAN_HEARTBEAT_SCHEMA,
    handler=_handle_heartbeat,
    check_fn=_check_kanban_mode,
    emoji="💓",
)

registry.register(
    name="kanban_comment",
    toolset="kanban",
    schema=KANBAN_COMMENT_SCHEMA,
    handler=_handle_comment,
    check_fn=_check_kanban_mode,
    emoji="💬",
)

registry.register(
    name="kanban_attach",
    toolset="kanban",
    schema=KANBAN_ATTACH_SCHEMA,
    handler=_handle_attach,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_attach_url",
    toolset="kanban",
    schema=KANBAN_ATTACH_URL_SCHEMA,
    handler=_handle_attach_url,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_attachments",
    toolset="kanban",
    schema=KANBAN_ATTACHMENTS_SCHEMA,
    handler=_handle_attachments,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_create",
    toolset="kanban",
    schema=KANBAN_CREATE_SCHEMA,
    handler=_handle_create,
    check_fn=_check_kanban_mode,
    emoji="➕",
)

registry.register(
    name="kanban_unblock",
    toolset="kanban",
    schema=KANBAN_UNBLOCK_SCHEMA,
    handler=_handle_unblock,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="▶",
)

registry.register(
    name="kanban_link",
    toolset="kanban",
    schema=KANBAN_LINK_SCHEMA,
    handler=_handle_link,
    check_fn=_check_kanban_mode,
    emoji="🔗",
)
