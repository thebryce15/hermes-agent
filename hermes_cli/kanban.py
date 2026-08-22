"""CLI for the Hermes Kanban board — ``hermes kanban …`` subcommand.

Exposes the full Kanban command surface documented in the design spec
(``docs/hermes-kanban-v1-spec.pdf``).  All DB work is delegated to
``kanban_db``.  This module adds:

  * Argparse subcommand construction (``build_parser``).
  * Argument dispatch (``kanban_command``).
  * Output formatting (plain text + ``--json``).
  * A short shared helper that parses a single slash-style string
    (used by ``/kanban …`` in CLI and gateway) and forwards it to the
    argparse surface.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import time
import re
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb


_IPC_SUPPORTED_ACTIONS = frozenset({
    "create", "assign", "link", "unlink", "claim", "comment", "complete",
    "block", "unblock", "heartbeat", "attach", "list", "ls", "show",
    "attachments", "runs",
})
_IPC_MUTATION_ACTIONS = frozenset({
    "create", "assign", "link", "unlink", "claim", "comment", "complete",
    "block", "unblock", "heartbeat", "attach",
})
_REQUEST_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

_IPC_REFUSED_ACTIONS = frozenset({
    "init", "swarm", "specify", "decompose", "set-model", "attach-rm",
    "reclaim", "reassign", "edit", "schedule", "promote", "archive",
    "dispatch", "daemon", "gc", "repair", "boards", "notify-subscribe",
    "notify-list", "notify-unsubscribe", "diagnostics", "diag", "assignees",
    "tail", "watch", "stats", "log", "context",
})


def _writer_request(
    operation: str, args: Optional[dict[str, Any]] = None, *, request_key: Optional[str] = None
) -> dict[str, Any]:
    """Send a CLI request to the fixed privileged writer socket."""
    from hermes_cli.kanban_writer import writer_request

    if operation in _IPC_MUTATION_ACTIONS:
        if not isinstance(request_key, str) or not _REQUEST_KEY_RE.fullmatch(request_key):
            raise ValueError(
                f"{operation}: a stable --request-key is required before writer IPC"
            )
    return writer_request(operation=operation, args=args or {}, request_key=request_key)


def _ipc_result(response: Any) -> Any:
    return response.get("result") if isinstance(response, dict) else None


_IPC_LINK_FIELDS = frozenset({"parent_ids", "child_ids"})
_IPC_SHOW_FIELDS = _IPC_LINK_FIELDS | {"parent_results", "worker_context"}
_IPC_PATH_FIELDS = frozenset({
    "workspace_path", "stored_path", "attachment_path", "db_path",
})


def _require_ipc_projection(
    value: Any,
    *,
    operation: str,
    fields: frozenset[str] = _IPC_SHOW_FIELDS,
) -> dict[str, Any]:
    """Accept only the writer's truthful, path-free read projection."""
    if not isinstance(value, dict):
        raise ValueError(f"{operation}: writer projection unavailable")
    if _IPC_PATH_FIELDS.intersection(value):
        raise ValueError(f"{operation}: writer returned a path-bearing projection")
    missing = sorted(fields.difference(value))
    if missing:
        raise ValueError(
            f"{operation}: writer projection unavailable (missing {', '.join(missing)})"
        )
    if not isinstance(value.get("parent_ids"), list) or not isinstance(value.get("child_ids"), list):
        raise ValueError(f"{operation}: writer returned malformed link projection")
    if "parent_results" in fields and not isinstance(value.get("parent_results"), list):
        raise ValueError(f"{operation}: writer returned malformed parent-results projection")
    if "worker_context" in fields and not isinstance(value.get("worker_context"), str):
        raise ValueError(f"{operation}: writer returned malformed worker_context projection")
    return value


def _cli_refused(action: str, detail: str = "") -> int:
    suffix = f": {detail}" if detail else ""
    print(
        f"kanban {action} refused by the admitted writer protocol{suffix}",
        file=sys.stderr,
    )
    return 1


def _fixed_command(args: argparse.Namespace, action: str) -> int:
    """Invoke one admitted route without touching the board locally.

    The named ``_cmd_*`` functions remain importable for callers that used
    them as helpers before the writer cutover.  They now only normalize the
    action and enter the same IPC adapter as the top-level dispatcher.
    """
    if getattr(args, "kanban_action", None) != action:
        values = vars(args).copy()
        values["kanban_action"] = action
        args = argparse.Namespace(**values)
    return _ipc_cli_command(args)


def _writer_call_command(args: argparse.Namespace, action: str) -> int:
    """Named writer-call seam used by the legacy command names."""
    return _fixed_command(args, action)


def _flat_path_free_metadata(value: Any) -> bool:
    if value is None or not isinstance(value, dict):
        return value is None
    for key, item in value.items():
        if not isinstance(key, str) or any(
            token in key.casefold() for token in ("path", "file", "artifact")
        ):
            return False
        if isinstance(item, (dict, list, tuple)):
            return False
        if item is not None and not isinstance(item, (str, int, float, bool)):
            return False
        if isinstance(item, str) and ("/" in item or "\\" in item):
            return False
    return True


def _ipc_cli_command(args: argparse.Namespace) -> int:
    """Run the admitted CLI subset without opening the board directly."""
    action = getattr(args, "kanban_action", "")
    try:
        request_key: Optional[str] = None
        if action in _IPC_MUTATION_ACTIONS:
            request_key = getattr(args, "request_key", None)
            if not isinstance(request_key, str) or not _REQUEST_KEY_RE.fullmatch(request_key):
                return _cli_refused(
                    action,
                    "a stable --request-key is required; reuse it for retries and "
                    "choose a new key for a deliberate new mutation",
                )
        if action == "create":
            unsupported = []
            for name, default in (
                ("workspace", "scratch"), ("branch", None), ("project", None),
                ("triage", False), ("max_runtime", None), ("max_retries", None),
                ("model_override", None), ("provider_override", None),
                ("goal_mode", False), ("goal_max_turns", None),
                ("initial_status", "running"), ("skills", []),
                ("created_by", "user"),
            ):
                if getattr(args, name, default) != default:
                    unsupported.append(f"--{name.replace('_', '-')}")
            if unsupported:
                return _cli_refused("create", "unsupported fields: " + ", ".join(unsupported))
            result = _ipc_result(_writer_request(
                "create",
                {
                    "title": args.title,
                    "body": getattr(args, "body", None),
                    "parents": list(getattr(args, "parent", None) or ()),
                    "assignee": args.assignee,
                    "tenant": getattr(args, "tenant", None),
                    "priority": getattr(args, "priority", 0),
                },
                request_key=request_key,
            )) or {}
            if getattr(args, "json", False):
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"Created {result.get('task_id', '')}")
            return 0

        if action in {"list", "ls"}:
            if getattr(args, "session", None):
                return _cli_refused("list", "session identity is not accepted")
            assignee = getattr(args, "assignee", None)
            if getattr(args, "mine", False):
                return _cli_refused("list", "--mine requires caller identity")
            result = _ipc_result(_writer_request("list", {
                "assignee": assignee,
                "status": getattr(args, "status", None),
                "tenant": getattr(args, "tenant", None),
                "include_archived": bool(getattr(args, "archived", False)),
                "limit": getattr(args, "limit", 50),
                "order_by": getattr(args, "sort", None),
                "workflow_template_id": getattr(args, "workflow_template_id", None),
                "current_step_key": getattr(args, "current_step_key", None),
            }))
            if not isinstance(result, list):
                raise ValueError("list: writer projection unavailable")
            result = [
                _require_ipc_projection(row, operation="list", fields=_IPC_LINK_FIELDS)
                for row in result
            ]
            if getattr(args, "json", False):
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                for row in result:
                    print(f"{row.get('id', '')}  {row.get('status', ''):8s}  "
                          f"{row.get('assignee') or '(unassigned)':20s}  {row.get('title', '')}")
                if not result:
                    print("(no matching tasks)")
            return 0

        if action == "show":
            state_type = getattr(args, "state_type", None)
            state_name = getattr(args, "state_name", None)
            if (state_type is None) != (state_name is None):
                return _cli_refused("show", "pass both --state-type and --state-name")
            tid = args.task_id
            task = _ipc_result(_writer_request("show", {"task_id": tid}))
            if task is None:
                print(f"no such task: {tid}", file=sys.stderr)
                return 1
            task = _require_ipc_projection(task, operation="show")
            comments = _ipc_result(_writer_request("comments", {"task_id": tid}))
            events = _ipc_result(_writer_request("events", {"task_id": tid}))
            runs = _ipc_result(_writer_request("runs", {
                "task_id": tid,
                **({"state_type": state_type, "state_name": state_name}
                   if state_type is not None else {}),
            }))
            if not all(isinstance(value, list) for value in (comments, events, runs)):
                raise ValueError("show: writer read projection unavailable")
            payload = {
                "task": task,
                "parent_ids": list(task["parent_ids"]),
                "child_ids": list(task["child_ids"]),
                "parent_results": list(task["parent_results"]),
                "comments": comments,
                "events": events,
                "runs": runs,
                "worker_context": task["worker_context"],
            }
            if getattr(args, "json", False):
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(f"Task {tid}: {task.get('title', '')}")
                print(f"  status:    {task.get('status', '')}")
                print(f"  assignee:  {task.get('assignee') or '-'}")
                if task.get("body"):
                    print(f"\nBody:\n{task['body']}")
            return 0

        if action == "runs":
            payload = {"task_id": args.task_id}
            state_type = getattr(args, "state_type", None)
            state_name = getattr(args, "state_name", None)
            if (state_type is None) != (state_name is None):
                return _cli_refused("runs", "pass both --state-type and --state-name")
            if state_type is not None:
                payload.update(state_type=state_type, state_name=state_name)
            result = _ipc_result(_writer_request("runs", payload)) or []
            print(json.dumps(result, indent=2, ensure_ascii=False) if getattr(args, "json", False)
                  else "\n".join(str(row) for row in result))
            return 0

        if action == "attach":
            # The writer accepts inline bytes only.  In particular, never
            # open or resolve a caller-supplied path in this adapter.
            encoded = getattr(args, "data_b64", None)
            filename = getattr(args, "filename", None) or getattr(args, "name", None)
            if getattr(args, "path", None) is not None:
                return _cli_refused("attach", "filesystem paths are not admitted")
            if getattr(args, "author", None):
                return _cli_refused("attach", "upload identity is writer-owned")
            if not isinstance(encoded, str) or not encoded:
                return _cli_refused("attach", "provide bounded inline --data-b64 bytes")
            if not isinstance(filename, str) or not filename:
                return _cli_refused("attach", "--filename is required")
            result = _ipc_result(_writer_request(
                "attach",
                {
                    "task_id": args.task_id,
                    "filename": filename,
                    "content_type": getattr(args, "content_type", None),
                    "data_b64": encoded,
                },
                request_key=request_key,
            )) or {}
            attachment_id = result.get("attachment_id")
            if getattr(args, "json", False):
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"Attached {filename} to {args.task_id} (attachment {attachment_id})")
            return 0

        if action == "attachments":
            result = _ipc_result(_writer_request("attachments", {"task_id": args.task_id}))
            if not isinstance(result, list):
                raise ValueError("attachments: writer projection unavailable")
            print(json.dumps(result, indent=2, ensure_ascii=False) if getattr(args, "json", False)
                  else "\n".join(f"{row.get('filename', '')} ({row.get('size', 0)} bytes)" for row in result))
            return 0

        if action == "assign":
            profile = getattr(args, "profile", None)
            result = _ipc_result(_writer_request("assign", {
                "task_id": args.task_id,
                "assignee": None if str(profile).lower() in {"none", "-", "null"} else profile,
            }, request_key=request_key)) or {}
            print(f"Assigned {args.task_id} to {result.get('assignee') or '(unassigned)'}")
            return 0

        if action in {"link", "unlink"}:
            _writer_request(
                action, {"parent_id": args.parent_id, "child_id": args.child_id},
                request_key=request_key,
            )
            print(f"{'Linked' if action == 'link' else 'Unlinked'} {args.parent_id} -> {args.child_id}")
            return 0

        if action == "claim":
            result = _ipc_result(_writer_request("claim", {
                "task_id": args.task_id, "ttl_seconds": getattr(args, "ttl", None),
            }, request_key=request_key)) or {}
            print(f"Claimed {args.task_id}")
            if result.get("run_id") is not None:
                print(f"Run: {result['run_id']}")
            return 0

        if action == "comment":
            if getattr(args, "author", None):
                return _cli_refused("comment", "author identity is writer-owned")
            body = " ".join(args.text).strip()
            max_len = getattr(args, "max_len", None)
            if max_len is not None:
                if max_len < 1:
                    return _cli_refused("comment", "--max-len must be positive")
                body = body[:max_len]
            _writer_request(
                "comment", {"task_id": args.task_id, "body": body},
                request_key=request_key,
            )
            print(f"Comment added to {args.task_id}")
            return 0

        if action == "complete":
            ids = list(getattr(args, "task_ids", None) or ())
            if len(ids) != 1:
                return _cli_refused("complete", "the writer admits one task per request")
            raw_meta = getattr(args, "metadata", None)
            metadata = None
            if raw_meta:
                try:
                    metadata = json.loads(raw_meta)
                except (ValueError, json.JSONDecodeError) as exc:
                    return _cli_refused("complete", f"--metadata: {exc}")
            if metadata is not None and not _flat_path_free_metadata(metadata):
                return _cli_refused("complete", "metadata must be flat, scalar, and path-free")
            if getattr(args, "artifacts", None) or getattr(args, "created_cards", None):
                return _cli_refused("complete", "artifacts and created_cards are not admitted")
            _writer_request("complete", {
                "task_id": ids[0], "result": getattr(args, "result", None),
                "summary": getattr(args, "summary", None), "metadata": metadata,
                "expected_run_id": _worker_run_id_for(ids[0]),
            }, request_key=request_key)
            print(f"Completed {ids[0]}")
            return 0

        if action == "block":
            ids = [args.task_id] + list(getattr(args, "ids", None) or ())
            if len(ids) != 1:
                return _cli_refused("block", "the writer admits one task per request")
            reason = " ".join(getattr(args, "reason", None) or ()).strip()
            _writer_request("block", {
                "task_id": ids[0], "reason": reason,
                "kind": getattr(args, "kind", None),
                "expected_run_id": _worker_run_id_for(ids[0]),
            }, request_key=request_key)
            print(f"Blocked {ids[0]}")
            return 0

        if action == "unblock":
            ids = list(getattr(args, "task_ids", None) or ())
            if len(ids) != 1:
                return _cli_refused("unblock", "the writer admits one task per request")
            _writer_request(
                "unblock", {"task_id": ids[0]}, request_key=request_key
            )
            print(f"Unblocked {ids[0]}")
            return 0

        if action == "heartbeat":
            if getattr(args, "note", None):
                return _cli_refused("heartbeat", "notes are not admitted")
            _writer_request(
                "heartbeat", {"task_id": args.task_id}, request_key=request_key
            )
            print(f"Heartbeat recorded for {args.task_id}")
            return 0

    except (ValueError, RuntimeError, OSError) as exc:
        print(f"kanban {action}: {exc}", file=sys.stderr)
        return 1
    return _cli_refused(action)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "todo":     "◻",
    "ready":    "▶",
    "running":  "●",
    "scheduled":"⏱",
    "blocked":  "⊘",
    "done":     "✓",
    "archived": "—",
}


def _fmt_ts(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _fmt_task_line(t: kb.Task) -> str:
    icon = _STATUS_ICONS.get(t.status, "?")
    assignee = t.assignee or "(unassigned)"
    tenant = f" [{t.tenant}]" if t.tenant else ""
    return f"{icon} {t.id}  {t.status:8s}  {assignee:20s}{tenant}  {t.title}"


def _task_to_dict(t: kb.Task) -> dict[str, Any]:
    return {
        "id": t.id,
        "title": t.title,
        "body": t.body,
        "assignee": t.assignee,
        "status": t.status,
        "priority": t.priority,
        "tenant": t.tenant,
        "workspace_kind": t.workspace_kind,
        "workspace_path": t.workspace_path,
        "branch_name": t.branch_name,
        "project_id": t.project_id,
        "created_by": t.created_by,
        "created_at": t.created_at,
        "started_at": t.started_at,
        "completed_at": t.completed_at,
        "result": t.result,
        "skills": list(t.skills) if t.skills else [],
        "max_retries": t.max_retries,
        "model_override": t.model_override,
        "provider_override": t.provider_override,
        "session_id": t.session_id,
        "workflow_template_id": t.workflow_template_id,
        "current_step_key": t.current_step_key,
    }


def _run_state_kwargs(args: argparse.Namespace) -> Optional[dict[str, str]]:
    st = getattr(args, "state_type", None)
    sn = getattr(args, "state_name", None)
    if (st is None) != (sn is None):
        return None
    if st is None:
        return {}
    return {"state_type": st, "state_name": sn}


def _parse_workspace_flag(value: str) -> tuple[str, Optional[str]]:
    """Parse ``--workspace`` into ``(kind, path|None)``.

    Accepts: ``scratch``, ``worktree``, ``worktree:<path>``, ``dir:<path>``.
    """
    if not value:
        return ("scratch", None)
    v = value.strip()
    if v in {"scratch", "worktree"}:
        return (v, None)
    for prefix, kind in (("dir:", "dir"), ("worktree:", "worktree")):
        if not v.startswith(prefix):
            continue
        path = v[len(prefix):].strip()
        if not path:
            raise argparse.ArgumentTypeError(
                f"--workspace {prefix} requires a path after the colon"
            )
        return (kind, os.path.expanduser(path))
    raise argparse.ArgumentTypeError(
        f"unknown --workspace value {value!r}: use scratch, worktree, "
        "worktree:<path>, or dir:<path>"
    )


def _parse_branch_flag(value: Optional[str]) -> Optional[str]:
    """Normalize an optional branch name from ``kanban create --branch``."""
    if value is None:
        return None
    branch = value.strip()
    if not branch:
        raise argparse.ArgumentTypeError("--branch requires a non-empty name")
    if branch.startswith("-"):
        raise argparse.ArgumentTypeError("--branch must not start with '-'")
    if any(ch.isspace() for ch in branch):
        raise argparse.ArgumentTypeError("--branch must not contain whitespace")
    return branch


def _check_dispatcher_presence(
    hermes_home: Optional[Path] = None,
) -> tuple[bool, str]:
    """Return ``(running, message)``.

    - ``running=True``: a gateway is alive for this HERMES_HOME and its
      config has ``kanban.dispatch_in_gateway`` on (default). Message
      is a short status line.
    - ``running=False``: either no gateway is running, or the gateway
      is running but the config flag is off. Message is human guidance
      explaining the next step.

    Used by ``hermes kanban create`` (and callers) to warn when a task
    will sit in ``ready`` because nothing is there to pick it up.
    Defensive against import failures and config-read errors — if the
    probe itself errors, we return ``(True, "")`` so we don't spam
    false warnings (better to miss a warning than to cry wolf).

    ``hermes_home`` scopes the probe to a named profile's directory. The
    dashboard plugin API passes it because the dashboard backend process can
    be running under a different HERMES_HOME than the profile the request
    targets, which otherwise produced a "no gateway is running" warning
    against a perfectly healthy profile gateway (#71211). CLI callers leave
    it ``None`` and keep the existing process-level behavior.
    """
    try:
        from gateway.status import resolve_gateway_liveness  # type: ignore
    except Exception:
        return (True, "")  # can't probe — silent
    try:
        # Same shared ladder the dashboard status endpoints use, so a
        # PID-file-less (launch-service-managed) or cross-container gateway
        # is not misreported as absent. use_cache=False: this is a one-shot
        # CLI/create-time probe, not a polling loop, and it must observe the
        # gateway's state right now rather than a cached snapshot.
        liveness = resolve_gateway_liveness(
            profile_dir=hermes_home, use_cache=False
        )
    except Exception:
        return (True, "")  # probe errored — silent
    if liveness.probe_error:
        # The resolver swallows per-rung failures so status endpoints never
        # 500. This caller must still fail OPEN: an unreadable probe means
        # "can't tell", not "no gateway", and warning on it cries wolf.
        return (True, "")
    pid = liveness.pid

    # Even if the gateway is up, dispatch_in_gateway may be off.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        dispatch_on = bool(cfg.get("kanban", {}).get("dispatch_in_gateway", True))
    except Exception:
        dispatch_on = True  # can't tell — assume default

    if pid and dispatch_on:
        return (True, f"gateway pid={pid}, dispatch enabled")
    if pid and not dispatch_on:
        return (
            False,
            "Gateway is running but kanban.dispatch_in_gateway=false in "
            "config.yaml — the task will sit in 'ready' until you flip it "
            "back on and restart the gateway, OR run the legacy "
            "standalone daemon (`hermes kanban daemon --force`)."
        )
    return (
        False,
        "No gateway is running — the task will sit in 'ready' until you "
        "start it. Run:\n"
        "    hermes gateway start\n"
        "The gateway hosts an embedded dispatcher (tick interval 60s by "
        "default); your task will be picked up on the next tick after "
        "the gateway comes up."
    )


# ---------------------------------------------------------------------------
# Argparse builder
# ---------------------------------------------------------------------------

def _add_request_key_argument(
    parser: argparse.ArgumentParser, *, create_alias: bool = False,
) -> None:
    flags = ("--request-key", "--idempotency-key") if create_alias else ("--request-key",)
    parser.add_argument(
        *flags,
        dest="request_key",
        required=True,
        help=(
            "Stable mutation identity. Reuse it when retrying after an unknown "
            "reply; choose a new key for a deliberate new mutation."
        ),
    )


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``kanban`` subcommand tree under an existing subparsers.

    Returns the top-level ``kanban`` parser so caller can ``set_defaults``.
    """
    kanban_parser = parent_subparsers.add_parser(
        "kanban",
        help="Multi-profile collaboration board (tasks, links, comments)",
        description=(
            "Durable SQLite-backed task board shared across Hermes profiles. "
            "Tasks are claimed atomically, can depend on other tasks, and "
            "are executed by a named profile in an isolated workspace. "
            "See https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban "
            "or docs/hermes-kanban-v1-spec.pdf for the full design."
        ),
    )
    # --- global --board flag ---
    # Applies to every subcommand below. When set, scopes all reads and
    # writes to that board's DB. When omitted, resolves via the
    # HERMES_KANBAN_BOARD env var, then the persisted current-board
    # file, then "default". See kanban_db.get_current_board().
    kanban_parser.add_argument(
        "--board",
        default=None,
        metavar="<slug>",
        help=(
            "Board slug to operate on. Defaults to the current board "
            "(set via `hermes kanban boards switch <slug>` or the "
            "HERMES_KANBAN_BOARD env var). Use `hermes kanban boards list` "
            "to see all boards."
        ),
    )
    sub = kanban_parser.add_subparsers(dest="kanban_action")

    # --- init ---
    sub.add_parser("init", help="Create kanban.db if missing (idempotent)")

    # --- boards (new in v2: multi-project support) ---
    p_boards = sub.add_parser(
        "boards",
        help="Manage kanban boards (one board per project / workstream)",
        description=(
            "Boards let you separate unrelated streams of work "
            "(projects, repos, domains) into isolated queues. Each "
            "board has its own DB, workspaces directory, and dispatcher "
            "loop — tasks on one board cannot collide with tasks on "
            "another. The first board is 'default' and always exists."
        ),
    )
    boards_sub = p_boards.add_subparsers(dest="boards_action")

    b_list = boards_sub.add_parser(
        "list", aliases=["ls"],
        help="List all boards with task counts",
    )
    b_list.add_argument("--json", action="store_true")
    b_list.add_argument("--all", action="store_true",
                        help="Include archived boards too")

    b_create = boards_sub.add_parser(
        "create", aliases=["new"],
        help="Create a new board",
    )
    b_create.add_argument("slug",
                          help="Board slug (kebab-case, e.g. atm10-server)")
    b_create.add_argument("--name", default=None,
                          help="Human-readable display name (defaults to Title Case of slug)")
    b_create.add_argument("--description", default=None,
                          help="Optional description")
    b_create.add_argument("--icon", default=None,
                          help="Optional emoji or single-character icon for the dashboard")
    b_create.add_argument("--color", default=None,
                          help="Optional hex color (e.g. '#8b5cf6') for the dashboard")
    b_create.add_argument("--switch", action="store_true",
                          help="Switch to the new board after creating it")
    b_create.add_argument("--default-workdir", default=None,
                          help="Default workspace path for tasks created on this board")

    b_rm = boards_sub.add_parser(
        "rm", aliases=["remove", "delete"],
        help="Archive (default) or delete a board",
    )
    b_rm.add_argument("slug")
    b_rm.add_argument("--delete", action="store_true",
                      help="Hard-delete the board directory instead of archiving it. "
                           "Default is to move it to boards/_archived/ so it's recoverable.")

    b_switch = boards_sub.add_parser(
        "switch", aliases=["use"],
        help="Set the active board for subsequent CLI calls",
    )
    b_switch.add_argument("slug")

    boards_sub.add_parser(
        "show", aliases=["current"],
        help="Print the currently-active board slug",
    )

    b_rename = boards_sub.add_parser(
        "rename",
        help="Change a board's human-readable display name (slug is immutable)",
    )
    b_rename.add_argument("slug")
    b_rename.add_argument("name", help="New display name")

    b_set_wd = boards_sub.add_parser(
        "set-default-workdir",
        help="Set the default workspace path for tasks on a board",
    )
    b_set_wd.add_argument("slug")
    b_set_wd.add_argument("path", nargs="?", default=None,
                          help="Absolute path to use as default workdir. Omit to clear.")

    # --- create ---
    p_create = sub.add_parser("create", help="Create a new task")
    p_create.add_argument("title", help="Task title")
    p_create.add_argument("--body", default=None, help="Optional opening post")
    p_create.add_argument("--assignee", default=None, help="Profile name to assign")
    p_create.add_argument("--parent", action="append", default=[],
                          help="Parent task id (repeatable)")
    p_create.add_argument("--workspace", default="scratch",
                          help="scratch | worktree | worktree:<path> | dir:<path> "
                               "(default: scratch)")
    p_create.add_argument("--branch", default=None,
                          help="Branch name for worktree tasks, e.g. wt/t6-wire")
    p_create.add_argument("--project", default=None,
                          help="Link to a project (id or slug). Anchors the task's "
                               "worktree under the project's primary repo with a "
                               "deterministic branch. See `hermes project list`.")
    p_create.add_argument("--tenant", default=None, help="Tenant namespace")
    p_create.add_argument("--priority", type=int, default=0, help="Priority tiebreaker")
    p_create.add_argument("--triage", action="store_true",
                          help="Park in triage — a specifier will flesh out the spec and promote to todo")
    _add_request_key_argument(p_create, create_alias=True)
    p_create.add_argument("--max-runtime", default=None,
                          help="Per-task runtime cap. Accepts seconds (300) or "
                               "durations (90s, 30m, 2h, 1d). When exceeded, "
                               "the dispatcher SIGTERMs (then SIGKILLs) the worker "
                               "and re-queues the task.")
    p_create.add_argument("--created-by", default="user",
                          help="Author name recorded on the task (default: user)")
    p_create.add_argument("--skill", action="append", default=[], dest="skills",
                          help="Skill to force-load into the worker "
                               "(repeatable). The kanban lifecycle is already "
                               "injected automatically. Example: "
                               "--skill translation --skill github-code-review")
    p_create.add_argument("--max-retries", type=int, default=None,
                          metavar="N",
                          help="Per-task override for the consecutive-failure "
                               "circuit breaker. Trip on the Nth failure — "
                               "e.g. --max-retries 1 blocks on the first "
                               "failure (no retries), --max-retries 3 allows "
                               "two retries. Omit to use the dispatcher's "
                               "kanban.failure_limit config "
                               f"(default {kb.DEFAULT_FAILURE_LIMIT}).")
    p_create.add_argument("--model", default=None, dest="model_override",
                          help="Pin the worker to this model (passed as "
                               "-m <model>) without changing the profile's "
                               "configured model. Combine with --provider "
                               "when the model belongs to a different "
                               "backend than the profile's default.")
    p_create.add_argument("--provider", default=None, dest="provider_override",
                          help="Provider the --model belongs to (passed as "
                               "--provider <name> to the worker). Requires "
                               "--model.")
    p_create.add_argument("--goal", action="store_true", dest="goal_mode",
                          help="Run the worker in a goal loop: after each "
                               "turn a judge checks the response against the "
                               "card title/body and, if not done, the worker "
                               "keeps going in the same session until the "
                               "judge agrees it's complete (or the turn "
                               "budget runs out, which blocks the card for "
                               "review). Best for open-ended cards one shot "
                               "rarely finishes.")
    p_create.add_argument("--goal-max-turns", type=int, default=None,
                          metavar="N", dest="goal_max_turns",
                          help="Turn budget for --goal workers (default 20). "
                               "Ignored without --goal.")
    p_create.add_argument("--initial-status",
                          choices=sorted(kb.VALID_INITIAL_STATUSES),
                          default="running",
                          help="Initial card status. Use 'blocked' for cards "
                               "that require immediate human ops (R3 gate) "
                               "to skip the brief running-to-blocked transition.")
    p_create.add_argument("--json", action="store_true", help="Emit JSON output")

    # --- swarm ---
    p_swarm = sub.add_parser(
        "swarm",
        help="Create a Kanban Swarm v1 graph (parallel workers → verifier → synthesizer)",
    )
    p_swarm.add_argument("goal", help="Swarm goal / final outcome")
    p_swarm.add_argument(
        "--worker",
        action="append",
        default=[],
        metavar="PROFILE:TITLE[:SKILL,SKILL]",
        help="Parallel worker card (repeatable)",
    )
    p_swarm.add_argument("--verifier", required=True, help="Verifier profile")
    p_swarm.add_argument("--synthesizer", required=True, help="Synthesizer/writer profile")
    p_swarm.add_argument("--tenant", default=None, help="Tenant namespace")
    p_swarm.add_argument("--priority", type=int, default=0, help="Priority tiebreaker")
    p_swarm.add_argument("--created-by", default=None, help="Creator/anchor profile")
    p_swarm.add_argument("--idempotency-key", default=None, help="Dedup key for the root card")
    p_swarm.add_argument("--json", action="store_true", help="Emit JSON output")

    # --- list ---
    p_list = sub.add_parser("list", aliases=["ls"], help="List tasks")
    p_list.add_argument("--mine", action="store_true",
                        help="Filter by $HERMES_PROFILE as assignee")
    p_list.add_argument("--assignee", default=None)
    p_list.add_argument("--status", default=None,
                        choices=sorted(kb.VALID_STATUSES))
    p_list.add_argument("--tenant", default=None)
    p_list.add_argument("--session", default=None,
                        help="Filter by originating chat/agent session id "
                             "(set on tasks created from inside an ACP loop)")
    p_list.add_argument("--archived", action="store_true",
                        help="Include archived tasks")
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument(
        "--sort",
        default=None,
        choices=sorted(kb.VALID_SORT_ORDERS.keys()),
        help="Sort order for listed tasks (default: priority)",
    )
    p_list.add_argument(
        "--workflow-template-id",
        default=None,
        metavar="ID",
        help="Restrict to tasks with this workflow_template_id",
    )
    p_list.add_argument(
        "--step-key",
        default=None,
        dest="current_step_key",
        metavar="KEY",
        help="Restrict to tasks with this current_step_key",
    )

    # --- show ---
    p_show = sub.add_parser("show", help="Show a task with comments + events")
    p_show.add_argument("task_id")
    p_show.add_argument("--json", action="store_true")
    p_show.add_argument(
        "--state-type",
        choices=("status", "outcome"),
        default=None,
        help="With --state-name: filter listed runs by task_runs column",
    )
    p_show.add_argument(
        "--state-name",
        default=None,
        metavar="VALUE",
        help="With --state-type: keep runs whose column equals this value",
    )

    # --- assign ---
    p_assign = sub.add_parser("assign", help="Assign or reassign a task")
    p_assign.add_argument("task_id")
    p_assign.add_argument("profile", help="Profile name (or 'none' to unassign)")
    _add_request_key_argument(p_assign)

    # --- set-model (per-task model/provider override) ---
    p_set_model = sub.add_parser(
        "set-model",
        help="Set or clear a task's model/provider override "
             "(takes effect on the next dispatch)",
    )
    p_set_model.add_argument("task_id")
    p_set_model.add_argument(
        "model", nargs="?", default=None,
        help="Model to pin the worker to (or 'none' to clear the override)",
    )
    p_set_model.add_argument(
        "--provider", default=None,
        help="Provider the model belongs to (worker is spawned with "
             "--provider <name>). Cleared together with the model.",
    )

    # --- reclaim / reassign (recovery) ---
    p_reclaim = sub.add_parser(
        "reclaim",
        help="Release an active worker claim on a running task",
    )
    p_reclaim.add_argument("task_id")
    p_reclaim.add_argument(
        "--reason", default=None,
        help="Human-readable reason (recorded on the reclaimed event)",
    )

    p_reassign = sub.add_parser(
        "reassign",
        help="Reassign a task to a different profile, optionally reclaiming first",
    )
    p_reassign.add_argument("task_id")
    p_reassign.add_argument(
        "profile",
        help="New profile name (or 'none' to unassign)",
    )
    p_reassign.add_argument(
        "--reclaim", action="store_true",
        help="Release any active claim before reassigning (required if task is running)",
    )
    p_reassign.add_argument(
        "--reason", default=None,
        help="Human-readable reason (recorded on the reclaimed event)",
    )

    # --- diagnostics (board-wide health) ---
    p_diag = sub.add_parser(
        "diagnostics",
        aliases=["diag"],
        help="List active diagnostics on the current board",
    )
    p_diag.add_argument(
        "--severity",
        choices=["warning", "error", "critical"],
        default=None,
        help="Only show diagnostics at or above this severity",
    )
    p_diag.add_argument(
        "--task",
        default=None,
        help="Only show diagnostics for one task id",
    )
    p_diag.add_argument(
        "--json", action="store_true",
        help="Emit JSON (structured) instead of the default human table",
    )

    # --- link / unlink ---
    p_link = sub.add_parser("link", help="Add a parent->child dependency")
    p_link.add_argument("parent_id")
    p_link.add_argument("child_id")
    _add_request_key_argument(p_link)
    p_unlink = sub.add_parser("unlink", help="Remove a parent->child dependency")
    p_unlink.add_argument("parent_id")
    p_unlink.add_argument("child_id")
    _add_request_key_argument(p_unlink)

    # --- claim ---
    p_claim = sub.add_parser(
        "claim",
        help="Atomically claim a ready task (prints resolved workspace path)",
    )
    p_claim.add_argument("task_id")
    p_claim.add_argument("--ttl", type=int, default=kb.DEFAULT_CLAIM_TTL_SECONDS,
                         help="Claim TTL in seconds (default: 900)")
    _add_request_key_argument(p_claim)

    # --- comment / complete / block / unblock / archive ---
    p_comment = sub.add_parser("comment", help="Append a comment")
    p_comment.add_argument("task_id")
    p_comment.add_argument("text", nargs="+", help="Comment body")
    p_comment.add_argument("--author", default=None,
                           help="Author name (default: $HERMES_PROFILE or 'user')")
    p_comment.add_argument("--max-len", type=int, default=None,
                           help="Trim the stored comment body to this many characters")
    _add_request_key_argument(p_comment)

    # --- attach / attachments / attach-rm ---
    p_attach = sub.add_parser(
        "attach",
        help="Attach bounded inline bytes to a task through the writer",
    )
    p_attach.add_argument("task_id")
    # ``path`` remains an optional compatibility parse slot so old invocations
    # fail closed in the adapter with a useful message rather than opening it.
    p_attach.add_argument(
        "path", nargs="?", default=None,
        help="Deprecated filesystem path; rejected by the writer boundary",
    )
    p_attach.add_argument(
        "--filename", "--name", dest="filename", default=None,
        help="Basename stored by the writer (required with --data-b64)",
    )
    p_attach.add_argument(
        "--data-b64", dest="data_b64", default=None,
        help="Bounded base64-encoded attachment bytes",
    )
    p_attach.add_argument("--content-type", default=None,
                          help="Optional MIME type")
    p_attach.add_argument("--json", action="store_true")
    _add_request_key_argument(p_attach)

    p_attachments = sub.add_parser("attachments", help="List a task's attachments")
    p_attachments.add_argument("task_id")
    p_attachments.add_argument("--json", action="store_true")

    p_attach_rm = sub.add_parser("attach-rm", help="Delete an attachment by id")
    p_attach_rm.add_argument("attachment_id", type=int)

    p_complete = sub.add_parser("complete", help="Mark one or more tasks done")
    p_complete.add_argument("task_ids", nargs="+",
                            help="One or more task ids (only --result applies to all of them)")
    p_complete.add_argument("--result", default=None, help="Result summary")
    p_complete.add_argument("--summary", default=None,
                            help="Structured handoff summary for downstream tasks. "
                                 "Falls back to --result if omitted.")
    p_complete.add_argument("--metadata", default=None,
                            help='JSON dict of structured facts (e.g. \'{"changed_files": [...], '
                                 '"tests_run": 12}\'). Stored on the closing run.')
    _add_request_key_argument(p_complete)

    p_edit = sub.add_parser(
        "edit",
        help="Edit recovery fields on an already-completed task",
    )
    p_edit.add_argument("task_id")
    p_edit.add_argument(
        "--result",
        required=True,
        help="Backfilled task result text for a done task",
    )
    p_edit.add_argument(
        "--summary",
        default=None,
        help="Structured handoff summary. Falls back to --result if omitted.",
    )
    p_edit.add_argument(
        "--metadata",
        default=None,
        help="JSON dict of structured facts to store on the latest completed run.",
    )

    p_block = sub.add_parser("block", help="Mark one or more tasks blocked")
    p_block.add_argument("task_id")
    p_block.add_argument("reason", nargs="*", help="Reason (also appended as a comment)")
    p_block.add_argument("--ids", nargs="+", default=None,
                         help="Additional task ids to block with the same reason (bulk mode)")
    p_block.add_argument(
        "--kind", default=None, choices=sorted(kb.VALID_BLOCK_KINDS),
        help=(
            "Typed block reason. 'dependency' waits in todo (auto-promoted "
            "when parents finish, no human); 'needs_input'/'capability' go to "
            "blocked for a human; 'transient' marks a maybe-flaky failure. "
            "Repeated same-kind re-blocks after unblock route the task to "
            "triage to break unblock loops. Omit for a generic block."
        ),
    )
    _add_request_key_argument(p_block)

    p_schedule = sub.add_parser("schedule", help="Park one or more tasks in Scheduled (waiting on time, not human input)")
    p_schedule.add_argument("task_id")
    p_schedule.add_argument("reason", nargs="*", help="Reason/timing note (also appended as a comment)")
    p_schedule.add_argument("--ids", nargs="+", default=None,
                            help="Additional task ids to schedule with the same reason (bulk mode)")

    p_unblock = sub.add_parser(
        "unblock",
        help="Return blocked/scheduled tasks to ready, or todo while parents remain open",
    )
    p_unblock.add_argument(
        "--reason",
        default=None,
        help="Optional reason/note — recorded as a comment before unblocking. Quote multi-word reasons.",
    )
    p_unblock.add_argument("task_ids", nargs="+")
    _add_request_key_argument(p_unblock)

    p_promote = sub.add_parser(
        "promote",
        help="Manually move one or more todo/blocked tasks to ready (recovery path)",
    )
    p_promote.add_argument("task_id")
    p_promote.add_argument(
        "reason",
        nargs="*",
        help="Audit-trail reason (recorded on the task_events row)",
    )
    p_promote.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Additional task ids to promote with the same reason (bulk mode)",
    )
    p_promote.add_argument(
        "--force",
        action="store_true",
        help="Promote even if parent dependencies are not yet done/archived",
    )
    p_promote.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the promotion without mutating state",
    )
    p_promote.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="Emit machine-readable JSON result",
    )

    p_archive = sub.add_parser("archive", help="Archive one or more tasks")
    p_archive.add_argument("task_ids", nargs="*",
                           help="Task ids to archive (default mode)")
    p_archive.add_argument(
        "--rm",
        dest="purge_ids",
        nargs="+",
        default=None,
        help="Permanently delete already-archived task ids from the board",
    )

    # --- tail ---
    p_tail = sub.add_parser("tail", help="Follow a task's event stream")
    p_tail.add_argument("task_id")
    p_tail.add_argument("--interval", type=float, default=1.0)

    # --- dispatch ---
    p_disp = sub.add_parser(
        "dispatch",
        help="One dispatcher pass: reclaim stale, promote ready, spawn workers",
    )
    p_disp.add_argument("--dry-run", action="store_true",
                        help="Don't actually spawn processes; just print what would happen")
    p_disp.add_argument("--max", type=int, default=None,
                        help="Cap number of spawns this pass")
    p_disp.add_argument("--failure-limit", type=int,
                        default=kb.DEFAULT_SPAWN_FAILURE_LIMIT,
                        help=f"Auto-block a task after this many consecutive non-success attempts "
                             f"(spawn_failed, timed_out, or crashed; default: {kb.DEFAULT_SPAWN_FAILURE_LIMIT})")
    p_disp.add_argument("--json", action="store_true")

    # --- daemon (deprecated) ---
    p_daemon = sub.add_parser(
        "daemon",
        help="DEPRECATED — dispatcher now runs in the gateway. Use `hermes gateway start`.",
    )
    p_daemon.add_argument("--interval", type=float, default=60.0,
                          help="Seconds between dispatch ticks (default: 60)")
    p_daemon.add_argument("--max", type=int, default=None,
                          help="Cap number of spawns per tick")
    p_daemon.add_argument("--failure-limit", type=int,
                          default=kb.DEFAULT_SPAWN_FAILURE_LIMIT)
    p_daemon.add_argument("--pidfile", default=None,
                          help="Write the daemon's PID to this file on start")
    p_daemon.add_argument("--verbose", "-v", action="store_true",
                          help="Log each tick's outcome to stdout")
    # Undocumented escape hatch for users who truly cannot run the gateway.
    # Intentionally excluded from --help so nobody discovers it casually and
    # keeps the old double-dispatcher pattern alive.
    p_daemon.add_argument("--force", action="store_true",
                          help=argparse.SUPPRESS)

    # --- watch ---
    p_watch = sub.add_parser(
        "watch",
        help="Live-stream task_events to the terminal (Ctrl+C to exit)",
    )
    p_watch.add_argument("--assignee", default=None,
                         help="Only show events for tasks assigned to this profile")
    p_watch.add_argument("--tenant", default=None,
                         help="Only show events from tasks in this tenant")
    p_watch.add_argument("--kinds", default=None,
                         help="Comma-separated event kinds to include "
                              "(e.g. 'completed,blocked,gave_up,crashed,timed_out')")
    p_watch.add_argument("--interval", type=float, default=0.5,
                         help="Poll interval in seconds (default: 0.5)")

    # --- stats ---
    p_stats = sub.add_parser(
        "stats", help="Per-status + per-assignee counts + oldest-ready age",
    )
    p_stats.add_argument("--json", action="store_true")

    # --- notify subscribe / list / remove ---
    p_nsub = sub.add_parser(
        "notify-subscribe",
        help="Subscribe a gateway source to a task's terminal events "
             "(used by /kanban subscribe in the gateway adapter)",
    )
    p_nsub.add_argument("task_id")
    p_nsub.add_argument("--platform", required=True)
    p_nsub.add_argument("--chat-id", required=True)
    p_nsub.add_argument("--chat-type", default="", help="dm / group / channel (used by wake routing)")
    p_nsub.add_argument("--thread-id", default=None)
    p_nsub.add_argument("--user-id", default=None)
    p_nsub.add_argument(
        "--notifier-profile", default=None,
        help="Profile gateway that owns/delivers this subscription (default: active profile)",
    )

    p_nlist = sub.add_parser(
        "notify-list",
        help="List notification subscriptions (optionally for a single task)",
    )
    p_nlist.add_argument("task_id", nargs="?", default=None)
    p_nlist.add_argument("--json", action="store_true")

    p_nrm = sub.add_parser(
        "notify-unsubscribe",
        help="Remove a gateway subscription from a task",
    )
    p_nrm.add_argument("task_id")
    p_nrm.add_argument("--platform", required=True)
    p_nrm.add_argument("--chat-id", required=True)
    p_nrm.add_argument("--thread-id", default=None)

    # --- log ---
    p_log = sub.add_parser(
        "log",
        help="Print the worker log for a task (from <kanban-root>/kanban/logs/)",
    )
    p_log.add_argument("task_id")
    p_log.add_argument("--tail", type=int, default=None,
                       help="Only print the last N bytes")

    # --- runs (per-attempt history for a task) ---
    p_runs = sub.add_parser(
        "runs",
        help="Show attempt history for a task (one row per run: profile, "
             "outcome, elapsed, summary)",
    )
    p_runs.add_argument("task_id")
    p_runs.add_argument("--json", action="store_true")
    p_runs.add_argument(
        "--state-type",
        choices=("status", "outcome"),
        default=None,
        help="With --state-name: filter runs by task_runs column",
    )
    p_runs.add_argument(
        "--state-name",
        default=None,
        metavar="VALUE",
        help="With --state-type: keep runs whose column equals this value",
    )

    # --- heartbeat (worker liveness signal) ---
    p_hb = sub.add_parser(
        "heartbeat",
        help="Emit a heartbeat event for a running task (worker liveness signal)",
    )
    p_hb.add_argument("task_id")
    p_hb.add_argument("--note", default=None,
                      help="Optional short note attached to the heartbeat event")
    _add_request_key_argument(p_hb)

    # --- assignees ---
    p_asg = sub.add_parser(
        "assignees",
        help="List known profiles + per-profile task counts "
             "(union of ~/.hermes/profiles/ and current assignees on the board)",
    )
    p_asg.add_argument("--json", action="store_true")

    # --- context --- (for spawned workers)
    p_ctx = sub.add_parser(
        "context",
        help="Print the full context a worker sees for a task "
             "(title + body + parent results + comments).",
    )
    p_ctx.add_argument("task_id")

    # --- specify --- (triage → todo via auxiliary LLM)
    p_specify = sub.add_parser(
        "specify",
        help="Flesh out a triage-column task into a concrete spec "
             "(title + body) and promote it to todo. Uses the auxiliary "
             "LLM configured under auxiliary.triage_specifier.",
    )
    p_specify.add_argument(
        "task_id",
        nargs="?",
        default=None,
        help="Task id to specify (required unless --all is given)",
    )
    p_specify.add_argument(
        "--all",
        dest="all_triage",
        action="store_true",
        help="Specify every task currently in the triage column",
    )
    p_specify.add_argument(
        "--tenant",
        default=None,
        help="When used with --all, restrict the sweep to this tenant",
    )
    p_specify.add_argument(
        "--author",
        default=None,
        help="Author name recorded on the audit comment "
             "(default: $HERMES_PROFILE or 'specifier')",
    )
    p_specify.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per task on stdout",
    )

    # --- decompose --- (triage → fan-out via auxiliary LLM + orchestrator)
    p_decompose = sub.add_parser(
        "decompose",
        help="Decompose a triage-column task into a graph of child tasks "
             "routed to specialist profiles by description. Falls back to "
             "specify-style single-task promotion when the task doesn't "
             "benefit from fan-out. Uses auxiliary.kanban_decomposer.",
    )
    p_decompose.add_argument(
        "task_id",
        nargs="?",
        default=None,
        help="Task id to decompose (required unless --all is given)",
    )
    p_decompose.add_argument(
        "--all",
        dest="all_triage",
        action="store_true",
        help="Decompose every task currently in the triage column",
    )
    p_decompose.add_argument(
        "--tenant",
        default=None,
        help="When used with --all, restrict the sweep to this tenant",
    )
    p_decompose.add_argument(
        "--author",
        default=None,
        help="Author name recorded on the audit comment "
             "(default: $HERMES_PROFILE or 'decomposer')",
    )
    p_decompose.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per task on stdout",
    )

    # --- gc ---
    p_gc = sub.add_parser(
        "gc", help="Garbage-collect archived-task workspaces, old events, and old logs",
    )
    p_gc.add_argument("--event-retention-days", type=int, default=30,
                      help="Delete task_events older than N days for terminal tasks (default: 30)")
    p_gc.add_argument("--log-retention-days", type=int, default=30,
                      help="Delete worker log files older than N days (default: 30)")

    # --- repair ---
    p_repair = sub.add_parser(
        "repair",
        help="Check kanban.db integrity and auto-repair index-only corruption",
        description=(
            "Runs PRAGMA integrity_check on the board's DB and reports the "
            "result. When the failure consists only of index-scoped errors "
            "('wrong # of entries in index <name>' / 'row N missing from "
            "index <name>'), the corrupt file is quarantined to a "
            ".corrupt.<hash>.bak sibling first and the damaged indexes are "
            "rebuilt with REINDEX — the same narrow auto-repair the "
            "connect-time guard applies. Any other corruption class is "
            "reported and left untouched (fail-closed). Exits 0 when the DB "
            "is healthy or was repaired, non-zero when it is still corrupt."
        ),
    )
    p_repair.add_argument("--json", action="store_true",
                          help="Emit the repair report as JSON")

    kanban_parser.set_defaults(_kanban_parser=kanban_parser)
    return kanban_parser


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

def kanban_command(args: argparse.Namespace) -> int:
    """Entry point from ``hermes kanban …`` argparse dispatch.

    Returns a shell-style exit code (0 on success, non-zero on error).
    """
    action = getattr(args, "kanban_action", None)
    if not action:
        # No subaction given: print help via the stored parser reference.
        parser = getattr(args, "_kanban_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print(
                "usage: hermes kanban <action> [options]\n"
                "Run 'hermes kanban --help' for the full list of actions.",
                file=sys.stderr,
            )
        return 0

    # Fast-fail for clearer CLI UX only. The durable trust boundary is lower in
    # hermes_cli.kanban_db, because children can import DB mutators directly.
    if _is_delegated_child_cli_mutation(args):
        print(
            "kanban: delegate_task child contexts cannot mutate Kanban tasks via the CLI",
            file=sys.stderr,
        )
        return 1

    # U1-D0B admits only the fixed writer operations.  Reject board
    # selection and every legacy command before any DB initialization,
    # board resolution, filesystem work, or direct SQL can occur.
    if getattr(args, "board", None):
        return _cli_refused(action, "board selection is fixed by the privileged writer")
    if action == "boards" or action not in _IPC_SUPPORTED_ACTIONS:
        return _cli_refused(action)
    return _ipc_cli_command(args)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _profile_author() -> str:
    """Best-effort author name for an interactive CLI call."""
    for env in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        v = os.environ.get(env)
        if v:
            return v
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "user"
    except Exception:
        return "user"


_DELEGATED_CHILD_DENIED_ACTIONS: frozenset[str] = frozenset({
    "init",
    "create",
    "swarm",
    "assign",
    "reclaim",
    "reassign",
    "link",
    "unlink",
    "claim",
    "comment",
    "attach",
    "attach-rm",
    "complete",
    "edit",
    "block",
    "schedule",
    "unblock",
    "promote",
    "archive",
    "dispatch",
    "daemon",
    "repair",
    "heartbeat",
    "notify-subscribe",
    "notify-unsubscribe",
    "specify",
    "decompose",
    "gc",
})

_DELEGATED_CHILD_DENIED_BOARD_ACTIONS: frozenset[str] = frozenset({
    "create",
    "new",
    "rm",
    "remove",
    "delete",
    "switch",
    "use",
    "rename",
    "set-default-workdir",
})


def _is_delegated_child_cli_mutation(args: argparse.Namespace) -> bool:
    action = getattr(args, "kanban_action", None)
    if action == "boards":
        boards_action = getattr(args, "boards_action", None) or "list"
        if boards_action not in _DELEGATED_CHILD_DENIED_BOARD_ACTIONS:
            return False
    elif action not in _DELEGATED_CHILD_DENIED_ACTIONS:
        return False
    try:
        from agent.delegation_context import is_delegated_child_process_context

        return is_delegated_child_process_context()
    except Exception:
        return bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))


# ---------------------------------------------------------------------------
# Boards management (hermes kanban boards …)
# ---------------------------------------------------------------------------

def _dispatch_boards(args: argparse.Namespace) -> int:
    return _cli_refused("boards", "board management is outside the writer protocol")


def _board_task_counts(slug: str) -> dict[str, int]:
    _cli_refused("boards", "board counts are outside the writer protocol")
    return {}


def _cmd_boards_list(args: argparse.Namespace) -> int:
    return _cli_refused("boards list", "board management is outside the writer protocol")


def _cmd_boards_create(args: argparse.Namespace) -> int:
    return _cli_refused("boards create", "board management is outside the writer protocol")


def _cmd_boards_rm(args: argparse.Namespace) -> int:
    return _cli_refused("boards rm", "board management is outside the writer protocol")


def _cmd_boards_switch(args: argparse.Namespace) -> int:
    return _cli_refused("boards switch", "board management is outside the writer protocol")


def _cmd_boards_show(args: argparse.Namespace) -> int:
    return _cli_refused("boards show", "board management is outside the writer protocol")


def _cmd_boards_rename(args: argparse.Namespace) -> int:
    return _cli_refused("boards rename", "board management is outside the writer protocol")


def _cmd_boards_set_default_workdir(args: argparse.Namespace) -> int:
    return _cli_refused("boards set-default-workdir", "board management is outside the writer protocol")


# ---------------------------------------------------------------------------


def _parse_duration(val) -> Optional[int]:
    """Parse ``30s`` / ``5m`` / ``2h`` / ``1d`` or a raw integer → seconds.

    Returns None for empty input. Raises ValueError on malformed input so
    the CLI can surface a usage error cleanly.
    """
    if val is None or val == "":
        return None
    s = str(val).strip().lower()
    # Bare integer → seconds.
    try:
        return int(s)
    except ValueError:
        pass
    # Suffixed form.
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s and s[-1] in units:
        try:
            n = float(s[:-1])
        except ValueError as exc:
            raise ValueError(f"malformed duration {val!r}") from exc
        return int(n * units[s[-1]])
    raise ValueError(f"malformed duration {val!r} (expected 30s, 5m, 2h, 1d, or a number)")


def _worker_run_id_for(task_id: str) -> Optional[int]:
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    raw = os.environ.get("HERMES_KANBAN_RUN_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _ready_queue_nonempty() -> bool:
    return bool(_cli_refused("daemon", "queue inspection is outside the writer protocol"))


def _cmd_init(args: argparse.Namespace) -> int:
    return _cli_refused("init", "database initialization is writer-owned")


def _cmd_swarm(args: argparse.Namespace) -> int:
    return _cli_refused("swarm", "swarm admission is outside the writer protocol")


def _cmd_specify(args: argparse.Namespace) -> int:
    return _cli_refused("specify", "specification is outside the writer protocol")


def _cmd_decompose(args: argparse.Namespace) -> int:
    return _cli_refused("decompose", "decomposition is outside the writer protocol")


def _cmd_set_model(args: argparse.Namespace) -> int:
    return _cli_refused("set-model", "model mutation is outside the writer protocol")


def _cmd_attach_rm(args: argparse.Namespace) -> int:
    return _cli_refused("attach-rm", "attachment deletion is outside the writer protocol")


def _cmd_reclaim(args: argparse.Namespace) -> int:
    return _cli_refused("reclaim", "reclaim is outside the writer protocol")


def _cmd_reassign(args: argparse.Namespace) -> int:
    return _cli_refused("reassign", "reassign is outside the writer protocol")


def _cmd_edit(args: argparse.Namespace) -> int:
    return _cli_refused("edit", "editing is outside the writer protocol")


def _cmd_schedule(args: argparse.Namespace) -> int:
    return _cli_refused("schedule", "scheduling is outside the writer protocol")


def _cmd_promote(args: argparse.Namespace) -> int:
    return _cli_refused("promote", "promotion is outside the writer protocol")


def _cmd_archive(args: argparse.Namespace) -> int:
    return _cli_refused("archive", "archival mutation is outside the writer protocol")


def _cmd_dispatch(args: argparse.Namespace) -> int:
    return _cli_refused("dispatch", "dispatch is outside the writer protocol")


def _cmd_daemon(args: argparse.Namespace) -> int:
    return _cli_refused("daemon", "daemon control is outside the writer protocol")


def _cmd_gc(args: argparse.Namespace) -> int:
    return _cli_refused("gc", "garbage collection is outside the writer protocol")


def _cmd_repair(args: argparse.Namespace) -> int:
    return _cli_refused("repair", "repair is outside the writer protocol")


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    return _cli_refused("diagnostics", "diagnostics are outside the writer protocol")


def _cmd_tail(args: argparse.Namespace) -> int:
    return _cli_refused("tail", "tailing is outside the writer protocol")


def _cmd_watch(args: argparse.Namespace) -> int:
    return _cli_refused("watch", "watch is outside the writer protocol")


def _cmd_stats(args: argparse.Namespace) -> int:
    return _cli_refused("stats", "stats are outside the writer protocol")


def _cmd_log(args: argparse.Namespace) -> int:
    return _cli_refused("log", "worker logs are outside the writer protocol")


def _cmd_context(args: argparse.Namespace) -> int:
    return _cli_refused("context", "worker context is exposed only through show")


def _cmd_assignees(args: argparse.Namespace) -> int:
    return _cli_refused("assignees", "assignee enumeration is outside the writer protocol")


def _cmd_notify_subscribe(args: argparse.Namespace) -> int:
    return _cli_refused("notify-subscribe", "notifications are outside this CLI route")


def _cmd_notify_list(args: argparse.Namespace) -> int:
    return _cli_refused("notify-list", "notifications are outside this CLI route")


def _cmd_notify_unsubscribe(args: argparse.Namespace) -> int:
    return _cli_refused("notify-unsubscribe", "notifications are outside this CLI route")


def _cmd_create(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "create")


def _cmd_assign(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "assign")


def _cmd_link(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "link")


def _cmd_unlink(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "unlink")


def _cmd_claim(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "claim")


def _cmd_comment(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "comment")


def _cmd_complete(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "complete")


def _cmd_block(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "block")


def _cmd_unblock(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "unblock")


def _cmd_heartbeat(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "heartbeat")


def _cmd_attach(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "attach")


def _cmd_list(args: argparse.Namespace) -> int:
    return _writer_call_command(args, getattr(args, "kanban_action", "list") or "list")


def _cmd_show(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "show")


def _cmd_attachments(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "attachments")


def _cmd_runs(args: argparse.Namespace) -> int:
    return _writer_call_command(args, "runs")


# ---------------------------------------------------------------------------
# Slash-command entry point (used by /kanban from CLI and gateway)
# ---------------------------------------------------------------------------

_SLASH_KANBAN_HELP = """\
**/kanban** — manage the shared task board.

Common subcommands:
  `list` (alias `ls`)   List tasks on the current board
  `show <id>`           Task details + comments + events
  `stats`               Per-status / per-assignee counts
  `create <title>…`     Create a task (auto-subscribes you to events)
  `comment <id> <msg>`  Append a comment
  `attach <id> --filename <name> --data-b64 <bytes>`  Attach inline bytes
  `complete <id>…`      Mark task(s) done
  `block <id> [reason]` Mark blocked; `schedule <id> [reason]` parks time-delay work; `unblock <id>` to revive
  `assign <id> <profile>`  Reassign
  `boards list`         Show all boards
  `assignees`           Known profiles + counts
  `context <id>`        Full worker-context dump
  `runs <id>`           Attempt history
  `log <id>`            Worker log

Run `/kanban <subcommand> -h` for arguments. \
Read-only commands are safe while an agent is running.\
"""


def run_slash(rest: str) -> str:
    """Execute a ``/kanban …`` string and return captured stdout/stderr.

    ``rest`` is everything after ``/kanban`` (may be empty).  Used from
    both the interactive CLI (``self._handle_kanban_command``) and the
    gateway (``_handle_kanban_command``) so formatting is identical.
    """
    import io
    import contextlib

    tokens = shlex.split(rest) if rest and rest.strip() else []

    # Bare ``/kanban`` or ``/kanban help`` / ``--help`` / ``-h`` / ``?``:
    # show the curated short-help block instead of dumping argparse's full
    # usage tree (which is enormous and reads as garbage in a chat
    # bubble).  Per-subcommand help still works via ``/kanban foo -h``.
    if not tokens or tokens[0] in {"help", "--help", "-h", "?"}:
        return _SLASH_KANBAN_HELP

    # Single argparse tree rooted at "/kanban".  build_parser() expects a
    # subparsers action to attach to, so build a throwaway one and pull
    # the kanban_parser back out — then drive it directly so usage/error
    # text reads as ``/kanban`` (not ``/kanban-wrap kanban``).
    _wrap = argparse.ArgumentParser(prog="/kanban-wrap", add_help=False)
    _wrap.exit_on_error = False  # type: ignore[attr-defined]
    _top_sub = _wrap.add_subparsers(dest="_top")
    kanban_parser = build_parser(_top_sub)
    kanban_parser.prog = "/kanban"
    kanban_parser.exit_on_error = False  # type: ignore[attr-defined]
    for _action in kanban_parser._actions:
        if isinstance(_action, argparse._SubParsersAction):
            for _name, _choice in _action.choices.items():
                _choice.prog = f"/kanban {_name}"
                _choice.exit_on_error = False  # type: ignore[attr-defined]

    def _usage_for_error() -> str:
        if tokens:
            for _action in kanban_parser._actions:
                if isinstance(_action, argparse._SubParsersAction):
                    subparser = _action.choices.get(tokens[0])
                    if subparser is not None:
                        return subparser.format_usage().rstrip()
        return kanban_parser.format_usage().rstrip()

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    # ``-h`` / ``--help`` makes argparse print to stdout and SystemExit(0).
    # Capture both streams so neither the help text nor the error text
    # bypasses our buffer.
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            args = kanban_parser.parse_args(tokens)
    except SystemExit as exc:
        out = buf_out.getvalue().rstrip()
        err = buf_err.getvalue().rstrip()
        # Help dump (exit 0) → return the captured help text directly.
        if exc.code in {0, None} and out:
            return out
        body = err or out
        return f"⚠ /kanban usage error\n{body}" if body else "⚠ /kanban usage error"
    except argparse.ArgumentError as exc:
        return f"⚠ /kanban usage error\n{_usage_for_error()}\n{exc}"

    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            kanban_command(args)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)

    out = buf_out.getvalue().rstrip()
    err = buf_err.getvalue().rstrip()
    if err and out:
        return f"{out}\n{err}"
    return err if err else (out or "(no output)")
