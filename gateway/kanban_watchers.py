"""Gateway Kanban watcher adapters for the privileged writer boundary.

The gateway no longer discovers boards or opens a Kanban database.  The
writer owns subscription enumeration and event claims; these methods only
forward the finite cursor operations when a subscription is already known.
Dispatcher/recovery/auto-decompose loops are deliberately refused in D0B.
"""

from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.kanban_writer import CANONICAL_SOCKET_PATH, WriterError, writer_request

logger = logging.getLogger("gateway.run")

_CANONICAL_BOARD = "bryceos"


def _resolve_auto_decompose_settings(
    load_config: Callable[[], Any],
) -> "tuple[bool, int]":
    """Keep the read-only config helper for callers; admission stays refused."""
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    enabled = bool(kcfg.get("auto_decompose", True))
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    return enabled, max(per_tick, 1)


def _acquire_singleton_lock(lock_path) -> "tuple[Optional[object], str]":
    """Refuse dispatcher lock acquisition before any path/file effect."""
    return None, "refused"


def _release_singleton_lock(handle) -> None:
    if handle is None:
        return
    try:
        from gateway.status import _release_file_lock
        _release_file_lock(handle)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


def _writer_socket() -> Path:
    return CANONICAL_SOCKET_PATH


def _check_board(board: Optional[str]) -> None:
    if board not in (None, "", _CANONICAL_BOARD):
        raise ValueError("only the canonical bryceos board is admitted")


def _writer_call(
    operation: str,
    args: dict[str, Any],
    *,
    mutation: bool = True,
    request_key: Optional[str] = None,
) -> Any:
    if mutation and not request_key:
        raise ValueError(f"{operation}: stable request_key is required before writer IPC")
    try:
        response = writer_request(
            _writer_socket(),
            operation,
            args,
            request_key=request_key if mutation else None,
        )
    except (OSError, WriterError) as exc:
        logger.warning("kanban writer %s failed: %s", operation, exc)
        return None
    return response.get("result") if isinstance(response, dict) else response


def _sub_identity(sub: dict[str, Any]) -> dict[str, Any]:
    """Project a subscription into the writer's closed cursor schema."""
    task_id = sub.get("task_id")
    platform = sub.get("platform")
    chat_id = sub.get("chat_id")
    if not task_id or not platform or chat_id is None:
        raise ValueError("subscription is missing task_id/platform/chat_id")
    return {
        "task_id": str(task_id),
        "platform": str(platform),
        "chat_id": str(chat_id),
        "thread_id": str(sub.get("thread_id") or ""),
    }


def _subscription_request_key(
    operation: str, sub: dict[str, Any], args: dict[str, Any],
) -> str:
    subscription_id = sub.get("id")
    if subscription_id is None:
        raise ValueError(
            f"{operation}: subscription id is required for retry-stable writer IPC"
        )
    material: dict[str, Any] = {
        "operation": operation,
        "subscription_id": str(subscription_id),
        "args": args,
    }
    if operation == "notify-claim":
        if sub.get("last_event_id") is None:
            raise ValueError(
                "notify-claim: subscription cursor is required for retry-stable writer IPC"
            )
        material["last_event_id"] = int(sub["last_event_id"])
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"gateway-{digest}"


class GatewayKanbanWatchersMixin:
    """Safe gateway hooks for the writer-owned notification boundary."""

    def _owns_kanban_dispatcher_lock(self) -> bool:
        return getattr(self, "_kanban_dispatcher_lock_handle", None) is not None

    def _release_kanban_dispatcher_lock(self) -> None:
        handle = getattr(self, "_kanban_dispatcher_lock_handle", None)
        self._kanban_dispatcher_lock_handle = None
        _release_singleton_lock(handle)

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Refuse subscription enumeration; the writer owns that read."""
        logger.warning(
            "kanban notifier watcher refused: subscription enumeration must use writer IPC"
        )
        return

    def _kanban_claim(
        self,
        sub: dict[str, Any],
        kinds: Optional[list[str]] = None,
        board: Optional[str] = None,
    ) -> Any:
        """Claim known-subscription events through the fixed writer operation."""
        _check_board(board)
        args = _sub_identity(sub)
        if kinds is not None:
            args["kinds"] = list(kinds)
        return _writer_call(
            "notify-claim", args,
            request_key=_subscription_request_key("notify-claim", sub, args),
        )

    def _kanban_advance(
        self, sub: dict[str, Any], cursor: int, board: Optional[str] = None,
    ) -> None:
        _check_board(board)
        args = _sub_identity(sub)
        args["new_cursor"] = int(cursor)
        _writer_call(
            "notify-advance", args,
            request_key=_subscription_request_key("notify-advance", sub, args),
        )

    def _kanban_unsub(self, sub: dict[str, Any], board: Optional[str] = None) -> None:
        _check_board(board)
        args = _sub_identity(sub)
        _writer_call(
            "notify-unsubscribe", args,
            request_key=_subscription_request_key("notify-unsubscribe", sub, args),
        )

    def _kanban_rewind(
        self,
        sub: dict[str, Any],
        claimed_cursor: int,
        old_cursor: int,
        board: Optional[str] = None,
    ) -> None:
        _check_board(board)
        args = _sub_identity(sub)
        args.update({
            "claimed_cursor": int(claimed_cursor),
            "old_cursor": int(old_cursor),
        })
        _writer_call(
            "notify-rewind", args,
            request_key=_subscription_request_key("notify-rewind", sub, args),
        )

    async def _deliver_kanban_artifacts(
        self,
        *,
        adapter,
        chat_id: str,
        metadata: dict,
        event_payload: Optional[dict],
        task,
    ) -> None:
        """Refuse path-based artifact delivery; attach accepts bytes only."""
        logger.warning(
            "kanban artifact delivery refused: arbitrary artifact paths are outside D0B"
        )
        return

    async def _kanban_dispatcher_watcher(self) -> None:
        """Dispatcher, recovery, and auto-decompose are writer-owned/refused."""
        logger.warning(
            "kanban dispatcher watcher refused: dispatch/recovery/auto-decompose are outside D0B"
        )
        return


__all__ = [
    "GatewayKanbanWatchersMixin",
    "_acquire_singleton_lock",
    "_release_singleton_lock",
    "_resolve_auto_decompose_settings",
]
