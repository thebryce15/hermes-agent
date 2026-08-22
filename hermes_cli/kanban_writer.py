"""Privileged, local-only admission for the canonical Kanban store.

This module is deliberately dormant: route adapters and service installation
are separate work. The writer owns one fixed DB/attachment root and accepts a
small JSON protocol over an AF_UNIX stream. Identity comes from SO_PEERCRED;
policy and UID/profile mapping are callbacks backed by the company manifest.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import dataclasses
import json
import os
import re
import socket
import sqlite3
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from hermes_cli import kanban_db as kb


MAX_FRAME_BYTES = 1 * 1024 * 1024
# Keep the decoded payload below the JSON/base64 frame ceiling.  The single
# exported value is also used by the public adapters; callers must not invent
# a larger HTTP/upload limit and then hand it to this IPC surface.
MAX_ATTACHMENT_BYTES = kb.KANBAN_ATTACHMENT_MAX_BYTES
_MAX_ATTACHMENT_B64_CHARS = ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4
CLIENT_READ_TIMEOUT_SECONDS = 1.0
CANONICAL_DB_PATH = Path(
    "/home/mordecai/.hermes/kanban/boards/bryceos/kanban.db"
)
CANONICAL_SOCKET_PATH = Path("/run/user/1000/bryceos/kanban-writer.sock")
CANONICAL_ATTACHMENTS_ROOT = CANONICAL_DB_PATH.parent / "attachments"
_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_PATH_RE = re.compile(
    r"(?:^~[\\/]|^[\\/]|^[A-Za-z]:[\\/]|^[A-Za-z][A-Za-z0-9+.-]*://|[\\/])"
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

FORBIDDEN_FIELDS = frozenset({
    "author", "created_by", "uploaded_by", "claimer", "session_id",
    "actor", "profile", "board", "db_path", "workspace_path",
    "attachment_path", "callable", "sql", "SQL",
})

READ_OPERATIONS = frozenset({
    "show", "list", "runs", "events", "comments", "attachments",
    "notify-read",
})
WRITE_OPERATIONS = frozenset({
    "create", "link", "unlink", "comment", "block", "unblock", "assign",
    "claim", "heartbeat", "complete", "attach", "notify-subscribe",
    "notify-unsubscribe", "notify-claim", "notify-advance", "notify-rewind",
})
OPERATIONS = READ_OPERATIONS | WRITE_OPERATIONS

# The protocol is closed. Values are validated separately below; this table
# prevents adapters from smuggling an old generic CLI/SQL shape through RPC.
OPERATION_FIELDS: dict[str, frozenset[str]] = {
    "show": frozenset({"task_id"}),
    "list": frozenset({
        "assignee", "status", "tenant", "include_archived", "limit", "order_by",
        "workflow_template_id", "current_step_key",
    }),
    "runs": frozenset({"task_id", "include_active", "state_type", "state_name"}),
    "events": frozenset({"task_id"}),
    "comments": frozenset({"task_id", "after_id"}),
    "attachments": frozenset({"task_id"}),
    "notify-read": frozenset({
        "task_id", "platform", "chat_id", "thread_id", "kinds",
    }),
    "notify-subscribe": frozenset({
        "task_id", "platform", "chat_id", "chat_type", "thread_id",
    }),
    "notify-unsubscribe": frozenset({
        "task_id", "platform", "chat_id", "thread_id",
    }),
    "notify-claim": frozenset({
        "task_id", "platform", "chat_id", "thread_id", "kinds",
    }),
    "notify-advance": frozenset({
        "task_id", "platform", "chat_id", "thread_id", "new_cursor",
    }),
    "notify-rewind": frozenset({
        "task_id", "platform", "chat_id", "thread_id", "claimed_cursor",
        "old_cursor",
    }),
    "create": frozenset({
        "title", "body", "parents", "assignee", "tenant", "priority",
    }),
    "link": frozenset({"parent_id", "child_id"}),
    "unlink": frozenset({"parent_id", "child_id"}),
    "comment": frozenset({"task_id", "body"}),
    "block": frozenset({"task_id", "reason", "kind", "expected_run_id"}),
    "unblock": frozenset({"task_id"}),
    "assign": frozenset({"task_id", "assignee"}),
    "claim": frozenset({"task_id", "ttl_seconds"}),
    "heartbeat": frozenset({"task_id", "ttl_seconds"}),
    "complete": frozenset({
        "task_id", "result", "summary", "metadata", "expected_run_id",
    }),
    "attach": frozenset({"task_id", "filename", "content_type", "data_b64"}),
}


class WriterError(RuntimeError):
    """Base class for protocol/admission failures."""


class WriterProtocolError(WriterError, ValueError):
    pass


class WriterAuthorizationError(WriterError, PermissionError):
    pass


class RequestConflictError(WriterError, ValueError):
    pass


@dataclass(frozen=True)
class Peer:
    pid: int
    uid: int
    gid: int
    profile: str


PeerProfile = Callable[[int], Optional[str]]
PolicyCallback = Callable[[str, str], bool]
AuthorityCallback = Callable[[int, str, Mapping[str, Any]], Mapping[str, Any]]


def _deny_peer(_uid: int) -> Optional[str]:
    return None


def _deny_policy(_profile: str, _operation: str) -> bool:
    return False


def _resolve_fixed_path(value: Path, *, name: str) -> Path:
    """Resolve a writer target while refusing existing symlink components."""
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    current = Path(path.anchor) if path.anchor else Path.cwd()
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise WriterProtocolError(f"{name} may not contain symlinks: {value}")
    return path.resolve()


@dataclass(frozen=True, init=False)
class WriterConfig:
    """Immutable writer-owned targets and explicit authorization callbacks."""

    db_path: Path
    socket_path: Path
    attachments_root: Path
    peer_profile: PeerProfile
    policy: PolicyCallback
    authority: Optional[AuthorityCallback] = None
    max_frame_bytes: int = MAX_FRAME_BYTES
    _fixture: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise WriterProtocolError(
            "construct WriterConfig.production() or WriterConfig.for_fixture()"
        )

    @classmethod
    def _build(
        cls,
        *,
        db_path: Path,
        socket_path: Path,
        attachments_root: Path,
        peer_profile: PeerProfile,
        policy: PolicyCallback,
        authority: Optional[AuthorityCallback],
        max_frame_bytes: int,
        fixture: bool,
    ) -> "WriterConfig":
        config = object.__new__(cls)
        object.__setattr__(config, "db_path", db_path)
        object.__setattr__(config, "socket_path", socket_path)
        object.__setattr__(config, "attachments_root", attachments_root)
        object.__setattr__(config, "peer_profile", peer_profile)
        object.__setattr__(config, "policy", policy)
        object.__setattr__(config, "authority", authority)
        object.__setattr__(config, "max_frame_bytes", max_frame_bytes)
        object.__setattr__(config, "_fixture", fixture)
        config.__post_init__()
        return config

    @classmethod
    def production(
        cls,
        *, max_frame_bytes: int = MAX_FRAME_BYTES,
    ) -> "WriterConfig":
        """Build the dormant production target with frozen paths and deny-all."""
        return cls._build(
            db_path=CANONICAL_DB_PATH,
            socket_path=CANONICAL_SOCKET_PATH,
            attachments_root=CANONICAL_ATTACHMENTS_ROOT,
            peer_profile=_deny_peer,
            policy=_deny_policy,
            authority=None,
            max_frame_bytes=max_frame_bytes,
            fixture=False,
        )

    canonical = production

    @classmethod
    def from_spark_adapter(
        cls,
        adapter: Any,
        *,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ) -> "WriterConfig":
        """Bind the dormant writer to Spark's manifest-backed adapter.

        The Hermes side owns no policy table.  Spark supplies both the UID
        mapping and the per-request authority decision; absent this explicit
        constructor the normal production path remains deny-all.
        """
        peer_profile = getattr(adapter, "peer_profile", None)
        authority = getattr(adapter, "authorize", None)
        bind_operations = getattr(adapter, "bind_protocol_operations", None)
        if (
            not callable(peer_profile)
            or not callable(authority)
            or not callable(bind_operations)
        ):
            raise WriterProtocolError(
                "Spark adapter must expose peer_profile(), authorize(), and "
                "bind_protocol_operations()"
            )
        bind_operations(OPERATIONS)
        return cls._build(
            db_path=CANONICAL_DB_PATH,
            socket_path=CANONICAL_SOCKET_PATH,
            attachments_root=CANONICAL_ATTACHMENTS_ROOT,
            peer_profile=peer_profile,
            policy=_deny_policy,
            authority=authority,
            max_frame_bytes=max_frame_bytes,
            fixture=False,
        )

    @classmethod
    def for_fixture(
        cls,
        *,
        db_path: Path,
        socket_path: Path,
        attachments_root: Path,
        peer_profile: PeerProfile,
        policy: PolicyCallback,
        authority: Optional[AuthorityCallback] = None,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ) -> "WriterConfig":
        """Explicit disposable test/fixture seam; never use for production."""
        return cls._build(
            db_path=db_path,
            socket_path=socket_path,
            attachments_root=attachments_root,
            peer_profile=peer_profile,
            policy=policy,
            authority=authority,
            max_frame_bytes=max_frame_bytes,
            fixture=True,
        )

    def __post_init__(self) -> None:
        if (
            not self._fixture
            and (
                os.name != "posix"
                or not hasattr(socket, "AF_UNIX")
                or not hasattr(socket, "SO_PEERCRED")
            )
        ):
            raise WriterProtocolError("privileged Kanban writer requires AF_UNIX + SO_PEERCRED")
        if not callable(self.peer_profile) or not callable(self.policy):
            raise WriterProtocolError("manifest peer/policy callbacks are required")
        if self.authority is not None and not callable(self.authority):
            raise WriterProtocolError("authority callback must be callable")
        if self.max_frame_bytes < 1024:
            raise WriterProtocolError("max_frame_bytes is too small")
        object.__setattr__(
            self, "db_path", _resolve_fixed_path(self.db_path, name="db_path")
        )
        object.__setattr__(
            self, "socket_path", _resolve_fixed_path(self.socket_path, name="socket_path")
        )
        object.__setattr__(
            self,
            "attachments_root",
            _resolve_fixed_path(self.attachments_root, name="attachments_root"),
        )
        if not self._fixture:
            expected = {
                "db_path": CANONICAL_DB_PATH,
                "socket_path": CANONICAL_SOCKET_PATH,
                "attachments_root": CANONICAL_ATTACHMENTS_ROOT,
            }
            for name, value in expected.items():
                if getattr(self, name) != _resolve_fixed_path(value, name=name):
                    raise WriterProtocolError(f"{name} is not the canonical writer target")


def canonical_request_digest(operation: str, args: Mapping[str, Any]) -> str:
    """Return the digest of the canonical operation and its closed arguments."""
    return kb.request_digest({"operation": operation, "args": dict(args)})


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _path_free_attachment_projection(attachment: Any) -> dict[str, Any]:
    result = _jsonable(attachment)
    if not isinstance(result, dict):
        return {}
    # A stored path is a writer-owned implementation detail, never an IPC
    # selector or a route-adapter input.
    result.pop("stored_path", None)
    return result


def _task_projection(
    conn: sqlite3.Connection,
    task: Any,
    *,
    include_context: bool,
) -> Optional[dict[str, Any]]:
    if task is None:
        return None
    result = _jsonable(task)
    if not isinstance(result, dict):
        return None
    result.pop("workspace_path", None)
    parents = kb.parent_ids(conn, task.id)
    children = kb.child_ids(conn, task.id)
    result["parent_ids"] = parents
    result["child_ids"] = children
    # Keep the old names as truthful aliases for existing route adapters.
    result["parents"] = list(parents)
    result["children"] = list(children)
    # Preserve the existing ``parent_results`` sink shape (JSON tuples become
    # two-item lists); adapters already consume this projection directly.
    result["parent_results"] = _jsonable(kb.parent_results(conn, task.id))
    if include_context:
        result["worker_context"] = kb.build_worker_context(
            conn, task.id, include_paths=False,
        )
    return result


def _walk_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_FIELDS:
                raise WriterProtocolError(f"forbidden field: {key}")
            _walk_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _walk_forbidden(child)


def _validate_metadata(metadata: Any) -> Optional[dict[str, Any]]:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise WriterProtocolError("complete metadata must be a flat object")
    if any(key in {"artifacts", "created_cards"} for key in metadata):
        raise WriterProtocolError("complete metadata cannot contain artifacts or created_cards")
    for key, value in metadata.items():
        if (
            not isinstance(key, str)
            or _PATH_RE.search(key)
            or any(token in key.casefold() for token in ("path", "file", "artifact"))
        ):
            raise WriterProtocolError("complete metadata keys must be path-free")
        if isinstance(value, (dict, list, tuple)):
            raise WriterProtocolError("complete metadata values must be flat scalars")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise WriterProtocolError("complete metadata values must be JSON scalars")
        if isinstance(value, str) and _PATH_RE.search(value):
            raise WriterProtocolError("complete metadata values must be path-free")
    return dict(metadata)


def _validate_request(raw: Mapping[str, Any]) -> tuple[str, dict[str, Any], Optional[str], Optional[str]]:
    if not isinstance(raw, dict):
        raise WriterProtocolError("request must be a JSON object")
    allowed = {"operation", "args", "request_key", "request_digest"}
    unknown = set(raw) - allowed
    if unknown:
        raise WriterProtocolError(f"unknown request field(s): {', '.join(sorted(unknown))}")
    operation = raw.get("operation")
    args = raw.get("args", {})
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise WriterProtocolError("unknown writer operation")
    if not isinstance(args, dict):
        raise WriterProtocolError("request args must be an object")
    unknown_args = set(args) - OPERATION_FIELDS[operation]
    if unknown_args:
        raise WriterProtocolError(
            f"unknown {operation} field(s): {', '.join(sorted(unknown_args))}"
        )
    _walk_forbidden(args)
    if operation == "attach":
        encoded = args.get("data_b64")
        if not isinstance(encoded, str):
            raise WriterProtocolError("attach requires inline data_b64")
        if len(encoded) > _MAX_ATTACHMENT_B64_CHARS:
            raise WriterProtocolError(
                f"attach data exceeds {MAX_ATTACHMENT_BYTES} decoded bytes"
            )
    key = raw.get("request_key")
    digest = raw.get("request_digest")
    if key is not None and (not isinstance(key, str) or not _KEY_RE.fullmatch(key)):
        raise WriterProtocolError("request_key must be a 1-128 character safe identifier")
    if digest is not None and (not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest)):
        raise WriterProtocolError("request_digest must be lowercase SHA-256 hex")
    if operation in WRITE_OPERATIONS and (key is None or digest is None):
        raise WriterProtocolError("mutations require request_key and request_digest")
    if (key is None) != (digest is None):
        raise WriterProtocolError("request_key and request_digest must be supplied together")
    if digest is not None and digest != canonical_request_digest(operation, args):
        raise WriterProtocolError("request_digest does not match canonical request")
    return operation, dict(args), key, digest


def _peer_credentials(conn: socket.socket) -> tuple[int, int, int]:
    if not hasattr(socket, "SO_PEERCRED"):
        raise WriterAuthorizationError("SO_PEERCRED is unavailable")
    try:
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return tuple(int(v) for v in struct.unpack("3i", raw))  # pid, uid, gid
    except (OSError, struct.error) as exc:
        raise WriterAuthorizationError("could not derive peer credentials") from exc


def _safe_int(value: Any, *, name: str, minimum: int = 0, maximum: int = 1000) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WriterProtocolError(f"{name} must be an integer")
    parsed = value
    if not minimum <= parsed <= maximum:
        raise WriterProtocolError(f"{name} is outside its bounded range")
    return parsed


class KanbanWriter:
    """One fixed-store writer and closed-operation dispatcher."""

    def __init__(self, config: WriterConfig):
        self.config = config

    def peer_from_credentials(self, credentials: tuple[int, int, int]) -> Peer:
        pid, uid, gid = credentials
        profile = self.config.peer_profile(uid)
        if not profile or not isinstance(profile, str):
            raise WriterAuthorizationError(f"peer uid {uid} is not admitted")
        return Peer(pid=pid, uid=uid, gid=gid, profile=profile)

    def dispatch(self, raw: Mapping[str, Any], peer: Peer) -> dict[str, Any]:
        operation, args, request_key, request_digest = _validate_request(raw)
        self._authorize(peer, operation, args)
        try:
            if self.config._fixture:
                conn = kb.connect(db_path=self.config.db_path)
            else:
                identity = kb.existing_db_identity(self.config.db_path)
                conn = kb.connect(
                    db_path=self.config.db_path,
                    require_existing=True,
                    expected_identity=identity,
                )
        except kb.ExistingKanbanDbError as exc:
            raise WriterProtocolError(str(exc)) from exc
        try:
            if request_key is not None:
                marker = kb.request_binding(conn, request_key)
                if marker is not None:
                    if marker["request_digest"] != request_digest:
                        raise RequestConflictError(
                            f"request key {request_key!r} was already used with different inputs"
                        )
                    if operation == "attach" and marker.get("kind") in {"attachment_pending", "attached"}:
                        result = self._dispatch_operation(
                            conn, operation, args, peer,
                            request_key=request_key, request_digest=request_digest,
                        )
                        return {"replayed": True, "result": _jsonable(result), "binding": marker}
                    return self._replay(marker)
            try:
                result = self._dispatch_operation(
                    conn, operation, args, peer,
                    request_key=request_key, request_digest=request_digest,
                )
            except sqlite3.IntegrityError:
                # A concurrent writer can win between the marker lookup and
                # the operation's transaction. The unique event marker makes
                # that race converge without a retrying side effect.
                if request_key is None:
                    raise
                marker = kb.request_binding(conn, request_key)
                if marker is None:
                    raise
                if marker["request_digest"] != request_digest:
                    raise RequestConflictError(
                        f"request key {request_key!r} was already used with different inputs"
                    )
                if operation == "attach" and marker.get("kind") in {"attachment_pending", "attached"}:
                    result = self._dispatch_operation(
                        conn, operation, args, peer,
                        request_key=request_key, request_digest=request_digest,
                    )
                    return {"replayed": True, "result": _jsonable(result), "binding": marker}
                return self._replay(marker)
            return {"replayed": False, "result": _jsonable(result)}
        finally:
            conn.close()

    def _authorize(
        self, peer: Peer, operation: str, args: Mapping[str, Any],
    ) -> None:
        """Authorize from Spark when explicitly bound, otherwise use deny-all/policy."""
        if self.config.authority is None:
            if not self.config.policy(peer.profile, operation):
                raise WriterAuthorizationError(
                    f"manifest policy refused profile={peer.profile!r} operation={operation!r}"
                )
            return
        try:
            decision = self.config.authority(peer.uid, operation, dict(args))
        except Exception as exc:  # authority callback is a trust boundary
            raise WriterAuthorizationError("Spark writer authority refused request") from exc
        if not isinstance(decision, Mapping):
            raise WriterAuthorizationError("Spark writer authority returned no decision")
        if (
            decision.get("authorized") is not True
            or decision.get("authorized_for_fixture") is not True
            or decision.get("live_production_enabled") is not False
            or decision.get("profile") != peer.profile
            or decision.get("operation") != operation
            or decision.get("canonical_board") != str(CANONICAL_DB_PATH)
        ):
            raise WriterAuthorizationError("Spark writer authority refused request")

    @staticmethod
    def _replay(marker: Mapping[str, Any]) -> dict[str, Any]:
        """Return the original operation projection plus audit binding."""
        return {
            "replayed": True,
            "result": marker.get("result") or {},
            "binding": dict(marker),
        }

    def _marker(self, key: Optional[str], digest: Optional[str], result: Optional[dict] = None) -> dict[str, Any]:
        return {
            "request_key": key,
            "request_digest": digest,
            "request_result": result or {},
        }

    def _dispatch_operation(
        self,
        conn: sqlite3.Connection,
        operation: str,
        args: dict[str, Any],
        peer: Peer,
        *,
        request_key: Optional[str],
        request_digest: Optional[str],
    ) -> Any:
        marker = self._marker(request_key, request_digest)
        if operation == "show":
            return _task_projection(
                conn, kb.get_task(conn, self._task_id(args)), include_context=True,
            )
        if operation == "list":
            limit = _safe_int(args.get("limit"), name="limit", maximum=1000)
            filters = {
                key: self._text(args, key)
                for key in (
                    "assignee", "status", "tenant", "order_by",
                    "workflow_template_id", "current_step_key",
                ) if key in args
            }
            tasks = kb.list_tasks(conn, limit=limit, **{
                **filters,
                "include_archived": self._bool(args, "include_archived", False),
            })
            return [
                _task_projection(conn, task, include_context=False)
                for task in tasks
            ]
        if operation == "runs":
            return kb.list_runs(
                conn, self._task_id(args),
                include_active=self._bool(args, "include_active", True),
                state_type=self._text(args, "state_type"),
                state_name=self._text(args, "state_name"),
            )
        if operation == "events":
            return kb.list_events(conn, self._task_id(args))
        if operation == "comments":
            task_id = self._task_id(args)
            if "after_id" in args:
                after_id = _safe_int(
                    args["after_id"], name="after_id", maximum=2**63 - 1,
                ) or 0
                return kb.list_comments_after(conn, task_id, after_id=after_id)
            return kb.list_comments(conn, task_id)
        if operation == "attachments":
            return [
                _path_free_attachment_projection(attachment)
                for attachment in kb.list_attachments(conn, self._task_id(args))
            ]
        if operation == "notify-read":
            return kb.unseen_events_for_sub(
                conn, task_id=self._task_id(args),
                platform=self._text(args, "platform") or "",
                chat_id=self._text(args, "chat_id") or "",
                thread_id=self._text(args, "thread_id"),
                kinds=self._strings(args.get("kinds"), name="kinds"),
            )
        if operation in {
            "notify-subscribe", "notify-unsubscribe", "notify-claim",
            "notify-advance", "notify-rewind", "comment", "block",
            "unblock", "assign", "claim", "heartbeat", "complete", "attach",
        }:
            self._require_task(conn, self._task_id(args))
        if operation in {
            "notify-subscribe", "notify-unsubscribe", "notify-claim",
            "notify-advance", "notify-rewind",
        }:
            task_id = self._task_id(args)
            platform = self._text(args, "platform", required=True) or ""
            chat_id = self._text(args, "chat_id", required=True) or ""
            thread_id = self._text(args, "thread_id")
            if operation == "notify-subscribe":
                kb.add_notify_sub(
                    conn,
                    task_id=task_id,
                    platform=platform,
                    chat_id=chat_id,
                    chat_type=self._text(args, "chat_type"),
                    thread_id=thread_id,
                    notifier_profile=peer.profile,
                    **marker,
                )
                return {"ok": True}
            if operation == "notify-unsubscribe":
                return {"ok": kb.remove_notify_sub(
                    conn,
                    task_id=task_id,
                    platform=platform,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    **marker,
                )}
            if operation == "notify-claim":
                old_cursor, new_cursor, events = kb.claim_unseen_events_for_sub(
                    conn,
                    task_id=task_id,
                    platform=platform,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    kinds=self._strings(args.get("kinds"), name="kinds"),
                    **marker,
                )
                return {
                    "old_cursor": old_cursor,
                    "new_cursor": new_cursor,
                    "events": events,
                }
            if operation == "notify-advance":
                cursor = _safe_int(
                    args.get("new_cursor"), name="new_cursor", maximum=2**63 - 1,
                )
                if cursor is None:
                    raise WriterProtocolError("new_cursor is required")
                kb.advance_notify_cursor(
                    conn,
                    task_id=task_id,
                    platform=platform,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    new_cursor=cursor,
                    **marker,
                )
                return {"ok": True, "cursor": cursor}
            claimed_cursor = _safe_int(
                args.get("claimed_cursor"),
                name="claimed_cursor", maximum=2**63 - 1,
            )
            old_cursor = _safe_int(
                args.get("old_cursor"), name="old_cursor", maximum=2**63 - 1,
            )
            if claimed_cursor is None or old_cursor is None:
                raise WriterProtocolError("notify-rewind cursors are required")
            return {"ok": kb.rewind_notify_cursor(
                conn,
                task_id=task_id,
                platform=platform,
                chat_id=chat_id,
                thread_id=thread_id,
                claimed_cursor=claimed_cursor,
                old_cursor=old_cursor,
                **marker,
            )}

        if operation == "create":
            return {"task_id": kb.create_task(
                conn, title=self._text(args, "title", required=True), body=self._text(args, "body"),
                assignee=self._text(args, "assignee"), tenant=self._text(args, "tenant"),
                created_by=peer.profile,
                priority=_safe_int(args.get("priority"), name="priority", minimum=-100, maximum=100) or 0,
                parents=self._ids(args.get("parents", []), name="parents"),
                idempotency_key=request_key, **marker,
            )}
        if operation == "link":
            self._require_task(conn, self._id(args, "parent_id"))
            self._require_task(conn, self._id(args, "child_id"))
            kb.link_tasks(
                conn, self._id(args, "parent_id"), self._id(args, "child_id"), **marker,
            )
            return {"ok": True}
        if operation == "unlink":
            self._require_task(conn, self._id(args, "parent_id"))
            self._require_task(conn, self._id(args, "child_id"))
            return {"ok": kb.unlink_tasks(
                conn, self._id(args, "parent_id"), self._id(args, "child_id"), **marker,
            )}
        if operation == "comment":
            comment_id = kb.add_comment(
                conn, self._task_id(args), peer.profile,
                self._text(args, "body", required=True), **marker,
            )
            return {"comment_id": comment_id}
        if operation == "block":
            blocked = kb.block_task(
                conn, self._task_id(args), reason=self._text(args, "reason"),
                kind=self._text(args, "kind"),
                expected_run_id=_safe_int(args.get("expected_run_id"), name="expected_run_id", maximum=2**63 - 1),
                **marker,
            )
            return {"ok": blocked}
        if operation == "unblock":
            return {"ok": kb.unblock_task(conn, self._task_id(args), **marker)}
        if operation == "assign":
            assignee = self._text(args, "assignee")
            if not assignee:
                raise WriterProtocolError("assign requires a target assignee")
            return {"ok": kb.assign_task(conn, self._task_id(args), assignee, **marker)}
        if operation in {"claim", "heartbeat"}:
            task_id = self._task_id(args)
            # The lock is derived from kernel credentials, never from a request
            # field. A reconnect gets a new peer pid and therefore cannot steal
            # a prior claim merely by replaying its profile name.
            lock = f"peer:{peer.uid}:{peer.pid}"
            ttl = _safe_int(args.get("ttl_seconds"), name="ttl_seconds", minimum=1, maximum=24 * 3600)
            if operation == "claim":
                claimed = kb.claim_task(
                    conn, task_id, ttl_seconds=ttl, claimer=lock, **marker,
                )
                return {
                    "task_id": task_id,
                    "run_id": claimed.current_run_id if claimed else None,
                } if claimed else {"ok": False}
            return {"ok": kb.heartbeat_claim(
                conn, task_id, ttl_seconds=ttl, claimer=lock, **marker,
            )}
        if operation == "complete":
            result = self._path_free_text(args.get("result"), name="result")
            summary = self._path_free_text(args.get("summary"), name="summary")
            metadata = _validate_metadata(args.get("metadata"))
            completed = kb.complete_task(
                conn, self._task_id(args), result=result, summary=summary, metadata=metadata,
                expected_run_id=_safe_int(args.get("expected_run_id"), name="expected_run_id", maximum=2**63 - 1),
                **marker,
            )
            return {"ok": completed}
        if operation == "attach":
            filename = self._text(args, "filename", required=True)
            if "/" in filename or "\\" in filename:
                raise WriterProtocolError("attach filename must be a basename")
            encoded = args.get("data_b64")
            if not isinstance(encoded, str):
                raise WriterProtocolError("attach requires inline data_b64")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise WriterProtocolError("attach data_b64 is invalid") from exc
            attachment_id = kb.store_attachment_bytes_fixed(
                conn, self._task_id(args), filename, data,
                attachments_root=self.config.attachments_root,
                content_type=self._text(args, "content_type"), uploaded_by=peer.profile,
                request_key=request_key, request_digest=request_digest,
                request_result=marker["request_result"],
            )
            return {"attachment_id": attachment_id}
        raise WriterProtocolError(f"unsupported operation: {operation}")

    @staticmethod
    def _id(args: Mapping[str, Any], name: str) -> str:
        value = args.get(name)
        if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
            raise WriterProtocolError(f"{name} must be a bounded string")
        return value

    @staticmethod
    def _require_task(conn: sqlite3.Connection, task_id: str) -> None:
        if kb.get_task(conn, task_id) is None:
            raise WriterProtocolError(f"unknown task: {task_id}")

    @classmethod
    def _task_id(cls, args: Mapping[str, Any]) -> str:
        return cls._id(args, "task_id")

    @staticmethod
    def _ids(value: Any, *, name: str) -> list[str]:
        if not isinstance(value, list) or len(value) > 100:
            raise WriterProtocolError(f"{name} must be a bounded list")
        return [KanbanWriter._id({name: item}, name=name) for item in value]

    @staticmethod
    def _strings(value: Any, *, name: str) -> Optional[list[str]]:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) > 100:
            raise WriterProtocolError(f"{name} must be a bounded string list")
        out = []
        for item in value:
            if not isinstance(item, str) or len(item) > 128:
                raise WriterProtocolError(f"{name} must contain bounded strings")
            out.append(item)
        return out

    @staticmethod
    def _bool(args: Mapping[str, Any], name: str, default: bool) -> bool:
        value = args.get(name, default)
        if not isinstance(value, bool):
            raise WriterProtocolError(f"{name} must be a boolean")
        return value

    @staticmethod
    def _text(args: Mapping[str, Any], name: str, *, required: bool = False) -> Optional[str]:
        value = args.get(name)
        if value is None:
            if required:
                raise WriterProtocolError(f"{name} is required")
            return None
        if not isinstance(value, str) or len(value) > 100_000:
            raise WriterProtocolError(f"{name} must be bounded text")
        return value

    @staticmethod
    def _path_free_text(value: Any, *, name: str) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or len(value) > 100_000:
            raise WriterProtocolError(f"{name} must be bounded text")
        if _PATH_RE.search(value):
            raise WriterProtocolError(f"{name} must be path-free")
        return value


class KanbanWriterServer:
    """Small one-request-per-connection AF_UNIX server."""

    def __init__(self, writer: KanbanWriter):
        self.writer = writer
        self._socket: Optional[socket.socket] = None
        self._socket_inode: Optional[int] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if (
            os.name != "posix"
            or not hasattr(socket, "AF_UNIX")
            or not hasattr(socket, "SO_PEERCRED")
        ):
            raise WriterProtocolError("privileged Kanban writer requires AF_UNIX + SO_PEERCRED")
        path = self.writer.config.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise WriterProtocolError(f"refusing symlink socket path: {path}")
        if path.exists():
            raise WriterProtocolError(
                f"writer socket already exists; refusing replacement: {path}"
            )
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(path))
            os.chmod(path, 0o660)
            sock.listen(32)
            self._socket_inode = path.lstat().st_ino
        except Exception:
            sock.close()
            with contextlib.suppress(OSError):
                path.unlink()
            raise
        self._socket = sock

    def serve_forever(self) -> None:
        if self._socket is None:
            self.start()
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                self._socket.settimeout(0.5)
                conn, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            with conn:
                conn.settimeout(CLIENT_READ_TIMEOUT_SECONDS)
                self._serve_connection(conn)

    def _serve_connection(self, conn: socket.socket) -> None:
        try:
            credentials = _peer_credentials(conn)
            peer = self.writer.peer_from_credentials(credentials)
            raw = self._read_json(conn)
            result = self.writer.dispatch(raw, peer)
            response = {"ok": True, **result}
        except Exception as exc:  # protocol boundary: never leak a traceback
            response = {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        try:
            conn.sendall(
                (json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n").encode()
            )
        except OSError:
            # A disconnected client must not terminate the single writer loop.
            return

    def _read_json(self, conn: socket.socket) -> Mapping[str, Any]:
        chunks = bytearray()
        while len(chunks) <= self.writer.config.max_frame_bytes:
            part = conn.recv(min(65536, self.writer.config.max_frame_bytes + 1 - len(chunks)))
            if not part:
                break
            chunks.extend(part)
            if b"\n" in part:
                break
        if not chunks or len(chunks) > self.writer.config.max_frame_bytes:
            raise WriterProtocolError("empty or oversized writer frame")
        line = bytes(chunks).split(b"\n", 1)[0]
        try:
            parsed = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WriterProtocolError("writer frame is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise WriterProtocolError("writer frame must be a JSON object")
        return parsed

    def stop(self) -> None:
        self._stop.set()
        sock = self._socket
        self._socket = None
        if sock is None:
            return
        inode = self._socket_inode
        self._socket_inode = None
        sock.close()
        if inode is None:
            return
        try:
            if self.writer.config.socket_path.lstat().st_ino != inode:
                return
            self.writer.config.socket_path.unlink()
        except OSError:
            pass


def _writer_request(
    socket_path: Path,
    operation: str,
    args: Optional[Mapping[str, Any]] = None,
    *,
    request_key: Optional[str] = None,
    fixture: bool = False,
) -> dict[str, Any]:
    """Send one closed request to a writer; no fallback transport exists."""
    body = dict(args or {})
    frame: dict[str, Any] = {"operation": operation, "args": body}
    if request_key is not None:
        frame["request_key"] = request_key
        frame["request_digest"] = canonical_request_digest(operation, body)
    # Refuse malformed or non-idempotent mutations before opening the socket.
    _validate_request(frame)
    if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
        raise WriterProtocolError("privileged Kanban writer requires AF_UNIX")
    socket_path = _resolve_fixed_path(socket_path, name="socket_path")
    if not fixture and socket_path != _resolve_fixed_path(
        CANONICAL_SOCKET_PATH, name="socket_path"
    ):
        raise WriterProtocolError("socket_path is not the canonical writer target")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(CLIENT_READ_TIMEOUT_SECONDS)
        conn.connect(str(Path(socket_path).resolve()))
        conn.sendall((json.dumps(frame, ensure_ascii=False, sort_keys=True) + "\n").encode())
        chunks = bytearray()
        while len(chunks) <= MAX_FRAME_BYTES:
            part = conn.recv(min(65536, MAX_FRAME_BYTES + 1 - len(chunks)))
            if not part:
                break
            chunks.extend(part)
            if b"\n" in part:
                break
    if not chunks:
        raise WriterError("writer closed connection without a response")
    response = json.loads(bytes(chunks).split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(response, dict):
        raise WriterError("writer response is malformed")
    if not response.get("ok"):
        error = response.get("error") or {}
        raise WriterError(str(error.get("message") or "writer request failed"))
    return response


def writer_request(
    socket_path: Optional[Path] = None,
    operation: Optional[str] = None,
    args: Optional[Mapping[str, Any]] = None,
    *,
    request_key: Optional[str] = None,
) -> dict[str, Any]:
    """Send one request to the frozen canonical writer socket."""
    if operation is None:
        raise WriterProtocolError("writer operation is required")
    return _writer_request(
        CANONICAL_SOCKET_PATH if socket_path is None else socket_path,
        operation, args,
        request_key=request_key,
    )


def writer_request_for_fixture(
    socket_path: Path,
    operation: str,
    args: Optional[Mapping[str, Any]] = None,
    *,
    request_key: Optional[str] = None,
) -> dict[str, Any]:
    """Explicit disposable socket seam for tests/fixtures only."""
    return _writer_request(
        socket_path, operation, args,
        request_key=request_key, fixture=True,
    )


__all__ = [
    "FORBIDDEN_FIELDS", "OPERATIONS", "OPERATION_FIELDS", "Peer", "WriterConfig",
    "CANONICAL_DB_PATH", "CANONICAL_SOCKET_PATH", "CANONICAL_ATTACHMENTS_ROOT",
    "MAX_ATTACHMENT_BYTES",
    "KanbanWriter", "KanbanWriterServer", "WriterError", "WriterProtocolError",
    "WriterAuthorizationError", "RequestConflictError", "canonical_request_digest",
    "writer_request", "writer_request_for_fixture",
]
