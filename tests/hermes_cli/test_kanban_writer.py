"""Behavior tests for the dormant privileged Kanban writer."""

from __future__ import annotations

import base64
import os
import socket
import sqlite3
import struct
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_writer as writer_module
from hermes_cli.kanban_writer import (
    CANONICAL_ATTACHMENTS_ROOT,
    CANONICAL_DB_PATH,
    CANONICAL_SOCKET_PATH,
    FORBIDDEN_FIELDS,
    OPERATIONS,
    OPERATION_FIELDS,
    KanbanWriter,
    KanbanWriterServer,
    MAX_ATTACHMENT_BYTES,
    Peer,
    RequestConflictError,
    WriterAuthorizationError,
    WriterConfig,
    WriterProtocolError,
    canonical_request_digest,
    writer_request,
    writer_request_for_fixture,
)


def _config(tmp_path: Path, *, policy=None, peer_profile=None) -> WriterConfig:
    return WriterConfig.for_fixture(
        db_path=tmp_path / "kanban.db",
        socket_path=tmp_path / "writer.sock",
        attachments_root=tmp_path / "attachments",
        peer_profile=peer_profile or (lambda uid: "worker" if uid == 1000 else None),
        policy=policy or (lambda _profile, _operation: True),
    )


def _peer() -> Peer:
    return Peer(pid=321, uid=1000, gid=1000, profile="worker")


def _production_like_writer(tmp_path: Path) -> KanbanWriter:
    """Exercise the production open path without requiring POSIX in CI."""
    config = _config(tmp_path)
    object.__setattr__(config, "_fixture", False)
    return KanbanWriter(config)


def _frame(operation: str, args: dict, request_key: str) -> dict:
    return {
        "operation": operation,
        "args": args,
        "request_key": request_key,
        "request_digest": canonical_request_digest(operation, args),
    }


def test_public_writer_request_forwards_idempotency_key_unchanged(monkeypatch):
    supplied_key = "adapter:create:01JEXACTKEY"
    supplied_args = {"title": "boundary"}
    captured = {}

    def capture(socket_path, operation, args, *, request_key=None, fixture=False):
        captured.update({
            "socket_path": socket_path,
            "operation": operation,
            "args": args,
            "request_key": request_key,
            "fixture": fixture,
        })
        return {"ok": True}

    monkeypatch.setattr(writer_module, "_writer_request", capture)
    assert writer_request(
        operation="create", args=supplied_args, request_key=supplied_key,
    ) == {"ok": True}
    assert captured == {
        "socket_path": CANONICAL_SOCKET_PATH,
        "operation": "create",
        "args": supplied_args,
        "request_key": supplied_key,
        "fixture": False,
    }


def test_public_writer_request_refuses_mutation_without_idempotency_key(monkeypatch):
    monkeypatch.setattr(
        writer_module.socket, "socket",
        lambda *_args, **_kwargs: pytest.fail("socket must not be opened"),
    )
    with pytest.raises(WriterProtocolError, match="mutations require request_key"):
        writer_request(operation="create", args={"title": "missing key"})


def test_fixture_authority_receives_top_level_request_key_after_frame_validation(tmp_path):
    calls = []

    def authorize(uid, operation, request, request_key, peer_pid):
        calls.append((uid, operation, request, request_key, peer_pid))
        return {
            "authorized": True,
            "authorized_for_fixture": True,
            "live_production_enabled": False,
            "profile": "worker",
            "operation": operation,
            "canonical_board": str(CANONICAL_DB_PATH),
        }

    config = WriterConfig.for_fixture(
        db_path=tmp_path / "kanban.db",
        socket_path=tmp_path / "writer.sock",
        attachments_root=tmp_path / "attachments",
        peer_profile=lambda uid: "worker" if uid == 1000 else None,
        policy=lambda _profile, _operation: True,
        authority=authorize,
    )
    writer = KanbanWriter(config)
    raw = _frame("create", {"title": "exact key"}, "authority-key")

    writer.dispatch(raw, _peer())
    assert calls == [(1000, "create", {"title": "exact key"}, "authority-key", 321)]

    with pytest.raises(WriterProtocolError, match="mutations require request_key"):
        writer.dispatch({"operation": "create", "args": {"title": "missing"}}, _peer())
    with pytest.raises(WriterProtocolError, match="request_key must be"):
        writer.dispatch(
            {"operation": "create", "args": {"title": "raw"}, "request_key": "raw key"},
            _peer(),
        )
    assert len(calls) == 1


def test_bound_authority_without_peer_pid_parameter_fails_closed_before_db(tmp_path):
    def legacy_authorize(_uid, _operation, _request, _request_key):
        return {
            "authorized": True,
            "authorized_for_fixture": True,
            "live_production_enabled": False,
            "profile": "worker",
            "operation": "show",
            "canonical_board": str(CANONICAL_DB_PATH),
        }

    writer = KanbanWriter(
        WriterConfig.for_fixture(
            db_path=tmp_path / "kanban.db",
            socket_path=tmp_path / "writer.sock",
            attachments_root=tmp_path / "attachments",
            peer_profile=lambda uid: "worker" if uid == 1000 else None,
            policy=lambda _profile, _operation: True,
            authority=legacy_authorize,
        )
    )
    with pytest.raises(WriterAuthorizationError, match="Spark writer authority refused"):
        writer.dispatch(_frame("create", {"title": "must refuse"}, "authority-key"), _peer())
    assert not (tmp_path / "kanban.db").exists()


def test_bound_authority_with_malformed_decision_fails_closed_before_db(tmp_path):
    def malformed_authorize(_uid, _operation, _request, _request_key, _peer_pid):
        return {"authorized": True}

    writer = KanbanWriter(
        WriterConfig.for_fixture(
            db_path=tmp_path / "kanban.db",
            socket_path=tmp_path / "writer.sock",
            attachments_root=tmp_path / "attachments",
            peer_profile=lambda uid: "worker" if uid == 1000 else None,
            policy=lambda _profile, _operation: True,
            authority=malformed_authorize,
        )
    )
    with pytest.raises(WriterAuthorizationError, match="Spark writer authority refused"):
        writer.dispatch(_frame("create", {"title": "must refuse"}, "authority-key"), _peer())
    assert not (tmp_path / "kanban.db").exists()


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_production_config_freezes_canonical_targets_and_fixture_is_explicit(tmp_path):
    with pytest.raises(WriterProtocolError, match=r"production\(\)|for_fixture"):
        WriterConfig(
            db_path=tmp_path / "other.db",
            socket_path=tmp_path / "other.sock",
            attachments_root=tmp_path / "other-attachments",
            peer_profile=lambda _uid: "worker",
            policy=lambda _profile, _operation: True,
        )

    config = WriterConfig.production()
    assert config.db_path == CANONICAL_DB_PATH
    assert config.socket_path == CANONICAL_SOCKET_PATH
    assert config.attachments_root == CANONICAL_ATTACHMENTS_ROOT
    assert config.peer_profile(1000) is None
    assert config.policy("worker", "create") is False
    assert _config(tmp_path).db_path == (tmp_path / "kanban.db").resolve()


def test_request_markers_migrate_and_bind_to_runs(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path=db_path) as conn:
        digest = kb.request_digest({"operation": "create", "args": {"title": "x"}})
        task_id = kb.create_task(
            conn,
            title="x",
            request_key="create-1",
            request_digest=digest,
        )

        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_events)")
        }
        assert {"request_key", "request_digest", "request_result"} <= columns
        indexes = {
            row["name"]
            for row in conn.execute("PRAGMA index_list(task_events)")
        }
        assert "idx_events_request_key" in indexes

        binding = kb.request_binding(conn, "create-1")
        assert binding is not None
        assert binding["task_id"] == task_id
        assert binding["request_digest"] == digest
        assert binding["result"] == {"task_id": task_id}

        claim_digest = kb.request_digest(
            {"operation": "claim", "args": {"task_id": task_id}}
        )
        claimed = kb.claim_task(
            conn,
            task_id,
            claimer="peer:1000:321",
            request_key="claim-1",
            request_digest=claim_digest,
        )
        assert claimed is not None
        claim_binding = kb.request_binding(conn, "claim-1")
        assert claim_binding is not None
        assert claim_binding["run_id"] is not None
        assert conn.execute(
            "SELECT task_id FROM task_runs WHERE id = ?",
            (claim_binding["run_id"],),
        ).fetchone()["task_id"] == task_id


def test_mutation_noops_still_bind_request_markers(tmp_path):
    with kb.connect(db_path=tmp_path / "kanban.db") as conn:
        task_id = kb.create_task(conn, title="done", initial_status="running")
        sibling_id = kb.create_task(conn, title="sibling", initial_status="running")
        assert kb.complete_task(conn, task_id)

        def request(operation, key, args, call):
            digest = kb.request_digest({"operation": operation, "args": args})
            assert call(request_key=key, request_digest=digest) is not True
            binding = kb.request_binding(conn, key)
            assert binding is not None
            assert binding["request_digest"] == digest

        request(
            "claim", "noop-claim", {"task_id": task_id},
            lambda **marker: kb.claim_task(conn, task_id, **marker),
        )
        request(
            "heartbeat", "noop-heartbeat", {"task_id": task_id},
            lambda **marker: kb.heartbeat_claim(conn, task_id, claimer="peer", **marker),
        )
        request(
            "complete", "noop-complete", {"task_id": task_id},
            lambda **marker: kb.complete_task(conn, task_id, **marker),
        )
        request(
            "block", "noop-block", {"task_id": task_id},
            lambda **marker: kb.block_task(conn, task_id, **marker),
        )
        request(
            "unblock", "noop-unblock", {"task_id": task_id},
            lambda **marker: kb.unblock_task(conn, task_id, **marker),
        )
        request(
            "unlink", "noop-unlink", {"parent_id": task_id, "child_id": sibling_id},
            lambda **marker: kb.unlink_tasks(conn, task_id, sibling_id, **marker),
        )


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_replay_and_digest_conflict_are_single_projection(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    args = {"title": "one"}
    first = writer.dispatch(_frame("create", args, "req-1"), _peer())
    second = writer.dispatch(_frame("create", args, "req-1"), _peer())

    assert first["replayed"] is False
    task_id = first["result"]["task_id"]
    assert second["replayed"] is True
    assert second["result"] == {"task_id": task_id}
    assert second["binding"]["task_id"] == task_id

    with pytest.raises(RequestConflictError):
        writer.dispatch(_frame("create", {"title": "two"}, "req-1"), _peer())

    with kb.connect(db_path=tmp_path / "kanban.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE request_key = 'req-1'"
        ).fetchone()[0] == 1


def test_create_records_authenticated_peer_profile_as_creator(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    created = writer.dispatch(
        _frame("create", {"title": "owned"}, "create-owned"), _peer(),
    )

    with kb.connect(db_path=tmp_path / "kanban.db") as conn:
        task = kb.get_task(conn, created["result"]["task_id"])
    assert task is not None
    assert task.created_by == "worker"

    forged = {"title": "forged", "created_by": "company-director"}
    with pytest.raises(WriterProtocolError, match="created_by"):
        writer.dispatch(_frame("create", forged, "create-forged"), _peer())


@pytest.mark.parametrize("kind", ["missing", "zero"])
def test_production_dispatch_refuses_missing_or_zero_byte_db_before_connect(
    tmp_path, monkeypatch, kind,
):
    writer = _production_like_writer(tmp_path)
    if kind == "zero":
        writer.config.db_path.touch()
    called = False
    real_connect = kb.connect

    def observed_connect(*args, **kwargs):
        nonlocal called
        called = True
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb, "connect", observed_connect)
    with pytest.raises(WriterProtocolError, match="existing non-empty regular file"):
        writer.dispatch(_frame("list", {}, f"list-{kind}"), _peer())
    assert called is False
    assert writer.config.db_path.exists() is (kind == "zero")


def test_production_dispatch_refuses_symlink_db_before_connect(tmp_path, monkeypatch):
    writer = _production_like_writer(tmp_path)
    target = tmp_path / "target.db"
    sqlite3.connect(target).close()
    try:
        writer.config.db_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    called = False

    def observed_connect(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("connect must not run")

    monkeypatch.setattr(kb, "connect", observed_connect)
    with pytest.raises(WriterProtocolError, match="symlink"):
        writer.dispatch(_frame("list", {}, "list-symlink"), _peer())
    assert called is False


def test_production_connect_refuses_target_replaced_after_precheck(tmp_path, monkeypatch):
    writer = _production_like_writer(tmp_path)
    conn = kb.connect(db_path=writer.config.db_path)
    conn.close()
    original_identity = kb.existing_db_identity(writer.config.db_path)
    replacement = tmp_path / "replacement.db"
    replacement_conn = sqlite3.connect(replacement)
    replacement_conn.execute("CREATE TABLE replacement (id INTEGER)")
    replacement_conn.close()
    real_connect = kb.connect

    def replace_then_connect(*args, **kwargs):
        writer.config.db_path.unlink()
        replacement.replace(writer.config.db_path)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb, "connect", replace_then_connect)
    with pytest.raises(WriterProtocolError, match="changed after validation"):
        writer.dispatch(_frame("list", {}, "list-replaced"), _peer())
    assert kb.existing_db_identity(writer.config.db_path) != original_identity


def test_production_dispatch_never_initializes_an_existing_uninitialized_db(tmp_path):
    writer = _production_like_writer(tmp_path)
    conn = sqlite3.connect(writer.config.db_path)
    conn.execute("CREATE TABLE sentinel (id INTEGER)")
    conn.close()

    with pytest.raises(WriterProtocolError, match="not initialized"):
        writer.dispatch(_frame("list", {}, "list-uninitialized"), _peer())

    inspection = sqlite3.connect(writer.config.db_path)
    try:
        tables = {
            row[0] for row in inspection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        inspection.close()
    assert "sentinel" in tables
    assert "tasks" not in tables
    assert "task_events" not in tables


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_create_replay_refuses_legacy_idempotency_row_without_marker(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path=db_path) as conn:
        task_id = kb.create_task(conn, title="legacy", idempotency_key="legacy-key")

    writer = KanbanWriter(_config(tmp_path))
    with pytest.raises(ValueError, match="matching atomic request marker"):
        writer.dispatch(_frame("create", {"title": "retry"}, "legacy-key"), _peer())

    with kb.connect(db_path=db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert kb.request_binding(conn, "legacy-key") is None
        assert conn.execute("SELECT id FROM tasks").fetchone()[0] == task_id


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_rejects_open_shapes_and_completion_paths(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    task_id = writer.dispatch(
        _frame("create", {"title": "shape validation"}, "shape-task"), _peer()
    )["result"]["task_id"]

    with pytest.raises(WriterProtocolError, match="unknown comment field"):
        writer.dispatch(
            _frame("comment", {"task_id": "t_1", "body": "x", "author": "caller"}, "r1"),
            _peer(),
        )

    with pytest.raises(WriterProtocolError, match="metadata values must be flat"):
        writer.dispatch(
            _frame(
                "complete",
                {"task_id": task_id, "metadata": {"nested": {"x": 1}}},
                "r2",
            ),
            _peer(),
        )

    with pytest.raises(WriterProtocolError, match="path-free"):
        writer.dispatch(
            _frame(
                "complete",
                {"task_id": task_id, "metadata": {"output": "relative/file.txt"}},
                "r3",
            ),
            _peer(),
        )


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_fixed_root_inline_attachment_is_replayable(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    created = writer.dispatch(_frame("create", {"title": "attach"}, "create"), _peer())
    task_id = created["result"]["task_id"]
    attach_args = {
        "task_id": task_id,
        "filename": "note.txt",
        "content_type": "text/plain",
        "data_b64": base64.b64encode(b"hello").decode("ascii"),
    }
    first = writer.dispatch(_frame("attach", attach_args, "attach-1"), _peer())
    second = writer.dispatch(_frame("attach", attach_args, "attach-1"), _peer())
    assert first["result"]["attachment_id"] > 0
    assert second["replayed"] is True
    assert second["result"] == first["result"]

    with kb.connect(db_path=tmp_path / "kanban.db") as conn:
        attachment = kb.list_attachments(conn, task_id)[0]
    attachment_path = Path(attachment.stored_path).resolve()
    assert attachment_path.read_bytes() == b"hello"
    assert attachment_path.is_relative_to((tmp_path / "attachments").resolve())


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_refuses_preexisting_unbound_attachment_without_suffix(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    created = writer.dispatch(_frame("create", {"title": "attach"}, "create"), _peer())
    task_id = created["result"]["task_id"]
    task_dir = tmp_path / "attachments" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "note.txt").write_bytes(b"crash-leftover")
    args = {
        "task_id": task_id,
        "filename": "note.txt",
        "data_b64": base64.b64encode(b"retry").decode("ascii"),
    }

    with pytest.raises(kb.AttachmentConflict, match="reconcile"):
        writer.dispatch(_frame("attach", args, "attach-conflict"), _peer())

    assert (task_dir / "note.txt").read_bytes() == b"crash-leftover"
    assert not (task_dir / "note (1).txt").exists()
    with kb.connect(db_path=tmp_path / "kanban.db") as conn:
        assert kb.list_attachments(conn, task_id) == []


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_attachment_does_not_follow_dangling_symlink(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    created = writer.dispatch(_frame("create", {"title": "symlink"}, "create"), _peer())
    task_id = created["result"]["task_id"]
    task_dir = tmp_path / "attachments" / task_id
    task_dir.mkdir(parents=True)
    target = tmp_path / "outside.bin"
    (task_dir / "note.txt").symlink_to(target)
    args = {
        "task_id": task_id,
        "filename": "note.txt",
        "data_b64": base64.b64encode(b"safe").decode("ascii"),
    }

    with pytest.raises(kb.AttachmentConflict, match="symlink"):
        writer.dispatch(_frame("attach", args, "attach-symlink"), _peer())

    assert not target.exists()
    assert not (task_dir / "note (1).txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="symlink hardening is proven on POSIX")
def test_writer_rejects_task_directory_redirect_to_another_task(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    first = writer.dispatch(_frame("create", {"title": "first"}, "first"), _peer())
    second = writer.dispatch(_frame("create", {"title": "second"}, "second"), _peer())
    first_id = first["result"]["task_id"]
    second_id = second["result"]["task_id"]
    root = tmp_path / "attachments"
    root.mkdir()
    (root / second_id).mkdir()
    (root / first_id).symlink_to(root / second_id, target_is_directory=True)
    args = {
        "task_id": first_id,
        "filename": "note.txt",
        "data_b64": base64.b64encode(b"must stay inside first task").decode("ascii"),
    }

    with pytest.raises(kb.AttachmentConflict, match="symlink"):
        writer.dispatch(_frame("attach", args, "attach-task-redirect"), _peer())

    assert not (root / second_id / "note.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_server_uses_peer_credentials_and_no_fallback(tmp_path):
    uid = os.getuid()
    writer = KanbanWriter(
        _config(
            tmp_path,
            peer_profile=lambda observed_uid: "worker" if observed_uid == uid else None,
        )
    )
    server = KanbanWriterServer(writer)
    server.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = writer_request_for_fixture(
            tmp_path / "writer.sock", "create", {"title": "socket"}, request_key="sock-1"
        )
        assert response["ok"] is True
        assert response["result"]["task_id"]
    finally:
        server.stop()
        thread.join(timeout=2)
    assert not (tmp_path / "writer.sock").exists()


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_server_refuses_to_replace_existing_socket(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    first = KanbanWriterServer(writer)
    first.start()
    second = KanbanWriterServer(writer)
    try:
        with pytest.raises(WriterProtocolError, match="already exists"):
            second.start()
        second.stop()
        assert (tmp_path / "writer.sock").exists()
    finally:
        first.stop()


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_server_idle_client_does_not_block_next_request(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    server = KanbanWriterServer(writer)
    server.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        idle.connect(str(tmp_path / "writer.sock"))
        try:
            time.sleep(1.25)
            disconnected = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            disconnected.connect(str(tmp_path / "writer.sock"))
            disconnected.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            disconnected.close()
            response = writer_request_for_fixture(
                tmp_path / "writer.sock", "create", {"title": "after idle"},
                request_key="after-idle",
            )
            assert response["ok"] is True
        finally:
            idle.close()
    finally:
        server.stop()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_writer_census_is_closed_and_has_no_privileged_request_fields():
    assert set(OPERATION_FIELDS) == set(OPERATIONS)
    exposed = set().union(*OPERATION_FIELDS.values())
    assert not exposed & FORBIDDEN_FIELDS
    assert {"db_path", "board", "sql", "callable"} <= FORBIDDEN_FIELDS


@pytest.mark.skipif(os.name != "posix", reason="the writer is AF_UNIX-only")
def test_writer_rejects_unknown_peer_profile(tmp_path):
    writer = KanbanWriter(_config(tmp_path, peer_profile=lambda _uid: None))
    with pytest.raises(WriterAuthorizationError):
        writer.peer_from_credentials((1, 2, 3))


def test_writer_read_projections_are_path_free_and_complete(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    parent = writer.dispatch(_frame("create", {"title": "parent"}, "parent"), _peer())
    parent_id = parent["result"]["task_id"]
    child = writer.dispatch(
        _frame("create", {"title": "child", "parents": [parent_id]}, "child"), _peer()
    )
    child_id = child["result"]["task_id"]

    shown = writer.dispatch(_frame("show", {"task_id": child_id}, "read-1"), _peer())
    task = shown["result"]
    assert task["parent_ids"] == [parent_id]
    assert task["parents"] == [parent_id]
    assert task["child_ids"] == []
    assert "workspace_path" not in task
    assert "Workspace: scratch @" not in task["worker_context"]

    listed = writer.dispatch(_frame("list", {"limit": 10}, "read-2"), _peer())
    listed_child = next(row for row in listed["result"] if row["id"] == child_id)
    assert listed_child["parent_ids"] == [parent_id]
    assert listed_child["child_ids"] == []
    assert "workspace_path" not in listed_child


@pytest.mark.skipif(os.name != "posix", reason="production writer requires AF_UNIX + SO_PEERCRED")
def test_production_writer_binds_spark_authority_callback_without_policy_copy():
    class SparkFixture:
        def bind_protocol_operations(self, operations):
            assert frozenset(operations) == OPERATIONS

        def peer_profile(self, uid):
            return "worker" if uid == 61001 else None

        def authorize(self, uid, operation, request, request_key, peer_pid):
            assert request_key == "show-key"
            assert peer_pid == 1
            if uid != 61001 or operation != "show":
                raise ValueError("refused")
            return {
                "authorized": True,
                "authorized_for_fixture": True,
                "live_production_enabled": False,
                "profile": "worker",
                "operation": operation,
                "canonical_board": str(CANONICAL_DB_PATH),
            }

    writer = KanbanWriter(WriterConfig.from_spark_adapter(SparkFixture()))
    peer = writer.peer_from_credentials((1, 61001, 61002))
    writer._authorize(peer, "show", {"task_id": "t_fixture"}, "show-key")
    with pytest.raises(WriterAuthorizationError):
        writer.peer_from_credentials((1, 61999, 61999))


def test_fixed_attachment_pending_marker_reconciles_exact_orphan(tmp_path):
    writer = KanbanWriter(_config(tmp_path))
    created = writer.dispatch(_frame("create", {"title": "attach"}, "create"), _peer())
    task_id = created["result"]["task_id"]
    data = b"crash-recoverable"
    args = {
        "task_id": task_id,
        "filename": "note.txt",
        "data_b64": base64.b64encode(data).decode("ascii"),
    }
    key = "attach-reconcile"
    digest = canonical_request_digest("attach", args)
    import hashlib
    expected = {
        "filename": "note.txt",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }
    with kb.connect(db_path=tmp_path / "kanban.db") as conn:
        with kb.write_txn(conn):
            kb._append_event(
                conn, task_id, "attachment_pending", expected,
                request_key=key, request_digest=digest, request_result=expected,
            )
    task_dir = tmp_path / "attachments" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "note.txt").write_bytes(data)

    result = writer.dispatch(_frame("attach", args, key), _peer())
    assert result["result"]["attachment_id"] > 0
    with kb.connect(db_path=tmp_path / "kanban.db") as conn:
        assert len(kb.list_attachments(conn, task_id)) == 1
        marker = kb.request_binding(conn, key)
        assert marker["kind"] == "attached"
        assert marker["result"]["sha256"] == expected["sha256"]
    assert not (task_dir / "note (1).txt").exists()
    assert MAX_ATTACHMENT_BYTES == kb.KANBAN_ATTACHMENT_MAX_BYTES
