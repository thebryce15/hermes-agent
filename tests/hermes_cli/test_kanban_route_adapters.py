"""Behavior checks for the fixed Kanban route adapters.

The tests inject the writer transport at the adapter boundary.  They never
open a board, socket, or attachment path.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


def _show_projection() -> dict:
    return {
        "id": "t_child",
        "title": "child",
        "status": "todo",
        "parent_ids": ["t_parent"],
        "child_ids": ["t_grandchild"],
        "parent_results": [["t_parent", "handoff"]],
        "worker_context": "# Kanban task t_child: child",
    }


def test_tools_consume_writer_projection_and_subscription(monkeypatch):
    from tools import kanban_tools as tools

    calls: list[tuple[str, dict]] = []
    show = _show_projection()

    def fake_request(operation, args=None, *, request_key=None):
        calls.append((operation, dict(args or {})))
        if operation == "show":
            return {"result": show}
        if operation == "comments":
            return {"result": [{"id": 1, "body": "note"}]}
        if operation == "events":
            return {"result": [{"id": 2, "kind": "created"}]}
        if operation == "runs":
            return {"result": []}
        if operation == "create":
            return {"result": {"task_id": "t_new"}}
        if operation == "notify-subscribe":
            return {"result": {"ok": True, "subscription_id": 9}}
        raise AssertionError(operation)

    monkeypatch.setattr(tools, "_writer_request", fake_request)
    monkeypatch.setattr(
        tools,
        "_maybe_auto_subscribe",
        lambda task_id, request_key: {"ok": True, "subscription_id": 9},
    )

    shown = json.loads(tools._handle_show({"task_id": "t_child"}))
    assert shown["parent_ids"] == ["t_parent"]
    assert shown["child_ids"] == ["t_grandchild"]
    assert shown["parent_results"] == [["t_parent", "handoff"]]
    assert shown["worker_context"].startswith("# Kanban task")

    created = json.loads(tools._handle_create({
        "title": "new", "assignee": "worker", "request_key": "tool-create-1",
    }))
    assert created["subscribed"] is True
    assert created["subscription"] == {"ok": True, "subscription_id": 9}
    assert [name for name, _ in calls] == [
        "show", "comments", "events", "runs", "create",
    ]


def test_tools_refuse_missing_read_projection(monkeypatch):
    from tools import kanban_tools as tools

    monkeypatch.setattr(
        tools,
        "_writer_request",
        lambda operation, args=None, **kwargs: {"result": {"id": "t1"}},
    )
    output = tools._handle_show({"task_id": "t1"})
    assert "projection unavailable" in output


def test_tool_attachment_limit_is_writer_owned(monkeypatch):
    from hermes_cli import kanban_writer
    from tools import kanban_tools as tools

    monkeypatch.setattr(kanban_writer, "MAX_ATTACHMENT_BYTES", 123456)
    assert tools._decoded_attachment_limit() == 123456


def test_core_attachment_cap_fits_the_writer_frame():
    from hermes_cli.kanban_writer import MAX_ATTACHMENT_BYTES, MAX_FRAME_BYTES

    encoded_chars = ((MAX_ATTACHMENT_BYTES + 2) // 3) * 4
    assert encoded_chars < MAX_FRAME_BYTES


def test_cli_show_and_list_use_truthful_projection(monkeypatch, capsys):
    from hermes_cli import kanban as cli

    projection = _show_projection()
    row = {
        "id": "t_child",
        "title": "child",
        "status": "todo",
        "assignee": "worker",
        "parent_ids": ["t_parent"],
        "child_ids": [],
    }

    def fake_request(operation, args=None, *, request_key=None):
        if operation == "show":
            return {"result": projection}
        if operation == "list":
            return {"result": [row]}
        if operation in {"comments", "events", "runs"}:
            return {"result": []}
        raise AssertionError(operation)

    monkeypatch.setattr(cli, "_writer_request", fake_request)
    show_args = SimpleNamespace(
        kanban_action="show", task_id="t_child", state_type=None,
        state_name=None, json=True,
    )
    assert cli._ipc_cli_command(show_args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["parent_ids"] == ["t_parent"]
    assert shown["worker_context"].startswith("# Kanban task")

    list_args = SimpleNamespace(
        kanban_action="list", session=None, mine=False, assignee=None,
        status=None, tenant=None, archived=False, limit=50, sort=None,
        workflow_template_id=None, current_step_key=None, json=True,
    )
    assert cli._ipc_cli_command(list_args) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["parent_ids"] == ["t_parent"]


def test_cli_attach_sends_inline_bytes_only(monkeypatch, capsys):
    from hermes_cli import kanban as cli

    calls: list[tuple[str, dict]] = []

    def fake_request(operation, args=None, *, request_key=None):
        calls.append((operation, dict(args or {})))
        return {"result": {"attachment_id": 7}}

    monkeypatch.setattr(cli, "_writer_request", fake_request)
    args = SimpleNamespace(
        kanban_action="attach", task_id="t_child", path=None,
        filename="note.txt", name=None, data_b64="aGVsbG8=",
        content_type="text/plain", author=None, json=True,
        request_key="cli-attach-1", idempotency_key=None,
    )
    assert cli._ipc_cli_command(args) == 0
    assert json.loads(capsys.readouterr().out)["attachment_id"] == 7
    assert calls == [(
        "attach",
        {
            "task_id": "t_child", "filename": "note.txt",
            "content_type": "text/plain", "data_b64": "aGVsbG8=",
        },
    )]

    calls.clear()
    args.path = "C:/not-admitted.txt"
    assert cli._ipc_cli_command(args) == 1
    assert calls == []


def test_slash_notification_schema_only_sends_chat_type_on_subscribe(monkeypatch):
    from gateway import slash_commands

    calls: list[tuple[str, dict]] = []

    def fake_call(operation, args=None, *, mutation=False, request_key=None):
        calls.append((operation, dict(args or {})))
        return {"ok": True}

    monkeypatch.setattr(slash_commands, "_kanban_writer_call", fake_call)
    event = SimpleNamespace(
        text="/kanban notify-read t_child",
        source=SimpleNamespace(
            platform=SimpleNamespace(value="telegram"),
            chat_id="chat-1", chat_type="group", thread_id="thread-1",
        ),
        message_id="message-1",
    )
    result = asyncio.run(slash_commands.GatewaySlashCommandsMixin._handle_kanban_command(None, event))
    assert '"ok": true' in result
    assert calls[-1][0] == "notify-read"
    assert "chat_type" not in calls[-1][1]

    calls.clear()
    event.text = "/kanban notify-subscribe t_child"
    result = asyncio.run(slash_commands.GatewaySlashCommandsMixin._handle_kanban_command(None, event))
    assert '"ok": true' in result
    assert calls[-1][0] == "notify-subscribe"
    assert calls[-1][1]["chat_type"] == "group"


def test_tool_and_cli_mutations_require_stable_request_keys(monkeypatch, capsys):
    from hermes_cli import kanban as cli
    from tools import kanban_tools as tools

    tool_keys: list[str | None] = []

    def fake_tool_request(operation, args=None, *, request_key=None):
        tool_keys.append(request_key)
        return {"result": {"comment_id": 1}}

    monkeypatch.setattr(tools, "_writer_request", fake_tool_request)
    tool_args = {"task_id": "t1", "body": "note", "request_key": "tool-comment-1"}
    assert json.loads(tools._handle_comment(dict(tool_args)))["ok"] is True
    assert json.loads(tools._handle_comment(dict(tool_args)))["ok"] is True
    tool_args["request_key"] = "tool-comment-2"
    assert json.loads(tools._handle_comment(dict(tool_args)))["ok"] is True
    assert tool_keys == ["tool-comment-1", "tool-comment-1", "tool-comment-2"]
    assert "request_key is required" in tools._handle_comment({"task_id": "t1", "body": "note"})
    assert len(tool_keys) == 3

    cli_keys: list[str | None] = []

    def fake_cli_request(operation, args=None, *, request_key=None):
        cli_keys.append(request_key)
        return {"result": {"comment_id": 1}}

    monkeypatch.setattr(cli, "_writer_request", fake_cli_request)
    args = SimpleNamespace(
        kanban_action="comment", task_id="t1", text=["note"], author=None,
        max_len=None, request_key="cli-comment-1", idempotency_key=None,
    )
    assert cli._ipc_cli_command(args) == 0
    assert cli._ipc_cli_command(args) == 0
    args.request_key = "cli-comment-2"
    assert cli._ipc_cli_command(args) == 0
    capsys.readouterr()
    assert cli_keys == ["cli-comment-1", "cli-comment-1", "cli-comment-2"]
    args.request_key = None
    assert cli._ipc_cli_command(args) == 1
    assert len(cli_keys) == 3


def test_auto_heartbeat_rotates_after_each_acknowledged_success(monkeypatch):
    from tools import kanban_tools as tools

    keys: list[str | None] = []

    def fake_request(operation, args=None, *, request_key=None):
        assert operation == "heartbeat"
        keys.append(request_key)
        return {"ok": True, "result": {"ok": True}}

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t-heartbeat")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")
    monkeypatch.setattr(tools, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(tools, "_writer_request", fake_request)
    tools._auto_heartbeat_last_attempt = 0.0
    with tools._auto_heartbeat_lock:
        tools._auto_heartbeat_pending_keys.clear()

    assert tools.heartbeat_current_worker_from_env() is True
    assert tools.heartbeat_current_worker_from_env() is True
    assert keys[0]
    assert keys[1]
    assert keys[1] != keys[0]


def test_auto_heartbeat_reuses_key_after_lost_reply_then_rotates(monkeypatch):
    from tools import kanban_tools as tools

    keys: list[str | None] = []
    attempts = 0

    def fake_request(operation, args=None, *, request_key=None):
        nonlocal attempts
        attempts += 1
        keys.append(request_key)
        if attempts == 1:
            raise TimeoutError("writer applied heartbeat but reply was lost")
        return {"ok": True, "result": {"ok": True}}

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t-heartbeat")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "18")
    monkeypatch.setattr(tools, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(tools, "_writer_request", fake_request)
    tools._auto_heartbeat_last_attempt = 0.0
    with tools._auto_heartbeat_lock:
        tools._auto_heartbeat_pending_keys.clear()

    assert tools.heartbeat_current_worker_from_env() is False
    assert tools.heartbeat_current_worker_from_env() is True
    assert keys[1] == keys[0]
    assert tools.heartbeat_current_worker_from_env() is True
    assert keys[2] != keys[1]


def test_dashboard_mutations_reuse_header_key_and_forbid_actor(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import ValidationError
    from plugins.kanban.dashboard import plugin_api

    with pytest.raises(ValidationError):
        plugin_api.CommentBody(body="note", actor="caller-controlled")

    calls: list[tuple[str, str | None]] = []

    def fake_writer(operation, args=None, *, mutation=False, request_key=None):
        calls.append((operation, request_key))
        return {"comment_id": 1}

    monkeypatch.setattr(plugin_api, "_writer", fake_writer)
    monkeypatch.setattr(plugin_api, "_ensure_task", lambda task_id: {"id": task_id})
    app = FastAPI()
    app.include_router(plugin_api.router)
    client = TestClient(app)

    assert client.post("/tasks/t1/comments", json={"body": "note"}).status_code == 422
    assert client.post(
        "/tasks/t1/comments",
        json={"body": "note", "actor": "caller-controlled"},
        headers={"Idempotency-Key": "dashboard-comment-1"},
    ).status_code == 422
    for request_key in (
        "dashboard-comment-1",
        "dashboard-comment-1",
        "dashboard-comment-2",
    ):
        response = client.post(
            "/tasks/t1/comments",
            json={"body": "note"},
            headers={"Idempotency-Key": request_key},
        )
        assert response.status_code == 200
    assert calls == [
        ("comment", "dashboard-comment-1"),
        ("comment", "dashboard-comment-1"),
        ("comment", "dashboard-comment-2"),
    ]


def test_watcher_derives_retry_stable_key_from_subscription_cursor(monkeypatch):
    from gateway import kanban_watchers

    keys: list[str | None] = []

    def fake_request(socket_path, operation, args=None, *, request_key=None):
        keys.append(request_key)
        return {"result": {"events": []}}

    monkeypatch.setattr(kanban_watchers, "writer_request", fake_request)
    watcher = kanban_watchers.GatewayKanbanWatchersMixin()
    sub = {
        "id": 7, "task_id": "t1", "platform": "telegram",
        "chat_id": "chat-1", "thread_id": "", "last_event_id": 12,
    }
    watcher._kanban_claim(sub)
    watcher._kanban_claim(dict(sub))
    sub["last_event_id"] = 13
    watcher._kanban_claim(sub)
    assert keys[0] == keys[1]
    assert keys[2] != keys[1]


def test_slash_rejects_unknown_options_and_uses_message_identity(monkeypatch):
    from gateway import slash_commands

    keys: list[str | None] = []

    def fake_call(operation, args=None, *, mutation=False, request_key=None):
        keys.append(request_key)
        return {"comment_id": 1}

    monkeypatch.setattr(slash_commands, "_kanban_writer_call", fake_call)
    event = SimpleNamespace(
        text="/kanban comment t1 note --bogus value",
        source=SimpleNamespace(
            platform=SimpleNamespace(value="telegram"), chat_id="chat-1",
            chat_type="group", thread_id="thread-1",
        ),
        message_id="message-1",
    )
    result = asyncio.run(slash_commands.GatewaySlashCommandsMixin._handle_kanban_command(None, event))
    assert "unsupported option" in result
    assert keys == []

    event.text = "/kanban comment t1 note --SQL SELECT"
    result = asyncio.run(slash_commands.GatewaySlashCommandsMixin._handle_kanban_command(None, event))
    assert "writer-owned" in result
    assert keys == []

    event.text = "/kanban comment t1 note"
    asyncio.run(slash_commands.GatewaySlashCommandsMixin._handle_kanban_command(None, event))
    asyncio.run(slash_commands.GatewaySlashCommandsMixin._handle_kanban_command(None, event))
    event.message_id = "message-2"
    asyncio.run(slash_commands.GatewaySlashCommandsMixin._handle_kanban_command(None, event))
    assert keys[0] == keys[1]
    assert keys[2] != keys[1]


def test_update_slash_refuses_before_file_or_process_effect(monkeypatch):
    import subprocess
    from pathlib import Path
    from gateway import slash_commands

    def forbidden(*args, **kwargs):
        raise AssertionError("update effect was reached")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    event = SimpleNamespace(text="/update")
    result = asyncio.run(
        slash_commands.GatewaySlashCommandsMixin._handle_update_command(None, event)
    )
    assert result == "/update refused by D0B admission boundary"


def test_dashboard_rejects_hollow_projection(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from plugins.kanban.dashboard import plugin_api

    monkeypatch.setattr(plugin_api, "_writer", lambda operation, args=None, **kwargs: {"id": "t1"})
    with pytest.raises(fastapi.HTTPException) as exc:
        plugin_api.get_board(board=None)
    assert exc.value.status_code == 503
    assert "projection unavailable" in str(exc.value.detail)
