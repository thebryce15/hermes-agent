"""Machine-checked U1-D0B route and direct-sink census.

This is intentionally a narrow source-shape gate.  The packet requires an
AST check because a wrapper can be correct while an imported helper, legacy
CLI handler, or shortest direct sink remains callable.  The test records the
finite admitted operation set and the explicit refusal partition without
freezing line numbers or whole-file snapshots.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hermes_cli.kanban_writer import (
    FORBIDDEN_FIELDS,
    OPERATION_FIELDS,
    OPERATIONS,
    WriterProtocolError,
    _validate_request,
)


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_OPERATIONS = frozenset(
    {
        "show",
        "list",
        "runs",
        "events",
        "comments",
        "attachments",
        "notify-read",
        "create",
        "link",
        "unlink",
        "comment",
        "block",
        "unblock",
        "assign",
        "claim",
        "heartbeat",
        "complete",
        "attach",
        "notify-subscribe",
        "notify-unsubscribe",
        "notify-claim",
        "notify-advance",
        "notify-rewind",
    }
)

EXPECTED_FORBIDDEN_FIELDS = frozenset(
    {
        "author",
        "created_by",
        "uploaded_by",
        "claimer",
        "session_id",
        "actor",
        "profile",
        "board",
        "db_path",
        "workspace_path",
        "attachment_path",
        "callable",
        "sql",
        "SQL",
    }
)


# These are the existing kanban_db APIs the writer may delegate to.  The
# partition is deliberately explicit: adding a public direct mutator requires
# a review of this test instead of silently becoming an admitted sink.
DB_FIXED_APIS = frozenset(
    {
        "get_task",
        "list_tasks",
        "list_runs",
        "get_run",
        "list_events",
        "list_comments",
        "list_comments_after",
        "list_attachments",
        "list_notify_subs",
        "count_notify_subs",
        "unseen_events_for_sub",
        "parent_ids",
        "child_ids",
        "build_worker_context",
        "create_task",
        "assign_task",
        "claim_task",
        "claim_review_task",
        "heartbeat_claim",
        "heartbeat_worker",
        "complete_task",
        "link_tasks",
        "unlink_tasks",
        "add_comment",
        "block_task",
        "unblock_task",
        "store_attachment_bytes",
        "store_attachment_bytes_fixed",
        "add_notify_sub",
        "remove_notify_sub",
        "claim_unseen_events_for_sub",
        "advance_notify_cursor",
        "rewind_notify_cursor",
    }
)

DB_REFUSED_APIS = frozenset(
    {
        "write_txn",
        "scoped_current_board",
        "set_current_board",
        "clear_current_board",
        "connect",
        "connect_closing",
        "write_board_metadata",
        "create_board",
        "remove_board",
        "repair_db",
        "init_db",
        "set_model_override",
        "set_reasoning_effort",
        "recompute_ready",
        "release_stale_claims",
        "reclaim_task",
        "reassign_task",
        "add_attachment",
        "delete_attachment",
        "edit_completed_task_result",
        "promote_task",
        "specify_triage_task",
        "decompose_triage_task",
        "archive_task",
        "delete_archived_task",
        "delete_task",
        "resolve_workspace",
        "set_workspace_path",
        "set_branch_name",
        "schedule_task",
        "reap_worker_zombies",
        "enforce_max_runtime",
        "detect_stale_running",
        "detect_crashed_workers",
        "dispatch_once",
        "run_daemon",
        "gc_events",
        "gc_worker_logs",
    }
)


# Route identities are keyed by source path.  ``fixed`` means the function is
# an adapter for a named writer operation (or a pure read projection); ``F``
# means the public/direct path must refuse before its old sink.
FIXED_ROUTES: dict[str, frozenset[str]] = {
    "tools/kanban_tools.py": frozenset(
        {
            "_writer_request",
            "_handle_show",
            "_handle_list",
            "_handle_create",
            "_handle_link",
            "_handle_comment",
            "_handle_block",
            "_handle_unblock",
            "_handle_heartbeat",
            "_handle_complete",
            "_handle_attach",
            "_handle_attach_url",
            "_handle_attachments",
            "_maybe_auto_subscribe",
            "heartbeat_current_worker_from_env",
            "inject_new_comments_from_env",
        }
    ),
    "hermes_cli/kanban.py": frozenset(
        {
            "_writer_request",
            "_ipc_cli_command",
            "kanban_command",
            "_cmd_create",
            "_cmd_assign",
            "_cmd_link",
            "_cmd_unlink",
            "_cmd_claim",
            "_cmd_comment",
            "_cmd_complete",
            "_cmd_block",
            "_cmd_unblock",
            "_cmd_heartbeat",
            "_cmd_attach",
            "_cmd_list",
            "_cmd_show",
            "_cmd_attachments",
            "_cmd_runs",
            "run_slash",
        }
    ),
    "plugins/kanban/dashboard/plugin_api.py": frozenset(
        {
            "_writer",
            "_read_list",
            "_ensure_task",
            "get_board",
            "get_task",
            "create_task",
            "list_task_attachments",
            "upload_task_attachment",
            "add_comment",
            "add_link",
            "delete_link",
        }
    ),
    "gateway/kanban_watchers.py": frozenset(
        {
            "_writer_call",
            "_kanban_claim",
            "_kanban_advance",
            "_kanban_unsub",
            "_kanban_rewind",
        }
    ),
    "gateway/slash_commands.py": frozenset(
        {
            "_kanban_writer_call",
            "_handle_kanban_command",
        }
    ),
}

REFUSED_ROUTES: dict[str, frozenset[str]] = {
    "hermes_cli/kanban.py": frozenset(
        {
            "_dispatch_boards",
            "_board_task_counts",
            "_ready_queue_nonempty",
            "_cmd_boards_list",
            "_cmd_boards_create",
            "_cmd_boards_rm",
            "_cmd_boards_switch",
            "_cmd_boards_show",
            "_cmd_boards_rename",
            "_cmd_boards_set_default_workdir",
            "_cmd_init",
            "_cmd_swarm",
            "_cmd_specify",
            "_cmd_decompose",
            "_cmd_set_model",
            "_cmd_attach_rm",
            "_cmd_reclaim",
            "_cmd_reassign",
            "_cmd_edit",
            "_cmd_schedule",
            "_cmd_promote",
            "_cmd_archive",
            "_cmd_dispatch",
            "_cmd_daemon",
            "_cmd_gc",
            "_cmd_repair",
            "_cmd_diagnostics",
            "_cmd_tail",
            "_cmd_watch",
            "_cmd_stats",
            "_cmd_log",
            "_cmd_context",
            "_cmd_assignees",
            "_cmd_notify_subscribe",
            "_cmd_notify_list",
            "_cmd_notify_unsubscribe",
        }
    ),
    "plugins/kanban/dashboard/plugin_api.py": frozenset(
        {
            "download_attachment",
            "remove_attachment",
            "update_task",
            "delete_task",
            "bulk_update",
            "list_diagnostics",
            "list_active_workers",
            "get_run_endpoint",
            "inspect_run_endpoint",
            "terminate_run_endpoint",
            "reclaim_task_endpoint",
            "specify_task_endpoint",
            "reassign_task_endpoint",
            "estimate_text_endpoint",
            "estimate_task_endpoint",
            "get_home_channels",
            "subscribe_home",
            "unsubscribe_home",
            "get_stats",
            "get_assignees",
            "get_task_log",
            "dispatch",
            "list_kanban_projects",
            "list_boards",
            "create_board_endpoint",
            "rename_board",
            "delete_board",
            "switch_board",
            "decompose_task_endpoint",
            "get_orchestration_settings",
            "set_orchestration_settings",
            "stream_events",
        }
    ),
    "gateway/kanban_watchers.py": frozenset(
        {
            "_acquire_singleton_lock",
            "_kanban_notifier_watcher",
            "_deliver_kanban_artifacts",
            "_kanban_dispatcher_watcher",
        }
    ),
    "hermes_cli/kanban_swarm.py": frozenset(
        {"create_swarm", "post_blackboard_update"}
    ),
    "hermes_cli/kanban_specify.py": frozenset(
        {"specify_task", "list_triage_ids"}
    ),
    "hermes_cli/kanban_decompose.py": frozenset(
        {"decompose_task", "list_triage_ids"}
    ),
    "hermes_cli/projects_cmd.py": frozenset(
        {
            "projects_command",
            "_cmd_create",
            "_cmd_list",
            "_cmd_show",
            "_cmd_add_folder",
            "_cmd_remove_folder",
            "_cmd_rename",
            "_cmd_set_primary",
            "_cmd_use",
            "_cmd_archive",
            "_cmd_restore",
            "_cmd_bind_board",
            "_sync_board_default_workdir",
            "wrapper",
        }
    ),
    "hermes_cli/backup.py": frozenset(
        {
            "run_backup",
            "run_import",
            "create_quick_snapshot",
            "restore_quick_snapshot",
            "prune_quick_snapshots",
            "run_quick_backup",
            "create_pre_update_backup",
            "create_pre_migration_backup",
        }
    ),
    "gateway/platforms/base.py": frozenset(),
    "cli.py": frozenset(
        {
            "_collect_query_images",
            "_try_attach_clipboard_image",
            "_run_kanban_goal_loop_q",
            "_task_status",
            "_block",
        }
    ),
    "hermes_cli/cli_commands_mixin.py": frozenset(
        {"_handle_image_command", "_handle_goal_command"}
    ),
    "agent/turn_finalizer.py": frozenset({"finalize_turn"}),
    "tui_gateway/server.py": frozenset(
        {
            "_append_model_switch_marker",
            "handle_request",
            "dispatch",
            "_sess",
            "_sess_nowait",
            "_get_db",
            "_db_for_profile",
            "_profile_db",
            "_finalize_session",
            "_init_session",
            "_persist_branch_seed",
            "_persist_live_session_runtime",
            "_persist_live_session_system_prompt",
            "_persist_model_switch",
            "_persist_session_git_meta",
            "_register_session_cwd",
            "_session_db",
            "_set_session_cwd",
            "_sync_session_key_after_compress",
            "_ensure_session_db_row",
            "_collect_kanban_notifications",
        }
    ),
    "gateway/slash_commands.py": frozenset(
        {
            "_handle_goal_command",
            "_handle_subgoal_command",
            "_handle_undo_command",
            "_handle_set_home_command",
            "_handle_retry_command",
            "_handle_resume_command",
            "_handle_background_command",
            "_handle_model_command",
            "_handle_restart_command",
            "_handle_memory_command",
            "_handle_skills_command",
            "_handle_update_command",
        }
    ),
    "hermes_cli/main.py": frozenset({"_pin_kanban_board_env"}),
}


REFUSED_METHOD_ROUTES: dict[str, frozenset[str]] = {
    "tui_gateway/methods_session.py": frozenset(
        {
            "session.create",
            "session.resume",
            "session.cwd.set",
            "session.activate",
            "session.delete",
            "session.title",
            "message.react",
            "handoff.request",
            "handoff.fail",
            "session.undo",
            "session.compress",
            "session.save",
            "session.close",
            "session.branch",
            "session.interrupt",
            "session.status",
            "spawn_tree.save",
            "spawn_tree.list",
            "spawn_tree.load",
            "session.steer",
            "session.redirect",
        }
    )
}

READ_ONLY_METHOD_ROUTES = frozenset(
    {
        "session.list",
        "session.most_recent",
        "session.active_list",
        "session.usage",
        "session.context_breakdown",
        "handoff.state",
        "session.history",
    }
)

ROUTE_FILES = (
    frozenset(FIXED_ROUTES)
    | frozenset(REFUSED_ROUTES)
    | frozenset(REFUSED_METHOD_ROUTES)
)


SINK_ATTRIBUTES = frozenset(
    {
        "connect",
        "connect_closing",
        "execute",
        "executemany",
        "executescript",
        "open",
        "write",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "mkdir",
        "run",
        "Popen",
        "call",
        "check_output",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "system",
    }
)


def _read_tree(relative: str) -> tuple[Path, ast.Module]:
    path = ROOT / relative
    assert path.is_file(), f"census route file is missing: {relative}"
    text = path.read_text(encoding="utf-8")
    return path, ast.parse(text, filename=str(path))


def _functions(tree: ast.AST) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    found: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, []).append(node)
    return found


def _method_handlers(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Index deferred TUI handlers by stable ``@method`` route."""

    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "method"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                continue
            route = decorator.args[0].value
            if route in found:
                raise AssertionError(f"duplicate @method route: {route}")
            found[route] = node
    return found


def _call_name(call: ast.Call) -> str:
    return ast.unparse(call.func)


def _sink_calls(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    """Return direct sink-shaped calls in one function, excluding nested defs."""

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[ast.Call] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is not function:
                return
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is not function:
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            full_name = _call_name(node)
            # ``Context.run`` and ``sys.stdout.write`` are not the filesystem
            # or subprocess sinks this census is guarding.  Keep the actual
            # ``subprocess.run``/file-handle writes in scope.
            if name == "run" and not full_name.startswith("subprocess."):
                name = ""
            if name == "write" and full_name.startswith(
                ("sys.stdout.", "sys.stderr.", "buf.")
            ):
                name = ""
            if name in SINK_ATTRIBUTES:
                self.calls.append(node)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(function)
    return visitor.calls


def _has_refusal_prefix(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the public/direct function refuses before its old body."""

    statements = list(function.body)
    if statements and isinstance(statements[0], ast.Expr):
        value = statements[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            statements.pop(0)
    for statement in statements[:4]:
        if isinstance(statement, ast.Return):
            value = ast.unparse(statement.value) if statement.value else ""
            if any(token in value.casefold() for token in ("refuse", "deny", "error")):
                return True
        if isinstance(statement, ast.Raise):
            value = ast.unparse(statement.exc) if statement.exc else ""
            if any(token in value.casefold() for token in ("refuse", "deny", "error")):
                return True
        if isinstance(statement, ast.Expr):
            value = ast.unparse(statement.value)
            if any(token in value.casefold() for token in ("refuse", "deny")):
                return True
    return False


def _has_refusal_guard(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a route contains an explicit refusal helper/exception."""

    if _has_refusal_prefix(function):
        return True
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = _call_name(node).casefold()
            if any(token in name for token in ("refuse", "deny", "reject")):
                return True
        elif isinstance(node, ast.Raise) and node.exc:
            value = ast.unparse(node.exc).casefold()
            if any(token in value for token in ("refuse", "deny", "reject")):
                return True
    return False


def _has_writer_call(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if (
                "writer_request" in name
                or "_writer_call" in name
                or name.endswith("._writer")
                or name == "_writer"
                # Dashboard read routes deliberately share these two narrow
                # helpers; mapping the helpers themselves keeps the proof
                # local and avoids a whole-module call-graph snapshot.
                or name in {"_read_list", "_ensure_task", "_handle_attach"}
            ):
                return True
        elif isinstance(node, ast.Name) and (
            "writer_request" in node.id
            or "_writer_call" in node.id
            or node.id == "_writer"
        ):
            # Async gateway routes pass the writer callable to
            # ``asyncio.to_thread`` instead of calling it inline.
            return True
    return False


def _has_direct_store_selector(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name.endswith("connect_closing") or name.endswith("sqlite3.connect"):
                return True
            if name in {"kb.connect", "kb.kanban_db_path", "kb.scoped_current_board"}:
                return True
    return False


def test_no_unlisted_direct_writer_or_raw_sql_file_sink() -> None:
    """Every enumerated route is fixed IPC or an explicit fail-closed F path."""

    assert OPERATIONS == EXPECTED_OPERATIONS
    assert set(OPERATION_FIELDS) == EXPECTED_OPERATIONS
    assert FORBIDDEN_FIELDS == EXPECTED_FORBIDDEN_FIELDS
    assert not set().union(*OPERATION_FIELDS.values()) & EXPECTED_FORBIDDEN_FIELDS

    # Closed request envelopes reject forbidden identity/store/SQL fields even
    # when nested in an otherwise allowed value, not just at the top level.
    for field in sorted(EXPECTED_FORBIDDEN_FIELDS):
        with pytest.raises(WriterProtocolError, match="forbidden field"):
            _validate_request(
                {
                    "operation": "list",
                    "args": {"tenant": {field: "caller-controlled"}},
                }
            )
    with pytest.raises(WriterProtocolError, match="unknown writer operation"):
        _validate_request({"operation": "dispatch", "args": {}})
    with pytest.raises(WriterProtocolError, match="unknown show field"):
        _validate_request(
            {"operation": "show", "args": {"task_id": "t", "path": "x"}}
        )

    # The exact source surface is part of the contract.  Missing a named
    # function is a census failure, not permission to silently drop coverage.
    failures: list[str] = []
    for relative in sorted(ROUTE_FILES):
        _, tree = _read_tree(relative)
        defs = _functions(tree)
        expected_names = FIXED_ROUTES.get(relative, frozenset()) | REFUSED_ROUTES.get(
            relative, frozenset()
        )
        for name in sorted(expected_names):
            if name not in defs:
                failures.append(f"{relative}:{name} missing from the census")

        for name in sorted(FIXED_ROUTES.get(relative, frozenset())):
            for function in defs.get(name, ()):
                findings: list[str] = []
                if not (_has_writer_call(function) or name in {
                    "finalize_turn",
                    "_pin_kanban_board_env",
                    "kanban_command",
                    "run_slash",
                }):
                    findings.append("not an IPC/read adapter")
                sinks = _sink_calls(function)
                if sinks:
                    findings.append(
                        "direct sink(s): "
                        + ", ".join(
                            f"{_call_name(call)}@{call.lineno}" for call in sinks
                        )
                    )
                if _has_direct_store_selector(function):
                    findings.append("selects a DB/board directly")
                if findings:
                    failures.append(f"{relative}:{name}: " + "; ".join(findings))

        for name in sorted(REFUSED_ROUTES.get(relative, frozenset())):
            for function in defs.get(name, ()):
                no_op_refusal = (relative, name) in {
                    ("cli.py", "_collect_query_images"),
                    ("cli.py", "_try_attach_clipboard_image"),
                    ("tui_gateway/server.py", "_collect_kanban_notifications"),
                    ("hermes_cli/main.py", "_pin_kanban_board_env"),
                }
                sinks = _sink_calls(function)
                if not (_has_refusal_guard(function) or no_op_refusal):
                    detail = "not an explicit F refusal"
                    if sinks:
                        detail += "; sinks=" + ", ".join(
                            f"{_call_name(call)}@{call.lineno}" for call in sinks
                        )
                    failures.append(f"{relative}:{name}: {detail}")
                elif sinks and not _has_refusal_prefix(function):
                    failures.append(
                        f"{relative}:{name}: direct sink(s) without an "
                        "unconditional F refusal: "
                        + ", ".join(
                            f"{_call_name(call)}@{call.lineno}" for call in sinks
                        )
                    )

    # Split TUI handler modules reuse the source name ``_`` for every deferred
    # function, so classify them by their stable @method route.  The semantic
    # scan also exposes sibling session/persistence handlers that would
    # otherwise sit outside the ordinary function-name census.
    mutating_method_calls = frozenset(
        {
            "SessionDB",
            "_append_spawn_tree_index",
            "_compress_session_history",
            "_enable_gateway_prompts",
            "_enqueue_prompt",
            "_ensure_session_db_row",
            "_get_compute_host_supervisor().interrupt",
            "_init_session",
            "_record_inflight_correction",
            "_register_session_cwd",
            "_schedule_agent_build",
            "_schedule_session_cap_enforcement",
            "_set_session_cwd",
            "_spawn_tree_session_dir",
            "_spawn_trees_root",
            "_sync_session_key_after_compress",
            "_teardown_popped_session",
            "_tts_stream_stop",
            "agent.redirect",
            "agent.steer",
            "db.append_messages_batch",
            "db.create_session",
            "db.delete_session",
            "db.fail_handoff",
            "db.reopen_session",
            "db.request_handoff",
            "db.set_message_reaction",
            "db.set_session_title",
            "path.write_text",
            "request_hard_interrupt",
            "saved_dir.mkdir",
            "scoped_db.set_session_title",
        }
    )
    server_defs = _functions(_read_tree("tui_gateway/server.py")[1])
    for relative, refused_routes in sorted(REFUSED_METHOD_ROUTES.items()):
        handlers = _method_handlers(_read_tree(relative)[1])
        relevant_routes = {
            route
            for route in handlers
            if route.startswith(("session.", "handoff.", "spawn_tree."))
            or route == "message.react"
        }
        overlap = refused_routes & READ_ONLY_METHOD_ROUTES
        for route in sorted(overlap):
            failures.append(
                f"{relative}:@method({route!r}) appears in both deferred partitions"
            )
        partition = refused_routes | READ_ONLY_METHOD_ROUTES
        for route in sorted(relevant_routes - partition):
            failures.append(
                f"{relative}:@method({route!r}) omitted from deferred partition"
            )
        for route in sorted(partition - relevant_routes):
            failures.append(
                f"{relative}:@method({route!r}) is partitioned but not discovered"
            )

        for route in sorted(refused_routes):
            function = handlers.get(route)
            if function is None:
                failures.append(f"{relative}:@method({route!r}) missing from census")
                continue
            sinks = _sink_calls(function)
            if not _has_refusal_prefix(function):
                detail = "not an explicit F refusal"
                if sinks:
                    detail += "; sinks=" + ", ".join(
                        f"{_call_name(call)}@{call.lineno}" for call in sinks
                    )
                failures.append(f"{relative}:@method({route!r}): {detail}")

        # A verified read-only route has no sink or known mutating call in its
        # handler or transitive server.py helper closure.  An explicitly
        # refused helper is a traversal stop: its old body is unreachable.
        for route in sorted(READ_ONLY_METHOD_ROUTES & relevant_routes):
            pending = [handlers[route]]
            inspected: set[int] = set()
            route_findings: set[str] = set()
            while pending:
                function = pending.pop()
                identity = id(function)
                if identity in inspected:
                    continue
                inspected.add(identity)
                if function is not handlers[route] and _has_refusal_prefix(function):
                    continue
                route_findings.update(
                    f"sink {_call_name(call)}@{call.lineno}"
                    for call in _sink_calls(function)
                )
                calls = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                ]
                for call in calls:
                    name = _call_name(call)
                    if name in mutating_method_calls:
                        route_findings.add(f"mutating call {name}@{call.lineno}")
                    if isinstance(call.func, ast.Name):
                        pending.extend(server_defs.get(call.func.id, ()))
            for finding in sorted(route_findings):
                failures.append(
                    f"{relative}:@method({route!r}) read-only violation: {finding}"
                )

    # Every cache/send method in the base platform interface is explicitly F;
    # a new one must not quietly become a delivery/file route.
    _, platform_tree = _read_tree("gateway/platforms/base.py")
    platform_defs = _functions(platform_tree)
    platform_names = {
        name
        for name in platform_defs
        if name in {"connect", "send", "edit_message", "delete_message"}
        or name.startswith("cache_")
        or name.startswith("send_")
    }
    if not platform_names:
        failures.append("gateway/platforms/base.py has no platform sink methods")
    for name in sorted(platform_names):
        for function in platform_defs.get(name, ()):
            if not _has_refusal_prefix(function):
                failures.append(
                    f"gateway/platforms/base.py:{name}: not an explicit F refusal"
                )

    # Public direct kanban_db mutators are the one existing sink family the
    # writer may call.  No newly added public function with a mutation-shaped
    # sink may bypass this partition.
    _, db_tree = _read_tree("hermes_cli/kanban_db.py")
    db_defs = _functions(db_tree)
    if not DB_FIXED_APIS.isdisjoint(DB_REFUSED_APIS):
        failures.append("kanban_db.py fixed/refused partitions overlap")
    for name in sorted(DB_FIXED_APIS | DB_REFUSED_APIS):
        if name not in db_defs:
            failures.append(f"kanban_db.py:{name} missing from the partition")

    mutation_tokens = (
        "write_txn",
        "executescript",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "mkdir",
        "subprocess",
    )
    for name, functions in db_defs.items():
        if name.startswith("_"):
            continue
        for function in functions:
            source = ast.unparse(function)
            if any(token in source for token in mutation_tokens):
                if name not in DB_FIXED_APIS | DB_REFUSED_APIS:
                    failures.append(
                        f"kanban_db.py:{name} has an unclassified direct mutator"
                    )

    # No route adapter may introduce another SQLite connection, raw SQL
    # executor, or file/subprocess sink.  kanban_db.py is the sole existing
    # sink module; kanban_writer.py may call only its canonical kb.connect.
    adapter_files = ROUTE_FILES - {"hermes_cli/kanban_db.py"}
    for relative in sorted(adapter_files):
        names = _functions(_read_tree(relative)[1])
        relevant_names = set(FIXED_ROUTES.get(relative, ())) | set(
            REFUSED_ROUTES.get(relative, ())
        )
        if relative == "hermes_cli/kanban.py":
            # This module is the legacy CLI's direct-sink hotspot.  Include
            # every command/board helper by identity so a newly added handler
            # cannot hide behind the dispatcher guard.
            relevant_names |= {
                name
                for name in names
                if name.startswith("_cmd_")
                or name.startswith("_board_")
                or name == "_ready_queue_nonempty"
            }
        if relative == "gateway/platforms/base.py":
            relevant_names |= {
                name
                for name in names
                if name in {"connect", "edit_message", "delete_message"}
                or name.startswith("cache_")
                or name in {"send"}
                or name.startswith("send_")
            }
        for owner in sorted(relevant_names):
            if owner in set(FIXED_ROUTES.get(relative, ())) | set(
                REFUSED_ROUTES.get(relative, ())
            ):
                continue
            for function in names.get(owner, ()):
                direct = _sink_calls(function)
                if direct:
                    allowed_refusal = owner in REFUSED_ROUTES.get(relative, ())
                    if relative == "gateway/platforms/base.py":
                        allowed_refusal = True
                    if not allowed_refusal:
                        failures.append(
                            f"{relative}:{owner}: unclassified direct sink(s): "
                            + ", ".join(
                                f"{_call_name(call)}@{call.lineno}" for call in direct
                            )
                        )

    writer_path, writer_tree = _read_tree("hermes_cli/kanban_writer.py")
    writer_classes = [
        node.name for node in writer_tree.body if isinstance(node, ast.ClassDef)
    ]
    if writer_classes.count("KanbanWriter") != 1:
        failures.append("kanban_writer.py must define exactly one KanbanWriter")
    if writer_classes.count("KanbanWriterServer") != 1:
        failures.append(
            "kanban_writer.py must define exactly one KanbanWriterServer"
        )
    if any(
        isinstance(node, ast.Call)
        and _call_name(node) in {"sqlite3.connect", "sqlite3.Connection"}
        for node in ast.walk(writer_tree)
    ):
        failures.append(f"{writer_path} contains a direct sqlite3 connection")
    writer_text = writer_path.read_text(encoding="utf-8")
    for forbidden in ("sidecar", "policy_store", "receipt_store"):
        if forbidden in writer_text.casefold():
            failures.append(f"kanban_writer.py contains forbidden {forbidden} text")

    if failures:
        ordered = sorted(set(failures))
        pytest.fail(f"census failures ({len(ordered)}): " + " | ".join(ordered))
