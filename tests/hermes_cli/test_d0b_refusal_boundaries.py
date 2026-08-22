"""Behavior proof for the U1-D0B fail-closed persistence boundaries."""

from __future__ import annotations

from agent.turn_finalizer import finalize_turn
from tui_gateway import methods_session


class _ExplodingAgent:
    def __getattr__(self, name):
        raise AssertionError(f"finalize_turn reached agent effect surface {name}")


def test_finalize_turn_refuses_before_cleanup_or_persistence() -> None:
    result = finalize_turn(
        _ExplodingAgent(),
        final_response="must not be finalized",
        api_call_count=0,
        interrupted=False,
        failed=False,
        messages=[],
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
    )

    assert result == {
        "completed": False,
        "error": "finalize_turn refused by the D0B admission boundary",
        "final_response": None,
        "refused": True,
    }


def test_spawn_tree_handlers_refuse_before_path_or_file_effects() -> None:
    class _FakeServer:
        _methods = {}

        @staticmethod
        def _profile_scoped(function):
            return function

        @staticmethod
        def _d0b_tui_refused(route, rid):
            return {"refused": True, "route": route, "id": rid}

        @staticmethod
        def _spawn_tree_session_dir(*args, **kwargs):
            raise AssertionError("spawn-tree directory effect ran")

        @staticmethod
        def _append_spawn_tree_index(*args, **kwargs):
            raise AssertionError("spawn-tree index effect ran")

        @staticmethod
        def _spawn_trees_root(*args, **kwargs):
            raise AssertionError("spawn-tree root mkdir effect ran")

    server = _FakeServer()
    # HandlerRegistry rebinds against the instance namespace, so materialize
    # the fake server's class-level seams there just like server.py globals.
    server._methods = {}
    server._profile_scoped = server._profile_scoped
    server._d0b_tui_refused = server._d0b_tui_refused
    server._spawn_tree_session_dir = server._spawn_tree_session_dir
    server._append_spawn_tree_index = server._append_spawn_tree_index
    server._spawn_trees_root = server._spawn_trees_root
    methods_session.register(server)

    for route in ("spawn_tree.save", "spawn_tree.list", "spawn_tree.load"):
        assert server._methods[route]("rid-1", {"subagents": [{"id": "a"}]}) == {
            "refused": True,
            "route": route,
            "id": "rid-1",
        }
