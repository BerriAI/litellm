from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from cli_harness.rate_limiter import RateBucket, RateLimiter
from codex.harness import CodexHarness
from codex.rollout import (
    Compaction,
    Subagent,
    rollout_compactions,
    rollout_final_messages,
    rollout_parent_thread,
    rollout_subagent,
)

THREAD_ID: Final = "01a18fea-b2e8-7ed2-a29b-1ce4d2b138c3"
USER_PROMPT: Final = {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "Run `cat secret.txt` and remember the value it prints."}],
}


def _line(record_type: str, payload: Mapping[str, object]) -> str:
    return json.dumps({"timestamp": "2026-10-31T03:03:32.000Z", "type": record_type, "payload": payload})


def _rollout(*compactions: Mapping[str, object]) -> str:
    return "\n".join(
        (
            _line("session_meta", {"id": THREAD_ID, "cli_version": "0.165.0"}),
            _line("response_item", USER_PROMPT),
            *(_line("compacted", compaction) for compaction in compactions),
            _line("event_msg", {"type": "task_complete"}),
        )
    )


REMOTE: Final = {
    "message": "",
    "replacement_history": [USER_PROMPT, {"type": "compaction", "encrypted_content": "gAAAAABq5VrT"}],
}
LOCAL: Final = {
    "message": "The user asked to read secret.txt, which printed 4a6e834a61f8d792.",
    "replacement_history": [
        USER_PROMPT,
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Summary: 4a6e834a61f8d792"}]},
    ],
}


class TestRolloutCompactions:
    def test_a_compaction_that_kept_the_providers_compaction_item_is_remote(self) -> None:
        assert rollout_compactions(_rollout(REMOTE)) == (Compaction(remote=True),)

    def test_a_compaction_that_codex_summarized_itself_is_local(self) -> None:
        assert rollout_compactions(_rollout(LOCAL, {"message": "an older rollout with no replacement history"})) == (
            Compaction(remote=False),
            Compaction(remote=False),
        )

    def test_a_rollout_without_compactions_has_none(self) -> None:
        assert rollout_compactions(_rollout()) == ()


def _harness(home: Path) -> CodexHarness:
    return CodexHarness(
        home=home,
        api_key="unused",
        rate_bucket=RateBucket.OPENAI,
        limiter=RateLimiter({}, home, time.monotonic, time.sleep),
    )


def _subagent_rollout(
    *, thread_id: str, parent: str | None, final_messages: tuple[str | None, ...], forked: bool = False
) -> str:
    meta: Final = (
        {"id": thread_id, "cli_version": "0.165.0"}
        | ({"parent_thread_id": parent} if parent else {})
        | ({"forked_from_id": parent} if forked else {})
    )
    inherited: Final = (_line("session_meta", {"id": parent, "cli_version": "0.165.0"}),) if forked else ()
    return "\n".join(
        (
            _line("session_meta", meta),
            *inherited,
            _line("event_msg", {"type": "task_started", "turn_id": "t1"}),
            _line(
                "response_item",
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "thinking"}]},
            ),
            *(
                _line("event_msg", {"type": "task_complete", "turn_id": "t1", "last_agent_message": message})
                for message in final_messages
            ),
        )
    )


class TestRolloutSubagents:
    def test_a_subagent_rollout_names_the_thread_that_spawned_it(self) -> None:
        rollout: Final = _subagent_rollout(thread_id="child", parent=THREAD_ID, final_messages=())
        assert rollout_parent_thread(rollout) == THREAD_ID

    def test_a_top_level_rollout_has_no_parent(self) -> None:
        assert rollout_parent_thread(_rollout()) is None

    def test_a_subagent_that_inherited_its_parents_conversation_is_forked(self) -> None:
        rollout: Final = _subagent_rollout(thread_id="child", parent=THREAD_ID, final_messages=("hi",), forked=True)
        assert rollout_subagent(rollout) == Subagent(forked=True, replies=("hi",))

    def test_a_subagent_that_started_fresh_is_not_forked(self) -> None:
        rollout: Final = _subagent_rollout(thread_id="child", parent=THREAD_ID, final_messages=("hi",))
        assert rollout_subagent(rollout) == Subagent(forked=False, replies=("hi",))

    def test_the_final_messages_are_the_last_message_of_each_finished_turn(self) -> None:
        rollout: Final = _subagent_rollout(thread_id="child", parent=THREAD_ID, final_messages=("first", None, "last"))
        assert rollout_final_messages(rollout) == ("first", "last")


def test_the_harness_reads_only_the_subagents_of_the_thread_it_is_asked_about(tmp_path: Path) -> None:
    codex: Final = _harness(tmp_path)
    sessions: Final = codex.codex_home / "sessions" / "2026" / "11" / "06"
    sessions.mkdir(parents=True)
    rollouts: Final = {
        THREAD_ID: _subagent_rollout(thread_id=THREAD_ID, parent=None, final_messages=("the parent's answer",)),
        "child-a": _subagent_rollout(thread_id="child-a", parent=THREAD_ID, final_messages=("reply a",)),
        "child-b": _subagent_rollout(thread_id="child-b", parent=THREAD_ID, final_messages=("reply b",), forked=True),
        "stranger": _subagent_rollout(thread_id="stranger", parent="other-thread", final_messages=("not ours",)),
    }
    for thread_id, rollout in rollouts.items():
        _ = (sessions / f"rollout-2026-11-06T17-01-05-{thread_id}.jsonl").write_text(rollout)

    assert codex.subagents(THREAD_ID) == (
        Subagent(forked=False, replies=("reply a",)),
        Subagent(forked=True, replies=("reply b",)),
    )


def test_the_harness_reads_only_the_rollout_of_the_thread_it_is_asked_about(tmp_path: Path) -> None:
    codex: Final = _harness(tmp_path)
    sessions: Final = codex.codex_home / "sessions" / "2026" / "10" / "30"
    sessions.mkdir(parents=True)
    _ = (sessions / f"rollout-2026-10-30T20-03-32-{THREAD_ID}.jsonl").write_text(_rollout(REMOTE))
    _ = (sessions / "rollout-2026-10-30T20-04-00-01a18feb-9d3e-7ee1-b19f-e35d8894e5da.jsonl").write_text(
        _rollout(LOCAL)
    )

    assert codex.compactions(THREAD_ID) == (Compaction(remote=True),)
