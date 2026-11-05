from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from cli_harness.rate_limiter import RateBucket, RateLimiter
from codex.harness import CodexHarness
from codex.rollout import Compaction, rollout_compactions

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


def test_the_harness_reads_only_the_rollout_of_the_thread_it_is_asked_about(tmp_path: Path) -> None:
    codex: Final = CodexHarness(
        home=tmp_path,
        api_key="unused",
        rate_bucket=RateBucket.OPENAI,
        limiter=RateLimiter({}, tmp_path, time.monotonic, time.sleep),
    )
    sessions: Final = codex.codex_home / "sessions" / "2026" / "10" / "30"
    sessions.mkdir(parents=True)
    _ = (sessions / f"rollout-2026-10-30T20-03-32-{THREAD_ID}.jsonl").write_text(_rollout(REMOTE))
    _ = (sessions / "rollout-2026-10-30T20-04-00-01a18feb-9d3e-7ee1-b19f-e35d8894e5da.jsonl").write_text(
        _rollout(LOCAL)
    )

    assert codex.compactions(THREAD_ID) == (Compaction(remote=True),)
