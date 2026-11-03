from __future__ import annotations

import configparser
import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from codex.run import Command, codex_run

E2E_PYTEST_INI: Final = Path(__file__).resolve().parents[2] / "e2e" / "pytest.ini"

ROUND_TRIP: Final = "\n".join(
    (
        '{"type":"thread.started","thread_id":"01a18f9f-9d08-72e0-9be3-aa34cf7489d2"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"I will run it."}}',
        '{"type":"item.started","item":{"id":"item_1","type":"command_execution","command":"/bin/zsh -lc \'cat'
        ' secret.txt\'","aggregated_output":"","exit_code":null,"status":"in_progress"}}',
        '{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"/bin/zsh -lc \'cat'
        ' secret.txt\'","aggregated_output":"marker-7f3a91\\n","exit_code":0,"status":"completed"}}',
        '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"marker-7f3a91"}}',
        '{"type":"turn.completed","usage":{"input_tokens":22547,"cached_input_tokens":11227,'
        '"cache_write_input_tokens":11314,"output_tokens":67,"reasoning_output_tokens":0}}',
    )
)


def _line(event: Mapping[str, object]) -> str:
    return json.dumps(event)


class TestCodexRun:
    def test_the_answer_is_the_last_agent_message_and_each_finished_command_keeps_its_output(self) -> None:
        run: Final = codex_run(exit_code=0, stdout=ROUND_TRIP, stderr="")

        assert run.thread_id == "01a18f9f-9d08-72e0-9be3-aa34cf7489d2"
        assert run.commands == (
            Command(command="/bin/zsh -lc 'cat secret.txt'", output="marker-7f3a91\n", exit_code=0),
        )
        assert run.answer == "marker-7f3a91"
        assert run.usage is not None and run.usage.input_tokens == 22547
        assert run.errors == ()

    def test_only_a_patch_that_applied_counts_as_patching_its_files(self) -> None:
        def patch(status: str, path: str) -> str:
            item: Final = {
                "id": path,
                "type": "file_change",
                "status": status,
                "changes": [{"path": path, "kind": "update"}],
            }
            return _line({"type": "item.completed", "item": item})

        run: Final = codex_run(
            exit_code=0, stdout="\n".join((patch("completed", "/w/app.py"), patch("failed", "/w/other.py"))), stderr=""
        )

        assert run.patched_files == (Path("/w/app.py"),)

    def test_a_turn_that_gave_up_on_a_5xx_is_rerun_by_the_suites_network_rule(self) -> None:
        failed: Final = _line(
            {"type": "turn.failed", "error": {"message": "exceeded retry limit, last status: 502 Bad Gateway"}}
        )

        run: Final = codex_run(exit_code=1, stdout=failed, stderr="")

        assert run.errors[0].status_code == 502
        rerun_patterns: Final = _only_rerun_patterns(E2E_PYTEST_INI)
        assert any(re.search(pattern, repr(run)) for pattern in rerun_patterns), rerun_patterns

    def test_an_event_or_item_type_it_does_not_model_is_skipped(self) -> None:
        newer_codex: Final = "\n".join(
            (
                _line({"type": "thread.goal_set", "goal": "ship it"}),
                _line({"type": "item.completed", "item": {"id": "item_9", "type": "todo_list", "items": []}}),
                ROUND_TRIP,
            )
        )

        assert codex_run(exit_code=0, stdout=newer_codex, stderr="") == codex_run(
            exit_code=0, stdout=ROUND_TRIP, stderr=""
        )


def _only_rerun_patterns(ini: Path) -> tuple[str, ...]:
    parser: Final = configparser.ConfigParser()
    _ = parser.read(ini)
    args: Final = shlex.split(parser["pytest"]["addopts"])
    return tuple(value for flag, value in zip(args, args[1:]) if flag == "--only-rerun")
