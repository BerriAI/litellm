from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker
from models import SpendLogRow
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict, JsonValue

pytestmark = pytest.mark.e2e

EFFORT: Final = "high"
HIGH_EFFORT: Final = ("--config", f'model_reasoning_effort="{EFFORT}"')
MIN_TURN_REQUESTS: Final = 2


class _Reasoning(BaseModel):
    model_config = ConfigDict(frozen=True)

    effort: str | None = None


class _WithReasoning(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning: _Reasoning | None = None


def _requested_efforts(rows: list[SpendLogRow]) -> tuple[str | None, ...]:
    assert all(row.proxy_server_request for row in rows), (
        "the gateway logged no request body, so this test needs store_prompts_in_spend_logs on",
        rows,
    )
    return _efforts(row.proxy_server_request for row in rows)


def _efforts(bodies: Iterable[JsonValue]) -> tuple[str | None, ...]:
    parsed: Final = tuple(_WithReasoning.model_validate(body) for body in bodies)
    return tuple(body.reasoning.effort if body.reasoning else None for body in parsed)


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexReasoningEffort:
    def test_reasons_at_high_effort_over_numbers_it_read_through_the_gateway(
        self, codex: CodexHarness, setup: ModelSetup, proxy: ProxyClient
    ) -> None:
        numbers: Final = tuple(100 + int(unique_marker()[:4], 16) % 900 for _ in range(3))
        workspace: Final = codex.new_workspace(files={"numbers.txt": "\n".join(map(str, numbers)) + "\n"})
        run: Final = codex.exec_with_flags(
            "Run `cat numbers.txt` and no other command, then work out the sum of the squares of the numbers it"
            " printed yourself. Reply with that sum as digits only.",
            model=setup.model,
            cwd=workspace,
            flags=HIGH_EFFORT,
        )
        assert run.exit_code == 0, run
        assert str(sum(number * number for number in numbers)) in run.answer, run
        assert run.usage is not None and run.usage.reasoning_output_tokens > 0, run
        rows: Final = proxy.poll_logs_for_key(setup.api_key, min_rows=MIN_TURN_REQUESTS)
        assert len(rows) >= MIN_TURN_REQUESTS, (
            "the gateway logged fewer requests than one command round trip makes",
            rows,
        )
        requested: Final = _requested_efforts(rows)
        assert all(effort == EFFORT for effort in requested), requested
        applied: Final = _efforts(row.response for row in rows)
        assert all(effort in (EFFORT, None) for effort in applied), applied
