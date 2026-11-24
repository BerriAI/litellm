from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker
from models import SpendLogRow
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e

FAST_TIER: Final = "priority"
MIN_TURN_REQUESTS: Final = 2


class _ClientRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_tier: str | None = None


def _requested_tiers(rows: list[SpendLogRow]) -> tuple[str | None, ...]:
    assert all(row.proxy_server_request for row in rows), (
        "the gateway logged no request body, so this test needs store_prompts_in_spend_logs on",
        rows,
    )
    return tuple(_ClientRequest.model_validate(row.proxy_server_request).service_tier for row in rows)


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexFastMode:
    def test_answers_with_what_a_command_printed_under_the_real_model_name(
        self, real_name_codex: CodexHarness, real_name_setup: ModelSetup, proxy: ProxyClient
    ) -> None:
        offers_fast: Final = FAST_TIER in real_name_codex.service_tiers()[real_name_setup.model]
        marker: Final = unique_marker()
        workspace: Final = real_name_codex.new_workspace(files={"secret.txt": marker})
        run: Final = real_name_codex.exec(
            "Run `cat secret.txt`, then reply with exactly what it printed.",
            model=real_name_setup.model,
            cwd=workspace,
            fast=True,
        )
        assert run.exit_code == 0, run
        assert any(marker in command.output for command in run.commands), run
        assert marker in run.answer, run
        rows: Final = proxy.poll_logs_for_key(real_name_setup.api_key, min_rows=MIN_TURN_REQUESTS)
        assert len(rows) >= MIN_TURN_REQUESTS, (
            "the gateway logged fewer requests than one command round trip makes",
            rows,
        )
        tiers: Final = _requested_tiers(rows)
        if offers_fast:
            assert all(tier == FAST_TIER for tier in tiers), tiers
        else:
            assert FAST_TIER not in tiers, tiers
