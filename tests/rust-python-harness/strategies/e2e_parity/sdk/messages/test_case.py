from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest

from .....shared.parity.replay import replay_server
from .....shared.reporting.models import Coverage, HarnessCase, RunStatus
from .....shared.reporting.strategy import ModuleCaseSpec
from ...runner import run_e2e_cases
from . import case


async def test_python_fallback_cannot_satisfy_messages_parity() -> None:
    route_rule: Final = case.RouteRule

    def python_only(route: case.Route, _rollout: case.Rollout) -> case.RouteRule:
        return route_rule(route, case.Rollout.PYTHON_ONLY)

    with (
        replay_server() as provider,
        patch.dict(os.environ, {"LITELLM_USER_AGENT": case.BASELINE_USER_AGENT}),
        patch.object(case, "RouteRule", python_only),
    ):
        try:
            with pytest.raises(AssertionError, match="fell back to Python"):
                await case.compare(provider, case.recorded_response(case.RESPONSE), {})
        finally:
            await case.GLOBAL_LOGGING_WORKER.stop()


def test_messages_parity_runs_against_the_native_extension() -> None:
    messages: Final = HarnessCase(
        strategy_id="e2e_parity",
        strategy_label="End-to-end parity",
        sdk_function="messages",
        surface="sdk",
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module=case.__name__),
    )

    code, run = run_e2e_cases((messages,), Path(__file__).parent, lambda _: None)

    assert code == 0, run.failures
    assert run.results[messages.key].status is RunStatus.PASSED
    assert len(run.results[messages.key].collected) == 4
