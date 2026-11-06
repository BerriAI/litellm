from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.models.workflow_execution import EngineRun, RunInput, RunState, StepTrace
from litellm.repositories.workflow_execution_repository import _aggregate_attempts

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


def pending_run() -> EngineRun:
    return EngineRun(
        run_id="run", session_id="session", workflow_id="workflow", snapshot_id="snapshot", team_id="team",
        created_by="owner", status=RunState.pending, created_at=NOW, updated_at=NOW, request_digest="digest",
        input=RunInput(
            messages=({"role": "user", "content": "plan"},), deadline=NOW + timedelta(minutes=5), draft=True,
        ),
    )


@pytest.mark.parametrize("cost", [None, 0.0, 0.125])
def test_attempt_aggregation_keeps_unknown_cost_and_unique_receipts(cost: float | None) -> None:
    first: Final = StepTrace(
        step_id="step", run_id="run", node_id="node", state=RunState.failed, input=None, attempts=1,
        deadline=NOW, started_at=NOW, prompt_tokens=4, completion_tokens=2, response_cost=cost,
        request_ids=("fallback-a", "fallback-b"),
    )
    latest: Final = first.model_copy(update={"attempts": 2, "state": RunState.completed, "output": False,
                                           "request_ids": ("fallback-b", "retry"), "response_cost": None})
    assert _aggregate_attempts((latest, first)) == latest.model_copy(update={
        "prompt_tokens": 8, "completion_tokens": 4, "response_cost": cost,
        "request_ids": ("fallback-b", "retry", "fallback-a"),
    })
