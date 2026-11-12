import json
from datetime import datetime, timezone

import httpx
import pytest

from litellm.models.task_insights import TASK_INSIGHTS_ANALYSIS_TAG, PendingRequest, TaskDefinition
from litellm.task_insights.jev import MAX_JEV_OPTIONS, GatewayJevDecider

_AT = datetime(2026, 11, 11, tzinfo=timezone.utc)
_REQUEST = PendingRequest(request_id="r", text_hash="h", request_text="Fix the flaky test", context_text="CI work")


def _task(task_id: str) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id,
        name=f"Name {task_id}",
        description=f"Description {task_id}",
        inclusion_criteria=f"Includes {task_id}",
        exclusion_criteria=f"Excludes {task_id}",
        status="frozen",
        source="discovered",
        map_x=0,
        map_y=0,
        created_at=_AT,
    )


_TASKS = (_task("debug"), _task("docs"))


def _answer(choice: str, probabilities: dict[str, float], fit: float) -> dict[str, object]:
    return {
        "model": "jev-1.14.0",
        "answers": {
            "task": {"type": "choice", "choice": choice, "confidence": 0.8, "probabilities": probabilities},
            "fits": {"type": "noul", "noul": fit},
        },
        "usage": {"input_tokens": 400, "output_tokens": 30},
    }


async def _decide(body: dict[str, object] | None = None, status: int = 200, tasks=_TASKS):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body or {})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        decider = GatewayJevDecider(
            client, url="http://gateway/v1/systemone", api_key="sk-test", model="task-insights-decision"
        )
        return await decider.decide(tasks, _REQUEST), seen


@pytest.mark.asyncio
async def test_assigns_the_chosen_task_when_its_probability_and_the_fit_clear_the_thresholds():
    decision, seen = await _decide(_answer("debug", {"debug": 0.82, "docs": 0.18}, fit=0.9))

    assert (decision.outcome, decision.task_id, decision.top_probability, decision.fit_probability) == (
        "assigned",
        "debug",
        0.82,
        0.9,
    )
    assert decision.decision_model == "jev-1.14.0"
    sent = json.loads(seen[0].content)
    assert sent["model"] == "task-insights-decision"
    assert seen[0].headers["x-litellm-tags"] == TASK_INSIGHTS_ANALYSIS_TAG
    assert [entry["excludes"] for entry in sent["state"]["task_definitions"]] == ["Excludes debug", "Excludes docs"]
    assert sent["questions"]["task"]["criteria"].keys() == {"debug", "docs"}
    assert sent["questions"]["fits"]["type"] == "noul"


@pytest.mark.asyncio
async def test_a_low_fit_makes_the_request_unknown_but_keeps_the_probabilities():
    decision, _ = await _decide(_answer("debug", {"debug": 0.95, "docs": 0.05}, fit=0.2))

    assert (decision.outcome, decision.task_id) == ("unknown", None)
    assert decision.probabilities == {"debug": 0.95, "docs": 0.05}


@pytest.mark.asyncio
async def test_the_threshold_reads_the_chosen_options_probability_not_the_largest():
    decision, _ = await _decide(_answer("docs", {"debug": 0.6, "docs": 0.4}, fit=0.9))

    assert (decision.outcome, decision.top_probability) == ("unknown", 0.4)


@pytest.mark.asyncio
async def test_a_choice_outside_the_taxonomy_is_a_failed_decision():
    decision, _ = await _decide(_answer("refactor", {"refactor": 1.0}, fit=1.0))

    assert (decision.outcome, decision.error) == ("failed", "InvalidTaskChoice")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "status", "error"), [({}, 502, "HTTP502"), ({"answers": "nope"}, 200, "UnreadableDecision")]
)
async def test_gateway_errors_become_failed_decisions(body, status, error):
    decision, _ = await _decide(body, status=status)

    assert (decision.outcome, decision.error) == ("failed", error)


@pytest.mark.asyncio
async def test_an_oversized_taxonomy_is_rejected_without_a_call():
    decision, seen = await _decide(tasks=tuple(_task(str(index)) for index in range(MAX_JEV_OPTIONS + 1)))

    assert (decision.outcome, decision.error, seen) == ("failed", "TooManyTasks", [])
