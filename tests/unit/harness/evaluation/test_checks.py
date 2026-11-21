from __future__ import annotations

import asyncio
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm import EvalStatus, EvalTarget, EvalTask, JudgeCheck, PredicateCheck, aagent_eval
from litellm.harness.options import ToolLoopOptions
from litellm.harness.types import Harness, Usage
from litellm.types.utils import ModelResponse

_OBJECT: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = TypeAdapter(tuple[dict[str, object], ...])


def _agent_response() -> ModelResponse:
    return ModelResponse(
        id="agent-reply",
        model="eval-agent",
        choices=[{"message": {"role": "assistant", "content": "The answer meets the rubric."}}],
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )


async def test_should_use_gateway_cost_headers_when_reporting_agent_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(
            200,
            headers={"x-litellm-response-cost": "0.4"},
            json={
                "id": "agent-reply",
                "object": "chat.completion",
                "created": 0,
                "model": "eval-agent",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        report: Final = await aagent_eval(
            (EvalTask("answer", "Answer", check=PredicateCheck(lambda result: bool(result.text))),),
            (EvalTarget(Harness.TOOL_LOOP, "litellm_proxy/eval-agent", label="candidate"),),
            metadata={"experiment": "unit-eval"},
            api_base="https://gateway.example/v1",
            api_key="transport-only",
        )

    assert report.runs[0].status is EvalStatus.PASS
    assert report.runs[0].cost == 0.4
    assert report.runs[0].usage == Usage(input_tokens=11, output_tokens=7, calls=1)
    assert report.summaries[0].cost == 0.4
    request: Final = requests.get_nowait()
    metadata: Final = _OBJECT.validate_json(request.headers["x-litellm-spend-logs-metadata"])
    assert metadata == {
        "experiment": "unit-eval",
        "agent_eval_suite": report.suite_id,
        "agent_eval_run": report.runs[0].run_id,
        "agent_eval_task": "answer",
        "agent_eval_target": "candidate",
        "agent_eval_trial": 1,
    }
    assert requests.empty()


@pytest.mark.parametrize(
    ("verdict", "expected_status"),
    (
        ('{"passed": true, "reason": "The answer follows the rubric."}', EvalStatus.PASS),
        ('{"passed": false, "reason": "The answer omits a required detail."}', EvalStatus.FAIL),
        ('{"passed": "true", "reason": "Malformed boolean."}', EvalStatus.ERROR),
        ("This is not a JSON verdict.", EvalStatus.ERROR),
    ),
)
async def test_should_grade_with_an_explicit_judge_and_preserve_its_separate_usage(
    monkeypatch: pytest.MonkeyPatch, verdict: str, expected_status: EvalStatus
) -> None:
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return httpx.Response(
            200,
            headers={"x-litellm-response-cost": "0.125"},
            json={
                "id": "judge-reply",
                "object": "chat.completion",
                "created": 0,
                "model": "eval-judge",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": verdict},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        report: Final = await aagent_eval(
            (EvalTask("quality", "Answer", check=JudgeCheck("Follow the rubric", "openai/eval-judge")),),
            (
                EvalTarget(
                    Harness.TOOL_LOOP,
                    "openai/eval-agent",
                    label="candidate",
                    options=ToolLoopOptions(completion_kwargs={"mock_response": _agent_response()}),
                ),
            ),
            metadata={"experiment": "unit-eval", "agent_eval_trial": "cannot-override"},
            api_base="https://judge.example/v1",
            api_key="transport-only",
        )

    run: Final = report.runs[0]
    assert run.status is expected_status
    assert run.result is not None
    assert run.check is not None
    assert run.check.judge_usage == Usage(input_tokens=20, output_tokens=5, calls=1)
    assert run.check.judge_cost == 0.125
    assert report.summaries[0].judge_cost == 0.125
    assert report.summaries[0].cost == run.cost
    assert (run.check.error is not None) == (expected_status is EvalStatus.ERROR)

    request: Final = requests.get_nowait()
    metadata: Final = _OBJECT.validate_json(request.headers["x-litellm-spend-logs-metadata"])
    assert metadata == {
        "experiment": "unit-eval",
        "agent_eval_suite": report.suite_id,
        "agent_eval_run": run.run_id,
        "agent_eval_task": "quality",
        "agent_eval_target": "candidate",
        "agent_eval_trial": 1,
        "agent_eval_stage": "judge",
    }
    payload: Final = _OBJECT.validate_json(request.content)
    messages: Final = _MESSAGES.validate_python(payload["messages"])
    assert "Follow the rubric" in str(messages[0]["content"])
    answer: Final = _OBJECT.validate_json(str(messages[1]["content"]))
    assert answer["task"] == "Answer"
    assert answer["answer"] == run.result.text
    assert not payload.get("stream")
    assert requests.empty()
    assert "transport-only" not in report.to_json()
    assert "api_key" not in report.to_json()


@pytest.mark.parametrize("failure", ("provider", "timeout"))
async def test_should_report_unknown_judge_cost_after_provider_errors(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("judge request timed out", request=request)
        return httpx.Response(401, json={"error": {"message": "judge access denied", "type": "authentication_error"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        monkeypatch.setattr(litellm, "num_retries", 0)
        report: Final = await aagent_eval(
            (EvalTask("quality", "Answer", check=JudgeCheck("Follow the rubric", "openai/eval-judge")),),
            (
                EvalTarget(
                    Harness.TOOL_LOOP,
                    "openai/eval-agent",
                    options=ToolLoopOptions(completion_kwargs={"mock_response": _agent_response()}),
                ),
            ),
            api_base="https://unavailable-judge.example/v1",
            api_key="transport-only",
        )

    run: Final = report.runs[0]
    assert run.status is EvalStatus.ERROR
    assert run.result is not None
    assert run.result.text == "The answer meets the rubric."
    assert run.check is not None
    assert run.check.error is not None
    assert run.check.judge_cost is None
    assert not report.summaries[0].metrics_complete
    assert report.summaries[0].errors == 1
    assert report.summaries[0].cost == run.result.cost
    assert "Known subtotals" in report.to_markdown()
