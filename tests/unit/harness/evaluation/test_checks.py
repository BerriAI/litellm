from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm import Agent, EvalStatus, EvalTarget, EvalTask, JudgeCheck, PredicateCheck, aagent_eval
from litellm.harness.config import UNSET, Unset
from litellm.harness.options import ToolLoopOptions
from litellm.harness.sandbox import local
from litellm.harness.types import Approval, Harness, Usage
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


def _completion_response(content: str, calls: Sequence[Mapping[str, object]] = ()) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "agent-reply",
            "object": "chat.completion",
            "created": 0,
            "model": "eval-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        **({"tool_calls": list(calls)} if calls else {}),
                    },
                    "finish_reason": "tool_calls" if calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    )


@pytest.mark.parametrize("override_connection", (False, True))
async def test_should_reuse_agent_instructions_output_connection_and_metadata_in_fresh_trials(
    monkeypatch: pytest.MonkeyPatch, override_connection: bool
) -> None:
    class Classification(BaseModel):
        model_config = ConfigDict(frozen=True)
        category: str

    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        return _completion_response('{"category":"billing"}')

    definition: Final = Agent(
        Harness.TOOL_LOOP,
        model="litellm_proxy/eval-agent",
        instructions="Classify support tickets. Reply with the category as JSON.",
        output=Classification,
        api_base="https://agent.example/v1",
        api_key="definition-key",
        metadata={"agent": "support", "precedence": "agent", "agent_eval_trial": "discard"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        report: Final = await aagent_eval(
            (
                EvalTask(
                    "billing",
                    "I was charged twice.",
                    check=PredicateCheck(
                        lambda result: isinstance(result.output, Classification) and result.output.category == "billing"
                    ),
                ),
            ),
            agent=definition,
            sandbox=local,
            trials=2,
            metadata={"precedence": "suite", "agent_eval_run": "discard"},
            api_base="https://suite.example/v1" if override_connection else None,
            api_key="suite-key" if override_connection else None,
        )

    assert tuple(run.status for run in report.runs) == (EvalStatus.PASS, EvalStatus.PASS)
    assert len({run.result.session_id for run in report.runs if run.result is not None}) == 2
    for run in report.runs:
        request: Final = requests.get_nowait()
        assert request.url.host == ("suite.example" if override_connection else "agent.example")
        assert request.headers["authorization"] == f"Bearer {'suite-key' if override_connection else 'definition-key'}"
        payload: Final = _OBJECT.validate_json(request.content)
        assert _MESSAGES.validate_python(payload["messages"]) == (
            {"role": "system", "content": definition.instructions},
            {"role": "user", "content": "I was charged twice."},
        )
        assert payload["response_format"]
        assert _OBJECT.validate_json(request.headers["x-litellm-spend-logs-metadata"]) == {
            "agent": "support",
            "precedence": "suite",
            "agent_eval_suite": report.suite_id,
            "agent_eval_run": run.run_id,
            "agent_eval_task": "billing",
            "agent_eval_target": run.target,
            "agent_eval_trial": run.trial,
        }
    assert requests.empty()
    assert definition.metadata["precedence"] == "agent"
    assert "definition-key" not in report.to_json() and "suite-key" not in report.to_json()


@pytest.mark.parametrize("limit", (UNSET, 2, None))
async def test_should_preserve_tools_approvals_and_agent_turn_limits_unless_overridden(
    monkeypatch: pytest.MonkeyPatch, limit: int | None | Unset
) -> None:
    executed: Final[SimpleQueue[str]] = SimpleQueue()
    approvals: Final[SimpleQueue[str]] = SimpleQueue()
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def lookup_account() -> str:
        executed.put("lookup_account")
        return "billing"

    def deny(approval: Approval) -> bool:
        approvals.put(approval.tool)
        return False

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        payload: Final = _OBJECT.validate_json(request.content)
        messages: Final = _MESSAGES.validate_python(payload["messages"])
        if any(message["role"] == "tool" for message in messages):
            return _completion_response("done")
        return _completion_response(
            "",
            tuple(
                {"id": name, "type": "function", "function": {"name": "lookup_account", "arguments": "{}"}}
                for name in ("first", "second")
            ),
        )

    definition: Final = Agent(
        Harness.TOOL_LOOP,
        model="litellm_proxy/eval-agent",
        api_base="https://agent.example/v1",
        api_key="definition-key",
        tools=(lookup_account,),
        permissions="ask",
        on_approval=deny,
        max_turns=1,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        report: Final = await aagent_eval(
            (EvalTask("account", "Lookup my account", check=PredicateCheck(lambda result: result.text == "done")),),
            agent=definition,
            sandbox=local,
            max_turns=limit,
        )

    run: Final = report.runs[0]
    assert run.status is (EvalStatus.FAIL if limit is UNSET else EvalStatus.PASS)
    assert run.stop_reason == ("max_turns" if limit is UNSET else "done")
    assert executed.empty()
    assert approvals.qsize() == (1 if limit is UNSET else 2)
    assert _OBJECT.validate_json(requests.get_nowait().content)["tools"]
    assert definition.max_turns == 1


async def test_should_compare_named_agent_definitions_using_their_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = _OBJECT.validate_json(request.content)
        return _completion_response("billing" if payload["model"] == "production" else "technical")

    production: Final = Agent(
        Harness.TOOL_LOOP,
        model="litellm_proxy/production",
        instructions="Classify the ticket.",
        api_base="https://agent.example/v1",
        api_key="definition-key",
    )
    candidate: Final = Agent(
        Harness.TOOL_LOOP,
        model="litellm_proxy/candidate",
        instructions="Classify the ticket.",
        api_base="https://agent.example/v1",
        api_key="definition-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        report: Final = await aagent_eval(
            (EvalTask("billing", "Charged twice", check=PredicateCheck(lambda result: result.text == "billing")),),
            (EvalTarget(agent=production, label="production"), EvalTarget(agent=candidate, label="candidate")),
            sandbox=local,
            trials=2,
        )

    assert tuple((summary.target, summary.passed, summary.total) for summary in report.summaries) == (
        ("production", 2, 2),
        ("candidate", 0, 2),
    )


async def test_should_use_the_agent_connection_for_its_judge_when_the_suite_omits_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        payload: Final = _OBJECT.validate_json(request.content)
        return _completion_response(
            '{"passed":true,"reason":"The answer follows the rubric."}'
            if payload["model"] == "eval-judge"
            else "billing"
        )

    definition: Final = Agent(
        Harness.TOOL_LOOP,
        model="litellm_proxy/eval-agent",
        api_base="https://agent.example/v1",
        api_key="definition-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        report: Final = await aagent_eval(
            (EvalTask("billing", "Charged twice", check=JudgeCheck("Choose billing", "litellm_proxy/eval-judge")),),
            agent=definition,
            sandbox=local,
        )

    assert report.runs[0].status is EvalStatus.PASS
    agent_request: Final = requests.get_nowait()
    judge_request: Final = requests.get_nowait()
    assert agent_request.url == judge_request.url
    assert judge_request.url.host == "agent.example"
    assert judge_request.headers["authorization"] == "Bearer definition-key"
    assert _OBJECT.validate_json(judge_request.content)["model"] == "eval-judge"
    assert requests.empty()


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
            sandbox=local,
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
            sandbox=local,
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
            sandbox=local,
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
