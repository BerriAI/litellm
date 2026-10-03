"""Verifiers: LiteLLM guardrails and reported scores become the number the decision maker learns from."""

import pytest
from fastapi import HTTPException

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import LLMAsAJudgeGuardrail
from litellm.router_strategy.oracle_router.verifier import (
    GuardrailVerifier,
    ProgramOutcome,
    ReportedVerifier,
    Verifier,
    build_verifier,
    clamp_score,
    recorded_status,
    response_text,
)
from litellm.types.router import OracleVerifierConfig
from litellm.types.utils import Choices, Message, ModelResponse


def _outcome(score=None, cost=0.0, answer="done", messages=()) -> ProgramOutcome:
    return ProgramOutcome(
        program_id="p", model="smart", prompt="task", response_text=answer, cost=cost, messages=messages, score=score
    )


@pytest.mark.asyncio
async def test_reported_verifier_uses_the_reported_score_or_its_default():
    assert await ReportedVerifier().verify(_outcome(score=0.8)) == 0.8
    assert await ReportedVerifier().verify(_outcome(score=4.0)) == 1.0
    assert await ReportedVerifier(default=0.25).verify(_outcome()) == 0.25


class _Guardrail:
    """A guardrail that blocks the way LiteLLM's do: by raising, with no status record."""

    def __init__(self, blocks: bool) -> None:
        self.blocks = blocks
        self.seen: list[dict] = []

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        self.seen.append({"inputs": inputs, "input_type": input_type, "request_data": request_data})
        if self.blocks:
            raise GuardrailRaisedException(guardrail_name="strict", message="blocked", blocked_content=True)
        return inputs


class _RecordingGuardrail:
    """A guardrail that reports through LiteLLM's status record, like llm_as_a_judge, and may also raise."""

    def __init__(self, status: str, error: Exception | None = None, name: str = "judge") -> None:
        self.status, self.error, self.name = status, error, name

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        self.seen = dict(request_data["metadata"])  # what the guardrail was handed, before it records anything
        records = request_data["metadata"].setdefault("standard_logging_guardrail_information", [])
        records.append({"guardrail_name": self.name, "guardrail_status": self.status})  # appended, like the real ones
        if self.error is not None:
            raise self.error
        return inputs


@pytest.mark.asyncio
async def test_guardrail_verifier_hands_the_judge_the_transcript_and_the_feedback_payload():
    passing, blocking = _Guardrail(blocks=False), _Guardrail(blocks=True)
    lookup = {"ok": passing, "strict": blocking}.get
    transcript = ({"role": "user", "content": "task"}, {"role": "assistant", "content": "final answer"})
    outcome = ProgramOutcome(
        program_id="p",
        model="smart",
        prompt="task",
        response_text="final answer",
        cost=0.0,
        messages=transcript,
        payload={"reference": "42"},
    )
    assert await GuardrailVerifier("ok", lookup=lookup).verify(outcome) == 1.0
    assert passing.seen == [
        {
            "inputs": {"texts": ["final answer"], "model": "smart", "structured_messages": list(transcript)},
            "input_type": "response",
            "request_data": {"model": "smart", "metadata": {"program_id": "p", "reference": "42"}},
        }
    ]
    assert await GuardrailVerifier("strict", lookup=lookup).verify(outcome) == 0.0
    with pytest.raises(LookupError):
        await GuardrailVerifier("missing", lookup=lookup).verify(outcome)


@pytest.mark.asyncio
async def test_guardrail_verifier_reads_the_recorded_verdict_before_the_exception():
    judge = {
        "passed": _RecordingGuardrail("success", name="passed"),
        "logged": _RecordingGuardrail("guardrail_intervened", name="logged"),  # on_failure: log -> no exception
        "blocked": _RecordingGuardrail(
            "guardrail_intervened", HTTPException(status_code=422, detail="judge"), "blocked"
        ),
        "flagged": _RecordingGuardrail("guardrail_flagged", name="flagged"),
    }
    assert await GuardrailVerifier("passed", lookup=judge.get).verify(_outcome()) == 1.0
    assert await GuardrailVerifier("logged", lookup=judge.get).verify(_outcome()) == 0.0
    assert await GuardrailVerifier("blocked", lookup=judge.get).verify(_outcome()) == 0.0
    assert await GuardrailVerifier("flagged", lookup=judge.get).verify(_outcome()) == 0.0


@pytest.mark.asyncio
async def test_feedback_payload_cannot_pre_seed_the_status_record_or_rename_the_program():
    judge = _RecordingGuardrail("guardrail_intervened", name="judge")  # on_failure: log -> no exception
    forged = {"standard_logging_guardrail_information": [0], "program_id": "someone-else", "container": "c1"}
    outcome = ProgramOutcome(
        program_id="p", model="smart", prompt="task", response_text="wrong", cost=0.0, messages=(), payload=forged
    )
    assert await GuardrailVerifier("judge", lookup={"judge": judge}.get).verify(outcome) == 0.0
    assert judge.seen == {"program_id": "p", "container": "c1"}


@pytest.mark.asyncio
async def test_guardrail_that_could_not_evaluate_is_a_failed_verification_not_a_pass():
    judge = {
        "down": _RecordingGuardrail("guardrail_failed_to_respond", name="down"),  # llm_as_a_judge fails open
        "skipped": _RecordingGuardrail("not_run", name="skipped"),
        "crashed": _RecordingGuardrail("success", RuntimeError("boom"), "crashed"),
    }
    with pytest.raises(RuntimeError, match="did not evaluate"):
        await GuardrailVerifier("down", lookup=judge.get).verify(_outcome())
    with pytest.raises(RuntimeError, match="not_run"):
        await GuardrailVerifier("skipped", lookup=judge.get).verify(_outcome())

    class _NoVerdict(_Guardrail):
        async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
            raise GuardrailRaisedException(guardrail_name="g", message="timeout", blocked_content=False)

    with pytest.raises(GuardrailRaisedException):
        await GuardrailVerifier("g", lookup={"g": _NoVerdict(blocks=True)}.get).verify(_outcome())
    assert await GuardrailVerifier("crashed", lookup=judge.get).verify(_outcome()) == 1.0  # the record wins


@pytest.mark.asyncio
async def test_guardrail_exceptions_without_a_record_are_classified():
    class _Raises:
        def __init__(self, error: Exception) -> None:
            self.error = error

        async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
            raise self.error

    lookup = {
        "http": _Raises(HTTPException(status_code=400, detail="Violated guardrail policy")),
        "infra": _Raises(ConnectionError("provider down")),
    }.get
    assert await GuardrailVerifier("http", lookup=lookup).verify(_outcome()) == 0.0
    with pytest.raises(ConnectionError):
        await GuardrailVerifier("infra", lookup=lookup).verify(_outcome())


def _judge(monkeypatch, on_failure: str, verdict) -> LLMAsAJudgeGuardrail:
    """The real llm_as_a_judge guardrail with only its judge-model call stubbed."""
    guardrail = LLMAsAJudgeGuardrail(
        guardrail_name="answer-judge",
        judge_model="judge",
        criteria=[{"name": "task_completed", "weight": 100, "description": "the task was completed"}],
        overall_threshold=50,
        on_failure=on_failure,
    )

    async def run_judge(messages, text_under_review, input_type="response"):
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    monkeypatch.setattr(guardrail, "_run_judge", run_judge)
    return guardrail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("on_failure", "overall_score", "expected"),
    [
        ("block", 90, 1.0),  # passes the threshold: recorded as success
        ("block", 10, 0.0),  # fails: the judge raises 422 and records guardrail_intervened
        ("log", 10, 0.0),  # fails but only logs: no exception, still recorded as guardrail_intervened
    ],
)
async def test_real_llm_as_a_judge_verdicts_become_scores(monkeypatch, on_failure, overall_score, expected):
    guardrail = _judge(monkeypatch, on_failure, {"overall_score": overall_score, "verdicts": []})
    verifier = GuardrailVerifier("answer-judge", lookup={"answer-judge": guardrail}.get)
    assert await verifier.verify(_outcome(answer="final answer")) == expected


@pytest.mark.asyncio
async def test_real_llm_as_a_judge_that_fails_open_is_not_a_pass(monkeypatch):
    guardrail = _judge(monkeypatch, "block", RuntimeError("judge model unreachable"))  # the guardrail returns normally
    verifier = GuardrailVerifier("answer-judge", lookup={"answer-judge": guardrail}.get)
    with pytest.raises(RuntimeError, match="did not evaluate"):
        await verifier.verify(_outcome(answer="final answer"))


def test_recorded_status_reads_both_buckets_and_takes_the_latest():
    data = {
        "metadata": {
            "standard_logging_guardrail_information": [{"guardrail_name": "a", "guardrail_status": "success"}]
        },
        "litellm_metadata": {
            "standard_logging_guardrail_information": {"guardrail_name": "a", "guardrail_status": "guardrail_flagged"}
        },
    }
    assert recorded_status(data, "a") == "success"  # a single dict, not a list, is not a valid record set
    data["litellm_metadata"]["standard_logging_guardrail_information"] = [
        {"guardrail_name": "a", "guardrail_status": "guardrail_intervened"}
    ]
    assert recorded_status(data, "a") == "guardrail_intervened"
    assert recorded_status(data, "other") is None
    assert recorded_status({"metadata": "junk"}, "a") is None


def test_clamp_score_rejects_non_finite_scores():
    assert clamp_score(1.7) == 1.0 and clamp_score(-2) == 0.0
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            clamp_score(bad)


@pytest.mark.asyncio
async def test_reported_verifier_rejects_a_nan_score():
    with pytest.raises(ValueError, match="finite"):
        await ReportedVerifier().verify(_outcome(score=float("nan")))


@pytest.mark.asyncio
async def test_guardrail_verifier_without_a_transcript_sends_the_initial_request():
    passing = _Guardrail(blocks=False)
    await GuardrailVerifier("ok", lookup={"ok": passing}.get).verify(_outcome())
    assert passing.seen[0]["inputs"]["structured_messages"] == [{"role": "user", "content": "task"}]


def test_response_text_reads_chat_completions_only():
    assert response_text(ModelResponse(choices=[Choices(message=Message(content="hi", role="assistant"))])) == "hi"
    assert response_text({"choices": []}) == ""
    assert response_text(None) == ""


class CustomVerifier:
    async def verify(self, outcome: ProgramOutcome) -> float:
        return 0.5


class NotAVerifier:
    pass


@pytest.mark.asyncio
async def test_build_verifier_by_type():
    assert isinstance(build_verifier(OracleVerifierConfig(type="reported")), ReportedVerifier)
    assert isinstance(build_verifier(OracleVerifierConfig(type="guardrail", guardrail="g")), GuardrailVerifier)
    custom = build_verifier(OracleVerifierConfig(type="custom", path=f"{__name__}:CustomVerifier"))
    assert isinstance(custom, Verifier) and await custom.verify(_outcome()) == 0.5
    with pytest.raises(TypeError):
        build_verifier(OracleVerifierConfig(type="custom", path=f"{__name__}:NotAVerifier"))
    with pytest.raises(ValueError, match=r"verifier\.guardrail"):
        OracleVerifierConfig(type="guardrail")
