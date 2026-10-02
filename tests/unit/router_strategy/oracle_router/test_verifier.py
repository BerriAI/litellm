"""Verifiers: LiteLLM guardrails and reported scores become the number the decision maker learns from."""

import pytest

from litellm.router_strategy.oracle_router.verifier import (
    GuardrailVerifier,
    ProgramOutcome,
    ReportedVerifier,
    Verifier,
    build_verifier,
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
    def __init__(self, blocks: bool) -> None:
        self.blocks = blocks
        self.seen: list[dict] = []

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        self.seen.append({"inputs": inputs, "input_type": input_type, "request_data": request_data})
        if self.blocks:
            raise ValueError("blocked")
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
