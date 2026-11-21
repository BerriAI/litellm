from typing import Final

import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.decisions.transformation import systemone_request_to_ir
from litellm.llms.nadir.decisions.transformation import NADIR_DECISIONS_CONFIG, nadir_bucket_body
from litellm.types.decisions import (
    DecisionsIRChoiceAnswer,
    DecisionsIRChoiceProbability,
    DecisionsIRMessages,
    DecisionsIRRequest,
    DecisionsIRState,
    DecisionsRequestBody,
    OpenAIDecisionInputMessage,
    UnsupportedDecisionsRequest,
)

_TIER_QUESTION: Final = {
    "tier": {
        "type": "choice",
        "instructions": "Which tier?",
        "criteria": {"SIMPLE": "facts", "MEDIUM": "edits", "COMPLEX": "design", "REASONING": "proofs"},
    }
}
_VERDICT: Final = {
    "bucket": "medium",
    "routing_tier": "complex",
    "confidence": 0.72,
    "probabilities": {"simple": 0.1, "medium": 0.6, "complex": 0.3},
    "selected_model": None,
    "decode": "complex_gate",
}


def _request(state: object = "What is 2+2?", questions: object = _TIER_QUESTION) -> DecisionsIRRequest:
    return systemone_request_to_ir(DecisionsRequestBody.model_validate({"state": state, "questions": questions}))


def test_routing_tier_wins_over_bucket_and_answers_on_the_question_s_own_choice_keys() -> None:
    parsed: Final = NADIR_DECISIONS_CONFIG.parse_response(_VERDICT, _request())
    assert parsed.answers == (
        DecisionsIRChoiceAnswer(
            choice="COMPLEX",
            confidence=0.72,
            probabilities=(
                DecisionsIRChoiceProbability(value="SIMPLE", probability=0.1),
                DecisionsIRChoiceProbability(value="MEDIUM", probability=0.6),
                DecisionsIRChoiceProbability(value="COMPLEX", probability=0.3),
            ),
        ),
    )


def test_bucket_is_used_without_routing_tier_and_confidence_falls_back_to_its_probability() -> None:
    lowercase: Final = _request(
        questions={"t": {"type": "choice", "criteria": {"simple": None, "medium": None, "complex": None}}}
    )
    answer: Final = NADIR_DECISIONS_CONFIG.parse_response(
        {"bucket": "simple", "confidence": None, "probabilities": {"simple": 0.8, "medium": 0.2}}, lowercase
    ).answers[0]
    assert isinstance(answer, DecisionsIRChoiceAnswer)
    assert (answer.choice, answer.confidence) == ("simple", 0.8)


def test_the_router_s_tier_answer_uses_canonical_labels() -> None:
    translated: Final = NADIR_DECISIONS_CONFIG.classifier_response(_VERDICT, "bucket")
    assert translated["model"] == "bucket"
    assert translated["answers"] == {
        "tier": {
            "type": "choice",
            "choice": "COMPLEX",
            "probabilities": {"SIMPLE": 0.1, "MEDIUM": 0.6, "COMPLEX": 0.3},
            "confidence": 0.72,
        }
    }
    assert (
        NADIR_DECISIONS_CONFIG.classifier_response({"routing_tier": "medium"}, "bucket")["answers"]["tier"][
            "confidence"
        ]
        == 0.0
    )


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"routing_tier": "reasoning"},
        {"routing_tier": "simple", "confidence": 1.5},
        {"routing_tier": "simple", "probabilities": {"simple": -0.1}},
    ],
)
def test_a_verdict_the_router_cannot_use_is_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _ = NADIR_DECISIONS_CONFIG.classifier_response(body, "bucket")
    with pytest.raises(ValidationError):
        _ = NADIR_DECISIONS_CONFIG.parse_response(body, _request())


@pytest.mark.parametrize(
    "questions",
    [
        {"q": {"type": "noul", "instructions": "Is this hard?"}},
        {"q": {"type": "choice", "criteria": {"simple": None, "complex": None}}},
        {"q": {"type": "choice", "criteria": {"easy": None, "hard": None, "harder": None}}},
        {**_TIER_QUESTION, "second": {"type": "choice", "criteria": {"simple": None, "medium": None, "complex": None}}},
    ],
)
def test_anything_but_one_tier_choice_question_is_refused_before_a_call(questions: object) -> None:
    refused: Final = NADIR_DECISIONS_CONFIG.transform_decisions_request(
        "bucket", _request(questions=questions), "nadir"
    )
    assert isinstance(refused, UnsupportedDecisionsRequest)
    assert "simple, medium and complex" in refused.reason


def test_the_health_check_asks_a_question_nadir_can_answer() -> None:
    request: Final = _request(questions=NADIR_DECISIONS_CONFIG.health_check_questions)
    assert not isinstance(
        NADIR_DECISIONS_CONFIG.transform_decisions_request("bucket", request, "nadir"), UnsupportedDecisionsRequest
    )


@pytest.mark.parametrize(
    ("decision_input", "expected"),
    [
        (DecisionsIRState(state="hi"), {"prompt": "hi"}),
        (
            DecisionsIRState(
                state={"messages": [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "hi"}]}
            ),
            {"messages": [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "hi"}]},
        ),
        (DecisionsIRState(state={"messages": []}), {"prompt": '{"messages": []}'}),
        (DecisionsIRState(state={"ticket": 7}), {"prompt": '{"ticket": 7}'}),
        (
            DecisionsIRMessages(
                messages=(
                    OpenAIDecisionInputMessage(content="first"),
                    OpenAIDecisionInputMessage.model_validate(
                        {"content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}
                    ),
                )
            ),
            {"messages": [{"role": "user", "content": "first"}, {"role": "user", "content": "a\nb"}]},
        ),
    ],
)
def test_the_bucket_body_carries_the_conversation_when_there_is_one(
    decision_input: DecisionsIRState | DecisionsIRMessages, expected: dict[str, object]
) -> None:
    assert nadir_bucket_body(decision_input) == {**expected, "source": "litellm"}


@pytest.mark.parametrize(
    "base",
    ["https://api.getnadir.com/v1", "https://api.getnadir.com/v1/", "https://api.getnadir.com", "https://n.test/"],
)
def test_bucket_url_accepts_the_documented_base_with_or_without_v1(base: str) -> None:
    url: Final = NADIR_DECISIONS_CONFIG.get_complete_url(base, "bucket")
    assert url.endswith("/v1/bucket")
    assert "/v1/v1/" not in url


def test_environment_key_only_goes_to_the_environment_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NADIR_API_BASE", raising=False)
    monkeypatch.setenv("NADIR_API_KEY", "ndr-env")
    connect: Final = NADIR_DECISIONS_CONFIG.resolve_connection

    assert connect(None, None) == ("https://api.getnadir.com/v1", "ndr-env")
    assert connect("https://api.getnadir.com/v1/", None) == ("https://api.getnadir.com/v1", "ndr-env")
    assert connect("https://collector.invalid", None) == ("https://collector.invalid", None)
    assert connect("https://nadir.internal", "ndr-own") == ("https://nadir.internal", "ndr-own")

    monkeypatch.setenv("NADIR_API_BASE", "https://nadir.internal/v1")
    assert connect(None, None) == ("https://nadir.internal/v1", "ndr-env")
    assert connect("https://api.getnadir.com/v1", None) == ("https://api.getnadir.com/v1", None)


def test_no_key_anywhere_is_an_anonymous_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NADIR_API_BASE", raising=False)
    monkeypatch.delenv("NADIR_API_KEY", raising=False)
    assert NADIR_DECISIONS_CONFIG.resolve_connection(None, None) == ("https://api.getnadir.com/v1", None)
    assert not NADIR_DECISIONS_CONFIG.api_key_required
