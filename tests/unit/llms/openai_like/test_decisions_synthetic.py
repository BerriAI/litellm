# Unit tests for the chat-synthetic Decisions provider (openai_like).

import math
from typing import Any, Final

import pytest

from litellm.llms.base_llm.decisions.transformation import systemone_request_to_ir
from litellm.llms.openai_like.decisions.transformation import (
    OpenAILikeDecisionsConfig,
    build_chat_messages,
    chat_response_to_ir,
)
from litellm.types.decisions import DecisionsIRRequest

_CHOICE_STATE: Final = "PR title: Add rate limiter. Diff touches gateway middleware only."


def _ir_request(body: dict[str, Any]) -> DecisionsIRRequest:
    from litellm.types.decisions import DecisionsRequestBody

    return systemone_request_to_ir(DecisionsRequestBody.model_validate(body))


def test_build_prompt_contains_letter_table() -> None:
    ir = _ir_request(
        {
            "state": _CHOICE_STATE,
            "questions": {
                "approve": {
                    "type": "choice",
                    "instructions": "Should this PR be approved?",
                    "criteria": {
                        "option_approve": "Changes are safe and tested",
                        "option_reject": "Risky or untested",
                    },
                }
            },
        }
    )
    messages = build_chat_messages(ir)
    assert not isinstance(messages, tuple) or len(messages) == 2
    system_prompt, state = messages  # type: ignore[misc]
    assert state == _CHOICE_STATE
    assert "A. option_approve" in system_prompt
    assert "B. option_reject" in system_prompt
    assert "Should this PR be approved?" in system_prompt
    assert "single letter" in system_prompt


def test_predicate_letters_are_yes_no() -> None:
    ir = _ir_request(
        {
            "state": "The sky is blue.",
            "questions": {"truth": {"type": "noul", "criteria": {"true": "factually correct"}}},
        }
    )
    system_prompt, _state = build_chat_messages(ir)  # type: ignore[misc]
    assert "A (yes/true)" in system_prompt
    assert "B (no/false)" in system_prompt


def test_more_than_26_options_is_unsupported() -> None:
    criteria = {f"option_{i}": f"description {i}" for i in range(27)}
    ir = _ir_request({"state": "x", "questions": {"q": {"type": "choice", "criteria": criteria}}})
    result = build_chat_messages(ir)
    assert hasattr(result, "reason") and "at most 26" in result.reason


def test_image_input_is_unsupported() -> None:
    from litellm.types.decisions import (
        DecisionsIRMessages,
        OpenAIDecisionInputImage,
        OpenAIDecisionInputMessage,
    )

    ir = DecisionsIRRequest(
        input=DecisionsIRMessages(
            messages=(
                OpenAIDecisionInputMessage(
                    role="user",
                    content=[OpenAIDecisionInputImage(type="input_image", image_url="https://x/y.png")],
                ),
            )
        ),
        questions=(),
    )
    result = build_chat_messages(ir)
    assert hasattr(result, "reason") and "text input only" in result.reason


def test_transform_request_fixed_params() -> None:
    ir = _ir_request(
        {
            "state": "Is it raining?",
            "questions": {"rain": {"type": "noul", "criteria": {"true": "raining"}}},
        }
    )
    config = OpenAILikeDecisionsConfig()
    body = config.transform_decisions_request(model="my-model", request=ir, custom_llm_provider="openai_like")
    assert isinstance(body, dict)
    assert body["model"] == "my-model"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 1  # single-question request stays at one token
    assert body["stream"] is False
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    messages = body["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Is it raining?"


def test_transform_request_multi_question_token_budget() -> None:
    ir = _ir_request(
        {
            "state": "context",
            "questions": {
                "a": {"type": "noul", "criteria": {"true": "t"}},
                "b": {"type": "choice", "criteria": {"x": "X", "y": "Y"}},
                "c": {"type": "noul", "criteria": {"false": "f"}},
            },
        }
    )
    config = OpenAILikeDecisionsConfig()
    body = config.transform_decisions_request(model="m", request=ir, custom_llm_provider="openai_like")
    assert isinstance(body, dict)
    assert body["max_tokens"] == 8 * 3


def test_choice_answer_with_logprobs() -> None:
    ir = _ir_request(
        {
            "state": "context",
            "questions": {
                "pick": {"type": "choice", "criteria": {"option_a": "first", "option_b": "second"}},
            },
        }
    )
    # top_logprobs: A=-0.2 (0.8187), B=-1.8 (0.1653) → normalized ≈ 0.832 / 0.168
    payload = {
        "model": "my-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "A"},
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.2},
                                {"token": "B", "logprob": -1.8},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 1},
    }
    response = chat_response_to_ir(payload, ir)
    assert len(response.answers) == 1
    answer = response.answers[0]
    assert answer.choice == "option_a"
    expected_a = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.8))
    assert answer.probabilities[0].probability == pytest.approx(expected_a)
    assert answer.probabilities[1].probability == pytest.approx(1.0 - expected_a)
    assert answer.confidence == pytest.approx(expected_a)
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 1


def test_choice_answer_without_logprobs_falls_back_to_point_mass() -> None:
    ir = _ir_request(
        {
            "state": "context",
            "questions": {"pick": {"type": "choice", "criteria": {"option_a": "first", "option_b": "second"}}},
        }
    )
    payload = {
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "B"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    response = chat_response_to_ir(payload, ir)
    answer = response.answers[0]
    assert answer.choice == "option_b"
    assert answer.confidence == 1.0
    assert [p.probability for p in answer.probabilities] == [0.0, 1.0]


def test_predicate_answer_maps_letter_to_probability() -> None:
    ir = _ir_request(
        {
            "state": "The sky is blue.",
            "questions": {"truth": {"type": "noul", "criteria": {"true": "fact"}}},
        }
    )
    payload = {
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "A"},
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.1},
                                {"token": "B", "logprob": -2.3},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
    }
    response = chat_response_to_ir(payload, ir)
    answer = response.answers[0]
    expected_true = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.3))
    assert answer.probability == pytest.approx(expected_true)


def test_score_answer_indexes_levels() -> None:
    ir = _ir_request(
        {
            "state": "essay text",
            "questions": {"grade": {"type": "score", "criteria": ["poor", "ok", "great"]}},
        }
    )
    payload = {
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "C"}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 1},
    }
    response = chat_response_to_ir(payload, ir)
    answer = response.answers[0]
    assert answer.score == 2.0
    assert [p.value for p in answer.probabilities] == [0, 1, 2]
    assert answer.probabilities[2].probability == 1.0


def test_multi_question_parses_letter_sequence() -> None:
    ir = _ir_request(
        {
            "state": "context",
            "questions": {
                "first": {"type": "noul", "criteria": {"true": "t"}},
                "second": {"type": "choice", "criteria": {"x": "X", "y": "Y"}},
            },
        }
    )
    payload = {
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "A B"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    response = chat_response_to_ir(payload, ir)
    assert response.answers[0].probability == 1.0
    assert response.answers[1].choice == "y"  # second letter (B) maps to the second option


def test_unparseable_output_yields_refusals() -> None:
    ir = _ir_request(
        {
            "state": "context",
            "questions": {"pick": {"type": "choice", "criteria": {"x": "X", "y": "Y"}}},
        }
    )
    payload = {
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "I cannot decide."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 4},
    }
    response = chat_response_to_ir(payload, ir)
    from litellm.types.decisions import DecisionsIRRefusal

    assert isinstance(response.answers[0], DecisionsIRRefusal)


def test_missing_choices_raises() -> None:
    ir = _ir_request({"state": "x", "questions": {"q": {"type": "noul", "criteria": {"true": "t"}}}})
    with pytest.raises(ValueError):
        chat_response_to_ir({"model": "m"}, ir)
