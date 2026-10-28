import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import (
    ir_to_systemone_request,
    ir_to_systemone_response,
    parse_systemone_response,
    systemone_request_to_ir,
)
from litellm.llms.openai.decisions.transformation import openai_request_to_ir
from litellm.types.decisions import (
    MAX_DECISION_QUESTIONS,
    DecisionsIRRefusal,
    DecisionsIRRequest,
    DecisionsIRScoreAnswer,
    DecisionsRequestBody,
    OpenAIDecisionRequestBody,
    UnsupportedDecisionsRequest,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_SYSTEMONE_BODY: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_OPENAI_BODY: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(OpenAIDecisionRequestBody)

_SYSTEMONE_REQUEST: Final[Mapping[str, object]] = {
    "state": {"ticket": 1234, "text": "Screen cracked"},
    "questions": {
        "damaged": {
            "type": "noul",
            "instructions": "Is the item damaged?",
            "criteria": {"true": "Visible damage", "false": None},
            "provider_field": "dropped",
        },
        "rubric_only": {"type": "noul", "criteria": {"true": {"signal": "refund"}}},
        "action": {
            "type": "choice",
            "instructions": {"policy": "refund-v2"},
            "criteria": {"refund": "Within 30 days", "escalate": None},
        },
        "severity": {"type": "score", "criteria": ["minor", {"label": "major"}]},
    },
}


def _openai_ir(raw: Mapping[str, object]) -> DecisionsIRRequest:
    return openai_request_to_ir(_OPENAI_BODY.validate_python(raw))


def test_a_systemone_request_reaches_a_systemone_provider_unchanged() -> None:
    ir: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python(_SYSTEMONE_REQUEST))

    assert ir_to_systemone_request("jev-latest", ir) == {"model": "jev-latest", **_SYSTEMONE_REQUEST}


@pytest.mark.parametrize(
    ("names", "keys"),
    (
        (("damaged", "action"), ("damaged", "action")),
        (("damaged", None), ("0", "1")),
        (("damaged", "damaged"), ("0", "1")),
    ),
    ids=("all_named", "one_unnamed", "duplicate_names"),
)
def test_openai_questions_are_keyed_by_name_only_when_every_name_is_unique(
    names: tuple[str | None, str | None], keys: tuple[str, str]
) -> None:
    questions: Final = [
        {"type": "predicate", "instructions": f"Question {index}?", **({} if name is None else {"name": name})}
        for index, name in enumerate(names)
    ]

    body: Final = ir_to_systemone_request("jev-latest", _openai_ir({"input": "review", "questions": questions}))

    assert tuple(_SYSTEMONE_BODY.validate_python(body).questions) == keys


def test_openai_messages_become_systemone_state_text() -> None:
    ir: Final = _openai_ir(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "The package arrived broken."},
                        {"type": "input_text", "text": "I want a refund."},
                    ],
                },
                {"role": "user", "content": "Order 1234."},
            ],
            "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
        }
    )

    body: Final = ir_to_systemone_request("jev-latest", ir)

    assert body["state"] == "The package arrived broken.\n\nI want a refund.\n\nOrder 1234."


def test_openai_input_images_reach_a_systemone_provider_as_its_images_in_message_order() -> None:
    ir: Final = _openai_ir(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is the screen cracked?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"},
                    ],
                },
                {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/webp;base64,BB=="}]},
                {"role": "user", "content": "Order 1234."},
            ],
            "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
        }
    )

    body: Final = ir_to_systemone_request("clef", ir)

    assert body["state"] == "Is the screen cracked?\n\nOrder 1234."
    assert body["images"] == ["data:image/png;base64,AA==", "data:image/webp;base64,BB=="]


def test_systemone_images_reach_a_systemone_provider_in_the_callers_form() -> None:
    images: Final = ["data:image/png;base64,AA==", {"content_type": "image/jpeg", "base64": "BB=="}]
    ir: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python({**_SYSTEMONE_REQUEST, "images": images}))

    assert ir_to_systemone_request("clef", ir) == {"model": "clef", **_SYSTEMONE_REQUEST, "images": images}


def test_the_largest_openai_request_accepted_translates_to_a_valid_systemone_request() -> None:
    ir: Final = _openai_ir(
        {
            "input": "review",
            "questions": [
                {"type": "predicate", "instructions": f"Question {index}?"} for index in range(MAX_DECISION_QUESTIONS)
            ],
        }
    )

    translated: Final = _SYSTEMONE_BODY.validate_python(ir_to_systemone_request("jev-latest", ir))

    assert len(translated.questions) == MAX_DECISION_QUESTIONS


def test_a_systemone_response_reaches_the_caller_unchanged_with_provider_extras() -> None:
    payload: Final = {
        "model": "jev-latest",
        "answers": {
            "damaged": {"type": "noul", "noul": 0.95, "rationale": "crack visible"},
            "action": {
                "type": "choice",
                "choice": "refund",
                "confidence": 0.8,
                "probabilities": {"refund": 0.9, "escalate": 0.1},
                "calibrated": True,
            },
            "severity": {
                "type": "score",
                "score": 0.4,
                "confidence": 0.6,
                "legend": {"0": "minor", "1": {"label": "major"}},
                "probabilities": {"0": 0.6, "1": 0.4},
                "raw_logits": [0.1, 0.2],
            },
        },
        "usage": {"input_tokens": 383, "output_tokens": 2, "cost": 0.25},
        "latency_ms": 3722.17,
    }
    ir: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python(_SYSTEMONE_REQUEST))

    response: Final = ir_to_systemone_response(parse_systemone_response(payload, ir), ir)

    assert response.model_dump(mode="json") == payload


def test_missing_or_mismatched_systemone_answers_are_refusals_left_out_of_the_response() -> None:
    ir: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python(_SYSTEMONE_REQUEST))
    payload: Final = {
        "model": "jev-latest",
        "answers": {
            "damaged": {"type": "noul", "noul": 0.95},
            "action": {"type": "noul", "noul": 0.5},
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }

    parsed: Final = parse_systemone_response(payload, ir)

    assert parsed.answers[1:] == (DecisionsIRRefusal(), DecisionsIRRefusal(), DecisionsIRRefusal())
    assert tuple(ir_to_systemone_response(parsed, ir).answers) == ("damaged",)


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"model": "jev-latest", "usage": {"input_tokens": 10, "output_tokens": 1}}, ("answers",)),
        (
            {"model": "jev-latest", "answers": {"damaged": None}, "usage": {"input_tokens": 10, "output_tokens": 1}},
            ("answers", "damaged"),
        ),
    ],
    ids=["missing_answers", "null_answer"],
)
def test_a_malformed_systemone_response_names_the_field_at_fault(
    payload: Mapping[str, object], field: tuple[str, ...]
) -> None:
    ir: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python(_SYSTEMONE_REQUEST))

    with pytest.raises(ValidationError) as error:
        parse_systemone_response(payload, ir)

    assert [item["loc"] for item in error.value.errors()] == [field]


def test_each_media_kind_is_gated_by_its_own_support_flag() -> None:
    audio: Final = ["data:audio/wav;base64,YQ=="]
    videos: Final = ["data:video/mp4;base64,YQ=="]
    audio_only: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python({**_SYSTEMONE_REQUEST, "audio": audio}))
    video_only: Final = systemone_request_to_ir(
        _SYSTEMONE_BODY.validate_python({**_SYSTEMONE_REQUEST, "videos": videos})
    )

    assert ir_to_systemone_request("clef-omni", audio_only, supports_audio=True) == {
        "model": "clef-omni",
        **_SYSTEMONE_REQUEST,
        "audio": audio,
    }
    assert ir_to_systemone_request("clef-omni", video_only, supports_video=True) == {
        "model": "clef-omni",
        **_SYSTEMONE_REQUEST,
        "videos": videos,
    }
    assert isinstance(
        ir_to_systemone_request("clef-omni", audio_only, supports_video=True), UnsupportedDecisionsRequest
    )
    assert isinstance(
        ir_to_systemone_request("clef-omni", video_only, supports_audio=True), UnsupportedDecisionsRequest
    )


_ASSET_REQUEST: Final[Mapping[str, object]] = {
    "state": {"photo": "img-7f3a"},
    "questions": {"has_square": {"type": "noul", "instructions": "Is there a red square in `photo`?"}},
}


@pytest.mark.parametrize(
    "assets",
    [
        {"   ": {"type": "image", "format": "png", "data_base64": "iVBORw0KGgo="}},
        {"r" * 1025: {"type": "image", "format": "png", "data_base64": "iVBORw0KGgo="}},
        {"img-7f3a": {"type": "image", "format": "png", "data_base64": "not base64!"}},
    ],
    ids=["blank-reference", "reference-over-1024-bytes", "bad-base64"],
)
def test_assets_typesafe_would_reject_are_rejected_by_validation(assets: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError):
        _SYSTEMONE_BODY.validate_python({**_ASSET_REQUEST, "assets": assets})


def test_an_asset_reference_of_exactly_1024_bytes_is_accepted() -> None:
    reference: Final = "r" * 1024
    body: Final = _SYSTEMONE_BODY.validate_python(
        {**_ASSET_REQUEST, "assets": {reference: {"type": "image", "format": "png", "data_base64": "iVBO"}}}
    )

    assert tuple(body.assets or {}) == (reference,)


def test_a_box_answer_without_confidence_reaches_the_caller_unchanged() -> None:
    request: Final = {**_ASSET_REQUEST, "questions": {"square": {"type": "box", "instructions": "the square"}}}
    payload: Final = {
        "model": "jev-1.14.0",
        "answers": {
            "square": {
                "type": "box",
                "box": {"top": 0.5, "bottom": 0.75, "left": 0.25, "right": 0.5},
                "confidence": None,
            }
        },
        "usage": {"input_tokens": 737, "output_tokens": 20},
    }
    ir: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python(request))

    response: Final = ir_to_systemone_response(parse_systemone_response(payload, ir), ir)

    assert response.model_dump(mode="json") == payload


@pytest.mark.parametrize("provider", ("nace", "openrouter", "fireworks_ai", "typesafe"))
@pytest.mark.parametrize("legend", (["low", "high"], {"0": "low", "1": "high"}))
def test_score_legends_preserve_labels_probabilities_and_metadata(provider: str, legend: object) -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("drex-v2.0", LlmProviders(provider))
    assert config is not None
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": {"text": "The order arrived damaged"},
                "questions": {"severity": {"type": "score", "criteria": ["low", "high"]}},
            }
        )
    )
    payload: Final = {
        "model": "drex-v2.0",
        "answers": {
            "severity": {
                "type": "score",
                "score": 0.75,
                "confidence": 0.8,
                "legend": legend,
                "probabilities": {"0": 0.25, "1": 0.75},
            }
        },
        "usage": {"input_tokens": 20, "output_tokens": 30},
        "request_id": "test-request",
        "evaluation_time_ms": 8.5,
    }
    original: Final = json.dumps(payload)
    response: Final = config.parse_response(payload, request)
    answer: Final = response.answers[0]
    assert isinstance(answer, DecisionsIRScoreAnswer)
    assert (answer.score, answer.confidence) == (0.75, 0.8)
    assert tuple((item.value, item.label, item.probability) for item in answer.probabilities) == (
        (0, "low", 0.25),
        (1, "high", 0.75),
    )
    assert response.model == payload["model"]
    assert (response.usage.input_tokens, response.usage.output_tokens) == (20, 30)
    assert response.extra == {"request_id": "test-request", "evaluation_time_ms": 8.5}
    assert json.dumps(payload) == original


@pytest.mark.parametrize(
    "payload", ([], {"answers": []}, {"answers": {"a": None}}, {"answers": {"a": {"type": "score", "legend": [1]}}})
)
@pytest.mark.parametrize("provider", ("nace", "openrouter", "fireworks_ai", "typesafe"))
def test_invalid_provider_response_is_an_api_error(payload: object, provider: str) -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("drex-v2.0", LlmProviders(provider))
    assert config is not None
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "text",
                "questions": {"a": {"type": "score", "criteria": ["low", "high"]}},
            }
        )
    )
    with pytest.raises(BaseLLMException, match="unexpected response") as error:
        config.transform_decisions_response("drex-v2.0", provider, httpx.Response(200, json=payload), request)
    assert error.value.status_code == 500


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    (
        ({}, ()),
        ({"1": 0.8}, ((1, "medium", 0.8),)),
        ({"0": 0.0, "1": 0.8}, ((0, "low", 0.0), (1, "medium", 0.8))),
        ({"2": 0.0}, ((2, "high", 0.0),)),
    ),
    ids=("empty", "missing", "explicit_zero", "only_zero"),
)
def test_score_probabilities_preserve_missing_entries_and_explicit_zeros(
    probabilities: Mapping[str, float], expected: tuple[tuple[int, str, float], ...]
) -> None:
    ir: Final = systemone_request_to_ir(
        _SYSTEMONE_BODY.validate_python(
            {
                "state": "Classify the fixture.",
                "questions": {"score": {"type": "score", "criteria": ["low", "medium", "high"]}},
            }
        )
    )
    payload: Final = {
        "model": "fixture",
        "answers": {
            "score": {
                "type": "score",
                "score": 1,
                "confidence": 0.8,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

    response: Final = parse_systemone_response(payload, ir)
    answer: Final = response.answers[0]

    assert isinstance(answer, DecisionsIRScoreAnswer)
    assert tuple((item.value, item.label, item.probability) for item in answer.probabilities) == expected
    assert (
        ir_to_systemone_response(response, ir).model_dump(mode="json")["answers"]["score"]["probabilities"]
        == probabilities
    )


@pytest.mark.parametrize("raw_content", (b"", b"not-json", b'{"answers":'))
@pytest.mark.parametrize("provider", ("nace", "openrouter", "fireworks_ai", "typesafe", "openai"))
def test_non_json_provider_response_is_an_api_error(raw_content: bytes, provider: str) -> None:
    config: Final = ProviderConfigManager.get_provider_decisions_config("drex-v2.0", LlmProviders(provider))
    assert config is not None
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "text",
                "questions": {"a": {"type": "score", "criteria": ["low", "high"]}},
            }
        )
    )
    with pytest.raises(BaseLLMException, match="unexpected response") as error:
        config.transform_decisions_response("drex-v2.0", provider, httpx.Response(200, content=raw_content), request)
    assert error.value.status_code == 500
