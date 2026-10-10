from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.llms.base_llm.decisions.transformation import (
    ir_to_systemone_request,
    ir_to_systemone_response,
    parse_systemone_response,
    systemone_request_to_ir,
)
from litellm.llms.cloudflare.decisions.transformation import CloudflareDecisionsConfig
from litellm.llms.openai.decisions.transformation import openai_request_to_ir
from litellm.llms.typesafe.decisions.transformation import TypeSafeDecisionsConfig
from litellm.types.decisions import (
    MAX_DECISION_QUESTIONS,
    DecisionsIRRefusal,
    DecisionsIRRequest,
    DecisionsRequestBody,
    OpenAIDecisionRequestBody,
    UnsupportedDecisionsRequest,
)

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

    assert isinstance(body, Mapping)
    assert body["state"] == "The package arrived broken.\n\nI want a refund.\n\nOrder 1234."


def test_image_input_is_unsupported_by_systemone_providers() -> None:
    ir: Final = _openai_ir(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is the screen cracked?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    ],
                }
            ],
            "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
        }
    )

    assert isinstance(ir_to_systemone_request("jev-latest", ir), UnsupportedDecisionsRequest)


def test_images_reach_a_systemone_provider_that_supports_them_in_order() -> None:
    ir: Final = _openai_ir(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is the screen cracked?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    ],
                },
                {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64,BB=="}]},
                {"role": "user", "content": "Order 1234."},
            ],
            "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
        }
    )

    body: Final = CloudflareDecisionsConfig().transform_decisions_request(
        model="clef", request=ir, custom_llm_provider="cloudflare"
    )

    assert body == {
        "model": "clef",
        "state": "Is the screen cracked?\n\nOrder 1234.",
        "questions": {"0": {"type": "noul", "instructions": "Is this a defect?"}},
        "images": ["data:image/png;base64,AA==", "data:image/jpeg;base64,BB=="],
    }


def test_images_are_unsupported_by_a_default_systemone_provider() -> None:
    ir: Final = _openai_ir(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is the screen cracked?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    ],
                }
            ],
            "questions": [{"type": "predicate", "instructions": "Is this a defect?"}],
        }
    )

    body: Final = TypeSafeDecisionsConfig().transform_decisions_request(
        model="jev-latest", request=ir, custom_llm_provider="typesafe"
    )

    assert isinstance(body, UnsupportedDecisionsRequest)


def test_systemone_images_normalize_to_data_urls_in_order() -> None:
    ir: Final = systemone_request_to_ir(
        _SYSTEMONE_BODY.validate_python(
            {
                **_SYSTEMONE_REQUEST,
                "images": [
                    "data:image/png;base64,AA==",
                    {"content_type": "image/jpeg", "base64": "BB=="},
                ],
            }
        )
    )

    body: Final = ir_to_systemone_request("clef", ir, supports_images=True)

    assert isinstance(body, Mapping)
    assert body["images"] == ["data:image/png;base64,AA==", "data:image/jpeg;base64,BB=="]


def test_systemone_images_are_unsupported_by_a_default_systemone_provider() -> None:
    ir: Final = systemone_request_to_ir(
        _SYSTEMONE_BODY.validate_python({**_SYSTEMONE_REQUEST, "images": ["data:image/png;base64,AA=="]})
    )

    assert isinstance(ir_to_systemone_request("jev-latest", ir), UnsupportedDecisionsRequest)


def test_a_request_without_images_sends_no_images_key() -> None:
    ir: Final = systemone_request_to_ir(_SYSTEMONE_BODY.validate_python(_SYSTEMONE_REQUEST))

    body: Final = ir_to_systemone_request("clef", ir, supports_images=True)

    assert "images" not in body


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
