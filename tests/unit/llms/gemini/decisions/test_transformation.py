from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.base_llm.decisions.transformation import (
    DecisionsResponseValidationError,
    ir_to_systemone_response,
    systemone_request_to_ir,
)
from litellm.llms.gemini.decisions.transformation import GeminiDecisionsConfig
from litellm.llms.openai.decisions.transformation import ir_to_openai_response, openai_request_to_ir
from litellm.types.decisions import (
    DecisionsIRRequest,
    DecisionsRequestBody,
    OpenAIDecisionRequestBody,
    UnsupportedDecisionsRequest,
)


@pytest.fixture
def request_ir() -> DecisionsIRRequest:
    return systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": {"item": "cracked mug"},
                "questions": {
                    "damaged": {
                        "type": "noul",
                        "instructions": "Damaged?",
                        "criteria": {"true": "Broken", "false": "Intact"},
                    },
                    "action": {
                        "type": "choice",
                        "instructions": "Next?",
                        "criteria": {"replace": "Broken", "keep": None},
                    },
                    "severity": {"type": "score", "instructions": "Severity?", "criteria": ["None", "Severe"]},
                },
            }
        )
    )


def test_systemone_questions_and_json_state_become_gemini_input(request_ir: DecisionsIRRequest) -> None:
    assert GeminiDecisionsConfig().transform_decisions_request("gemini-decision-1", request_ir, "gemini") == {
        "model": "gemini-decision-1",
        "input": ({"type": "text", "text": '{"item": "cracked mug"}'},),
        "questions": {
            "damaged": {
                "type": "noul",
                "instructions": "Damaged?",
                "criteria": {"true_description": "Broken", "false_description": "Intact"},
            },
            "action": {"type": "choice", "instructions": "Next?", "criteria": {"replace": "Broken", "keep": ""}},
            "severity": {"type": "score", "instructions": "Severity?", "criteria": ["None", "Severe"]},
        },
    }


def test_openai_questions_preserve_boolean_values_and_score_descriptions() -> None:
    request: Final = openai_request_to_ir(
        OpenAIDecisionRequestBody.model_validate(
            {
                "model": "gemini/gemini-decision-1",
                "input": "mug",
                "questions": [
                    {"type": "predicate", "instructions": "Damaged?"},
                    {
                        "type": "choice",
                        "instructions": "Replace?",
                        "choices": [{"value": True}, {"value": False, "description": "Keep"}],
                    },
                    {
                        "type": "score",
                        "instructions": "Severity?",
                        "levels": [{"label": "None"}, {"label": "Severe", "description": "Unusable"}],
                    },
                ],
            }
        )
    )
    assert GeminiDecisionsConfig().transform_decisions_request("gemini-decision-1", request, "gemini") == {
        "model": "gemini-decision-1",
        "input": ({"type": "text", "text": "mug"},),
        "questions": {
            "0": {"type": "noul", "instructions": "Damaged?"},
            "1": {"type": "choice", "instructions": "Replace?", "criteria": {"true": "", "false": "Keep"}},
            "2": {"type": "score", "instructions": "Severity?", "criteria": ["None", "Severe: Unusable"]},
        },
    }


def test_inline_images_audio_and_video_become_typed_blocks() -> None:
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "Inspect",
                "images": ["data:image/png;base64,YQ==", {"content_type": "image/jpeg", "base64": "Yg=="}],
                "audio": ["data:audio/wav;base64,Yw=="],
                "videos": [{"content_type": "video/mp4", "base64": "ZA=="}],
                "questions": {"ok": {"type": "noul", "instructions": "OK?"}},
            }
        )
    )
    assert GeminiDecisionsConfig().transform_decisions_request("gemini-decision-1", request, "gemini") == {
        "model": "gemini-decision-1",
        "input": (
            {"type": "text", "text": "Inspect"},
            {"type": "image", "data": "YQ==", "mime_type": "image/png"},
            {"type": "image", "data": "Yg==", "mime_type": "image/jpeg"},
            {"type": "audio", "data": "Yw==", "mime_type": "audio/wav"},
            {"type": "video", "data": "ZA==", "mime_type": "video/mp4"},
        ),
        "questions": {"ok": {"type": "noul", "instructions": "OK?"}},
    }


def test_boolean_choice_response_returns_boolean_not_wire_label() -> None:
    request: Final = openai_request_to_ir(
        OpenAIDecisionRequestBody.model_validate(
            {
                "model": "gemini/gemini-decision-1",
                "input": "mug",
                "questions": [
                    {
                        "type": "choice",
                        "name": "replace",
                        "instructions": "Replace?",
                        "choices": [{"value": True}, {"value": False}],
                    }
                ],
            }
        )
    )
    response: Final = GeminiDecisionsConfig().parse_response(
        {
            "answers": {
                "replace": {
                    "type": "choice",
                    "choice": "false",
                    "confidence": 0.9,
                    "probabilities": {"true": 0.1, "false": 0.9},
                }
            }
        },
        request,
    )
    assert ir_to_openai_response(response, request, "gemini/gemini-decision-1").model_dump(
        mode="json", exclude_none=True
    )["answers"] == [
        {
            "type": "choice",
            "name": "replace",
            "choice": False,
            "confidence": 0.9,
            "probabilities": [{"value": True, "probability": 0.1}, {"value": False, "probability": 0.9}],
        }
    ]


def test_criteria_only_questions_receive_required_instructions() -> None:
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "mug",
                "questions": {
                    "damaged": {"type": "noul", "criteria": {"true": "Broken", "false": "Intact"}},
                    "action": {"type": "choice", "criteria": {"replace": "Broken", "keep": "Intact"}},
                    "severity": {"type": "score", "criteria": ["None", "Severe"]},
                },
            }
        )
    )
    assert GeminiDecisionsConfig().transform_decisions_request("gemini-decision-1", request, "gemini") == {
        "model": "gemini-decision-1",
        "input": ({"type": "text", "text": "mug"},),
        "questions": {
            "damaged": {
                "type": "noul",
                "instructions": "Does the input match the true rather than false criteria?",
                "criteria": {"true_description": "Broken", "false_description": "Intact"},
            },
            "action": {
                "type": "choice",
                "instructions": "Choose the option that best matches the input.",
                "criteria": {"replace": "Broken", "keep": "Intact"},
            },
            "severity": {
                "type": "score",
                "instructions": "Rate the input against the ordered criteria.",
                "criteria": ["None", "Severe"],
            },
        },
    }


def test_response_preserves_answers_and_normalizes_billable_usage(request_ir: DecisionsIRRequest) -> None:
    answers: Final = {
        "damaged": {"type": "noul", "noul": 0.98, "confidence": 0.98},
        "action": {
            "type": "choice",
            "choice": "replace",
            "confidence": 0.9,
            "probabilities": {"replace": 0.9, "keep": 0.1},
        },
        "severity": {
            "type": "score",
            "score": 0.8,
            "confidence": 0.8,
            "legend": {"0": "None", "1": "Severe"},
            "probabilities": {"0": 0.2, "1": 0.8},
        },
    }
    usage: Final = {
        "total_input_tokens": 200,
        "total_output_tokens": 0,
        "total_tokens": 200,
        "input_tokens_by_modality": [{"modality": "text", "tokens": 200}],
    }
    response: Final = GeminiDecisionsConfig().parse_response({"answers": answers, "usage": usage}, request_ir)
    assert ir_to_systemone_response(response, request_ir).model_dump(exclude_none=True) == {
        "answers": answers,
        "usage": {**usage, "input_tokens": 200, "output_tokens": 0},
    }
    assert response.usage is not None
    assert response.usage.input_tokens == 200
    assert response.usage.output_tokens == 0


def test_missing_usage_is_not_invented(request_ir: DecisionsIRRequest) -> None:
    response: Final = GeminiDecisionsConfig().parse_response({"answers": {}}, request_ir)
    assert response.usage is None
    assert ir_to_systemone_response(response, request_ir).answers == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"answers": {}, "usage": {"total_input_tokens": -1}},
        {"answers": {}, "usage": {}},
        {"answers": {"damaged": {"type": "noul", "noul": "not a probability"}}},
        {"answers": {"action": {"type": "choice", "choice": "replace", "probabilities": {"replace": 1.0}}}},
    ],
)
def test_invalid_provider_response_uses_existing_500_contract(
    request_ir: DecisionsIRRequest,
    payload: Mapping[str, object],
) -> None:
    with pytest.raises(DecisionsResponseValidationError) as error:
        GeminiDecisionsConfig().transform_decisions_response(
            "gemini-decision-1",
            "gemini",
            httpx.Response(200, json=payload),
            request_ir,
        )
    assert error.value.status_code == 500


@pytest.mark.parametrize("base", ["https://example.com", "https://example.com/v1beta", "https://example.com/v1beta/"])
def test_url_and_key_auth_do_not_use_model_rpc_or_bearer_token(base: str) -> None:
    config: Final = GeminiDecisionsConfig()
    assert config.get_complete_url(base, "gemini-decision-1") == "https://example.com/v1beta/decisions"
    assert config.validate_environment(
        {"Authorization": "Bearer wrong", "X-Goog-Api-Key": "wrong", "X-Trace": "trace"},
        "gemini-decision-1",
        "test-key",
        {},
    ) == {"x-goog-api-key": "test-key", "Content-Type": "application/json", "X-Trace": "trace"}


@pytest.mark.parametrize("key_env", ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
@pytest.mark.parametrize("api_key", (None, ""))
def test_custom_base_requires_its_own_key(monkeypatch: pytest.MonkeyPatch, key_env: str, api_key: str | None) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv(key_env, "shared-google-key")
    with pytest.raises(ValueError, match="explicit api_key"):
        GeminiDecisionsConfig().resolve_connection("https://gateway.example", api_key)


@pytest.mark.parametrize("key_env", ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
def test_default_base_uses_environment_key_and_custom_base_uses_explicit_key(
    monkeypatch: pytest.MonkeyPatch, key_env: str
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    monkeypatch.setenv(key_env, "shared-google-key")
    config: Final = GeminiDecisionsConfig()
    assert config.resolve_connection(None, None) == (
        "https://generativelanguage.googleapis.com",
        "shared-google-key",
    )
    assert config.resolve_connection("https://gateway.example", "gateway-key") == (
        "https://gateway.example",
        "gateway-key",
    )
    monkeypatch.setenv("GEMINI_API_BASE", "https://configured-gateway.example")
    assert config.resolve_connection(None, None) == ("https://configured-gateway.example", "shared-google-key")


def test_box_questions_are_rejected_by_transformation_and_public_sdk() -> None:
    request: Final = systemone_request_to_ir(
        DecisionsRequestBody.model_validate(
            {
                "state": "Find mug",
                "questions": {"mug": {"type": "box", "instructions": "Find mug"}},
            }
        )
    )
    assert isinstance(
        GeminiDecisionsConfig().transform_decisions_request("gemini-decision-1", request, "gemini"),
        UnsupportedDecisionsRequest,
    )
    with pytest.raises(litellm.BadRequestError) as error:
        litellm.systemone(
            model="gemini/gemini-decision-1",
            state="Find mug",
            questions={"mug": {"type": "box", "instructions": "Find mug"}},
            api_key="test-key",
        )
    assert error.value.status_code == 400


def test_empty_systemone_questions_are_rejected_before_provider_call() -> None:
    with pytest.raises(litellm.BadRequestError) as error:
        litellm.systemone(model="gemini/gemini-decision-1", state="mug", questions={}, api_key="test-key")
    assert error.value.status_code == 400


def test_empty_decisions_questions_are_rejected_before_provider_call() -> None:
    with pytest.raises(litellm.BadRequestError) as error:
        litellm.decisions(model="gemini/gemini-decision-1", input="mug", questions=[], api_key="test-key")
    assert error.value.status_code == 400


def test_message_captions_keep_their_position_next_to_each_image() -> None:
    request: Final = openai_request_to_ir(
        OpenAIDecisionRequestBody.model_validate(
            {
                "model": "gemini/gemini-decision-1",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Before"},
                            {"type": "input_image", "image_url": "data:image/png;base64,YQ=="},
                            {"type": "input_text", "text": "After"},
                            {"type": "input_image", "image_url": "data:image/png;base64,Yg=="},
                        ],
                    },
                    {"role": "user", "content": "Done"},
                ],
                "questions": [{"type": "predicate", "instructions": "Did it change?"}],
            }
        )
    )
    transformed: Final = GeminiDecisionsConfig().transform_decisions_request("gemini-decision-1", request, "gemini")
    assert not isinstance(transformed, UnsupportedDecisionsRequest)
    assert transformed["input"] == (
        {"type": "text", "text": "Before"},
        {"type": "image", "data": "YQ==", "mime_type": "image/png"},
        {"type": "text", "text": "After"},
        {"type": "image", "data": "Yg==", "mime_type": "image/png"},
        {"type": "text", "text": "Done"},
    )
