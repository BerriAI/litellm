from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.predibase.chat.transformation import PredibaseConfig
from litellm.llms.predibase.common_utils import PredibaseError
from litellm.utils import Choices, Message, ModelResponse


def _build_model_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                finish_reason=None,
                index=0,
                message=Message(role="assistant", content=""),
            )
        ]
    )


def test_predibase_transform_request_non_stream():
    config = PredibaseConfig()
    request_data = config.transform_request(
        model="predibase-model",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"temperature": 0.2},
        litellm_params={},
        headers={},
    )

    assert request_data["inputs"]
    assert request_data["parameters"]["temperature"] == 0.2
    assert request_data["parameters"]["details"] is True
    assert "stream" not in request_data["parameters"]


def test_predibase_transform_request_custom_prompt(monkeypatch):
    config = PredibaseConfig()

    monkeypatch.setattr(
        "litellm.llms.predibase.chat.transformation.custom_prompt",
        lambda **kwargs: "custom-prompt",
    )

    request_data = config.transform_request(
        model="predibase-model",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={
            "custom_prompt_dict": {
                "predibase-model": {
                    "roles": {},
                    "initial_prompt_value": "",
                    "final_prompt_value": "",
                }
            }
        },
        headers={},
    )

    assert request_data["inputs"] == "custom-prompt"


def test_predibase_get_complete_url_stream_and_non_stream():
    config = PredibaseConfig()
    litellm_params = {"predibase_tenant_id": "tenant-123"}

    non_stream_url = config.get_complete_url(
        api_base="https://serving.example.com",
        api_key="test-key",
        model="predibase-model",
        optional_params={"stream": False},
        litellm_params=litellm_params,
    )
    stream_url = config.get_complete_url(
        api_base="https://serving.example.com",
        api_key="test-key",
        model="predibase-model",
        optional_params={"stream": True},
        litellm_params=litellm_params,
    )

    assert non_stream_url.endswith("/generate")
    assert stream_url.endswith("/generate_stream")


def test_predibase_transform_response_success_best_of(monkeypatch):
    config = PredibaseConfig()
    logging_obj = Mock()
    encoding = Mock()
    encoding.encode.return_value = [1, 2, 3]
    monkeypatch.setattr("litellm.token_counter", lambda messages: 5)

    raw_response = httpx.Response(
        status_code=200,
        json={
            "generated_text": "<|assistant|>primary-output</s>",
            "details": {
                "finish_reason": "eos_token",
                "tokens": [{"logprob": -0.2}, {"logprob": None}],
                "best_of_sequences": [
                    {
                        "generated_text": "<s>secondary-output</s>",
                        "finish_reason": "length",
                        "tokens": [{"logprob": -0.5}],
                    }
                ],
            },
        },
        headers={"x-request-id": "req-123"},
    )

    result = config.transform_response(
        model="predibase-model",
        raw_response=raw_response,
        model_response=_build_model_response(),
        logging_obj=logging_obj,
        request_data={"inputs": "hello", "parameters": {}},
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"best_of": 2},
        litellm_params={},
        encoding=encoding,
        api_key="test-key",
    )

    assert result.choices[0].message.content == "primary-output"
    assert len(result.choices) == 2
    assert result.choices[1].message.content == "secondary-output"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    assert (
        result._hidden_params["additional_headers"]["llm_provider-x-request-id"]
        == "req-123"
    )


def test_predibase_transform_response_invalid_json():
    config = PredibaseConfig()

    with pytest.raises(PredibaseError) as exc:
        config.transform_response(
            model="predibase-model",
            raw_response=httpx.Response(status_code=200, content=b"not-json"),
            model_response=_build_model_response(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={},
            encoding=Mock(),
            api_key="test-key",
        )

    assert exc.value.status_code == 422


def test_predibase_transform_response_error_field():
    config = PredibaseConfig()

    with pytest.raises(PredibaseError) as exc:
        config.transform_response(
            model="predibase-model",
            raw_response=httpx.Response(
                status_code=400, json={"error": "invalid request"}
            ),
            model_response=_build_model_response(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={},
            encoding=Mock(),
            api_key="test-key",
        )

    assert exc.value.status_code == 400
