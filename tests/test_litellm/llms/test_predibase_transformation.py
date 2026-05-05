from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from litellm.llms.predibase.chat.handler import PredibaseChatCompletion
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


def test_predibase_get_complete_url_missing_tenant_id():
    config = PredibaseConfig()

    with pytest.raises(ValueError, match="Missing Predibase Tenant ID"):
        config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="predibase-model",
            optional_params={},
            litellm_params={},
        )


def test_predibase_get_complete_url_with_tenant_id_key():
    config = PredibaseConfig()

    url = config.get_complete_url(
        api_base="https://serving.example.com",
        api_key="test-key",
        model="predibase-model",
        optional_params={},
        litellm_params={"tenant_id": "tenant-xyz"},
    )

    assert "tenant-xyz" in url
    assert url.endswith("/generate")


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


def test_predibase_transform_response_missing_generated_text():
    config = PredibaseConfig()

    with pytest.raises(PredibaseError, match="'generated_text' is not a key"):
        config.transform_response(
            model="predibase-model",
            raw_response=httpx.Response(status_code=200, json={"details": {}}),
            model_response=_build_model_response(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={},
            encoding=Mock(),
            api_key="test-key",
        )


def test_predibase_transform_response_non_dict_payload():
    config = PredibaseConfig()
    raw_response = Mock()
    raw_response.text = "[]"
    raw_response.status_code = 200
    raw_response.headers = {}
    raw_response.json.return_value = []

    with pytest.raises(PredibaseError, match="'completion_response' is not a dictionary"):
        config.transform_response(
            model="predibase-model",
            raw_response=raw_response,
            model_response=_build_model_response(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={},
            encoding=Mock(),
            api_key="test-key",
        )


def test_predibase_transform_response_best_of_with_empty_generated_text(monkeypatch):
    config = PredibaseConfig()
    logging_obj = Mock()
    encoding = Mock()
    encoding.encode.return_value = [1]
    monkeypatch.setattr("litellm.token_counter", lambda messages: 1)

    raw_response = httpx.Response(
        status_code=200,
        json={
            "generated_text": "primary-output",
            "details": {
                "finish_reason": "stop",
                "tokens": [],
                "best_of_sequences": [
                    {
                        "generated_text": "",
                        "finish_reason": "length",
                        "tokens": [],
                    }
                ],
            },
        },
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

    assert len(result.choices) == 2
    assert result.choices[1].message.content is None


def test_predibase_transform_response_best_of_from_request_data(monkeypatch):
    config = PredibaseConfig()
    logging_obj = Mock()
    encoding = Mock()
    encoding.encode.return_value = [1]
    monkeypatch.setattr("litellm.token_counter", lambda messages: 1)

    raw_response = httpx.Response(
        status_code=200,
        json={
            "generated_text": "primary-output",
            "details": {
                "finish_reason": "stop",
                "tokens": [],
                "best_of_sequences": [
                    {
                        "generated_text": "secondary-output",
                        "finish_reason": "length",
                        "tokens": [],
                    }
                ],
            },
        },
    )

    result = config.transform_response(
        model="predibase-model",
        raw_response=raw_response,
        model_response=_build_model_response(),
        logging_obj=logging_obj,
        request_data={"inputs": "hello", "parameters": {"best_of": 2}},
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        encoding=encoding,
        api_key="test-key",
    )

    assert len(result.choices) == 2
    assert result.choices[1].message.content == "secondary-output"


def test_predibase_transform_response_best_of_invalid_value_falls_back(monkeypatch):
    config = PredibaseConfig()
    logging_obj = Mock()
    encoding = Mock()
    encoding.encode.return_value = [1]
    monkeypatch.setattr("litellm.token_counter", lambda messages: 1)

    raw_response = httpx.Response(
        status_code=200,
        json={
            "generated_text": "primary-output",
            "details": {
                "finish_reason": "stop",
                "tokens": [],
                "best_of_sequences": [
                    {
                        "generated_text": "secondary-output",
                        "finish_reason": "length",
                        "tokens": [],
                    }
                ],
            },
        },
    )

    result = config.transform_response(
        model="predibase-model",
        raw_response=raw_response,
        model_response=_build_model_response(),
        logging_obj=logging_obj,
        request_data={"inputs": "hello", "parameters": {}},
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"best_of": "invalid-int"},
        litellm_params={},
        encoding=encoding,
        api_key="test-key",
    )

    # Invalid best_of should safely fall back to 0 and not append extra choices.
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "primary-output"


def test_predibase_transform_response_empty_output_sets_completion_tokens_zero(monkeypatch):
    config = PredibaseConfig()
    logging_obj = Mock()
    encoding = Mock()
    monkeypatch.setattr("litellm.token_counter", lambda messages: 3)

    raw_response = httpx.Response(
        status_code=200,
        json={"generated_text": "", "details": {"tokens": [], "finish_reason": "stop"}},
    )

    result = config.transform_response(
        model="predibase-model",
        raw_response=raw_response,
        model_response=_build_model_response(),
        logging_obj=logging_obj,
        request_data={"inputs": "hello", "parameters": {}},
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        encoding=encoding,
        api_key="test-key",
    )

    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 0


def test_predibase_get_complete_url_uses_env_base_url(monkeypatch):
    config = PredibaseConfig()
    monkeypatch.setenv("PREDIBASE_API_BASE", "https://env.predibase.com")

    url = config.get_complete_url(
        api_base=None,
        api_key="test-key",
        model="predibase-model",
        optional_params={},
        litellm_params={"predibase_tenant_id": "tenant-123"},
    )

    assert url.startswith("https://env.predibase.com/tenant-123/")


def test_predibase_transform_response_usage_fallbacks(monkeypatch):
    config = PredibaseConfig()
    logging_obj = Mock()
    encoding = Mock()
    encoding.encode.side_effect = RuntimeError("encoding failure")
    monkeypatch.setattr(
        "litellm.token_counter", lambda messages: (_ for _ in ()).throw(RuntimeError())
    )

    raw_response = httpx.Response(
        status_code=200,
        json={"generated_text": "ok", "details": {"tokens": [], "finish_reason": "stop"}},
    )

    result = config.transform_response(
        model="predibase-model",
        raw_response=raw_response,
        model_response=_build_model_response(),
        logging_obj=logging_obj,
        request_data={"inputs": "hello", "parameters": {}},
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        encoding=encoding,
        api_key="test-key",
    )

    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0


@pytest.mark.asyncio
async def test_predibase_async_completion_uses_default_config_when_none(monkeypatch):
    handler = PredibaseChatCompletion()
    mock_response = httpx.Response(status_code=200, json={"generated_text": "ok"})

    async_handler = Mock()
    async_handler.post = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(
        "litellm.llms.predibase.chat.handler.get_async_httpx_client",
        lambda **kwargs: async_handler,
    )

    default_config = Mock()
    default_config.transform_response.return_value = _build_model_response()
    monkeypatch.setattr("litellm.PredibaseConfig", lambda: default_config)

    result = await handler.async_completion(
        model="predibase-model",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://serving.example.com/x/generate",
        model_response=_build_model_response(),
        print_verbose=Mock(),
        encoding=Mock(),
        api_key="test-key",
        logging_obj=Mock(),
        stream=False,
        data={"inputs": "hello", "parameters": {}},
        optional_params={},
        timeout=10,
        litellm_params={},
        headers={"Authorization": "Bearer test"},
    )

    assert result is default_config.transform_response.return_value
    default_config.transform_response.assert_called_once()


@pytest.mark.asyncio
async def test_predibase_async_completion_uses_passed_config(monkeypatch):
    handler = PredibaseChatCompletion()
    mock_response = httpx.Response(status_code=200, json={"generated_text": "ok"})

    async_handler = Mock()
    async_handler.post = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(
        "litellm.llms.predibase.chat.handler.get_async_httpx_client",
        lambda **kwargs: async_handler,
    )

    passed_config = Mock()
    passed_config.transform_response.return_value = _build_model_response()

    result = await handler.async_completion(
        model="predibase-model",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://serving.example.com/x/generate",
        model_response=_build_model_response(),
        print_verbose=Mock(),
        encoding=Mock(),
        api_key="test-key",
        logging_obj=Mock(),
        stream=False,
        data={"inputs": "hello", "parameters": {}},
        optional_params={},
        timeout=10,
        litellm_params={},
        headers={"Authorization": "Bearer test"},
        predibase_config=passed_config,
    )

    assert result is passed_config.transform_response.return_value
    passed_config.transform_response.assert_called_once()


def test_predibase_completion_sync_returns_transform_response(monkeypatch):
    handler = PredibaseChatCompletion()
    expected = _build_model_response()

    def fake_validate_environment(self, **kwargs):
        return {"Authorization": "Bearer test"}

    def fake_get_complete_url(self, **kwargs):
        return "https://serving.example.com/tenant/deployments/v2/llms/model/generate"

    def fake_transform_request(self, **kwargs):
        return {"inputs": "hello", "parameters": {}}

    def fake_transform_response(self, **kwargs):
        return expected

    monkeypatch.setattr(PredibaseConfig, "validate_environment", fake_validate_environment)
    monkeypatch.setattr(PredibaseConfig, "get_complete_url", fake_get_complete_url)
    monkeypatch.setattr(PredibaseConfig, "transform_request", fake_transform_request)
    monkeypatch.setattr(PredibaseConfig, "transform_response", fake_transform_response)
    monkeypatch.setattr(
        "litellm.module_level_client.post",
        lambda *args, **kwargs: httpx.Response(status_code=200, json={"generated_text": "ok"}),
    )

    result = handler.completion(
        model="predibase-model",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://serving.example.com",
        custom_prompt_dict={},
        model_response=_build_model_response(),
        print_verbose=Mock(),
        encoding=Mock(),
        api_key="test-key",
        logging_obj=Mock(),
        optional_params={},
        litellm_params={},
        tenant_id="tenant-123",
        timeout=10,
        acompletion=False,
    )

    assert result is expected


def test_predibase_completion_passes_existing_config_to_async_completion(monkeypatch):
    handler = PredibaseChatCompletion()
    captured = {}

    def fake_validate_environment(self, **kwargs):
        captured["config_instance"] = self
        return {"Authorization": "Bearer test"}

    def fake_get_complete_url(self, **kwargs):
        return "https://serving.example.com/tenant/deployments/v2/llms/model/generate"

    def fake_transform_request(self, **kwargs):
        return {"inputs": "hello", "parameters": {}}

    def fake_async_completion(**kwargs):
        captured["async_kwargs"] = kwargs
        return "async-result"

    monkeypatch.setattr(PredibaseConfig, "validate_environment", fake_validate_environment)
    monkeypatch.setattr(PredibaseConfig, "get_complete_url", fake_get_complete_url)
    monkeypatch.setattr(PredibaseConfig, "transform_request", fake_transform_request)
    monkeypatch.setattr(handler, "async_completion", fake_async_completion)

    result = handler.completion(
        model="predibase-model",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://serving.example.com",
        custom_prompt_dict={},
        model_response=_build_model_response(),
        print_verbose=Mock(),
        encoding=Mock(),
        api_key="test-key",
        logging_obj=Mock(),
        optional_params={},
        litellm_params={},
        tenant_id="tenant-123",
        timeout=10,
        acompletion=True,
    )

    assert result == "async-result"
    assert captured["async_kwargs"]["predibase_config"] is captured["config_instance"]
