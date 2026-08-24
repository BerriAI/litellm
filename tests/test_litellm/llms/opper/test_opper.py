import json
import os
from unittest.mock import patch

import pytest

OPPER_API_BASE = "https://api.opper.ai/v3/compat"


@pytest.fixture(autouse=True)
def _without_ambient_opper_credentials(monkeypatch):
    """A developer's own OPPER_* environment must not decide where these tests point."""
    monkeypatch.delenv("OPPER_API_KEY", raising=False)
    monkeypatch.delenv("OPPER_API_BASE", raising=False)


def _completion_body(usage: dict) -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "anthropic/claude-haiku-4-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def test_opper_provider_registered():
    import litellm

    assert litellm.LlmProviders.OPPER.value == "opper"
    assert litellm.LlmProviders("opper") == litellm.LlmProviders.OPPER


def test_opper_provider_detection_by_prefix():
    """The provider/model remainder after the opper/ prefix is preserved intact."""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider("opper/anthropic/claude-haiku-4-5")

    assert model == "anthropic/claude-haiku-4-5"
    assert provider == "opper"
    assert api_base == OPPER_API_BASE


def test_opper_env_var_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    with patch.dict(
        os.environ,
        {
            "OPPER_API_KEY": "test-key",
            "OPPER_API_BASE": "https://gw.example.com/v3/compat",
        },
    ):
        model, provider, api_key, api_base = get_llm_provider("opper/anthropic/claude-haiku-4-5")

    assert provider == "opper"
    assert api_key == "test-key"
    assert api_base == "https://gw.example.com/v3/compat"


def test_opper_chat_config_registered():
    import litellm
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="anthropic/claude-haiku-4-5", provider=LlmProviders.OPPER
    )
    assert isinstance(config, litellm.OpperConfig)


@pytest.mark.respx()
def test_opper_cost_tracking_non_streaming(respx_mock, monkeypatch):
    """usage.cost from the gateway body is surfaced as the provider-reported response cost."""
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post(f"{OPPER_API_BASE}/chat/completions").respond(
        status_code=200,
        json=_completion_body({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 3.3e-05}),
    )

    response = litellm.completion(
        model="opper/anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="test-key",
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 3.3e-05


@pytest.mark.respx()
def test_opper_missing_cost_does_not_fail_response(respx_mock, monkeypatch):
    """A gateway body without usage.cost still returns content, with no cost claimed."""
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post(f"{OPPER_API_BASE}/chat/completions").respond(
        status_code=200,
        json=_completion_body({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
    )

    response = litellm.completion(
        model="opper/anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="test-key",
    )

    assert response.choices[0].message.content == "Hello!"
    additional_headers = response._hidden_params.get("additional_headers") or {}
    assert "llm_provider-x-litellm-response-cost" not in additional_headers


def test_opper_streaming_requests_usage_by_default():
    """Streaming requests ask for the usage chunk by default; explicit opt-out wins."""
    import litellm

    config = litellm.OpperConfig()
    messages = [{"role": "user", "content": "Hello"}]

    streaming_request = config.transform_request(
        model="anthropic/claude-haiku-4-5",
        messages=messages,
        optional_params={"stream": True},
        litellm_params={},
        headers={},
    )
    assert streaming_request["stream_options"] == {"include_usage": True}

    opted_out_request = config.transform_request(
        model="anthropic/claude-haiku-4-5",
        messages=messages,
        optional_params={"stream": True, "stream_options": {"include_usage": False}},
        litellm_params={},
        headers={},
    )
    assert opted_out_request["stream_options"] == {"include_usage": False}

    non_streaming_request = config.transform_request(
        model="anthropic/claude-haiku-4-5",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert "stream_options" not in non_streaming_request


@pytest.mark.respx()
def test_opper_streaming_dispatch_carries_cost(respx_mock, monkeypatch):
    """usage.cost from the terminal usage chunk survives dispatch into stream assembly."""
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    mock_chunks = [
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "cost": 3.3e-05,
                },
            }
        )
        + "\n\n",
        "data: [DONE]\n\n",
    ]

    respx_mock.post(f"{OPPER_API_BASE}/chat/completions").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content="".join(mock_chunks),
    )

    response = litellm.completion(
        model="opper/anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="test-key",
        stream=True,
    )
    yielded_chunks = list(response)

    request_body = json.loads(respx_mock.calls[0].request.content)
    assert request_body["stream_options"] == {"include_usage": True}

    assert [c.choices[0].delta.content for c in yielded_chunks] == ["Hello", None]
    assert all(getattr(c, "usage", None) is None for c in yielded_chunks)

    collected_usages = [getattr(c, "usage", None) for c in response.chunks]
    assert any(u is not None and u.cost == 3.3e-05 for u in collected_usages)
    complete_response = litellm.stream_chunk_builder(response.chunks, messages=[{"role": "user", "content": "Hello"}])
    assert complete_response.usage.cost == 3.3e-05


@pytest.mark.respx()
def test_opper_reasoning_effort_passes_through(respx_mock, monkeypatch):
    """reasoning_effort is not rejected client-side and reaches the request body."""
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    config = litellm.OpperConfig()
    assert "reasoning_effort" in config.get_supported_openai_params(model="anthropic/claude-haiku-4-5")

    respx_mock.post(f"{OPPER_API_BASE}/chat/completions").respond(
        status_code=200,
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "anthropic/claude-haiku-4-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    )

    litellm.completion(
        model="opper/anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="test-key",
        reasoning_effort="high",
    )

    request_body = json.loads(respx_mock.calls[0].request.content)
    assert request_body["reasoning_effort"] == "high"


def test_opper_preserves_cache_control_in_content():
    """Per-content cache_control markers are not stripped from the request."""
    import litellm

    config = litellm.OpperConfig()

    transformed_request = config.transform_request(
        model="anthropic/claude-haiku-4-5",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Long cached context",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "Question"},
                ],
            }
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )

    content = transformed_request["messages"][0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in content[1]


def test_opper_connection_error_reports_opper_endpoint():
    """Status-less failures carry Opper request metadata, not the OpenAI default."""
    import litellm
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type

    with pytest.raises(litellm.APIConnectionError) as excinfo:
        exception_type(
            model="anthropic/claude-haiku-4-5",
            original_exception=Exception("connection reset"),
            custom_llm_provider="opper",
        )

    assert "api.opper.ai" in str(excinfo.value.request.url)


def test_opper_get_llm_provider_with_explicit_provider():
    """With custom_llm_provider preset, the opper/ routing prefix is stripped from provider/model IDs."""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, _ = get_llm_provider("opper/anthropic/claude-haiku-4-5", custom_llm_provider="opper")
    assert model == "anthropic/claude-haiku-4-5"
    assert provider == "opper"

    model, provider, _, _ = get_llm_provider("opper/auto", custom_llm_provider="opper")
    assert model == "opper/auto"
    assert provider == "opper"


def test_opper_validate_environment():
    """validate_environment reports OPPER_API_KEY as the required credential."""
    import litellm

    with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}, clear=False):
        present = litellm.validate_environment("opper/anthropic/claude-haiku-4-5")
    assert present["keys_in_environment"] is True
    assert present["missing_keys"] == []

    env_without_key = {k: v for k, v in os.environ.items() if k != "OPPER_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        missing = litellm.validate_environment("opper/anthropic/claude-haiku-4-5")
    assert missing["keys_in_environment"] is False
    assert "OPPER_API_KEY" in missing["missing_keys"]


def test_opper_supported_params_dispatch():
    """The get_supported_openai_params dispatch resolves the Opper config."""
    import litellm
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,
    )

    assert litellm.OpperConfig().custom_llm_provider == "opper"
    params = get_supported_openai_params(model="anthropic/claude-haiku-4-5", custom_llm_provider="opper")
    assert params is not None
    assert "reasoning_effort" in params


@pytest.mark.respx()
def test_opper_unknown_model_raises_not_found(respx_mock, monkeypatch):
    """A 404 from the gateway maps to a non-retryable NotFoundError."""
    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post(f"{OPPER_API_BASE}/chat/completions").respond(
        status_code=404,
        json={
            "error": {
                "message": "model nonexistent/model not found",
                "type": "model_not_found",
            }
        },
    )

    with pytest.raises(litellm.NotFoundError):
        litellm.completion(
            model="opper/nonexistent/model",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
        )


@pytest.mark.respx()
def test_opper_async_completion_and_streaming(respx_mock, monkeypatch):
    """acompletion works non-streaming and streaming through the async dispatch path."""
    import asyncio

    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post(f"{OPPER_API_BASE}/chat/completions").respond(
        status_code=200,
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "anthropic/claude-haiku-4-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 3.3e-05},
        },
    )

    response = asyncio.run(
        litellm.acompletion(
            model="opper/anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
        )
    )
    assert response.choices[0].message.content == "Hello!"
    assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 3.3e-05

    stream_chunks = [
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-124",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}],
            }
        )
        + "\n\n",
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-124",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "anthropic/claude-haiku-4-5",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": 1.1e-05},
            }
        )
        + "\n\n",
        "data: [DONE]\n\n",
    ]
    respx_mock.post(f"{OPPER_API_BASE}/chat/completions").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content="".join(stream_chunks),
    )

    async def consume_stream():
        stream = await litellm.acompletion(
            model="opper/anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
            stream=True,
        )
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        return stream, collected

    stream, chunks = asyncio.run(consume_stream())
    assert any(c.choices and c.choices[0].delta.content == "Hi" for c in chunks)
    assert any(getattr(c, "usage", None) is not None and c.usage.cost == 1.1e-05 for c in stream.chunks)


@pytest.mark.respx()
def test_opper_get_models_prefixes_the_live_catalog(respx_mock):
    """The model list is fetched from the gateway, so it tracks the catalog instead of a static table."""
    import litellm

    route = respx_mock.get(f"{OPPER_API_BASE}/models").respond(
        status_code=200,
        json={
            "object": "list",
            "data": [
                {"id": "anthropic/claude-haiku-4-5", "object": "model"},
                {"id": "openai/gpt-5.6", "object": "model"},
            ],
        },
    )

    models = litellm.OpperConfig().get_models(api_key="test-key")

    assert models == ["opper/anthropic/claude-haiku-4-5", "opper/openai/gpt-5.6"]
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"


@pytest.mark.respx()
def test_opper_get_models_keeps_the_gateway_path(respx_mock):
    """The catalog lives under the gateway path, not at the host root like plain OpenAI."""
    import litellm

    route = respx_mock.get("https://gw.example.com/v3/compat/models").respond(
        status_code=200,
        json={"object": "list", "data": [{"id": "mistral/mistral-large", "object": "model"}]},
    )

    models = litellm.OpperConfig().get_models(api_key="test-key", api_base="https://gw.example.com/v3/compat/")

    assert models == ["opper/mistral/mistral-large"]
    assert route.called


@pytest.mark.respx()
def test_opper_get_models_raises_on_error_status(respx_mock):
    import litellm

    respx_mock.get(f"{OPPER_API_BASE}/models").respond(
        status_code=401,
        json={"error": "missing authorization header"},
    )

    with pytest.raises(ValueError, match="401"):
        litellm.OpperConfig().get_models(api_key="bad-key")


def test_opper_get_models_requires_an_api_key():
    import litellm

    with pytest.raises(ValueError, match="OPPER_API_KEY"):
        litellm.OpperConfig().get_models()


@pytest.mark.respx()
def test_opper_valid_models_come_from_the_provider_endpoint(respx_mock):
    """get_valid_models reaches the gateway catalog through the registered provider config."""
    import litellm

    api_base = "https://valid-models.example.com/v3/compat"
    respx_mock.get(f"{api_base}/models").respond(
        status_code=200,
        json={
            "object": "list",
            "data": [
                {"id": "anthropic/claude-sonnet-5", "object": "model"},
                {"id": "gemini/gemini-3-flash-preview", "object": "model"},
            ],
        },
    )

    models = litellm.get_valid_models(
        check_provider_endpoint=True,
        custom_llm_provider="opper",
        api_key="test-key",
        api_base=api_base,
    )

    assert models == ["opper/anthropic/claude-sonnet-5", "opper/gemini/gemini-3-flash-preview"]


def test_opper_is_selectable_in_the_add_model_form():
    from pathlib import Path

    import litellm

    path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    with open(path) as f:
        entries = [e for e in json.load(f) if e["litellm_provider"] == "opper"]

    assert len(entries) == 1, "opper must appear exactly once in provider_create_fields.json"

    entry = entries[0]
    assert entry["provider"] == "Opper"
    assert entry["provider_display_name"] == "Opper"
    assert entry["default_model_placeholder"].startswith("opper/")

    fields = {f["key"]: f for f in entry["credential_fields"]}
    assert fields["api_key"]["required"] is True
    assert fields["api_key"]["field_type"] == "password"
    assert fields["api_base"]["required"] is False
    assert fields["api_base"]["placeholder"] == OPPER_API_BASE
