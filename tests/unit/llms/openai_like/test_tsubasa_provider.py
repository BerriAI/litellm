import json
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from litellm import AuthenticationError, LlmProviders, RateLimitError, UnsupportedParamsError, acompletion, completion
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.utils import ProviderConfigManager, get_model_info, supports_response_schema


@pytest.mark.parametrize("model_id", ("tsubasa-fast", "tsubasa-pro"))
def test_tsubasa_routes_named_models_and_resolves_credentials(model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSUBASA_API_KEY", "test-credential")
    registration: Final = JSONProviderRegistry.get("tsubasa")
    assert registration is not None
    model, provider, api_key, api_base = get_llm_provider(model=f"tsubasa/{model_id}")
    assert (model, provider, api_key, api_base) == (
        model_id,
        registration.slug,
        "test-credential",
        registration.base_url,
    )
    config: Final = ProviderConfigManager.get_provider_chat_config(model=model_id, provider=LlmProviders.TSUBASA)
    assert config is not None
    assert (
        config.get_complete_url(api_base, api_key, model_id, {}, {}, stream=True)
        == f"{registration.base_url}/chat/completions"
    )
    assert config._get_openai_compatible_provider_info(None, "explicit-credential") == (
        registration.base_url,
        "explicit-credential",
    )
    assert config.map_openai_params({"max_completion_tokens": 64}, {}, model_id, False) == {"max_tokens": 64}
    info: Final = get_model_info(model=f"tsubasa/{model_id}")
    assert info["max_output_tokens"] <= info["max_input_tokens"]
    assert info["litellm_provider"] == registration.slug


@pytest.mark.parametrize("model_id", ("tsubasa-fast", "tsubasa-pro"))
@pytest.mark.parametrize("stream", (False, True))
def test_tsubasa_chat_transport(model_id: str, stream: bool) -> None:
    registration: Final = JSONProviderRegistry.get("tsubasa")
    assert registration is not None

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{registration.base_url}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-credential"
        assert json.loads(request.content) == {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
            **({"stream": True, "stream_options": {"include_usage": True}} if stream else {}),
        }
        result: Final = {
            "id": "chatcmpl-contract",
            "created": 1,
            "model": model_id,
            "object": "chat.completion.chunk" if stream else "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta" if stream else "message": {"role": "assistant", "content": "Hello"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if stream:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, request=request, json=result)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result: Final = completion(
            model=f"tsubasa/{model_id}",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-credential",
            max_completion_tokens=32,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            client=OpenAI(api_key="test-credential", base_url=registration.base_url, http_client=client),
        )
        if stream:
            chunks: Final = tuple(result)
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "Hello"
            assert chunks[-1].usage.total_tokens == 2
        else:
            assert result.choices[0].message.content == "Hello"
            assert result.usage.total_tokens == 2


@pytest.mark.parametrize("status, error", ((401, AuthenticationError), (429, RateLimitError)))
def test_tsubasa_preserves_upstream_error_status(status: int, error: type[Exception]) -> None:
    registration: Final = JSONProviderRegistry.get("tsubasa")
    assert registration is not None

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request, json={"error": {"message": "Request rejected"}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client, pytest.raises(error) as raised:
        completion(
            model="tsubasa/tsubasa-pro",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-credential",
            num_retries=0,
            client=OpenAI(api_key="test-credential", base_url=registration.base_url, http_client=client, max_retries=0),
        )
    assert raised.value.status_code == status


@pytest.mark.parametrize("model_id", ("tsubasa-fast", "tsubasa-pro"))
def test_tsubasa_rejects_tools_while_qualification_is_pending(model_id: str) -> None:
    with pytest.raises(UnsupportedParamsError, match="tools"):
        completion(
            model=f"tsubasa/{model_id}",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-credential",
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        )


async def test_tsubasa_forwards_json_schema_without_declaring_qualified_support() -> None:
    registration: Final = JSONProviderRegistry.get("tsubasa")
    assert registration is not None
    response_format: Final = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{registration.base_url}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-credential"
        assert json.loads(request.content) == {
            "model": "tsubasa-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": response_format,
        }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-contract",
                "created": 1,
                "model": "tsubasa-pro",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":"Hello"}',
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result: Final = await acompletion(
            model="tsubasa/tsubasa-pro",
            messages=[{"role": "user", "content": "Hello"}],
            response_format=response_format,
            api_key="test-credential",
            client=AsyncOpenAI(api_key="test-credential", base_url=registration.base_url, http_client=client),
        )
    assert json.loads(result.choices[0].message.content) == {"answer": "Hello"}
    assert supports_response_schema("tsubasa/tsubasa-pro") is False
