"""
Tests for the Apodex provider (https://platform.apodex.ai/docs).
"""

import json
from pathlib import Path

import httpx
import openai
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

REPO_ROOT = Path(__file__).parents[4]
CORE_MODEL = "apodex/apodex-1.1"
DEEP_RESEARCH_MODEL = "apodex/apodex-1-1-deep-research"

CHAT_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1712345678,
    "model": "apodex-1.1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok", "reasoning_content": "let me think"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_tokens_details": {"cached_tokens": 500},
    },
}

STREAM_BODY = (
    b'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1,"model":"apodex-1.1",'
    b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.fixture(autouse=True)
def _apodex_env(monkeypatch: pytest.MonkeyPatch):
    """Resolve models against the in-repo cost map, not the published one."""
    monkeypatch.setenv("APODEX_API_KEY", "sk-apodex-test")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    yield


def _openai_client(captured: dict, *, stream: bool = False) -> openai.OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        if stream:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=STREAM_BODY)
        return httpx.Response(200, json=CHAT_RESPONSE)

    return openai.OpenAI(
        api_key="sk-apodex-test",
        base_url="https://api.apodex.ai/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestApodexRegistration:
    def test_provider_enum_and_lists(self):
        assert LlmProviders.APODEX.value == "apodex"
        assert "apodex" in litellm.provider_list
        assert "apodex" in litellm.constants.openai_compatible_providers

    def test_json_provider_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        apodex = JSONProviderRegistry.get("apodex")
        assert apodex is not None
        assert apodex.base_url == "https://api.apodex.ai/v1"
        assert apodex.api_key_env == "APODEX_API_KEY"
        assert apodex.api_base_env == "APODEX_API_BASE"
        assert apodex.param_mappings["max_completion_tokens"] == "max_tokens"
        assert apodex.supported_endpoints == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
        assert JSONProviderRegistry.supports_responses_api("apodex") is True

    def test_provider_resolution(self):
        model, provider, _, api_base = litellm.get_llm_provider(model=CORE_MODEL)
        assert (model, provider, api_base) == ("apodex-1.1", "apodex", "https://api.apodex.ai/v1")

    def test_api_base_autodetection(self):
        _, provider, api_key, _ = litellm.get_llm_provider(model="apodex-1.1", api_base="https://api.apodex.ai/v1")
        assert provider == "apodex"
        assert api_key == "sk-apodex-test"

    def test_explicit_api_base_and_key_win(self):
        _, provider, api_key, api_base = litellm.get_llm_provider(
            model=CORE_MODEL, api_base="https://gateway.internal/v1", api_key="sk-override"
        )
        assert (provider, api_key, api_base) == ("apodex", "sk-override", "https://gateway.internal/v1")


class TestApodexStreamDefault:
    """Apodex defaults `stream` to true, so a non-streaming call must pin it to false.

    Regression guard: the OpenAI SDK omits `stream` when it is false, which would make
    litellm.completion() receive SSE and fail to parse it.
    """

    def test_chat_completion_pins_stream_false(self):
        captured: dict = {}
        response = litellm.completion(
            model=CORE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            client=_openai_client(captured),
        )

        assert captured["url"] == "https://api.apodex.ai/v1/chat/completions"
        assert captured["body"]["stream"] is False
        assert captured["body"]["model"] == "apodex-1.1"
        assert response.choices[0].message.reasoning_content == "let me think"

    def test_chat_completion_streaming_sends_stream_true(self):
        captured: dict = {}
        chunks = list(
            litellm.completion(
                model=CORE_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                client=_openai_client(captured, stream=True),
            )
        )

        assert captured["body"]["stream"] is True
        assert chunks

    def test_user_supplied_extra_body_is_preserved(self):
        captured: dict = {}
        litellm.completion(
            model=CORE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"mcp_servers": [{"name": "docs", "url": "https://example.com/mcp"}]},
            client=_openai_client(captured),
        )

        assert captured["body"]["stream"] is False
        assert captured["body"]["mcp_servers"] == [{"name": "docs", "url": "https://example.com/mcp"}]

    def test_responses_api_pins_stream_false(self):
        captured: dict = {}

        class CapturingHandler(HTTPHandler):
            def post(self, *args, **kwargs):
                captured.update(url=kwargs.get("url"), body=kwargs.get("json"))
                raise RuntimeError("captured")

        with pytest.raises(Exception):
            litellm.responses(model=DEEP_RESEARCH_MODEL, input="hi", client=CapturingHandler())

        assert captured["url"] == "https://api.apodex.ai/v1/responses"
        assert captured["body"]["stream"] is False
        assert captured["body"]["model"] == "apodex-1-1-deep-research"

    @pytest.mark.asyncio
    async def test_responses_api_streaming_sends_stream_true(self):
        captured: dict = {}

        class CapturingHandler(AsyncHTTPHandler):
            async def post(self, *args, **kwargs):
                captured.update(body=kwargs.get("json"))
                raise RuntimeError("captured")

        with pytest.raises(Exception):
            await litellm.aresponses(model=DEEP_RESEARCH_MODEL, input="hi", stream=True, client=CapturingHandler())

        assert captured["body"]["stream"] is True

    def test_flag_is_opt_in_for_other_json_providers(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        pinstripes = JSONProviderRegistry.get("pinstripes")
        assert pinstripes is not None
        assert "send_explicit_stream_false" not in pinstripes.special_handling

        config = ProviderConfigManager.get_provider_chat_config(
            model="ps/glm-4.5-air", provider=LlmProviders.PINSTRIPES
        )
        params = config.map_openai_params({}, {}, "ps/glm-4.5-air", False)
        assert "stream" not in params
        assert "stream" not in (params.get("extra_body") or {})


class TestApodexToolSupport:
    """Deep research tiers reject OpenAI-style tools; core models accept them."""

    def test_deep_research_drops_tool_params(self):
        config = ProviderConfigManager.get_provider_chat_config(
            model="apodex-1-1-deep-research", provider=LlmProviders.APODEX
        )
        supported = config.get_supported_openai_params("apodex-1-1-deep-research")
        assert "tools" not in supported
        assert "tool_choice" not in supported

    def test_core_model_keeps_tool_params(self):
        config = ProviderConfigManager.get_provider_chat_config(model="apodex-1.1", provider=LlmProviders.APODEX)
        supported = config.get_supported_openai_params("apodex-1.1")
        assert "tools" in supported
        assert "tool_choice" in supported

    def test_max_completion_tokens_maps_to_max_tokens(self):
        config = ProviderConfigManager.get_provider_chat_config(model="apodex-1.1", provider=LlmProviders.APODEX)
        params = config.map_openai_params({"max_completion_tokens": 512}, {}, "apodex-1.1", False)
        assert params["max_tokens"] == 512
        assert "max_completion_tokens" not in params


class TestApodexAnthropicMessages:
    """Apodex serves POST /v1/messages natively, so the payload is forwarded untranslated."""

    def test_native_passthrough_config(self):
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="apodex-1.1", provider=LlmProviders.APODEX
        )
        assert config is not None
        assert type(config).__name__ == "JSONProviderAnthropicMessagesConfig"
        assert (
            config.get_complete_url(
                api_base=None, api_key=None, model="apodex-1.1", optional_params={}, litellm_params={}
            )
            == "https://api.apodex.ai/v1/messages"
        )

    def test_headers_use_provider_api_key(self):
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="apodex-1.1", provider=LlmProviders.APODEX
        )
        assert config is not None
        headers, _ = config.validate_anthropic_messages_environment(
            headers={}, model="apodex-1.1", messages=[], optional_params={}, litellm_params={}
        )
        assert headers["authorization"] == "Bearer sk-apodex-test"
        assert headers["anthropic-version"] == "2023-06-01"


class TestApodexModelMetadata:
    @pytest.fixture(scope="class")
    def model_cost(self) -> dict:
        with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
            return json.load(f)

    def test_core_model_pricing(self, model_cost: dict):
        info = model_cost["apodex/apodex-1.1"]
        assert info["litellm_provider"] == "apodex"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 262144
        assert info["input_cost_per_token"] == 3e-07
        assert info["cache_read_input_token_cost"] == 3e-08
        assert info["output_cost_per_token"] == 3e-06
        # Requests over 200K input tokens are billed at 2x across every tier
        assert info["input_cost_per_token_above_200k_tokens"] == 6e-07
        assert info["cache_read_input_token_cost_above_200k_tokens"] == 6e-08
        assert info["output_cost_per_token_above_200k_tokens"] == 6e-06
        assert info["supports_prompt_caching"] is True
        assert info["supports_function_calling"] is True
        assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]

    def test_deep_research_model_pricing(self, model_cost: dict):
        info = model_cost["apodex/apodex-1-1-deep-research"]
        assert info["max_input_tokens"] == 131072
        assert info["max_output_tokens"] == 65536
        assert info["input_cost_per_token"] == 5e-06
        assert info["output_cost_per_token"] == 2e-05
        assert info["supports_function_calling"] is False
        assert info["supports_response_schema"] is False
        assert info["supports_prompt_caching"] is False
        assert info["supports_web_search"] is True
        assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses"]

    def test_every_apodex_model_is_registered(self, model_cost: dict):
        assert {key for key in model_cost if key.startswith("apodex/")} == {
            "apodex/apodex-1.1",
            "apodex/apodex-1.1-mini",
            "apodex/apodex-1-1-deep-research",
            "apodex/apodex-1-1-deep-solve",
            "apodex/apodex-1-1-deep-discover",
            "apodex/apodex-1-0-deep-research",
            "apodex/apodex-1-0-deep-solve",
            "apodex/apodex-1-0-deep-discover",
        }

    def test_backup_cost_map_in_sync(self, model_cost: dict):
        with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
            backup = json.load(f)
        for key in (key for key in model_cost if key.startswith("apodex/")):
            assert backup[key] == model_cost[key], f"{key} differs between main and backup cost maps"

    def test_cost_tracks_cached_input_separately(self):
        captured: dict = {}
        response = litellm.completion(
            model=CORE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            client=_openai_client(captured),
        )

        # 500 fresh input + 500 cached input + 100 output
        expected = 500 * 3e-07 + 500 * 3e-08 + 100 * 3e-06
        assert litellm.completion_cost(response, model=CORE_MODEL) == pytest.approx(expected)
