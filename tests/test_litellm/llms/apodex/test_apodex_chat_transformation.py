"""
Apodex chat completions transformation.
"""

import json

import httpx
import openai
import pytest

import litellm
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

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
    monkeypatch.delenv("APODEX_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    yield


def _client(captured: dict, *, stream: bool = False) -> openai.OpenAI:
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


def _chat_config(model: str):
    return ProviderConfigManager.get_provider_chat_config(model=model, provider=LlmProviders.APODEX)


class TestProviderResolution:
    def test_openai_compatible_provider_info_uses_apodex_credentials(self):
        config = _chat_config("apodex-1.1")
        assert config._get_openai_compatible_provider_info("https://override.test/v1", "sk-override") == (
            "https://override.test/v1",
            "sk-override",
        )

    def test_prefixed_model_resolves_to_the_default_base(self):
        model, provider, api_key, api_base = litellm.get_llm_provider(model=CORE_MODEL)
        assert (model, provider, api_key, api_base) == (
            "apodex-1.1",
            "apodex",
            "sk-apodex-test",
            "https://api.apodex.ai/v1",
        )

    def test_api_base_autodetection(self):
        _, provider, api_key, _ = litellm.get_llm_provider(model="apodex-1.1", api_base="https://api.apodex.ai/v1")
        assert provider == "apodex"
        assert api_key == "sk-apodex-test"

    def test_explicit_api_base_and_key_win(self):
        _, provider, api_key, api_base = litellm.get_llm_provider(
            model=CORE_MODEL, api_base="https://gateway.internal/v1", api_key="sk-override"
        )
        assert (provider, api_key, api_base) == ("apodex", "sk-override", "https://gateway.internal/v1")

    def test_api_base_env_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APODEX_API_BASE", "https://env.apodex.test/v1")
        _, _, _, api_base = litellm.get_llm_provider(model=CORE_MODEL)
        assert api_base == "https://env.apodex.test/v1"


class TestStreamDefault:
    """The Deep Research tiers default `stream` to true, so a non-streaming call must pin it false.

    Regression guard: the OpenAI chat handler pops `stream` out of the params it
    forwards, which would leave those tiers streaming SSE at a call that cannot
    parse it. The core models default to false but are pinned the same way.
    """

    def test_non_streaming_call_pins_stream_false(self):
        captured: dict = {}
        response = litellm.completion(
            model=CORE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            client=_client(captured),
        )

        assert captured["url"] == "https://api.apodex.ai/v1/chat/completions"
        assert captured["body"]["stream"] is False
        assert captured["body"]["model"] == "apodex-1.1"
        assert response.choices[0].message.reasoning_content == "let me think"

    def test_streaming_call_sends_stream_true(self):
        captured: dict = {}
        chunks = list(
            litellm.completion(
                model=CORE_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                client=_client(captured, stream=True),
            )
        )

        assert captured["body"]["stream"] is True
        assert chunks

    def test_deep_research_models_pin_stream_too(self):
        captured: dict = {}
        litellm.completion(
            model=DEEP_RESEARCH_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            client=_client(captured),
        )
        assert captured["body"]["stream"] is False

    def test_user_supplied_extra_body_is_preserved(self):
        """Deep research tiers reach external tools through `mcp_servers` in extra_body."""
        captured: dict = {}
        mcp_servers = [{"name": "docs", "url": "https://example.com/mcp"}]
        litellm.completion(
            model=DEEP_RESEARCH_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"mcp_servers": mcp_servers},
            client=_client(captured),
        )

        assert captured["body"]["stream"] is False
        assert captured["body"]["mcp_servers"] == mcp_servers

    def test_extra_body_cannot_override_non_streaming_pin(self):
        captured: dict = {}
        litellm.completion(
            model=DEEP_RESEARCH_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"stream": True},
            client=_client(captured),
        )

        assert captured["body"]["stream"] is False

    def test_transform_does_not_mutate_optional_params(self):
        config = _chat_config("apodex-1.1")
        optional_params = config.map_openai_params(
            non_default_params={}, optional_params={}, model="apodex-1.1", drop_params=False
        )
        original = optional_params.copy()

        config.transform_request(
            model="apodex-1.1",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={"custom_llm_provider": "apodex"},
            headers={},
        )

        assert optional_params == original


class TestSupportedParams:
    def test_responses_only_model_rejects_chat_completions(self):
        with pytest.raises(litellm.BadRequestError, match="only available through /v1/responses"):
            litellm.completion(
                model="apodex/apodex-1-1-deep-discover",
                messages=[{"role": "user", "content": "hi"}],
                client=_client({}),
            )

    def test_core_models_support_tools(self):
        supported = _chat_config("apodex-1.1").get_supported_openai_params("apodex-1.1")
        assert "tools" in supported
        assert "tool_choice" in supported
        assert "temperature" in supported
        assert "top_p" in supported

    def test_deep_research_rejects_tools_and_sampling_params(self):
        """The tiers document tools as unsupported and sampling params as ignored."""
        supported = _chat_config(DEEP_RESEARCH_MODEL).get_supported_openai_params("apodex-1-1-deep-research")
        for param in ("tools", "tool_choice", "function_call", "functions", "parallel_tool_calls"):
            assert param not in supported
        assert "temperature" not in supported
        assert "top_p" not in supported
        assert "max_tokens" in supported

    def test_tools_on_a_deep_research_model_raise(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="tools"):
            litellm.completion(
                model=DEEP_RESEARCH_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                client=_client({}),
            )

    def test_max_completion_tokens_is_renamed_to_max_tokens(self):
        captured: dict = {}
        litellm.completion(
            model=CORE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=512,
            client=_client(captured),
        )

        assert captured["body"]["max_tokens"] == 512
        assert "max_completion_tokens" not in captured["body"]


class TestCostTracking:
    def test_cached_input_is_billed_at_the_lower_rate(self):
        captured: dict = {}
        response = litellm.completion(
            model=CORE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            client=_client(captured),
        )

        # 500 fresh input + 500 cached input + 100 output
        expected = 500 * 3e-07 + 500 * 3e-08 + 100 * 3e-06
        assert litellm.completion_cost(response, model=CORE_MODEL) == pytest.approx(expected)
