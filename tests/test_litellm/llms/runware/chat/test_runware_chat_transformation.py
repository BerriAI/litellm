"""
Tests for the Runware provider.

Runware is an OpenAI-compatible inference provider registered purely via the
JSON provider registry (``litellm/llms/openai_like/providers.json``); it has no
hand-written transformation class. These tests therefore validate:

  * the JSON registration and the parsed ``SimpleProviderConfig`` defaults,
  * provider resolution through ``litellm.get_llm_provider`` (env var, explicit
    key, and ``api_base`` override),
  * the dynamically generated config class (URL building, credential
    resolution, supported-param filtering, OpenAI-param mapping),
  * end-to-end mocked chat completions (sync + async), request-body shape, and
    streaming.

Runware exposes both a bare model id (e.g. ``minimax-m2-7``) and an AIR id
(e.g. ``minimax:m2.7@0``) for the same model; the bare id is used throughout
these tests. The id is passed through verbatim to Runware, so its exact value
does not affect provider plumbing.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import completion
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

API_BASE = "https://api.runware.ai/v1"
CHAT_URL = f"{API_BASE}/chat/completions"
MODEL = "runware/minimax-m2-7"
BARE_MODEL = "minimax-m2-7"


def _new_config():
    """Build a fresh dynamically-generated config instance for runware."""
    return create_config_class(JSONProviderRegistry.get("runware"))()


def _completion_payload(content="Hi from Runware!"):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": BARE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


class TestRunwareProviderRegistration:
    """Runware is configured purely via providers.json (no transformation.py)."""

    def test_runware_registered_in_json_registry(self):
        assert JSONProviderRegistry.exists("runware")

        config = JSONProviderRegistry.get("runware")
        assert config is not None
        assert config.base_url == API_BASE
        assert config.api_key_env == "RUNWARE_API_KEY"

    def test_runware_listed(self):
        assert "runware" in JSONProviderRegistry.list_providers()

    def test_runware_config_defaults(self):
        config = JSONProviderRegistry.get("runware")
        # Only base_url + api_key_env are set; everything else takes defaults.
        assert config.base_class == "openai_gpt"
        assert config.param_mappings == {}
        assert config.constraints == {}
        assert config.special_handling == {}
        assert config.supported_endpoints == []
        assert config.api_base_env is None

    def test_runware_does_not_advertise_responses_api(self):
        # No supported_endpoints means the Responses API is not advertised.
        assert JSONProviderRegistry.supports_responses_api("runware") is False


class TestRunwareProviderResolution:
    def test_get_llm_provider_resolves_runware(self, monkeypatch):
        monkeypatch.setenv("RUNWARE_API_KEY", "test-runware-key")

        model, custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider(model=MODEL)
        )

        assert model == BARE_MODEL
        assert custom_llm_provider == "runware"
        assert dynamic_api_key == "test-runware-key"
        assert api_base == API_BASE

    def test_get_llm_provider_uses_explicit_api_key(self, monkeypatch):
        # An explicitly supplied key wins over the environment variable.
        monkeypatch.setenv("RUNWARE_API_KEY", "env-key")

        _, _, dynamic_api_key, _ = litellm.get_llm_provider(
            model=MODEL, api_key="sk-explicit"
        )
        assert dynamic_api_key == "sk-explicit"

    def test_get_llm_provider_honors_api_base_override(self):
        _, custom_llm_provider, _, api_base = litellm.get_llm_provider(
            model=MODEL, api_base="https://custom.example/v1"
        )
        assert custom_llm_provider == "runware"
        assert api_base == "https://custom.example/v1"


class TestRunwareDynamicConfig:
    def test_custom_llm_provider(self):
        assert _new_config().custom_llm_provider == "runware"

    def test_get_complete_url_appends_chat_completions(self):
        url = _new_config().get_complete_url(
            api_base=API_BASE,
            api_key="k",
            model=BARE_MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == CHAT_URL

    def test_get_complete_url_does_not_double_append(self):
        url = _new_config().get_complete_url(
            api_base=CHAT_URL,
            api_key="k",
            model=BARE_MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == CHAT_URL

    def test_get_complete_url_falls_back_to_base_url(self):
        # When no api_base is supplied, the provider's base_url is used.
        url = _new_config().get_complete_url(
            api_base=None,
            api_key="k",
            model=BARE_MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == CHAT_URL

    def test_provider_info_resolves_from_env(self, monkeypatch):
        monkeypatch.setenv("RUNWARE_API_KEY", "env-key")
        base, key = _new_config()._get_openai_compatible_provider_info(None, None)
        assert base == API_BASE
        assert key == "env-key"

    def test_provider_info_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("RUNWARE_API_KEY", "env-key")
        base, key = _new_config()._get_openai_compatible_provider_info(
            "https://x/v1", "explicit-key"
        )
        assert base == "https://x/v1"
        assert key == "explicit-key"


class TestRunwareParamHandling:
    def test_standard_sampling_params_pass_through(self):
        out = _new_config().map_openai_params(
            non_default_params={"temperature": 0.6, "top_p": 0.9},
            optional_params={},
            model=BARE_MODEL,
            drop_params=False,
        )
        assert out == {"temperature": 0.6, "top_p": 0.9}

    def test_max_tokens_is_not_remapped(self):
        # runware declares no param_mappings, so max_tokens stays max_tokens
        # (Runware's OpenAI-compatible endpoint accepts max_tokens directly).
        out = _new_config().map_openai_params(
            non_default_params={"max_tokens": 256},
            optional_params={},
            model=BARE_MODEL,
            drop_params=False,
        )
        assert out["max_tokens"] == 256

    def test_tool_params_dropped_without_model_registration(self):
        # Runware models are not (yet) present in
        # model_prices_and_context_window.json, so supports_function_calling()
        # is False and the dynamic config strips tool-related params. Until a
        # model entry with "supports_function_calling": true is added, tools
        # are silently dropped before the request is sent.
        cfg = _new_config()
        assert "tools" not in cfg.get_supported_openai_params(BARE_MODEL)
        out = cfg.map_openai_params(
            non_default_params={"tools": [{"type": "function"}]},
            optional_params={},
            model=BARE_MODEL,
            drop_params=False,
        )
        assert "tools" not in out

    def test_standard_params_still_supported(self):
        sp = _new_config().get_supported_openai_params(BARE_MODEL)
        assert "temperature" in sp
        assert "top_p" in sp


class TestRunwareCompletion:
    @pytest.fixture(autouse=True)
    def _disable_aiohttp(self):
        # respx mocks the httpx transport, so the aiohttp transport must be off.
        original = getattr(litellm, "disable_aiohttp_transport", False)
        litellm.disable_aiohttp_transport = True
        yield
        litellm.disable_aiohttp_transport = original

    @pytest.mark.respx()
    def test_completion_mock(self, respx_mock):
        respx_mock.post(CHAT_URL).respond(json=_completion_payload(), status_code=200)

        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": "say hi"}],
            api_key="test-runware-key",
            api_base=API_BASE,
        )

        assert response is not None
        assert response.choices[0].message.content == "Hi from Runware!"
        assert response.usage.total_tokens == 21

    @pytest.mark.respx()
    def test_completion_sends_correct_url_and_body(self, respx_mock):
        route = respx_mock.post(CHAT_URL).respond(
            json=_completion_payload(), status_code=200
        )

        completion(
            model=MODEL,
            messages=[{"role": "user", "content": "say hi"}],
            api_key="test-runware-key",
            api_base=API_BASE,
            temperature=0.6,
        )

        assert route.called
        request = route.calls.last.request
        assert str(request.url) == CHAT_URL
        body = json.loads(request.content)
        assert body["model"] == BARE_MODEL
        assert body["messages"] == [{"role": "user", "content": "say hi"}]
        assert body["temperature"] == 0.6

    @pytest.mark.respx()
    def test_completion_respects_api_base_override(self, respx_mock):
        custom_base = "https://gateway.internal/v1"
        route = respx_mock.post(f"{custom_base}/chat/completions").respond(
            json=_completion_payload(content="via gateway"), status_code=200
        )

        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": "say hi"}],
            api_key="test-runware-key",
            api_base=custom_base,
        )

        assert route.called
        assert response.choices[0].message.content == "via gateway"

    @pytest.mark.respx()
    def test_async_completion_mock(self, respx_mock):
        respx_mock.post(CHAT_URL).respond(
            json=_completion_payload(content="async hi"), status_code=200
        )

        async def _run():
            return await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": "say hi"}],
                api_key="test-runware-key",
                api_base=API_BASE,
            )

        response = asyncio.run(_run())
        assert response.choices[0].message.content == "async hi"


class TestRunwareStreaming:
    def test_streaming_yields_content_chunks(self):
        chunks = list(
            completion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_key="test-runware-key",
                stream=True,
                mock_response="Hello from Runware",
            )
        )
        assert len(chunks) > 0
        text = "".join((c.choices[0].delta.content or "") for c in chunks)
        assert "Hello from Runware" in text

    def test_async_streaming_yields_content_chunks(self):
        async def _run():
            collected = []
            stream = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_key="test-runware-key",
                stream=True,
                mock_response="Streamed from Runware",
            )
            async for chunk in stream:
                collected.append(chunk.choices[0].delta.content or "")
            return "".join(collected)

        text = asyncio.run(_run())
        assert "Streamed from Runware" in text
