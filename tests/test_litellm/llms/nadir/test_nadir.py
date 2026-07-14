"""
Unit tests for the Nadir provider (https://getnadir.com).

Nadir is an OpenAI-compatible intelligent router: the virtual model
``nadir/auto`` is classified server-side and routed to the cheapest model that
clears the quality bar. These tests cover provider resolution, credential
handling, and config wiring without making a live API call.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import get_llm_provider
from litellm.types.utils import LlmProviders


class TestNadirProviderResolution:
    def test_model_prefix_resolves_to_nadir(self):
        model, provider, dynamic_api_key, api_base = get_llm_provider(model="nadir/auto", api_key="sk-test")
        assert provider == "nadir"
        # The nadir/ prefix is stripped; the virtual router alias is sent upstream.
        assert model == "auto"

    def test_default_api_base(self):
        _, _, _, api_base = get_llm_provider(model="nadir/auto", api_key="sk-test")
        assert api_base == "https://api.getnadir.com/v1"

    def test_api_base_override(self):
        _, _, _, api_base = get_llm_provider(
            model="nadir/auto",
            api_key="sk-test",
            api_base="https://gateway.internal/v1",
        )
        assert api_base == "https://gateway.internal/v1"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-live-env")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto")
        assert dynamic_api_key == "sk-live-env"

    def test_endpoint_reverse_maps_to_nadir(self):
        # A caller passing only the Nadir base_url (no nadir/ prefix) is still
        # identified as the nadir provider.
        _, provider, _, _ = get_llm_provider(
            model="auto",
            api_base="https://api.getnadir.com/v1",
            api_key="sk-test",
        )
        assert provider == "nadir"


class TestNadirCredentialScoping:
    """The server NADIR_API_KEY must never be forwarded to a caller-supplied host."""

    def test_env_key_used_for_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto")
        assert dynamic_api_key == "sk-server-secret"

    def test_env_key_used_when_base_matches_default(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto", api_base="https://api.getnadir.com/v1/")
        assert dynamic_api_key == "sk-server-secret"

    def test_env_key_NOT_leaked_to_custom_base(self, monkeypatch):
        # A caller-controlled api_base without a caller key must NOT receive
        # the server's env credential.
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto", api_base="https://attacker.example/v1")
        assert dynamic_api_key is None

    def test_caller_key_used_for_custom_base(self, monkeypatch):
        # A caller directing at a custom base may still supply their own key.
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(
            model="nadir/auto",
            api_base="https://self-hosted.internal/v1",
            api_key="sk-caller-own",
        )
        assert dynamic_api_key == "sk-caller-own"

    def test_env_key_used_for_operator_configured_base(self, monkeypatch):
        # An operator-configured NADIR_API_BASE is trusted; passing that same
        # base explicitly still uses the env key.
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        monkeypatch.setenv("NADIR_API_BASE", "https://nadir.mycorp.internal/v1")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto", api_base="https://nadir.mycorp.internal/v1")
        assert dynamic_api_key == "sk-server-secret"


class TestNadirRegistration:
    def test_enum_member(self):
        assert LlmProviders.NADIR.value == "nadir"

    def test_config_loads(self):
        assert litellm.NadirConfig().__class__.__name__ == "NadirConfig"

    def test_supported_params_nonempty(self):
        params = litellm.NadirConfig().get_supported_openai_params(model="auto")
        assert isinstance(params, list) and len(params) > 0
        # Streaming is advertised; real token-by-token requires the SSE-enabled
        # Nadir backend, otherwise stream=False returns a single completion.
        assert "stream" in params

    def test_unsupported_params_are_not_advertised(self):
        # Nadir validates into its own request schema and drops anything
        # outside it, so the provider must not inherit OpenAI's full param
        # list. Advertising these would silently no-op at request time.
        params = litellm.NadirConfig().get_supported_openai_params(model="auto")
        for unsupported in (
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "parallel_tool_calls",
            "stop",
            "seed",
            "n",
            "logprobs",
            "stream_options",
            "user",
        ):
            assert unsupported not in params, f"{unsupported} is not honored by Nadir"


class TestNadirParamMapping:
    def test_get_optional_params_maps_nadir(self):
        # Exercises the nadir branch in litellm.utils.get_optional_params.
        params = litellm.get_optional_params(
            model="auto",
            custom_llm_provider="nadir",
            temperature=0.5,
            max_tokens=64,
        )
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 64

    def test_get_supported_openai_params_dispatcher(self):
        # Exercises the nadir branch in the top-level get_supported_openai_params.
        params = litellm.get_supported_openai_params(model="auto", custom_llm_provider="nadir")
        assert isinstance(params, list) and "stream" in params


class TestNadirEnvValidation:
    def test_validate_environment_detects_key(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-live-xyz")
        result = litellm.validate_environment(model="nadir/auto")
        assert result["keys_in_environment"] is True

    def test_validate_environment_flags_missing_key(self, monkeypatch):
        monkeypatch.delenv("NADIR_API_KEY", raising=False)
        result = litellm.validate_environment(model="nadir/auto")
        assert "NADIR_API_KEY" in result["missing_keys"]


class TestNadirDispatch:
    """Nadir must not ride the generic OpenAI-compatible path.

    That path never calls ``provider_config.transform_response``, so the cost
    Nadir reports would be dropped and every call would record 0.0 spend.
    """

    def test_not_in_openai_compatible_providers(self):
        assert "nadir" not in litellm.openai_compatible_providers

    def test_provider_config_resolves(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model="auto", provider=LlmProviders.NADIR)
        assert config.__class__.__name__ == "NadirConfig"


class TestNadirCostAttribution:
    """The routed model is a vendor name with no nadir/* pricing entry, so the
    cost Nadir reports is what must reach the cost calculator."""

    def _transform(self, payload):
        import httpx
        from litellm.types.utils import ModelResponse

        raw = httpx.Response(
            200,
            json=payload,
            request=httpx.Request("POST", "https://api.getnadir.com/v1/chat/completions"),
        )
        return litellm.NadirConfig().transform_response(
            model="auto",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    def _payload(self, **extra):
        base = {
            "id": "req-1",
            "object": "chat.completion",
            "created": 0,
            "model": "claude-haiku-4-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        base.update(extra)
        return base

    def test_reported_cost_reaches_the_cost_calculator(self):
        from litellm.cost_calculator import get_response_cost_from_hidden_params

        res = self._transform(self._payload(nadir_metadata={"cost": {"total_cost_usd": 0.00123}}))
        assert get_response_cost_from_hidden_params(res._hidden_params) == 0.00123

    def test_routed_model_is_preserved(self):
        # Overwriting this with "auto" would misattribute every request.
        res = self._transform(self._payload(nadir_metadata={"cost": {"total_cost_usd": 0.001}}))
        assert res.model == "claude-haiku-4-5"

    def test_missing_cost_does_not_fail_the_response(self):
        res = self._transform(self._payload(nadir_metadata={}))
        assert res.choices[0].message.content == "hi"
        assert "llm_provider-x-litellm-response-cost" not in res._hidden_params.get("additional_headers", {})


class TestNadirCompletionDispatch:
    """`_complete_nadir` is the branch that routes Nadir through the httpx
    handler instead of the OpenAI SDK path. These pin its contract without a
    network call."""

    def _call(self, **kwargs):
        from litellm.types.utils import ModelResponse

        captured = {}

        def fake_completion(**call_kwargs):
            captured.update(call_kwargs)
            return ModelResponse()

        with patch("litellm.main.base_llm_http_handler.completion", side_effect=fake_completion):
            litellm.completion(
                model="nadir/auto",
                messages=[{"role": "user", "content": "hi"}],
                **kwargs,
            )
        return captured

    def test_routes_through_the_http_handler_as_nadir(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        captured = self._call()
        assert captured["custom_llm_provider"] == "nadir"
        assert captured["api_base"] == "https://api.getnadir.com/v1"

    def test_env_key_is_used_for_the_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        assert self._call()["api_key"] == "sk-env"

    def test_caller_key_wins(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        assert self._call(api_key="sk-caller")["api_key"] == "sk-caller"

    def test_env_key_is_not_forwarded_to_a_caller_supplied_host(self, monkeypatch):
        # The scoping lives in get_llm_provider; _complete_nadir must not undo
        # it by re-reading NADIR_API_KEY at dispatch time.
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        captured = self._call(api_base="https://attacker.example/v1")
        assert captured["api_key"] != "sk-env"
        assert captured["api_base"] == "https://attacker.example/v1"

    def test_custom_api_base_is_honoured(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        captured = self._call(api_base="https://nadir.internal/v1", api_key="sk-own")
        assert captured["api_base"] == "https://nadir.internal/v1"
        assert captured["api_key"] == "sk-own"


class TestNadirConfigSurface:
    def test_get_config_returns_a_mapping(self):
        assert isinstance(litellm.NadirConfig.get_config(), dict)

    def test_provider_info_defaults_the_base(self):
        base, key = litellm.NadirConfig()._get_openai_compatible_provider_info(None, "sk-x")
        assert base == "https://api.getnadir.com/v1"
        assert key == "sk-x"

    def test_provider_info_honours_an_explicit_base(self):
        base, _ = litellm.NadirConfig()._get_openai_compatible_provider_info("https://nadir.internal/v1", "sk-x")
        assert base == "https://nadir.internal/v1"

    def test_non_json_body_does_not_fail_the_response(self):
        # Exercises the narrow except: a body that is not JSON at all.
        import httpx
        from litellm.types.utils import ModelResponse

        raw = httpx.Response(
            200,
            text="not json",
            request=httpx.Request("POST", "https://api.getnadir.com/v1/chat/completions"),
        )
        with patch.object(litellm.NadirConfig.__bases__[0], "transform_response", return_value=ModelResponse()):
            res = litellm.NadirConfig().transform_response(
                model="auto",
                raw_response=raw,
                model_response=ModelResponse(),
                logging_obj=MagicMock(),
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )
        assert "llm_provider-x-litellm-response-cost" not in res._hidden_params.get("additional_headers", {})
