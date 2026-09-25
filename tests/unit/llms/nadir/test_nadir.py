import json
import math
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import get_llm_provider
from litellm.types.utils import ModelResponse, Usage

NADIR_BASE = "https://api.getnadir.com/v1"
COST_HEADER = "llm_provider-x-litellm-response-cost"


def _transform(payload):
    raw = httpx.Response(
        200,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", f"{NADIR_BASE}/chat/completions"),
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


def _payload(**extra):
    return {
        "id": "req-1",
        "object": "chat.completion",
        "created": 0,
        "model": "claude-haiku-4-5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        **extra,
    }


def _cost(response, provider):
    return litellm.completion_cost(completion_response=response, custom_llm_provider=provider)


def _logged_cost(response):
    return litellm.response_cost_calculator(
        response_object=response,
        model="auto",
        custom_llm_provider="nadir",
        call_type="completion",
        optional_params={},
    )


class TestNadirProviderResolution:
    def test_model_prefix_resolves_to_nadir(self):
        model, provider, _, _ = get_llm_provider(model="nadir/auto", api_key="sk-test")
        assert (model, provider) == ("auto", "nadir")

    def test_default_api_base(self):
        _, _, _, api_base = get_llm_provider(model="nadir/auto", api_key="sk-test")
        assert api_base == NADIR_BASE

    def test_api_base_override(self):
        _, _, _, api_base = get_llm_provider(
            model="nadir/auto",
            api_key="sk-test",
            api_base="https://gateway.internal/v1",
        )
        assert api_base == "https://gateway.internal/v1"

    def test_endpoint_reverse_maps_to_nadir_with_the_env_key(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, provider, dynamic_api_key, _ = get_llm_provider(model="auto", api_base=NADIR_BASE)
        assert (provider, dynamic_api_key) == ("nadir", "sk-server-secret")

    def test_plaintext_endpoint_never_loads_the_env_key(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, provider, dynamic_api_key, _ = get_llm_provider(model="auto", api_base="http://api.getnadir.com/v1")
        assert provider == "nadir"
        assert dynamic_api_key is None


class TestNadirCredentialScoping:
    def test_env_key_used_for_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto")
        assert dynamic_api_key == "sk-server-secret"

    def test_env_key_used_when_base_matches_default(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto", api_base=f"{NADIR_BASE}/")
        assert dynamic_api_key == "sk-server-secret"

    def test_env_key_not_leaked_to_custom_base(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto", api_base="https://attacker.example/v1")
        assert dynamic_api_key is None

    def test_caller_key_used_for_custom_base(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        _, _, dynamic_api_key, _ = get_llm_provider(
            model="nadir/auto",
            api_base="https://self-hosted.internal/v1",
            api_key="sk-caller-own",
        )
        assert dynamic_api_key == "sk-caller-own"

    def test_env_key_used_for_operator_configured_base(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-server-secret")
        monkeypatch.setenv("NADIR_API_BASE", "https://nadir.mycorp.internal/v1")
        _, _, dynamic_api_key, _ = get_llm_provider(model="nadir/auto", api_base="https://nadir.mycorp.internal/v1")
        assert dynamic_api_key == "sk-server-secret"


class TestNadirParamMapping:
    def test_supported_params_are_mapped(self):
        params = litellm.get_optional_params(
            model="auto",
            custom_llm_provider="nadir",
            temperature=0.5,
            max_tokens=64,
        )
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 64

    def test_streaming_is_advertised_and_tools_are_not(self):
        params = litellm.get_supported_openai_params(model="auto", custom_llm_provider="nadir")
        assert "stream" in params
        assert "tools" not in params

    @pytest.mark.parametrize(
        "unsupported",
        [
            {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
            {"stop": ["\n"]},
            {"seed": 7},
            {"n": 2},
        ],
    )
    def test_params_nadir_would_silently_drop_are_rejected(self, unsupported):
        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.get_optional_params(model="auto", custom_llm_provider="nadir", **unsupported)

    def test_unsupported_params_are_dropped_when_asked(self):
        params = litellm.get_optional_params(
            model="auto",
            custom_llm_provider="nadir",
            drop_params=True,
            seed=7,
            temperature=0.2,
        )
        assert "seed" not in params
        assert params["temperature"] == 0.2


class TestNadirEnvValidation:
    def test_validate_environment_detects_key(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-live-xyz")
        result = litellm.validate_environment(model="nadir/auto")
        assert result["keys_in_environment"] is True

    def test_validate_environment_flags_missing_key(self, monkeypatch):
        monkeypatch.delenv("NADIR_API_KEY", raising=False)
        result = litellm.validate_environment(model="nadir/auto")
        assert "NADIR_API_KEY" in result["missing_keys"]


class TestNadirCostAttribution:
    def test_reported_cost_wins_over_model_pricing(self):
        res = _transform(_payload(nadir_metadata={"cost": {"total_cost_usd": 0.00123}}))
        assert _logged_cost(res) == pytest.approx(0.00123)
        assert _logged_cost(res) != _cost(res, "anthropic")

    def test_routed_model_is_preserved(self):
        res = _transform(_payload(nadir_metadata={"cost": {"total_cost_usd": 0.001}}))
        assert res.model == "claude-haiku-4-5"

    def test_missing_cost_prices_the_routed_model_from_its_own_entry(self):
        res = _transform(_payload())
        assert res.choices[0].message.content == "hi"
        assert COST_HEADER not in res._hidden_params.get("additional_headers", {})
        assert _cost(res, "nadir") == _logged_cost(res) == _cost(res, "anthropic") > 0

    def test_streamed_routed_model_prices_from_its_own_entry(self):
        res = ModelResponse(
            model="gemini-3.5-flash-lite",
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        assert _cost(res, "nadir") == _cost(res, "gemini") > 0

    @pytest.mark.parametrize("bad", [-0.001, math.nan, math.inf, -math.inf, True, "0.001", None])
    def test_invalid_reported_cost_falls_back_to_model_pricing(self, bad):
        res = _transform(_payload(nadir_metadata={"cost": {"total_cost_usd": bad}}))
        assert COST_HEADER not in res._hidden_params.get("additional_headers", {})
        assert _cost(res, "nadir") == _cost(res, "anthropic") > 0

    @pytest.mark.parametrize("metadata", ["oops", {"cost": "free"}, {"cost": None}, {}])
    def test_malformed_metadata_falls_back_to_model_pricing(self, metadata):
        res = _transform(_payload(nadir_metadata=metadata))
        assert COST_HEADER not in res._hidden_params.get("additional_headers", {})
        assert _cost(res, "nadir") == _cost(res, "anthropic") > 0


class TestNadirCompletionDispatch:
    def _call(self, **kwargs):
        captured = {}

        def fake_completion(**call_kwargs):
            captured.update(call_kwargs)
            return ModelResponse()

        with patch(  # test-quality-ok: these tests assert the dispatch wiring itself (nadir must reach base_llm_http_handler, and which credentials it is handed); faking HTTP would not observe that
            "litellm.main.base_llm_http_handler.completion", side_effect=fake_completion
        ):
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
        assert captured["api_base"] == NADIR_BASE

    def test_env_key_is_used_for_the_default_endpoint(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        assert self._call()["api_key"] == "sk-env"

    def test_caller_key_wins(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        assert self._call(api_key="sk-caller")["api_key"] == "sk-caller"

    def test_env_key_is_not_forwarded_to_a_caller_supplied_host(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        captured = self._call(api_base="https://attacker.example/v1")
        assert captured["api_key"] != "sk-env"
        assert captured["api_base"] == "https://attacker.example/v1"

    def test_global_key_is_not_forwarded_to_a_caller_supplied_host(self, monkeypatch):
        monkeypatch.delenv("NADIR_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", "sk-global")
        captured = self._call(api_base="https://attacker.example/v1")
        assert captured["api_key"] != "sk-global"

    def test_custom_api_base_is_honoured(self, monkeypatch):
        monkeypatch.setenv("NADIR_API_KEY", "sk-env")
        captured = self._call(api_base="https://nadir.internal/v1", api_key="sk-own")
        assert captured["api_base"] == "https://nadir.internal/v1"
        assert captured["api_key"] == "sk-own"
