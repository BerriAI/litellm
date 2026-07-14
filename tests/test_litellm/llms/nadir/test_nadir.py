"""
Unit tests for the Nadir provider (https://getnadir.com).

Nadir is an OpenAI-compatible intelligent router: the virtual model
``nadir/auto`` is classified server-side and routed to the cheapest model that
clears the quality bar. These tests cover provider resolution, credential
handling, and config wiring without making a live API call.
"""

import os
import sys
from unittest.mock import patch

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

    def test_in_openai_compatible_providers(self):
        assert "nadir" in litellm.openai_compatible_providers

    def test_config_loads(self):
        assert litellm.NadirConfig().__class__.__name__ == "NadirConfig"

    def test_supported_params_nonempty(self):
        params = litellm.NadirConfig().get_supported_openai_params(model="auto")
        assert isinstance(params, list) and len(params) > 0
        # Streaming is advertised; real token-by-token requires the SSE-enabled
        # Nadir backend, otherwise stream=False returns a single completion.
        assert "stream" in params


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
