"""
Tests for the MiroMind JSON-driven provider.

MiroMind is registered via litellm/llms/openai_like/providers.json plus an
entry in the LlmProviders enum. There is no per-provider Python module —
everything routes through the OpenAI-like dynamic config machinery (see
tests/test_litellm/llms/openai_like/responses/test_openai_like_responses.py
for the generic mechanics). The tests here only assert that the miromind-
specific entries are wired into the registry and resolve to the documented
shape (responses endpoint, MiroMind base URL, `MIROMIND_API_KEY` env var).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))


class TestMiroMindProviderRegistration:
    def test_provider_loaded_from_providers_json(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("miromind")

    def test_provider_supports_responses_api(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("miromind") is True

    def test_provider_in_llm_providers_enum(self):
        from litellm.types.utils import LlmProviders

        assert LlmProviders("miromind") == LlmProviders.MIROMIND


class TestMiroMindResponsesConfig:
    def _make_config(self):
        from litellm.llms.openai_like.dynamic_config import (
            create_responses_config_class,
        )
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("miromind")
        return create_responses_config_class(provider)()

    def test_custom_llm_provider(self):
        assert self._make_config().custom_llm_provider == "miromind"

    def test_complete_url_default_base(self):
        """
        With no api_base override and no MIROMIND_API_BASE env var, the
        URL is built from providers.json `base_url`.
        """
        from unittest.mock import patch

        with patch(
            "litellm.llms.openai_like.dynamic_config.get_secret_str",
            return_value=None,
        ):
            url = self._make_config().get_complete_url(
                api_base=None, litellm_params={}
            )
        assert url == "https://api.miromind.ai/v1/responses"

    def test_complete_url_api_base_override(self):
        url = self._make_config().get_complete_url(
            api_base="https://api-test.miromind.example/v1",
            litellm_params={},
        )
        assert url == "https://api-test.miromind.example/v1/responses"

    def test_validate_environment_sets_bearer_from_env(self):
        from unittest.mock import patch

        with patch(
            "litellm.llms.openai_like.dynamic_config.get_secret_str",
            return_value="sk-miromind-test",
        ):
            headers = self._make_config().validate_environment(
                headers={}, model="mirothinker-1-7-deepresearch-mini", litellm_params=None
            )
        assert headers["Authorization"] == "Bearer sk-miromind-test"


@pytest.mark.parametrize(
    "model",
    [
        "miromind/mirothinker-1-7-deepresearch",
        "miromind/mirothinker-1-7-deepresearch-mini",
    ],
)
class TestMiroMindModelRegistration:
    """Both flagship models must be declared in the source-of-truth
    model_prices_and_context_window.json with mode=responses and
    supports_native_streaming=true. We read the file directly (rather
    than via litellm.model_cost) because the runtime fetches the cost
    map from a remote URL at import time, which doesn't reflect the
    in-PR edits until merge."""

    @staticmethod
    def _load_local_cost_map():
        import json
        import pathlib

        # tests/test_litellm/llms/miromind/test_miromind.py → repo root → JSON
        repo_root = pathlib.Path(__file__).resolve().parents[4]
        path = repo_root / "model_prices_and_context_window.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_model_present(self, model):
        assert model in self._load_local_cost_map()

    def test_provider_and_mode(self, model):
        entry = self._load_local_cost_map()[model]
        assert entry["litellm_provider"] == "miromind"
        assert entry["mode"] == "responses"

    def test_supports_native_streaming(self, model):
        """Without this flag, LiteLLM falls back to fake_stream — which
        tries to parse the SSE response as JSON and breaks every Responses
        streaming call. See PR description for the failure mode."""
        entry = self._load_local_cost_map()[model]
        assert entry.get("supports_native_streaming") is True

    def test_advertises_responses_endpoint(self, model):
        entry = self._load_local_cost_map()[model]
        assert "/v1/responses" in entry.get("supported_endpoints", [])
