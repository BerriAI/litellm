"""
Tests for the Sarvam provider configuration and cost map entries.
"""

import json

import pytest
import respx

import litellm
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

SARVAM_MODELS = (
    "sarvam/sarvam-105b",
    "sarvam/sarvam-105b-conversations",
    "sarvam/glm5.3",
    "sarvam/glm5.3-flash",
    "sarvam/gemma4",
    "sarvam/deepseekv4.1-flash-1017",
)


class TestSarvamProviderConfig:
    def test_sarvam_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="sarvam/sarvam-105b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "sarvam-105b"
        assert provider == "sarvam"
        assert api_base == "https://api.sarvam.ai/v2"

    def test_sarvam_responses_api_enabled(self):
        from litellm.utils import ProviderConfigManager

        assert JSONProviderRegistry.supports_responses_api("sarvam") is True
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="sarvam",
            model="sarvam/sarvam-105b",
        )
        assert config is not None
        assert config.custom_llm_provider == "sarvam"

    def test_sarvam_listed_in_add_model_provider_fields(self):
        from pathlib import Path

        fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        entries = json.loads(fields_path.read_text())
        sarvam_entries = [entry for entry in entries if entry["litellm_provider"] == "sarvam"]

        assert len(sarvam_entries) == 1
        entry = sarvam_entries[0]
        assert entry["provider"] == "SARVAM"
        assert entry["provider_display_name"] == "Sarvam"
        assert entry["default_model_placeholder"] == "sarvam/sarvam-105b"
        field_keys = {field["key"] for field in entry["credential_fields"]}
        assert "api_key" in field_keys
        assert "api_base" in field_keys

    def test_sarvam_chat_completion_request(self):
        with respx.mock() as upstream:
            route = upstream.post("https://api.sarvam.ai/v2/chat/completions").respond(
                200,
                json={
                    "id": "chatcmpl_sarvam",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "sarvam-105b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                },
            )
            response = litellm.completion(
                model="sarvam/sarvam-105b",
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="sarvam-test-key",
                max_completion_tokens=200,
            )

        request = route.calls.last.request
        body = json.loads(request.content)
        assert route.call_count == 1
        assert str(request.url) == "https://api.sarvam.ai/v2/chat/completions"
        assert request.headers["authorization"] == "Bearer sarvam-test-key"
        # LiteLLM maps max_completion_tokens to max_tokens for OpenAI-compatible providers.
        assert body.get("max_completion_tokens", body.get("max_tokens")) == 200
        assert response.choices[0].message.content == "Hello"

    def test_sarvam_responses_request(self):
        with respx.mock() as upstream:
            route = upstream.post("https://api.sarvam.ai/v2/responses").respond(
                200,
                json={
                    "id": "resp_sarvam",
                    "object": "response",
                    "created_at": 1,
                    "model": "sarvam-105b",
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg_sarvam",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                },
            )
            response = litellm.responses(
                model="sarvam/sarvam-105b",
                input="Say hello",
                api_key="sarvam-test-key",
            )

        request = route.calls.last.request
        assert route.call_count == 1
        assert str(request.url) == "https://api.sarvam.ai/v2/responses"
        assert request.headers["authorization"] == "Bearer sarvam-test-key"
        assert response.output_text == "Hello"


@pytest.mark.usefixtures("local_model_cost_map")
class TestSarvamCostMap:
    @pytest.mark.parametrize("model", SARVAM_MODELS)
    def test_model_registered(self, model):
        assert model in litellm.model_cost

        entry = litellm.model_cost[model]
        assert entry["litellm_provider"] == "sarvam"
        assert entry["mode"] == "chat"

    @pytest.mark.parametrize("model", SARVAM_MODELS)
    def test_cached_input_cost_is_set(self, model):
        entry = litellm.model_cost[model]
        assert entry["cache_read_input_token_cost"] > 0

    def test_sarvam_m_kept_for_deprecation(self):
        # Source: https://docs.sarvam.ai/changelog/2026/6/5.md (2026-06-05)
        entry = litellm.model_cost["sarvam/sarvam-m"]
        assert entry["deprecation_date"] == "2026-06-05"
        assert entry["cache_creation_input_token_cost_above_1hr"] == 0
