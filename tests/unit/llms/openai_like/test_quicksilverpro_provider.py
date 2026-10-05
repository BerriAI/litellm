"""
Tests for the QuickSilver Pro provider configuration and integration.

QuickSilver Pro (`quicksilverpro`) is a JSON-configured OpenAI-compatible
provider (https://quicksilverpro.io, operated by MachineFi Inc.). Its price /
context catalog is committed to the model cost map; these tests check
LiteLLM's own behaviour against the committed entries rather than pinning
upstream numbers.
"""

import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

QSP_MODELS: Final = tuple(
    sorted(name for name in litellm.model_cost if name.startswith("quicksilverpro/"))
)


class TestQuickSilverProProviderConfig:
    def test_quicksilverpro_has_cost_map_entries(self):
        assert QSP_MODELS

    def test_quicksilverpro_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "QUICKSILVERPRO")
        assert LlmProviders.QUICKSILVERPRO.value == "quicksilverpro"
        assert "quicksilverpro" in litellm.provider_list

    def test_quicksilverpro_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("quicksilverpro")

        provider = JSONProviderRegistry.get("quicksilverpro")
        assert provider is not None
        assert provider.base_url == "https://api.quicksilverpro.io/v1"
        assert provider.api_key_env == "QUICKSILVERPRO_API_KEY"
        assert provider.api_base_env == "QUICKSILVERPRO_API_BASE"
        assert provider.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_quicksilverpro_supported_endpoints(self):
        """chat completions, Responses API and Anthropic /v1/messages are
        declared; image generations have no JSON-provider mechanism and are
        not."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("quicksilverpro")
        assert provider is not None
        assert provider.supported_endpoints == [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        ]

    def test_quicksilverpro_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "quicksilverpro" in openai_compatible_providers

    def test_quicksilverpro_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="quicksilverpro/claude-sonnet-5-5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "claude-sonnet-5-5"
        assert provider == "quicksilverpro"
        assert api_base == "https://api.quicksilverpro.io/v1"

    def test_quicksilverpro_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="quicksilverpro/claude-sonnet-5-5",
            custom_llm_provider=None,
            api_base="https://custom.quicksilverpro.example/v1",
            api_key="sk-test",
        )

        assert provider == "quicksilverpro"
        assert api_base == "https://custom.quicksilverpro.example/v1"
        assert api_key == "sk-test"

    def test_quicksilverpro_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="claude-sonnet-5-5",
            custom_llm_provider=None,
            api_base="https://api.quicksilverpro.io/v1",
            api_key=None,
        )
        assert provider == "quicksilverpro"
        assert api_base == "https://api.quicksilverpro.io/v1"

    def test_quicksilverpro_max_completion_tokens_mapped(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("quicksilverpro")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="claude-sonnet-5-5",
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 256
        assert "max_completion_tokens" not in optional_params

    def test_quicksilverpro_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "quicksilverpro-chat",
                    "litellm_params": {
                        "model": "quicksilverpro/claude-sonnet-5-5",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "quicksilverpro-chat"


class TestQuickSilverProModelMetadata:
    @staticmethod
    def _load(path_parts):
        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_all_models_present_in_model_cost(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model in QSP_MODELS:
            assert model in model_cost, f"{model} missing from model cost map"
            assert model_cost[model]["litellm_provider"] == "quicksilverpro"
            assert model_cost[model]["mode"] == "chat"

    def test_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(
            ("litellm", "model_prices_and_context_window_backup.json")
        )
        for model in QSP_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], (
                f"{model} differs between root and backup json"
            )

    @pytest.mark.parametrize("model", QSP_MODELS)
    def test_quicksilverpro_model_cost_and_capabilities(self, model: str):
        """Behaviour against the committed entries: cost_per_token must charge
        each model's own input/output price, and get_model_info must expose a
        coherent entry. Expected values are read from the loaded cost map, not
        hard-coded."""
        from litellm.cost_calculator import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            custom_llm_provider="quicksilverpro",
        )
        model_info = litellm.get_model_info(model)

        assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
        assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
        assert model_info["output_cost_per_token"] > 0
        max_output = model_info.get("max_output_tokens") or model_info["max_input_tokens"]
        assert (
            model_info["max_tokens"]
            == max_output
            <= model_info["max_input_tokens"]
        )
        assert model_info["litellm_provider"] == "quicksilverpro"
        assert model_info["mode"] == "chat"
        assert litellm.supports_vision(model) is model_info["supports_vision"]

    def test_quicksilverpro_cost_map_is_queryable(self):
        """The entries must be consumable through litellm's own cost map, not
        just present in the JSON file."""
        from litellm import model_cost

        for model in QSP_MODELS:
            assert model in model_cost, model
            assert model_cost[model]["litellm_provider"] == "quicksilverpro"


class TestQuickSilverProDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        path = (
            Path(litellm.__file__).parent
            / "proxy"
            / "public_endpoints"
            / "provider_create_fields.json"
        )
        with open(path) as f:
            return json.load(f)

    def test_quicksilverpro_is_selectable_in_the_add_model_form(self):
        entries = [
            e
            for e in self._provider_create_fields()
            if e["litellm_provider"] == "quicksilverpro"
        ]
        assert (
            len(entries) == 1
        ), "quicksilverpro must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "QUICKSILVERPRO"
        assert entry["provider_display_name"] == "QuickSilver Pro"
        assert entry["default_model_placeholder"].startswith("quicksilverpro/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False

    def test_quicksilverpro_supported_endpoints_matrix(self):
        matrix = json.loads(
            (
                Path(litellm.__file__).parent
                / "provider_endpoints_support_backup.json"
            ).read_text()
        )

        endpoints = matrix["providers"]["quicksilverpro"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is True
        assert endpoints["embeddings"] is False
        assert endpoints["image_generations"] is False


def test_quicksilverpro_chat_completion_request():
    """chat/completions hits the provider's base URL with bearer auth and maps
    max_completion_tokens -> max_tokens."""
    with respx.mock() as upstream:
        route: Final = upstream.post(
            "https://api.quicksilverpro.io/v1/chat/completions"
        ).respond(
            200,
            json={
                "id": "chatcmpl_quicksilverpro",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "claude-sonnet-5-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from QuickSilver Pro"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="quicksilverpro/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=128,
            api_key="quicksilverpro-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.quicksilverpro.io/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer quicksilverpro-test-key"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 128
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "Hello from QuickSilver Pro"


def test_quicksilverpro_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(
            "https://api.quicksilverpro.io/v1/responses"
        ).respond(
            200,
            json={
                "id": "resp_quicksilverpro",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "claude-sonnet-5-5",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_quicksilverpro",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hello from QuickSilver Pro",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="quicksilverpro/claude-sonnet-5-5",
            input="Say hello",
            api_key="quicksilverpro-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.quicksilverpro.io/v1/responses"
    assert request.headers["authorization"] == "Bearer quicksilverpro-test-key"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from QuickSilver Pro"


@pytest.mark.asyncio
async def test_quicksilverpro_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(
            "https://api.quicksilverpro.io/v1/messages"
        ).respond(
            200,
            json={
                "id": "msg_quicksilverpro",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "Hello from QuickSilver Pro"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="quicksilverpro/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="quicksilverpro-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.quicksilverpro.io/v1/messages"
    assert request.headers["authorization"] == "Bearer quicksilverpro-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from QuickSilver Pro"
