"""
Tests for Y-API provider configuration and integration.

Y-API is a JSON-configured OpenAI-compatible relay. Its model ids keep the
upstream organization prefix, so they contain a slash of their own
(`y-api/deepseek/deepseek-v4-flash`) -- provider resolution must split on the
first slash only.
"""

import litellm

BASE_URL = "https://api.y-api.bestvirtualgoods.com/v1"


class TestYApiProviderConfig:
    """Test Y-API provider configuration"""

    def test_y_api_in_provider_list(self):
        """Test that y-api is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "Y_API")
        assert LlmProviders.Y_API.value == "y-api"
        assert "y-api" in litellm.provider_list

    def test_y_api_json_config_exists(self):
        """Test that y-api is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("y-api")

        y_api = JSONProviderRegistry.get("y-api")
        assert y_api is not None
        assert y_api.base_url == BASE_URL
        assert y_api.api_key_env == "YAPI_API_KEY"
        assert y_api.api_base_env == "YAPI_API_BASE"
        assert y_api.param_mappings == {"max_tokens": "max_completion_tokens"}

    def test_y_api_supported_endpoints(self):
        """Test the endpoints declared for y-api"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        y_api = JSONProviderRegistry.get("y-api")
        assert y_api is not None
        assert "/v1/chat/completions" in y_api.supported_endpoints
        assert "/v1/responses" in y_api.supported_endpoints
        assert "/v1/messages" in y_api.supported_endpoints
        assert JSONProviderRegistry.supports_responses_api("y-api") is True

    def test_y_api_provider_resolution_with_nested_model_id(self):
        """Test that provider resolution finds y-api and keeps the nested model id"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="y-api/deepseek/deepseek-v4-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek/deepseek-v4-flash"
        assert provider == "y-api"
        assert api_base == BASE_URL

    def test_y_api_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="y-api/deepseek/deepseek-v4-flash",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "y-api"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_y_api_router_config(self):
        """Test that y-api can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "y-api-chat",
                    "litellm_params": {
                        "model": "y-api/deepseek/deepseek-v4-flash",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "y-api-chat"

    def test_y_api_supported_endpoints_matrix(self):
        """The documented matrix lists y-api with the endpoints it actually serves."""
        import json
        from pathlib import Path

        import litellm as _litellm

        matrix_path = (
            Path(_litellm.__file__).parent.parent / "provider_endpoints_support.json"
        )
        matrix = json.loads(matrix_path.read_text())

        assert "y-api" in matrix["providers"]
        endpoints = matrix["providers"]["y-api"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is True
        # embeddings is advertised false: GET /v1/models returns no embedding
        # model, and POST /v1/embeddings answers
        # "No available channel for model ... under group y-api (distributor)".
        assert endpoints["embeddings"] is False

    def test_y_api_runtime_supported_endpoints_matrix(self):
        """The runtime-served backup matrix (GET /public/supported_endpoints) lists y-api."""
        import json
        from pathlib import Path

        import litellm as _litellm

        backup_path = (
            Path(_litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())

        assert "y-api" in matrix["providers"]
        endpoints = matrix["providers"]["y-api"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is True


class TestYApiDashboardRegistration:
    """The Add Model form must offer Y-API.

    `test_every_backend_provider_is_listed_in_add_model_or_frozen_as_unlisted`
    fails if a provider exists in `LlmProviders` without an entry here, so this
    is the difference between "selectable in the UI" and "silently dropped into
    the frozen unlisted set".
    """

    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = (
            Path(litellm.__file__).parent
            / "proxy"
            / "public_endpoints"
            / "provider_create_fields.json"
        )
        with open(path) as f:
            return json.load(f)

    def test_y_api_is_selectable_in_the_add_model_form(self):
        entries = [
            e
            for e in self._provider_create_fields()
            if e["litellm_provider"] == "y-api"
        ]
        assert (
            len(entries) == 1
        ), "y-api must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "Y_API"
        assert entry["provider_display_name"] == "Y-API"
        assert entry["default_model_placeholder"].startswith("y-api/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
        # The base URL is surfaced as the placeholder so admins can see the
        # default without it being written into config.
        assert fields["api_base"]["placeholder"] == BASE_URL

    def test_y_api_is_returned_by_the_public_providers_fields_endpoint(self):
        """The endpoint the Add Model dropdown actually reads from."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.public_endpoints.public_endpoints import router

        app_instance = FastAPI()
        app_instance.include_router(router)
        test_client = TestClient(app_instance)

        response = test_client.get("/public/providers/fields")
        assert response.status_code == 200
        providers = {p["litellm_provider"] for p in response.json()}
        assert "y-api" in providers


class TestYApiTokenParamMapping:
    """`max_tokens` has to reach this relay as `max_completion_tokens`.

    The `openai/*` models y-api fronts reject the legacy name outright:

        Unsupported parameter: 'max_tokens' is not supported with this model.
        Use 'max_completion_tokens' instead.

    LiteLLM's `openai_gpt` base class does not translate it for this provider --
    the model ids are not in the OpenAI cost map, so the caller's `max_tokens`
    is forwarded verbatim. These tests drive the transformation the provider
    actually applies to caller params, rather than reading the config, because
    the config alone cannot show which parameter name leaves the process.
    """

    @staticmethod
    def _map(**params):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("y-api")
        config = create_config_class(provider)()
        return config.map_openai_params(
            non_default_params=dict(params),
            optional_params={},
            model="gpt-5.6-luna",
            drop_params=False,
        )

    def test_max_tokens_is_rewritten(self):
        mapped = self._map(max_tokens=16)
        assert mapped["max_completion_tokens"] == 16
        assert "max_tokens" not in mapped

    def test_max_completion_tokens_passes_through(self):
        mapped = self._map(max_completion_tokens=16)
        assert mapped["max_completion_tokens"] == 16
        assert "max_tokens" not in mapped

    def test_absent_cap_sends_neither(self):
        mapped = self._map()
        assert "max_tokens" not in mapped
        assert "max_completion_tokens" not in mapped
