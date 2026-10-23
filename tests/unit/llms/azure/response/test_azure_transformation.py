import json
from collections.abc import Callable
from copy import deepcopy
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.constants import AZURE_DEFAULT_RESPONSES_API_VERSION
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.azure.responses.o_series_transformation import (
    AzureOpenAIOSeriesResponsesAPIConfig,
)
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from tests.unit.proxy.conftest import httpx_transport


@pytest.mark.serial
def test_validate_environment_api_key_within_litellm_params():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    litellm_params = GenericLiteLLMParams(api_key="test-api-key")

    result = azure_openai_responses_apiconfig.validate_environment(
        headers={}, model="", litellm_params=litellm_params
    )

    expected = {"api-key": "test-api-key"}

    assert result == expected


@pytest.mark.serial
def test_validate_environment_api_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.api_key", "test-api-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-api-key"}

        assert result == expected


@pytest.mark.serial
def test_validate_environment_azure_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()

    with patch("litellm.azure_key", "test-azure-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-azure-key"}

        assert result == expected


@pytest.mark.serial
def test_validate_environment_azure_key_within_headers():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    headers = {"api-key": "test-api-key-from-headers"}
    litellm_params = GenericLiteLLMParams()

    result = azure_openai_responses_apiconfig.validate_environment(
        headers=headers, model="", litellm_params=litellm_params
    )

    expected = {"api-key": "test-api-key-from-headers"}

    assert result == expected


@pytest.mark.serial
def test_get_complete_url():
    """
    Test the get_complete_url function
    """
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    api_base = "https://litellm8397336933.openai.azure.com"
    litellm_params = {"api_version": "2024-05-01-preview"}

    result = azure_openai_responses_apiconfig.get_complete_url(
        api_base=api_base, litellm_params=litellm_params
    )

    expected = "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2024-05-01-preview"

    assert result == expected


@pytest.mark.parametrize(
    ("base_path", "expected_path"),
    [
        ("/foundry/openai/openai/v1", "/foundry/openai/openai/v1/responses?api-version=v1"),
        ("/foundry/openai/openai/v1/", "/foundry/openai/openai/v1/responses?api-version=v1"),
        ("/foundry/openai/openai/v1/responses", "/foundry/openai/openai/v1/responses?api-version=v1"),
        ("/foundry/openai/openai/v1/responses/", "/foundry/openai/openai/v1/responses?api-version=v1"),
        (
            "/openai/v1?redirect=/openai/v1/responses",
            "/openai/v1/responses?redirect=%2Fopenai%2Fv1%2Fresponses&api-version=v1",
        ),
        (
            "/openai/v1?tenant=example&api-version=preview",
            "/openai/v1/responses?tenant=example&api-version=preview",
        ),
        (
            "/openai/v1/responses?api-version=2025-04-01-preview",
            "/openai/v1/responses?api-version=2025-04-01-preview",
        ),
        (
            "/openai/v1/responses?redirect=/openai/responses",
            "/openai/v1/responses?redirect=%2Fopenai%2Fresponses&api-version=v1",
        ),
        ("/foo%2Fbar/openai/v1?tenant=example", "/foo%2Fbar/openai/v1/responses?tenant=example&api-version=v1"),
        (
            "/foo%3Fbar/openai/v1/responses?tenant=example",
            "/foo%3Fbar/openai/v1/responses?tenant=example&api-version=v1",
        ),
        ("/openai", "/openai/v1/responses?api-version=v1"),
        ("/foundry/openai", "/foundry/openai/v1/responses?api-version=v1"),
    ],
)
def test_explicit_v1_api_base_preserves_the_gateway_prefix_and_query(base_path: str, expected_path: str) -> None:
    result: Final = AzureOpenAIResponsesAPIConfig().get_complete_url(
        api_base=f"https://gateway.example.com{base_path}", litellm_params={"api_version": "v1"}
    )

    assert result == f"https://gateway.example.com{expected_path}"


@pytest.mark.parametrize(
    ("litellm_params", "expected_version"),
    [
        ({}, AZURE_DEFAULT_RESPONSES_API_VERSION),
        ({"api_version": "v1"}, "v1"),
        ({"api_version": "latest"}, "latest"),
        ({"api_version": "preview"}, "preview"),
        ({"api_version": "2025-04-01-preview"}, "2025-04-01-preview"),
        ({"api_version": "2024-10-21"}, AZURE_DEFAULT_RESPONSES_API_VERSION),
        ({"api_version": "2024-10-21", "responses_api_version": "preview"}, "preview"),
    ],
)
def test_explicit_v1_api_base_keeps_responses_api_version_selection(
    litellm_params: dict[str, str], expected_version: str
) -> None:
    result: Final = AzureOpenAIResponsesAPIConfig().get_complete_url(
        api_base="https://gateway.example.com/foundry/openai/openai/v1",
        litellm_params=litellm_params,
    )

    assert result == f"https://gateway.example.com/foundry/openai/openai/v1/responses?api-version={expected_version}"


@pytest.mark.serial
def test_response_id_path_requests_encode_response_id():
    config = AzureOpenAIResponsesAPIConfig()
    api_base = (
        "https://litellm8397336933.openai.azure.com/openai/responses"
        "?api-version=2024-05-01-preview"
    )

    url, params = config.transform_cancel_response_api_request(
        response_id="../../responses/other?x=1#frag",
        api_base=api_base,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        url
        == "https://litellm8397336933.openai.azure.com/openai/responses/..%2F..%2Fresponses%2Fother%3Fx%3D1%23frag/cancel?api-version=2024-05-01-preview"
    )
    assert params == {}


@pytest.mark.serial
def test_azure_o_series_responses_api_supported_params():
    """Test that Azure OpenAI O-series responses API excludes temperature from supported parameters."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()
    supported_params = config.get_supported_openai_params("o_series/gpt-o1")

    # Temperature should not be in supported params for O-series models
    assert "temperature" not in supported_params

    # Other parameters should still be supported
    assert "input" in supported_params
    assert "max_output_tokens" in supported_params
    assert "stream" in supported_params
    assert "top_p" in supported_params


@pytest.mark.serial
def test_azure_o_series_responses_api_drop_temperature_param():
    """Test that temperature parameter is dropped when drop_params is True for O-series models."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()

    # Create request params with temperature
    request_params = ResponsesAPIOptionalRequestParams(
        temperature=0.7, max_output_tokens=1000, stream=False, top_p=0.9
    )

    # Test with drop_params=True
    mapped_params_with_drop = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=True,
    )

    # Temperature should be dropped
    assert "temperature" not in mapped_params_with_drop
    # Other params should remain
    assert mapped_params_with_drop["max_output_tokens"] == 1000
    assert mapped_params_with_drop["top_p"] == 0.9

    # Test with drop_params=False
    mapped_params_without_drop = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=False,
    )

    # Temperature should still be present when drop_params=False
    assert mapped_params_without_drop["temperature"] == 0.7
    assert mapped_params_without_drop["max_output_tokens"] == 1000
    assert mapped_params_without_drop["top_p"] == 0.9


@pytest.mark.serial
def test_azure_o_series_responses_api_drop_params_no_temperature():
    """Test that map_openai_params works correctly when temperature is not present for O-series models."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()

    # Create request params without temperature
    request_params = ResponsesAPIOptionalRequestParams(
        max_output_tokens=1000, stream=False, top_p=0.9
    )

    # Should work fine even with drop_params=True
    mapped_params = config.map_openai_params(
        response_api_optional_params=request_params,
        model="o_series/gpt-o1",
        drop_params=True,
    )

    assert "temperature" not in mapped_params
    assert mapped_params["max_output_tokens"] == 1000
    assert mapped_params["top_p"] == 0.9


@pytest.mark.serial
def test_azure_regular_responses_api_supports_temperature():
    """Test that regular Azure OpenAI responses API (non-O-series) supports temperature parameter."""
    config = AzureOpenAIResponsesAPIConfig()
    supported_params = config.get_supported_openai_params("gpt-4o")

    # Regular Azure models should support temperature
    assert "temperature" in supported_params

    # Other parameters should still be supported
    assert "input" in supported_params
    assert "max_output_tokens" in supported_params
    assert "stream" in supported_params
    assert "top_p" in supported_params


@pytest.mark.serial
def test_o_series_model_detection():
    """Test that the O-series configuration correctly identifies O-series models."""
    config = AzureOpenAIOSeriesResponsesAPIConfig()

    # Test explicit o_series naming
    assert config.is_o_series_model("o_series/gpt-o1")
    assert config.is_o_series_model("azure/o_series/gpt-o3")

    # Test regular models
    assert not config.is_o_series_model("gpt-4o")
    assert not config.is_o_series_model("gpt-3.5-turbo")


@pytest.mark.serial
def test_provider_config_manager_o_series_selection():
    """Test that ProviderConfigManager returns the correct config for O-series vs regular models."""
    import litellm
    from litellm.utils import ProviderConfigManager

    # Test O-series model selection
    o_series_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE, model="o_series/gpt-o1"
    )
    assert isinstance(o_series_config, AzureOpenAIOSeriesResponsesAPIConfig)

    # Test regular model selection
    regular_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE, model="gpt-4o"
    )
    assert isinstance(regular_config, AzureOpenAIResponsesAPIConfig)
    assert not isinstance(regular_config, AzureOpenAIOSeriesResponsesAPIConfig)

    # Test with no model specified (should default to regular)
    default_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.AZURE, model=None
    )
    assert isinstance(default_config, AzureOpenAIResponsesAPIConfig)
    assert not isinstance(default_config, AzureOpenAIOSeriesResponsesAPIConfig)


_ARTIFACT_FIELD_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'


class TestAzureResponsesAPIConfig:
    def setup_method(self):
        self.config = AzureOpenAIResponsesAPIConfig()
        self.model = "gpt-4o"
        self.logging_obj = MagicMock()

    def test_azure_get_complete_url_with_version_types(self):
        """Test Azure get_complete_url with different API version types"""
        base_url = "https://litellm8397336933.openai.azure.com"

        # Test with preview version - should use openai/v1/responses
        result_preview = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "preview"},
        )
        assert (
            result_preview
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=preview"
        )

        # Test with latest version - should use openai/v1/responses
        result_latest = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "latest"},
        )
        assert (
            result_latest
            == "https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version=latest"
        )

        result_dated_preview = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": "2025-04-01-preview"},
        )
        assert (
            result_dated_preview
            == "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
        )

    @pytest.mark.parametrize("api_version", ["2024-10-21", "2024-06-01", "2025-01-01"])
    def test_dated_ga_api_version_routes_responses_like_an_unset_one(self, api_version: str):
        base_url: Final = "https://litellm8397336933.openai.azure.com"

        dated: Final = self.config.get_complete_url(api_base=base_url, litellm_params={"api_version": api_version})
        unset: Final = self.config.get_complete_url(api_base=base_url, litellm_params={})

        assert dated == unset
        assert httpx.URL(dated).params["api-version"] == AZURE_DEFAULT_RESPONSES_API_VERSION

    def test_azure_get_complete_url_with_default_api_version(self):
        """Test Azure get_complete_url uses default API version when none is provided"""
        from litellm.constants import AZURE_DEFAULT_RESPONSES_API_VERSION

        base_url = "https://litellm8397336933.openai.azure.com"

        # Test with no api_version provided - should use default
        result_no_version = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={},
        )
        expected_url = f"https://litellm8397336933.openai.azure.com/openai/v1/responses?api-version={AZURE_DEFAULT_RESPONSES_API_VERSION}"
        assert result_no_version == expected_url

        # Test with empty litellm_params - should use default
        result_empty_params = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={},
        )
        assert result_empty_params == expected_url

        # Test with None api_version - should use default
        result_none_version = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": None},
        )
        assert result_none_version == expected_url

    @pytest.mark.parametrize(
        ("api_version", "responses_api_version", "expected_path"),
        [
            ("2024-10-21", "2025-04-01-preview", "/openai/responses?api-version=2025-04-01-preview"),
            ("2025-04-01-preview", "v1", "/openai/v1/responses?api-version=v1"),
            (None, "2025-04-01-preview", "/openai/responses?api-version=2025-04-01-preview"),
        ],
    )
    def test_responses_api_version_wins_over_api_version(
        self, api_version: str | None, responses_api_version: str, expected_path: str
    ):
        base_url: Final = "https://litellm8397336933.openai.azure.com"

        url: Final = self.config.get_complete_url(
            api_base=base_url,
            litellm_params={"api_version": api_version, "responses_api_version": responses_api_version},
        )

        assert url == f"{base_url}{expected_path}"

    @pytest.mark.parametrize("responses_api_version", [None, "v1"])
    def test_api_version_in_api_base_query_wins_over_a_dated_ga_api_version(self, responses_api_version: str | None):
        api_base: Final = "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2025-04-01-preview"

        url: Final = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={"api_version": "2024-10-21", "responses_api_version": responses_api_version},
        )

        assert url == api_base

    def test_api_version_in_env_api_base_query_wins_over_a_dated_ga_api_version(self, monkeypatch: pytest.MonkeyPatch):
        env_api_base: Final = (
            "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
        )
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setenv("AZURE_API_BASE", env_api_base)

        url: Final = self.config.get_complete_url(api_base=None, litellm_params={"api_version": "2024-10-21"})

        assert url == env_api_base

    @pytest.mark.parametrize(
        ("api_base", "expected_path"),
        [
            ("https://apim.example.com/openai-gw/team", "/openai-gw/team/openai/v1/responses"),
            ("https://apim.example.com/openai/team-a", "/openai/team-a/openai/v1/responses"),
        ],
    )
    @pytest.mark.parametrize("litellm_params", [{"api_version": "2024-10-21"}, {"responses_api_version": "v1"}])
    def test_v1_route_only_rewrites_the_responses_route_segment_of_a_gateway_api_base(
        self, api_base: str, expected_path: str, litellm_params: dict[str, str]
    ):
        url: Final = httpx.URL(self.config.get_complete_url(api_base=api_base, litellm_params=litellm_params))

        assert url.host == "apim.example.com"
        assert url.path == expected_path

    def test_azure_cancel_response_api_request(self):
        """Test Azure cancel response API request transformation"""
        from litellm.types.router import GenericLiteLLMParams

        response_id = "resp_test123"
        api_base = "https://test.openai.azure.com/openai/responses?api-version=2024-05-01-preview"
        litellm_params = GenericLiteLLMParams(api_version="2024-05-01-preview")
        headers = {"Authorization": "Bearer test-key"}

        url, data = self.config.transform_cancel_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        expected_url = "https://test.openai.azure.com/openai/responses/resp_test123/cancel?api-version=2024-05-01-preview"
        assert url == expected_url
        assert data == {}

    def test_azure_list_input_items_request_url_path_before_query(self):
        from litellm.types.router import GenericLiteLLMParams

        api_base = "https://test.openai.azure.com/openai/responses?api-version=2025-03-01-preview"

        url, params = self.config.transform_list_input_items_request(
            response_id="resp_test123",
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(api_version="2025-03-01-preview"),
            headers={},
        )

        assert (
            url
            == "https://test.openai.azure.com/openai/responses/resp_test123/input_items?api-version=2025-03-01-preview"
        )
        assert params == {"limit": 20, "order": "desc"}

    def test_azure_cancel_response_api_response(self):
        """Test Azure cancel response API response transformation"""
        from unittest.mock import Mock

        from litellm.types.llms.openai import ResponsesAPIResponse

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "resp_test123",
            "object": "response",
            "created_at": 1234567890,
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "status": "cancelled",
        }
        mock_response.text = "test response"
        mock_response.status_code = 200

        # Mock logging object
        mock_logging_obj = Mock()

        result = self.config.transform_cancel_response_api_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
        )

        assert isinstance(result, ResponsesAPIResponse)
        assert result.id == "resp_test123"

    def test_azure_responses_api_tool_flattening_nested_to_flat(self):
        """Test that nested tools are flattened correctly"""
        from litellm.types.router import GenericLiteLLMParams

        # Setup
        nested_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response_api_params = {"tools": nested_tools}
        litellm_params = GenericLiteLLMParams()

        # Execute
        self.config.transform_responses_api_request(
            model=self.model,
            input="test input",
            response_api_optional_request_params=response_api_params,
            litellm_params=litellm_params,
            headers={},
        )

        # Verify
        expected_tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        assert response_api_params["tools"] == expected_tools

    def test_azure_responses_api_tool_flattening_already_flat(self):
        """Test that already flat tools are passed through unchanged"""
        from litellm.types.router import GenericLiteLLMParams

        # Setup
        flat_tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        # Make a copy to check it doesn't change
        response_api_params = {"tools": list(flat_tools)}
        litellm_params = GenericLiteLLMParams()

        # Execute
        self.config.transform_responses_api_request(
            model=self.model,
            input="test input",
            response_api_optional_request_params=response_api_params,
            litellm_params=litellm_params,
            headers={},
        )

        # Verify
        assert response_api_params["tools"] == flat_tools

    def test_azure_responses_api_tool_flattening_preserves_original(self):
        """Test that the original tool dictionary is not mutated"""
        from litellm.types.router import GenericLiteLLMParams

        # Setup
        original_tool = {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {}},
        }
        original_tool_copy = deepcopy(original_tool)

        response_api_params = {"tools": [original_tool]}
        litellm_params = GenericLiteLLMParams()

        # Execute
        self.config.transform_responses_api_request(
            model=self.model,
            input="test input",
            response_api_optional_request_params=response_api_params,
            litellm_params=litellm_params,
            headers={},
        )

        assert original_tool == original_tool_copy

    def test_azure_responses_api_tool_flattening_mixed_tools(self):
        """Test mixed nested and flat tools"""
        from litellm.types.router import GenericLiteLLMParams

        # Setup
        nested_tool = {
            "type": "function",
            "function": {"name": "nested", "parameters": {}},
        }
        flat_tool = {"type": "function", "name": "flat", "parameters": {}}

        response_api_params = {"tools": [nested_tool, flat_tool]}
        litellm_params = GenericLiteLLMParams()

        # Execute
        self.config.transform_responses_api_request(
            model=self.model,
            input="test input",
            response_api_optional_request_params=response_api_params,
            litellm_params=litellm_params,
            headers={},
        )

        # Verify
        assert len(response_api_params["tools"]) == 2

        # First tool should be flattened
        assert "function" not in response_api_params["tools"][0]
        assert response_api_params["tools"][0]["name"] == "nested"

        # Second tool should remain as is
        assert response_api_params["tools"][1] == flat_tool

    def test_azure_responses_api_tool_flattening_no_tools(self):
        """Test handling when no tools are present"""
        from litellm.types.router import GenericLiteLLMParams

        # Setup
        response_api_params = {}
        litellm_params = GenericLiteLLMParams()

        # Execute - should not crash
        self.config.transform_responses_api_request(
            model=self.model,
            input="test input",
            response_api_optional_request_params=response_api_params,
            litellm_params=litellm_params,
            headers={},
        )

        assert "tools" not in response_api_params

    def test_azure_responses_api_context_management_unsupported(self):
        """Test that context_management is not in Azure supported params.

        Azure does not support context_management (compaction). It should be
        excluded from supported params so it gets dropped.
        """
        supported = self.config.get_supported_openai_params(self.model)
        assert "context_management" not in supported

    def _anyof_tool(self):
        return {
            "type": "function",
            "name": "automation_update",
            "description": "Update an automation",
            "parameters": {
                "type": "object",
                "anyOf": [
                    {
                        "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                        "required": ["id", "enabled"],
                    },
                    {
                        "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                        "required": ["id", "schedule"],
                    },
                ],
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        }

    def test_azure_flattens_top_level_anyof_for_gpt4_family_deployment_name(self):
        result = self.config.transform_responses_api_request(
            model="gpt-4o",
            input="hi",
            response_api_optional_request_params={"tools": [self._anyof_tool()]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        parameters = result["tools"][0]["parameters"]
        assert "anyOf" not in parameters
        assert parameters["type"] == "object"
        assert set(parameters["properties"]) == {"id", "enabled", "schedule"}
        assert parameters["required"] == ["id"]

    def test_azure_flattens_via_base_model_for_arbitrary_deployment_name(self):
        result = self.config.transform_responses_api_request(
            model="my-eastus-deployment",
            input="hi",
            response_api_optional_request_params={"tools": [self._anyof_tool()]},
            litellm_params=GenericLiteLLMParams(model_info={"base_model": "azure/gpt-4o"}),
            headers={},
        )

        assert "anyOf" not in result["tools"][0]["parameters"]

    def test_azure_keeps_combinators_for_gpt5_base_model(self):
        tool = self._anyof_tool()

        result = self.config.transform_responses_api_request(
            model="my-eastus-deployment",
            input="hi",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=GenericLiteLLMParams(model_info={"base_model": "azure/gpt-5.4-mini"}),
            headers={},
        )

        assert result["tools"][0] is tool
        assert "anyOf" in result["tools"][0]["parameters"]

    def test_azure_drops_non_python_regex_pattern_while_keeping_gpt5_combinators(self):
        tool = {
            "type": "function",
            "name": "Artifact",
            "parameters": {
                "type": "object",
                "anyOf": [{"properties": {"field": {"type": "string", "pattern": _ARTIFACT_FIELD_PATTERN}}}],
                "properties": {"field": {"type": "string", "pattern": _ARTIFACT_FIELD_PATTERN}},
            },
        }

        result = self.config.transform_responses_api_request(
            model="my-eastus-deployment",
            input="hi",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=GenericLiteLLMParams(model_info={"base_model": "azure/gpt-5.4-mini"}),
            headers={},
        )

        assert result["tools"][0]["parameters"] == {
            "type": "object",
            "anyOf": [{"properties": {"field": {"type": "string"}}}],
            "properties": {"field": {"type": "string"}},
        }

    def test_azure_keeps_combinators_for_unrecognized_deployment_without_base_model(self):
        tool = self._anyof_tool()

        result = self.config.transform_responses_api_request(
            model="my-eastus-deployment",
            input="hi",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert result["tools"][0] is tool
        assert "anyOf" in result["tools"][0]["parameters"]


@pytest.fixture()
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the bundled cost map: the published map lags a key added in this repo."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))
    litellm.add_known_models(model_cost_map=litellm.model_cost)


def test_azure_responses_reasoning_effort_none_unlocks_temperature_off_the_azure_row(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    """The gate reads the azure/ cost-map entry for the bare deployment name rather than OpenAI's
    twin: with OpenAI's gpt-6-sol row turned off for none, azure/gpt-6-sol's row still lets none
    through and, only then, a non-default temperature."""
    monkeypatch.setitem(
        litellm.model_cost, "gpt-6-sol", {**litellm.model_cost["gpt-6-sol"], "supports_none_reasoning_effort": False}
    )
    params = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            temperature=0.2,
            reasoning={"effort": "none"},
        ),
        model="gpt-6-sol",
        drop_params=False,
    )
    assert params["temperature"] == 0.2
    assert params["reasoning"] == {"effort": "none"}


def test_azure_responses_gpt6_astra_rejects_temperature_while_reasoning(local_model_cost_map: None):
    with pytest.raises(litellm.UnsupportedParamsError):
        AzureOpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(
                temperature=0.2,
                reasoning={"effort": "low"},
            ),
            model="gpt-6-astra",
            drop_params=False,
        )


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
@pytest.mark.parametrize("level", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_azure_responses_gpt6_effort_levels_follow_the_azure_row(local_model_cost_map: None, model: str, level: str):
    """A bare deployment name reads the azure/ row, opt-out for every level but max, so the none
    and minimal Foundry refuses on this route (live 400s on 2026-10-01 for gpt-6-astra and 2026-10-02
    for gpt-6.1-sol) are dropped or refused here while the max it honors goes through whatever the
    route-blind flag says."""
    forwarded = (
        level == "max" or litellm.model_cost[f"azure/{model}"].get(f"supports_{level}_reasoning_effort") is not False
    )
    dropped = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(reasoning={"effort": level}),
        model=model,
        drop_params=True,
    )
    assert ("reasoning" in dropped) is forwarded
    if forwarded:
        kept = AzureOpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(reasoning={"effort": level}),
            model=model,
            drop_params=False,
        )
        assert kept["reasoning"] == {"effort": level}
    else:
        with pytest.raises(litellm.UnsupportedParamsError):
            AzureOpenAIResponsesAPIConfig().map_openai_params(
                response_api_optional_params=ResponsesAPIOptionalRequestParams(reasoning={"effort": level}),
                model=model,
                drop_params=False,
            )


def test_azure_gpt6_astra_keeps_max_on_this_route_whatever_its_row_says(local_model_cost_map: None):
    """Pinned on purpose: the row-derived test above exempts max by hand, so this one has to fail
    if the gate ever reads the flag. A Foundry gpt-6-astra deployment answered reasoning.effort max
    through /v1/responses with 200, the effort echoed back and reasoning tokens billed, on
    2026-10-01, while chat refuses the same level, as OpenAI's chat does. The row keeps max off for
    the adapter and the picker, and this gate still forwards it even under drop_params."""
    assert litellm.model_cost["azure/gpt-6-astra"]["supports_max_reasoning_effort"] is False
    kept = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(reasoning={"effort": "max"}),
        model="gpt-6-astra",
        drop_params=True,
    )
    assert kept["reasoning"] == {"effort": "max"}


@pytest.mark.parametrize("level", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_azure_responses_effort_gate_fires_when_azure_ai_prefix_survives_provider_remap(
    local_model_cost_map: None, level: str
):
    """litellm.responses(model="azure_ai/gpt-6.1-sol", api_base=<Foundry host>) is re-routed to the
    azure provider with the prefix intact, so this config sees the prefixed name. The gate used to
    skip every prefixed name as another provider's pass-through, and a live Foundry gpt-6.1-sol
    answered the forwarded none and minimal with 400 on 2026-10-02 even under drop_params. The
    prefixed name reads the azure_ai/ row and drops or refuses exactly what that row turns off."""
    forwarded = (
        level == "max"
        or litellm.model_cost["azure_ai/gpt-6.1-sol"].get(f"supports_{level}_reasoning_effort") is not False
    )
    dropped = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(
            max_output_tokens=60, reasoning={"effort": level}
        ),
        model="azure_ai/gpt-6.1-sol",
        drop_params=True,
    )
    assert ("reasoning" in dropped) is forwarded
    if forwarded:
        return
    with pytest.raises(litellm.UnsupportedParamsError):
        AzureOpenAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(
                max_output_tokens=60, reasoning={"effort": level}
            ),
            model="azure_ai/gpt-6.1-sol",
            drop_params=False,
        )


def test_azure_responses_prefixed_name_reads_the_foundry_row_not_the_azure_twin(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    """The rows are made to disagree on low, a level neither real row flags, so the prefixed name
    proving it reads azure_ai/gpt-6-sol cannot pass by reading azure/gpt-6-sol, which the bare
    deployment name keeps reading."""
    monkeypatch.setitem(
        litellm.model_cost,
        "azure_ai/gpt-6-sol",
        {**litellm.model_cost["azure_ai/gpt-6-sol"], "supports_low_reasoning_effort": False},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/gpt-6-sol",
        {**litellm.model_cost["azure/gpt-6-sol"], "supports_low_reasoning_effort": True},
    )
    prefixed = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(reasoning={"effort": "low"}),
        model="azure_ai/gpt-6-sol",
        drop_params=True,
    )
    assert "reasoning" not in prefixed
    bare = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=ResponsesAPIOptionalRequestParams(reasoning={"effort": "low"}),
        model="gpt-6-sol",
        drop_params=True,
    )
    assert bare["reasoning"] == {"effort": "low"}


_COMPLETED_RESPONSE: Final = {
    "id": "resp_dated_chat_deployment",
    "object": "response",
    "created_at": 1734366691,
    "status": "completed",
    "model": "gpt-5.4-nano",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": None,
    "temperature": None,
    "tool_choice": "auto",
    "tools": [],
    "top_p": None,
    "max_output_tokens": None,
    "previous_response_id": None,
    "reasoning": None,
    "truncation": None,
    "user": None,
}


_AZURE_RESOURCE: Final = "https://example-resource.openai.azure.com"


def _unset_api_version_responses_url() -> httpx.URL:
    return httpx.URL(AzureOpenAIResponsesAPIConfig().get_complete_url(api_base=_AZURE_RESOURCE, litellm_params={}))


def _azure_responses_create(api_version: str) -> str | None:
    response: Final = litellm.responses(
        model="azure/gpt-5.4-nano",
        input="hi",
        api_base=_AZURE_RESOURCE,
        api_key="azure-test-key",
        api_version=api_version,
    )
    return response.output[0].content[0].text


def _azure_chat_bridged_to_responses(api_version: str) -> str | None:
    response: Final = litellm.completion(
        model="azure/responses/gpt-5.4-nano",
        messages=[{"role": "user", "content": "hi"}],
        api_base=_AZURE_RESOURCE,
        api_key="azure-test-key",
        api_version=api_version,
    )
    return response.choices[0].message.content


@pytest.mark.parametrize("call_responses", [_azure_responses_create, _azure_chat_bridged_to_responses])
def test_dated_ga_api_version_sends_responses_to_the_unset_api_version_url(
    respx_mock: respx.MockRouter, call_responses: Callable[[str], str | None]
) -> None:
    route: Final = respx_mock.post(host="example-resource.openai.azure.com").mock(
        return_value=httpx.Response(200, json=_COMPLETED_RESPONSE)
    )

    text: Final = call_responses("2024-10-21")

    assert text == "Done."
    assert route.call_count == 1
    assert route.calls[0].request.url == _unset_api_version_responses_url()


_COMPLETED_RESPONSE_SSE: Final = "".join(
    f"data: {json.dumps(event)}\n\n"
    for event in (
        {
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Done.",
        },
        {"type": "response.completed", "sequence_number": 1, "response": _COMPLETED_RESPONSE},
    )
)


@pytest.fixture
def gateway_responses(respx_mock: respx.MockRouter, stream: bool) -> respx.Route:
    body: Final = _COMPLETED_RESPONSE_SSE if stream else json.dumps(_COMPLETED_RESPONSE)
    content_type: Final = "text/event-stream" if stream else "application/json"
    return respx_mock.post(
        "https://gateway.example.com/foundry/openai/openai/v1/responses?api-version=v1"
    ).mock(return_value=httpx.Response(200, text=body, headers={"content-type": content_type}))


@pytest.mark.parametrize("stream", [False, True])
def test_explicit_v1_base_sends_responses_to_the_gateway(gateway_responses: respx.Route, stream: bool) -> None:
    response: Final = litellm.responses(
        model="azure/gpt-6.1-sol",
        input="hi",
        api_base="https://gateway.example.com/foundry/openai/openai/v1",
        api_key="azure-test-key",
        api_version="v1",
        stream=stream,
    )
    text: Final = (
        "".join(event.delta for event in response if event.type == "response.output_text.delta")
        if stream
        else response.output[0].content[0].text
    )

    assert text == "Done."
    assert gateway_responses.call_count == 1


@pytest.mark.parametrize("stream", [False, True])
def test_explicit_v1_base_sends_bridged_chat_to_the_gateway(gateway_responses: respx.Route, stream: bool) -> None:
    response: Final = litellm.completion(
        model="azure/responses/gpt-6.1-sol",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://gateway.example.com/foundry/openai/openai/v1",
        api_key="azure-test-key",
        api_version="v1",
        stream=stream,
    )
    text: Final = (
        "".join(chunk.choices[0].delta.content or "" for chunk in response)
        if stream
        else response.choices[0].message.content
    )

    assert text == "Done."
    assert gateway_responses.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures(httpx_transport.__name__)
@pytest.mark.parametrize("stream", [False, True])
async def test_explicit_v1_base_sends_async_responses_to_the_gateway(
    gateway_responses: respx.Route, stream: bool
) -> None:
    response: Final = await litellm.aresponses(
        model="azure/gpt-6.1-sol",
        input="hi",
        api_base="https://gateway.example.com/foundry/openai/openai/v1",
        api_key="azure-test-key",
        api_version="v1",
        stream=stream,
    )
    text: Final = (
        "".join([event.delta async for event in response if event.type == "response.output_text.delta"])
        if stream
        else response.output[0].content[0].text
    )

    assert text == "Done."
    assert gateway_responses.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures(httpx_transport.__name__)
@pytest.mark.parametrize("stream", [False, True])
async def test_explicit_v1_base_sends_async_bridged_chat_to_the_gateway(
    gateway_responses: respx.Route, stream: bool
) -> None:
    response: Final = await litellm.acompletion(
        model="azure/responses/gpt-6.1-sol",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://gateway.example.com/foundry/openai/openai/v1",
        api_key="azure-test-key",
        api_version="v1",
        stream=stream,
    )
    text: Final = (
        "".join([chunk.choices[0].delta.content or "" async for chunk in response])
        if stream
        else response.choices[0].message.content
    )

    assert text == "Done."
    assert gateway_responses.call_count == 1


_LIFECYCLE_BODIES: Final = {
    "get": _COMPLETED_RESPONSE,
    "delete": {"id": "resp_dated", "object": "response", "deleted": True},
    "cancel": {**_COMPLETED_RESPONSE, "status": "cancelled"},
    "input_items": {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False},
}


def _call_lifecycle(operation: str) -> object:
    kwargs: Final = {
        "response_id": "resp_dated",
        "custom_llm_provider": "azure",
        "api_base": _AZURE_RESOURCE,
        "api_key": "azure-test-key",
        "api_version": "2024-10-21",
    }
    match operation:
        case "get":
            return litellm.get_responses(**kwargs)
        case "delete":
            return litellm.delete_responses(**kwargs)
        case "cancel":
            return litellm.cancel_responses(**kwargs)
        case _:
            return litellm.list_input_items(**kwargs)


@pytest.mark.parametrize(
    ("operation", "path_suffix"),
    [("get", ""), ("delete", ""), ("cancel", "/cancel"), ("input_items", "/input_items")],
)
def test_dated_ga_api_version_sends_response_lifecycle_calls_to_the_unset_api_version_route(
    respx_mock: respx.MockRouter, operation: str, path_suffix: str
) -> None:
    route: Final = respx_mock.route(host="example-resource.openai.azure.com").mock(
        return_value=httpx.Response(200, json=_LIFECYCLE_BODIES[operation])
    )

    _call_lifecycle(operation)

    unset_url: Final = _unset_api_version_responses_url()
    sent_url: Final = route.calls[0].request.url
    assert route.call_count == 1
    assert sent_url.path == f"{unset_url.path}/resp_dated{path_suffix}"
    assert sent_url.params["api-version"] == unset_url.params["api-version"]


def test_chat_completion_keeps_the_dated_ga_api_version(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(
        f"{_AZURE_RESOURCE}/openai/deployments/gpt-5.4-nano/chat/completions", params={"api-version": "2024-10-21"}
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-dated",
                "object": "chat.completion",
                "created": 1734366691,
                "model": "gpt-5.4-nano",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
        )
    )

    response: Final = litellm.completion(
        model="azure/gpt-5.4-nano",
        messages=[{"role": "user", "content": "hi"}],
        api_base=_AZURE_RESOURCE,
        api_key="azure-test-key",
        api_version="2024-10-21",
    )

    assert response.choices[0].message.content == "Done."
    assert route.call_count == 1


@pytest.mark.respx(assert_all_called=True)
def test_responses_api_version_routes_a_dated_chat_deployment_to_its_own_responses_version(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(
        "https://example-resource.openai.azure.com/openai/responses", params={"api-version": "2025-04-01-preview"}
    ).mock(return_value=httpx.Response(200, json=_COMPLETED_RESPONSE))

    response: Final = litellm.responses(
        model="azure/gpt-5.4-nano",
        input="hi",
        api_base="https://example-resource.openai.azure.com",
        api_key="azure-test-key",
        api_version="2024-10-21",
        responses_api_version="2025-04-01-preview",
    )
    request_body: Final = json.loads(route.calls[0].request.content)

    assert route.call_count == 1
    assert response.output[0].content[0].text == "Done."
    assert request_body["model"] == "gpt-5.4-nano"
    assert "responses_api_version" not in request_body
    assert "api_version" not in request_body


@pytest.mark.respx(assert_all_called=True)
def test_responses_api_version_reaches_the_url_when_chat_completion_bridges_to_responses(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(
        "https://example-resource.openai.azure.com/openai/v1/responses", params={"api-version": "v1"}
    ).mock(return_value=httpx.Response(200, json=_COMPLETED_RESPONSE))

    response: Final = litellm.completion(
        model="azure/responses/gpt-5.4-nano",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://example-resource.openai.azure.com",
        api_key="azure-test-key",
        api_version="2024-10-21",
        responses_api_version="v1",
    )
    request_body: Final = json.loads(route.calls[0].request.content)

    assert route.call_count == 1
    assert response.choices[0].message.content == "Done."
    assert "responses_api_version" not in request_body


def test_azure_responses_sends_the_deployment_name_when_azure_ai_prefix_survives_provider_remap():
    request = AzureOpenAIResponsesAPIConfig().transform_responses_api_request(
        model="azure_ai/gpt-5.4-nano",
        input="hi",
        response_api_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request["model"] == "gpt-5.4-nano"


@pytest.mark.asyncio
async def test_azure_responses_api_status_error():
    """
    Test that 'status' field is not sent in the final request body to Azure API.
    The status field should be filtered out from input messages before making the API call.
    """
    from unittest.mock import MagicMock
    import json

    request_data = {
        "model": "computer-use-preview",
        "input": [
            {"content": "tell me an interesting fact", "role": "user"},
            {
                "id": "rs_0ab687487834d9df0068e462a1b2d88197aabbc832c9ba5316",
                "summary": [],
                "type": "reasoning",
                "content": None,
                "encrypted_content": None,
                "status": "completed",
            },
            {
                "id": "msg_0ab687487834d9df0068e462a1df188197b74b1eef05102c18",
                "content": [
                    {
                        "annotations": [],
                        "text": "very good morning",
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {"role": "user", "content": "tell me another"},
        ],
        "include": [],
        "instructions": "You are a helpful assistant.",
        "reasoning": {"effort": "minimal"},
        "stream": False,
        "tools": [],
    }

    # Mock response
    mock_response_data = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1234567890,
        "model": "computer-use-preview",
        "status": "completed",
        "output": [
            {
                "id": "msg_123",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Here's an interesting fact."}
                ],
            }
        ],
    }

    captured_request_body = {}

    async def mock_post(*args, **kwargs):
        # Capture the request body
        nonlocal captured_request_body
        if "json" in kwargs:
            captured_request_body = kwargs["json"]
        elif "data" in kwargs:
            captured_request_body = json.loads(kwargs["data"])

        import httpx

        # Create a proper httpx Response object
        response_content = json.dumps(mock_response_data).encode("utf-8")
        response = httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=response_content,
            request=httpx.Request(method="POST", url="https://test.openai.azure.com"),
        )
        return response

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from unittest.mock import patch

    with patch.object(AsyncHTTPHandler, "post", new=mock_post):
        response = await litellm.aresponses(
            model="azure/computer-use-preview",
            truncation="auto",
            api_version="preview",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            input=request_data["input"],
        )

    # Verify that 'status' field is not present in any of the input messages
    print(
        "Final request body:", json.dumps(captured_request_body, indent=4, default=str)
    )
    assert "input" in captured_request_body, "Request body should contain 'input' field"

    expected_input = [
        {"content": "tell me an interesting fact", "role": "user"},
        {
            "id": "rs_0ab687487834d9df0068e462a1b2d88197aabbc832c9ba5316",
            "summary": [],
            "type": "reasoning",
        },
        {
            "id": "msg_0ab687487834d9df0068e462a1df188197b74b1eef05102c18",
            "content": [
                {
                    "annotations": [],
                    "text": "very good morning",
                    "type": "output_text",
                    "logprobs": [],
                }
            ],
            "role": "assistant",
            "type": "message",
        },
        {"role": "user", "content": "tell me another"},
    ]

    assert captured_request_body["input"] == expected_input, (
        f"Request body input should match expected format without 'status' field.\n"
        f"Expected: {json.dumps(expected_input, indent=2)}\n"
        f"Got: {json.dumps(captured_request_body['input'], indent=2)}"
    )


@pytest.mark.asyncio
async def test_azure_responses_api_headers_with_llm_provider_prefix():
    """
    Test that Azure-specific headers like 'x-request-id' and 'apim-request-id'
    are properly forwarded with 'llm_provider-' prefix in response._hidden_params["headers"].

    Issue: https://github.com/BerriAI/litellm/issues/16538

    The fix ensures that processed headers (with llm_provider- prefix) are stored
    in response._hidden_params["headers"] instead of additional_headers, making them
    accessible via completion.headers in the same way as the completion API.
    """
    import httpx

    mock_response_data = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1234567890,
        "model": "gpt-5-codex",
        "status": "completed",
        "output": [
            {
                "id": "msg_123",
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
    }

    # Mock headers that Azure returns - exactly like in the issue
    mock_headers = {
        "date": "Wed, 12 Nov 2025 15:31:28 GMT",
        "server": "uvicorn",
        "content-type": "application/json",
        "x-ratelimit-remaining-tokens": "5010000",
        "x-ratelimit-limit-tokens": "5010000",
        # These are the Azure-specific headers that should be forwarded with llm_provider- prefix
        "x-request-id": "12086715-aca3-4006-a29f-2f1e1d552043",
        "apim-request-id": "25664b0d-cf4b-4e10-8d27-c7272e7efd49",
        "x-ms-region": "Sweden Central",
    }

    async def mock_post(*args, **kwargs):
        response_content = json.dumps(mock_response_data).encode("utf-8")
        response = httpx.Response(
            status_code=200,
            headers=mock_headers,
            content=response_content,
            request=httpx.Request(method="POST", url="https://test.openai.azure.com"),
        )
        return response

    with patch.object(AsyncHTTPHandler, "post", new=mock_post):
        response = await litellm.aresponses(
            model="azure/gpt-5-codex",
            api_version="2025-03-01-preview",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            input="Hello, can you tell me a short joke?",
        )

    # Check that the response has the expected headers structure
    assert hasattr(response, "_hidden_params"), "Response should have _hidden_params"
    assert (
        "additional_headers" in response._hidden_params
    ), "Response _hidden_params should contain 'additional_headers' with the LLM provider headers"

    headers = response._hidden_params["additional_headers"]

    # Verify that Azure-specific headers are present with llm_provider- prefix
    assert "llm_provider-x-request-id" in headers, (
        f"Response should contain 'llm_provider-x-request-id' header. "
        f"Headers: {list(headers.keys())}"
    )
    assert "llm_provider-apim-request-id" in headers, (
        f"Response should contain 'llm_provider-apim-request-id' header. "
        f"Headers: {list(headers.keys())}"
    )

    # Verify the header values match
    assert (
        headers["llm_provider-x-request-id"] == "12086715-aca3-4006-a29f-2f1e1d552043"
    )
    assert (
        headers["llm_provider-apim-request-id"]
        == "25664b0d-cf4b-4e10-8d27-c7272e7efd49"
    )
    assert headers["llm_provider-x-ms-region"] == "Sweden Central"

    # Also verify openai-compatible headers are included
    assert "x-ratelimit-limit-tokens" in headers
    assert "x-ratelimit-remaining-tokens" in headers


@pytest.mark.parametrize(
    "model", ["gpt-5", "gpt-5-turbo", "GPT-5", "azure/gpt-5", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
)
def test_azure_gpt_models_resolve_to_responses_config_with_temperature(model: str) -> None:
    config: Final = ProviderConfigManager.get_provider_responses_api_config(provider=LlmProviders.AZURE, model=model)
    assert type(config) is AzureOpenAIResponsesAPIConfig
    assert "temperature" in config.get_supported_openai_params(model)


@pytest.mark.parametrize("model", ["o1", "o3"])
def test_azure_o_series_resolves_to_o_series_config_without_temperature(model: str) -> None:
    config: Final = ProviderConfigManager.get_provider_responses_api_config(provider=LlmProviders.AZURE, model=model)
    assert type(config) is AzureOpenAIOSeriesResponsesAPIConfig
    assert "temperature" not in config.get_supported_openai_params(model)


def test_openai_gpt5_resolves_to_responses_config_with_temperature() -> None:
    config: Final = ProviderConfigManager.get_provider_responses_api_config(provider=LlmProviders.OPENAI, model="gpt-5")
    assert type(config) is OpenAIResponsesAPIConfig
    assert "temperature" in config.get_supported_openai_params("gpt-5")
