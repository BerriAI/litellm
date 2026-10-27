"""
Unit tests for watsonx_proxy_route endpoint.

Tests the Watsonx pass-through endpoint that handles automatic IAM token management
and version parameter injection.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, Response

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    watsonx_proxy_route,
)


class TestWatsonxProxyRoute:
    """Tests for the Watsonx pass-through route."""

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_success_non_streaming(self):
        """Test successful non-streaming request through Watsonx proxy route."""
        # Setup mocks
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_request.json = AsyncMock(return_value={"stream": False, "input": "test"})
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation",
            {},
        )
        mock_provider_config.validate_environment.return_value = {
            "Authorization": "Bearer test-iam-token"
        }

        # Mock endpoint function
        mock_endpoint_func = AsyncMock(
            return_value={"model_id": "ibm/granite-13b-chat-v2", "results": []}
        )

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            result = await watsonx_proxy_route(
                endpoint="ml/v1/text/generation",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify provider config was called correctly
            mock_provider_config.get_complete_url.assert_called_once()
            mock_provider_config.validate_environment.assert_called_once()

            # Verify create_pass_through_route was called with correct parameters
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["endpoint"] == "ml/v1/text/generation"
            assert (
                call_args["target"]
                == "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation"
            )
            assert (
                call_args["custom_headers"]["Authorization"] == "Bearer test-iam-token"
            )
            assert call_args["is_streaming_request"] is False
            assert call_args["custom_llm_provider"] == "watsonx"
            assert (
                call_args["query_params"]["version"]
                == litellm.WATSONX_DEFAULT_API_VERSION
            )

            # Verify endpoint function was called
            mock_endpoint_func.assert_called_once_with(
                mock_request, mock_response, mock_user_api_key_dict
            )

            assert result == {"model_id": "ibm/granite-13b-chat-v2", "results": []}

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_success_streaming(self):
        """Test successful streaming request through Watsonx proxy route."""
        # Setup mocks
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_request.json = AsyncMock(return_value={"stream": True, "input": "test"})
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation_stream",
            {},
        )
        mock_provider_config.validate_environment.return_value = {
            "Authorization": "Bearer test-iam-token"
        }

        # Mock endpoint function
        mock_endpoint_func = AsyncMock(return_value="streaming_response")

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            result = await watsonx_proxy_route(
                endpoint="ml/v1/text/generation_stream",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify create_pass_through_route was called with streaming enabled
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["is_streaming_request"] is True

            assert result == "streaming_response"

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_get_request(self):
        """Test GET request through Watsonx proxy route."""
        # Setup mocks
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {"project_id": "test-project"}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/models",
            {},
        )
        mock_provider_config.validate_environment.return_value = {
            "Authorization": "Bearer test-iam-token"
        }

        # Mock endpoint function
        mock_endpoint_func = AsyncMock(return_value={"resources": []})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            result = await watsonx_proxy_route(
                endpoint="ml/v1/models",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify is_streaming_request is False for GET requests
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["is_streaming_request"] is False

            assert result == {"resources": []}

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_multipart_form_data(self):
        """Test multipart/form-data request through Watsonx proxy route."""
        # Setup mocks
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "multipart/form-data; boundary=----"}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Mock form data
        mock_form_data = {"file": "test_file", "stream": False}

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/tokenization",
            {},
        )
        mock_provider_config.validate_environment.return_value = {
            "Authorization": "Bearer test-iam-token"
        }

        # Mock endpoint function
        mock_endpoint_func = AsyncMock(return_value={"token_count": 10})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_form_data",
                return_value=mock_form_data,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            result = await watsonx_proxy_route(
                endpoint="ml/v1/text/tokenization",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify is_streaming_request is False for non-streaming form data
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["is_streaming_request"] is False

            assert result == {"token_count": 10}

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_no_provider_config(self):
        """Test that HTTPException is raised when provider config is not found."""
        # Setup mocks
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=None,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await watsonx_proxy_route(
                    endpoint="ml/v1/text/generation",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Watsonx passthrough config not found"

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_version_parameter_injection(self):
        """Test that version parameter is correctly injected into query params."""
        # Setup mocks
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_request.json = AsyncMock(return_value={"input": "test"})
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Mock provider config
        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation",
            {},
        )
        mock_provider_config.validate_environment.return_value = {
            "Authorization": "Bearer test-iam-token"
        }

        # Mock endpoint function
        mock_endpoint_func = AsyncMock(return_value={})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            await watsonx_proxy_route(
                endpoint="ml/v1/text/generation",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify version parameter is injected
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert "query_params" in call_args
            assert "version" in call_args["query_params"]
            assert (
                call_args["query_params"]["version"]
                == litellm.WATSONX_DEFAULT_API_VERSION
            )

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_custom_headers_from_validate_environment(self):
        """Test that custom headers from validate_environment are passed through."""
        # Setup mocks
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_request.json = AsyncMock(return_value={"input": "test"})
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Mock provider config with custom headers
        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation",
            {},
        )
        mock_provider_config.validate_environment.return_value = {
            "Authorization": "Bearer test-iam-token",
            "X-Custom-Header": "custom-value",
        }

        # Mock endpoint function
        mock_endpoint_func = AsyncMock(return_value={})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            await watsonx_proxy_route(
                endpoint="ml/v1/text/generation",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify custom headers are passed through
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert "custom_headers" in call_args
            assert (
                call_args["custom_headers"]["Authorization"] == "Bearer test-iam-token"
            )
            assert call_args["custom_headers"]["X-Custom-Header"] == "custom-value"

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_different_endpoints(self):
        """Test various Watsonx endpoint paths."""
        endpoints = [
            "ml/v1/text/generation",
            "ml/v1/text/tokenization",
            "ml/v1/deployments/test-deployment/text/generation",
            "ml/v1/models",
        ]

        for endpoint_path in endpoints:
            # Setup mocks
            mock_request = MagicMock(spec=Request)
            mock_request.method = "POST"
            mock_request.query_params = {}
            mock_request.headers = {"content-type": "application/json"}
            mock_request.json = AsyncMock(return_value={"input": "test"})
            mock_response = MagicMock(spec=Response)
            mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

            # Mock provider config
            mock_provider_config = MagicMock()
            mock_provider_config.get_complete_url.return_value = (
                f"https://us-south.ml.cloud.ibm.com/{endpoint_path}",
                {},
            )
            mock_provider_config.validate_environment.return_value = {
                "Authorization": "Bearer test-iam-token"
            }

            # Mock endpoint function
            mock_endpoint_func = AsyncMock(return_value={})

            with (
                patch(
                    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                    return_value=mock_provider_config,
                ),
                patch(
                    "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                    return_value=mock_endpoint_func,
                ) as mock_create_route,
            ):
                await watsonx_proxy_route(
                    endpoint=endpoint_path,
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

                # Verify endpoint is passed correctly
                mock_create_route.assert_called_once()
                call_args = mock_create_route.call_args[1]
                assert call_args["endpoint"] == endpoint_path
                assert (
                    call_args["target"]
                    == f"https://us-south.ml.cloud.ibm.com/{endpoint_path}"
                )

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_model_restriction_denied(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_request.json = AsyncMock(return_value={"model_id": "ibm/granite-13b-chat-v2", "input": "test"})
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.models = ["openai/gpt-4o"]
        mock_user_api_key_dict.team_models = None
        mock_user_api_key_dict.access_group_ids = None

        mock_provider_config = MagicMock()
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "watsonx-chat",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-13b-chat-v2",
                    "use_in_pass_through": True,
                    "api_key": "wx-key",
                    "api_base": "https://us-south.ml.cloud.ibm.com",
                },
            }
        ]

        from litellm.proxy._types import ModelAccessDeniedProxyException

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                AsyncMock(return_value=["openai/gpt-4o"]),
            ),
        ):
            with pytest.raises(ModelAccessDeniedProxyException) as exc_info:
                await watsonx_proxy_route(
                    endpoint="ml/v1/text/generation",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )
            assert exc_info.value.code == "403"
            assert exc_info.value.type == "key_model_access_denied"

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_model_restriction_allowed(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_request.json = AsyncMock(return_value={"model_id": "ibm/granite-13b-chat-v2", "input": "test"})
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.models = ["watsonx-chat"]
        mock_user_api_key_dict.team_models = None
        mock_user_api_key_dict.access_group_ids = None

        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation",
            {},
        )
        mock_provider_config.validate_environment.return_value = {"Authorization": "Bearer test-iam-token"}

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "watsonx-chat",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-13b-chat-v2",
                    "use_in_pass_through": True,
                    "api_key": "wx-key",
                    "api_base": "https://us-south.ml.cloud.ibm.com",
                },
            }
        ]
        mock_endpoint_func = AsyncMock(return_value={"results": []})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                AsyncMock(return_value=["watsonx-chat"]),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ),
        ):
            result = await watsonx_proxy_route(
                endpoint="ml/v1/text/generation",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )
            assert result == {"results": []}

    @pytest.mark.asyncio
    async def test_watsonx_proxy_route_match_project_id_from_query_params(self):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {"project_id": "proj-b"}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
        mock_user_api_key_dict.models = None
        mock_user_api_key_dict.team_models = None
        mock_user_api_key_dict.access_group_ids = None

        mock_provider_config = MagicMock()
        mock_provider_config.get_complete_url.return_value = (
            "https://us-south.ml.cloud.ibm.com/ml/v1/deployments",
            {},
        )
        mock_provider_config.validate_environment.return_value = {"Authorization": "Bearer test-iam-token"}

        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "watsonx-chat-a",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-13b-chat-v2",
                    "use_in_pass_through": True,
                    "api_key": "wx-key-a",
                    "project_id": "proj-a",
                    "api_base": "https://us-south.ml.cloud.ibm.com",
                },
            },
            {
                "model_name": "watsonx-chat-b",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-8b-code",
                    "use_in_pass_through": True,
                    "api_key": "wx-key-b",
                    "project_id": "proj-b",
                    "api_base": "https://us-south.ml.cloud.ibm.com",
                },
            },
        ]
        mock_endpoint_func = AsyncMock(return_value={"resources": []})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_passthrough_config",
                return_value=mock_provider_config,
            ),
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                AsyncMock(return_value=["watsonx-chat-a", "watsonx-chat-b"]),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ),
        ):
            result = await watsonx_proxy_route(
                endpoint="ml/v1/deployments",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )
            assert result == {"resources": []}
            mock_provider_config.validate_environment.assert_called_once()
            call_kwargs = mock_provider_config.validate_environment.call_args[1]
            assert call_kwargs["api_key"] == "wx-key-b"
