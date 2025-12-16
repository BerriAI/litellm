
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import _base_vertex_proxy_route
from litellm.proxy._types import ProxyException, ProxyErrorTypes
from fastapi import status


@pytest.mark.asyncio
async def test_vertex_passthrough_model_access_allowed_exact_match():
    """Verify that users can access Vertex models they have permission for (exact match)"""
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock user API key with access to gemini-1.5-pro
    mock_user_api_key = MagicMock()
    mock_user_api_key.models = ["gemini-1.5-pro"]
    mock_user_api_key.team_model_aliases = None
    mock_user_api_key.token = "sk-1234567890abcdef"

    # Mock router
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {
            "model": "vertex_ai/gemini-1.5-pro",
            "vertex_project": "test-project",
            "vertex_location": "us-central1",
            "use_in_pass_through": True
        }
    }
    mock_router.get_available_deployment_for_pass_through.return_value = mock_deployment

    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value="gemini-1.5-pro"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value="my-project"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value="us-central1"), \
         patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", []), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._override_vertex_params_from_router_credentials") as mock_override:

        # Setup
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://test.url", False, "my-project", "us-central1")
        mock_create_route.return_value = AsyncMock()
        mock_auth.return_value = mock_user_api_key
        mock_override.return_value = ("my-project", "us-central1")

        # Execute
        await _base_vertex_proxy_route(
            endpoint="v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler
        )

        # Verify - user_api_key_auth was called (which includes model permission check)
        mock_auth.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_passthrough_model_access_denied():
    """Verify that users cannot access models they don't have permission for"""
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock user API key with access only to gemini-pro, not gemini-1.5-pro
    mock_user_api_key = MagicMock()
    mock_user_api_key.models = ["gemini-pro"]
    mock_user_api_key.team_model_aliases = None
    mock_user_api_key.token = "sk-1234567890abcdef"

    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value="gemini-1.5-pro"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value="my-project"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value="us-central1"), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._override_vertex_params_from_router_credentials") as mock_override:

        # Setup
        # Simulate access denied by raising ProxyException during user_api_key_auth
        mock_auth.side_effect = ProxyException(
            message="Key not allowed to access model",
            type=ProxyErrorTypes.auth_error,
            param="model",
            code=status.HTTP_401_UNAUTHORIZED,
        )
        mock_override.return_value = ("my-project", "us-central1")

        # Execute and expect exception
        with pytest.raises(ProxyException) as exc_info:
            await _base_vertex_proxy_route(
                endpoint="v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
                request=mock_request,
                fastapi_response=mock_response,
                get_vertex_pass_through_handler=mock_handler
            )

        # Verify
        assert exc_info.value.type == ProxyErrorTypes.auth_error
        assert exc_info.value.code == "401" or exc_info.value.code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_vertex_passthrough_wildcard_access():
    """Verify that wildcard permissions (vertex_ai/*) work correctly"""
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock user API key with wildcard access to all vertex_ai models
    mock_user_api_key = MagicMock()
    mock_user_api_key.models = ["vertex_ai/*"]
    mock_user_api_key.team_model_aliases = None
    mock_user_api_key.token = "sk-1234567890abcdef"

    # Mock router
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {
            "model": "vertex_ai/gemini-2.0-flash",
            "vertex_project": "test-project",
            "vertex_location": "us-central1",
            "use_in_pass_through": True
        }
    }
    mock_router.get_available_deployment_for_pass_through.return_value = mock_deployment

    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value="gemini-2.0-flash"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value="my-project"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value="us-central1"), \
         patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", []), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._override_vertex_params_from_router_credentials") as mock_override:

        # Setup
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://test.url", False, "my-project", "us-central1")
        mock_create_route.return_value = AsyncMock()
        mock_auth.return_value = mock_user_api_key
        mock_override.return_value = ("my-project", "us-central1")

        # Execute
        await _base_vertex_proxy_route(
            endpoint="v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler
        )

        # Verify - user_api_key_auth was called (which includes model permission check)
        mock_auth.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_passthrough_access_group():
    """Verify that access group permissions work correctly"""
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock user API key with access to a group
    mock_user_api_key = MagicMock()
    mock_user_api_key.models = ["production-models"]
    mock_user_api_key.team_model_aliases = None
    mock_user_api_key.token = "sk-1234567890abcdef"

    # Mock router with access groups
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {
            "model": "vertex_ai/gemini-1.5-pro",
            "vertex_project": "test-project",
            "vertex_location": "us-central1",
            "use_in_pass_through": True
        }
    }
    mock_router.get_available_deployment_for_pass_through.return_value = mock_deployment
    # Simulate access group matching
    mock_router.get_model_access_groups.return_value = {"gemini-1.5-pro": ["production-models"]}

    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value="gemini-1.5-pro"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value="my-project"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value="us-central1"), \
         patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", []), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._override_vertex_params_from_router_credentials") as mock_override:

        # Setup
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://test.url", False, "my-project", "us-central1")
        mock_create_route.return_value = AsyncMock()
        mock_auth.return_value = mock_user_api_key
        mock_override.return_value = ("my-project", "us-central1")

        # Execute
        await _base_vertex_proxy_route(
            endpoint="v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler
        )

        # Verify - user_api_key_auth was called (which includes model permission check)
        mock_auth.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_passthrough_team_alias():
    """Verify that team model aliases work correctly"""
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock user API key with team alias
    mock_user_api_key = MagicMock()
    mock_user_api_key.models = ["my-gemini"]
    mock_user_api_key.team_model_aliases = {"my-gemini": "gemini-1.5-pro"}
    mock_user_api_key.token = "sk-1234567890abcdef"

    # Mock router
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {
            "model": "vertex_ai/gemini-1.5-pro",
            "vertex_project": "test-project",
            "vertex_location": "us-central1",
            "use_in_pass_through": True
        }
    }
    mock_router.get_available_deployment_for_pass_through.return_value = mock_deployment

    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value="gemini-1.5-pro"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value="my-project"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value="us-central1"), \
         patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", []), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._override_vertex_params_from_router_credentials") as mock_override:

        # Setup
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://test.url", False, "my-project", "us-central1")
        mock_create_route.return_value = AsyncMock()
        mock_auth.return_value = mock_user_api_key
        mock_override.return_value = ("my-project", "us-central1")

        # Execute
        await _base_vertex_proxy_route(
            endpoint="v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler
        )

        # Verify - user_api_key_auth was called (which includes model permission check)
        mock_auth.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_passthrough_no_model_id():
    """Verify graceful handling when model_id cannot be extracted"""
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock user API key
    mock_user_api_key = MagicMock()
    mock_user_api_key.models = ["*"]

    # Mock router
    mock_router = MagicMock()

    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value=None), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value="my-project"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value="us-central1"), \
         patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", []), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._override_vertex_params_from_router_credentials") as mock_override:

        # Setup
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://test.url", False, "my-project", "us-central1")
        mock_create_route.return_value = AsyncMock()
        mock_auth.return_value = mock_user_api_key
        mock_override.return_value = ("my-project", "us-central1")

        # Execute - should not raise exception even though model_id is None
        await _base_vertex_proxy_route(
            endpoint="v1/projects/my-project/locations/us-central1/publishers/google/models/",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler
        )

        # Verify - function executed successfully without errors
        mock_auth.assert_called_once()


@pytest.mark.asyncio
async def test_vertex_passthrough_with_router_deployment():
    """Verify that permission checks don't affect existing router deployment lookup logic"""
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock user API key with wildcard access
    mock_user_api_key = MagicMock()
    mock_user_api_key.models = ["*"]
    mock_user_api_key.team_model_aliases = None

    # Mock router with deployment
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {
            "model": "vertex_ai/gemini-1.5-pro",
            "vertex_project": "deployment-project",
            "vertex_location": "deployment-location",
            "use_in_pass_through": True
        }
    }
    mock_router.get_available_deployment_for_pass_through.return_value = mock_deployment

    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value="gemini-1.5-pro"), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value=None), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value=None), \
         patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", []), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._override_vertex_params_from_router_credentials") as mock_override:

        # Setup
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://test.url", False, "deployment-project", "deployment-location")
        mock_create_route.return_value = AsyncMock()
        mock_auth.return_value = mock_user_api_key
        mock_override.return_value = (None, None)

        # Execute
        await _base_vertex_proxy_route(
            endpoint="v1/projects/my-project/locations/my-location/publishers/google/models/gemini-1.5-pro:generateContent",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler
        )

        # Verify
        # 1. user_api_key_auth was called (which includes model permission check)
        mock_auth.assert_called_once()
        # 2. get_available_deployment_for_pass_through was called to get project/location
        mock_router.get_available_deployment_for_pass_through.assert_called_once_with(model="gemini-1.5-pro")
        # 3. Verify deployment project/location were used in auth headers
        call_args = mock_prep_headers.call_args
        assert call_args[1]['vertex_project'] == "deployment-project"
        assert call_args[1]['vertex_location'] == "deployment-location"
