
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _base_vertex_proxy_route,
)
from litellm.types.router import DeploymentTypedDict


@pytest.mark.asyncio
async def test_vertex_passthrough_load_balancing():
    """
    Test that _base_vertex_proxy_route uses llm_router.get_available_deployment_for_pass_through
    instead of get_model_list to ensure load balancing works with pass-through filtering.
    """
    # Setup mocks
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Mock the router
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {
            "model": "vertex_ai/gemini-pro",
            "vertex_project": "test-project-lb",
            "vertex_location": "us-central1-lb",
            "use_in_pass_through": True
        }
    }
    mock_router.get_available_deployment_for_pass_through.return_value = mock_deployment

    # Mock get_vertex_model_id_from_url to return a model ID
    with patch("litellm.llms.vertex_ai.common_utils.get_vertex_model_id_from_url", return_value="gemini-pro"), \
         patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_project_id_from_url", return_value=None), \
         patch("litellm.llms.vertex_ai.common_utils.get_vertex_location_from_url", return_value=None), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth:

        # Setup additional mocks to avoid side effects
        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://test.url", False, "test-project-lb", "us-central1-lb")

        mock_endpoint_func = AsyncMock()
        mock_create_route.return_value = mock_endpoint_func
        mock_auth.return_value = {}

        # Execute
        await _base_vertex_proxy_route(
            endpoint="https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent",
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler
        )

        # Verify
        # 1. Check that get_available_deployment_for_pass_through was called with the correct model ID
        mock_router.get_available_deployment_for_pass_through.assert_called_once_with(model="gemini-pro")

        # 2. Check that get_model_list was NOT called (this ensures we aren't doing the old logic)
        mock_router.get_model_list.assert_not_called()

        # 3. Verify that the project and location from the deployment were used (passed to _prepare_vertex_auth_headers)
        # The args are: request, vertex_credentials, router_credentials, vertex_project, vertex_location, ...
        # We check the 4th and 5th args (index 3 and 4)
        call_args = mock_prep_headers.call_args
        assert call_args[1]['vertex_project'] == "test-project-lb"
        assert call_args[1]['vertex_location'] == "us-central1-lb"


def test_get_available_deployment_for_pass_through_filters_correctly():
    """
    Test that get_available_deployment_for_pass_through filters deployments correctly
    """
    from litellm.router import Router

    # Configure router with both pass-through and non-pass-through deployments
    model_list = [
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-pro",
                "vertex_project": "project-1",
                "vertex_location": "us-central1",
                "use_in_pass_through": True,  # Supports pass-through
            }
        },
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-pro",
                "vertex_project": "project-2",
                "vertex_location": "us-west1",
                "use_in_pass_through": False,  # Does not support pass-through
            }
        },
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-pro",
                "vertex_project": "project-3",
                "vertex_location": "us-east1",
                # use_in_pass_through not set (defaults to False)
            }
        },
    ]

    router = Router(model_list=model_list, routing_strategy="simple-shuffle")

    # Test: Should only return project-1 (use_in_pass_through=True)
    deployment = router.get_available_deployment_for_pass_through(model="gemini-pro")

    assert deployment is not None
    assert deployment["litellm_params"]["vertex_project"] == "project-1"
    assert deployment["litellm_params"]["use_in_pass_through"] is True


def test_get_available_deployment_for_pass_through_no_deployments():
    """
    Test that correct error is thrown when there are no pass-through deployments
    """
    import litellm
    from litellm.router import Router

    model_list = [
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-pro",
                "vertex_project": "project-1",
                "vertex_location": "us-central1",
                "use_in_pass_through": False,  # Does not support pass-through
            }
        }
    ]

    router = Router(model_list=model_list)

    # Should throw BadRequestError
    with pytest.raises(litellm.BadRequestError) as exc_info:
        router.get_available_deployment_for_pass_through(model="gemini-pro")

    assert "use_in_pass_through=True" in str(exc_info.value)


def test_get_available_deployment_for_pass_through_load_balancing():
    """
    Test load balancing for pass-through deployments
    """
    from litellm.router import Router

    model_list = [
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-pro",
                "vertex_project": "project-1",
                "vertex_location": "us-central1",
                "use_in_pass_through": True,
                "rpm": 100,
            }
        },
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-pro",
                "vertex_project": "project-2",
                "vertex_location": "us-west1",
                "use_in_pass_through": True,
                "rpm": 200,  # Higher RPM should be selected more frequently
            }
        },
    ]

    router = Router(
        model_list=model_list,
        routing_strategy="simple-shuffle"
    )

    # Call multiple times and track selected deployments
    selections = {"project-1": 0, "project-2": 0}
    for _ in range(100):
        deployment = router.get_available_deployment_for_pass_through(model="gemini-pro")
        project = deployment["litellm_params"]["vertex_project"]
        selections[project] += 1

    # Due to rpm weight, project-2 should be selected more times
    assert selections["project-2"] > selections["project-1"]


@pytest.mark.asyncio
async def test_async_get_available_deployment_for_pass_through():
    """
    Test the async version of get_available_deployment_for_pass_through
    """
    from litellm.router import Router

    model_list = [
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-pro",
                "vertex_project": "project-1",
                "vertex_location": "us-central1",
                "use_in_pass_through": True,
            }
        }
    ]

    router = Router(
        model_list=model_list,
        routing_strategy="simple-shuffle"
    )

    deployment = await router.async_get_available_deployment_for_pass_through(
        model="gemini-pro",
        request_kwargs={}
    )

    assert deployment is not None
    assert deployment["litellm_params"]["use_in_pass_through"] is True


@pytest.mark.asyncio
async def test_vertex_passthrough_forwards_anthropic_beta_header():
    """
    Test that _prepare_vertex_auth_headers forwards the anthropic-beta header
    (and other important headers) from the incoming request when credentials are available.

    This test validates the fix for the issue where the 1M context window header
    (anthropic-beta: context-1m-2025-08-07) was being dropped when forwarding
    requests to Vertex AI.
    """
    from starlette.datastructures import Headers

    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _prepare_vertex_auth_headers,
    )

    # Create a mock request with anthropic-beta header
    mock_request = MagicMock()
    mock_request.headers = Headers({
        "authorization": "Bearer old-token",
        "anthropic-beta": "context-1m-2025-08-07",
        "content-type": "application/json",
        "user-agent": "test-client",
        "content-length": "1234",  # Should be removed
        "host": "localhost:4000",  # Should be removed
    })

    # Create mock vertex credentials
    mock_vertex_credentials = MagicMock()
    mock_vertex_credentials.vertex_project = "test-project"
    mock_vertex_credentials.vertex_location = "us-central1"
    mock_vertex_credentials.vertex_credentials = "test-credentials"

    # Create mock handler
    mock_handler = MagicMock()
    mock_handler.update_base_target_url_with_credential_location.return_value = (
        "https://us-central1-aiplatform.googleapis.com"
    )

    with patch.object(
        VertexBase,
        "_ensure_access_token_async",
        new_callable=AsyncMock,
        return_value=("test-auth-header", "test-project"),
    ) as mock_ensure_token, patch.object(
        VertexBase,
        "_get_token_and_url",
        return_value=("new-access-token", None),
    ) as mock_get_token:

        # Call the function
        (
            headers,
            base_target_url,
            headers_passed_through,
            vertex_project,
            vertex_location,
        ) = await _prepare_vertex_auth_headers(
            request=mock_request,
            vertex_credentials=mock_vertex_credentials,
            router_credentials=None,
            vertex_project="test-project",
            vertex_location="us-central1",
            base_target_url="https://us-central1-aiplatform.googleapis.com",
            get_vertex_pass_through_handler=mock_handler,
        )

        # Verify that allowlisted headers are preserved
        assert "anthropic-beta" in headers
        assert headers["anthropic-beta"] == "context-1m-2025-08-07"
        assert "content-type" in headers
        assert headers["content-type"] == "application/json"

        # Verify that the Authorization header is set with vendor credentials
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer new-access-token"

        # Verify that non-allowlisted headers are NOT forwarded (security)
        # Only anthropic-beta, content-type, and Authorization should be present
        assert "authorization" not in headers  # lowercase auth token not forwarded
        assert "user-agent" not in headers     # not in allowlist
        assert "content-length" not in headers  # not in allowlist
        assert "host" not in headers            # not in allowlist

        # Verify that headers_passed_through is False (since we have credentials)
        assert headers_passed_through is False


@pytest.mark.asyncio
async def test_vertex_passthrough_does_not_forward_litellm_auth_token():
    """
    Test that the LiteLLM authorization header is NOT forwarded to Vertex AI.

    This test validates the fix for the issue where both the LiteLLM auth token
    (lowercase 'authorization') and the Vertex AI token (uppercase 'Authorization')
    were being sent, causing 401 errors on the vendor side.

    The incoming request has:
      - authorization: Bearer <litellm_token>  (should NOT be forwarded)

    The outgoing request should only have:
      - Authorization: Bearer <vertex_token>  (vendor credentials)
    """
    from starlette.datastructures import Headers

    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _prepare_vertex_auth_headers,
    )

    # Create a mock request with ONLY the litellm auth token (no other headers)
    mock_request = MagicMock()
    mock_request.headers = Headers({
        "authorization": "Bearer sk-litellm-secret-key",  # LiteLLM token - should NOT be forwarded
        "Authorization": "Bearer sk-litellm-secret-key-uppercase",  # Also try uppercase
    })

    # Create mock vertex credentials
    mock_vertex_credentials = MagicMock()
    mock_vertex_credentials.vertex_project = "test-project"
    mock_vertex_credentials.vertex_location = "us-central1"
    mock_vertex_credentials.vertex_credentials = "test-credentials"

    # Create mock handler
    mock_handler = MagicMock()
    mock_handler.update_base_target_url_with_credential_location.return_value = (
        "https://us-central1-aiplatform.googleapis.com"
    )

    with patch.object(
        VertexBase,
        "_ensure_access_token_async",
        new_callable=AsyncMock,
        return_value=("test-auth-header", "test-project"),
    ), patch.object(
        VertexBase,
        "_get_token_and_url",
        return_value=("vertex-access-token", None),
    ):

        (
            headers,
            _base_target_url,
            _headers_passed_through,
            _vertex_project,
            _vertex_location,
        ) = await _prepare_vertex_auth_headers(
            request=mock_request,
            vertex_credentials=mock_vertex_credentials,
            router_credentials=None,
            vertex_project="test-project",
            vertex_location="us-central1",
            base_target_url="https://us-central1-aiplatform.googleapis.com",
            get_vertex_pass_through_handler=mock_handler,
        )

        # The ONLY Authorization header should be the Vertex token
        assert headers["Authorization"] == "Bearer vertex-access-token"

        # The LiteLLM token should NOT be present (neither lowercase nor as a duplicate)
        assert "authorization" not in headers
        assert headers.get("Authorization") != "Bearer sk-litellm-secret-key"
        assert headers.get("Authorization") != "Bearer sk-litellm-secret-key-uppercase"

        # Verify we only have the expected headers (Authorization + any allowlisted ones present)
        # Since the request only had auth headers, only Authorization should be in output
        assert set(headers.keys()) == {"Authorization"}


def test_forward_headers_from_request_x_pass_prefix():
    """
    Test that headers with 'x-pass-' prefix are forwarded with the prefix stripped.

    This allows users to force-forward arbitrary headers to the vendor API:
    - 'x-pass-anthropic-beta: value' becomes 'anthropic-beta: value'
    - 'x-pass-custom-header: value' becomes 'custom-header: value'

    This is tested on BasePassthroughUtils.forward_headers_from_request which is used
    by all pass-through endpoints (not just Vertex AI).
    """
    from litellm.passthrough.utils import BasePassthroughUtils

    # Simulate incoming request headers
    request_headers = {
        "x-pass-anthropic-beta": "context-1m-2025-08-07",
        "x-pass-custom-header": "custom-value",
        "x-pass-another-header": "another-value",
        "authorization": "Bearer sk-litellm-key",
        "x-litellm-api-key": "sk-1234",
        "content-type": "application/json",
    }

    # Start with empty headers dict (simulating custom headers from endpoint config)
    headers = {}

    # Call the method with forward_headers=False (default behavior)
    # x-pass- headers should still be forwarded
    result = BasePassthroughUtils.forward_headers_from_request(
        request_headers=request_headers,
        headers=headers,
        forward_headers=False,
    )

    # Verify x-pass- prefixed headers are forwarded with prefix stripped
    assert "anthropic-beta" in result
    assert result["anthropic-beta"] == "context-1m-2025-08-07"
    assert "custom-header" in result
    assert result["custom-header"] == "custom-value"
    assert "another-header" in result
    assert result["another-header"] == "another-value"

    # Verify other headers are NOT forwarded (since forward_headers=False)
    assert "authorization" not in result
    assert "x-litellm-api-key" not in result
    assert "content-type" not in result

    # Verify original x-pass- prefixed headers are NOT in output (only stripped versions)
    assert "x-pass-anthropic-beta" not in result
    assert "x-pass-custom-header" not in result


@pytest.mark.asyncio
async def test_vertex_passthrough_custom_model_name_replaced_in_url():
    """
    Test that when a passthrough URL contains a custom model_name (e.g., gcp/google/gemini-3-pro),
    the URL is rewritten to use the actual Vertex AI model name (e.g., gemini-3-pro)
    before being forwarded to Vertex AI.

    This prevents 404 errors from Vertex AI when custom model names are used in the config.

    Config example:
        model_name: gcp/google/gemini-3-pro
        litellm_params:
          model: vertex_ai/gemini-3-pro
          vertex_project: "my-project"
          vertex_location: "global"
          use_in_pass_through: true
    """
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_handler = MagicMock()

    # Deployment with custom model_name but real vertex model
    mock_deployment = {
        "litellm_params": {
            "model": "vertex_ai/gemini-3-pro",
            "vertex_project": "nv-gcpllmgwit-20250411173346",
            "vertex_location": "global",
            "use_in_pass_through": True,
        }
    }
    mock_router = MagicMock()
    mock_router.get_available_deployment_for_pass_through.return_value = mock_deployment

    # The URL contains project/location AND a custom model name with slashes
    test_endpoint = "v1/projects/nv-gcpllmgwit-20250411173346/locations/global/publishers/google/models/gcp/google/gemini-3-pro:generateContent"

    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router") as mock_pt_router, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._prepare_vertex_auth_headers", new_callable=AsyncMock) as mock_prep_headers, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route") as mock_create_route, \
         patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth", new_callable=AsyncMock) as mock_auth:

        mock_pt_router.get_vertex_credentials.return_value = MagicMock()
        mock_prep_headers.return_value = ({}, "https://global-aiplatform.googleapis.com", False, "nv-gcpllmgwit-20250411173346", "global")
        mock_endpoint_func = AsyncMock()
        mock_create_route.return_value = mock_endpoint_func
        mock_auth.return_value = {}

        mock_handler.get_default_base_target_url.return_value = "https://global-aiplatform.googleapis.com"

        await _base_vertex_proxy_route(
            endpoint=test_endpoint,
            request=mock_request,
            fastapi_response=mock_response,
            get_vertex_pass_through_handler=mock_handler,
        )

        # Verify the router was called with the custom model name (extracted from URL)
        mock_router.get_available_deployment_for_pass_through.assert_called_once_with(
            model="gcp/google/gemini-3-pro"
        )

        # Verify the target URL passed to create_pass_through_route contains
        # the REAL Vertex AI model name, not the custom one
        create_route_call = mock_create_route.call_args
        target_url = create_route_call.kwargs.get("target", "")
        assert "gcp/google/gemini-3-pro" not in target_url, \
            f"Custom model name should have been replaced in target URL. Got: {target_url}"
        assert "gemini-3-pro" in target_url, \
            f"Actual Vertex AI model name should be in target URL. Got: {target_url}"

