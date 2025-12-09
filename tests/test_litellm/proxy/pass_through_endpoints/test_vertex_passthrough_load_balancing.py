
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import _base_vertex_proxy_route
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

