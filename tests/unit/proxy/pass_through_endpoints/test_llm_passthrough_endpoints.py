import asyncio
import importlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import Router
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router import Deployment, LiteLLM_Params
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.fixture
def isolate_passthrough_endpoint_router_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    monkeypatch.setattr(
        passthrough_endpoint_router,
        "deployment_key_to_vertex_credentials",
        passthrough_endpoint_router.deployment_key_to_vertex_credentials.copy(),
    )


@pytest.mark.parametrize("reusable_credentials", [True, False])
@pytest.mark.usefixtures("isolate_passthrough_endpoint_router_state")
def test_initialize_deployment_for_pass_through_success(reusable_credentials):
    """
    Test successful initialization of a Vertex AI pass-through deployment
    """
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.types.utils import CredentialItem

    vertex_project = "test-project"
    vertex_location = "us-central1"
    vertex_credentials = json.dumps({"type": "service_account", "project_id": "test"})

    if not reusable_credentials:
        litellm_params = LiteLLM_Params(
            model="vertex_ai/test-model",
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            use_in_pass_through=True,
        )
    else:
        # add credentials to the credential accessor
        CredentialAccessor.upsert_credentials(
            [
                CredentialItem(
                    credential_name="vertex_credentials",
                    credential_values={
                        "vertex_project": vertex_project,
                        "vertex_location": vertex_location,
                        "vertex_credentials": vertex_credentials,
                    },
                    credential_info={},
                )
            ]
        )
        litellm_params = LiteLLM_Params(
            model="vertex_ai/test-model",
            litellm_credential_name="vertex_credentials",
            use_in_pass_through=True,
        )
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=litellm_params,
    )

    # Test the initialization
    router._initialize_deployment_for_pass_through(
        deployment=deployment,
        custom_llm_provider="vertex_ai",
    )

    # Verify the credentials were properly set
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    vertex_creds = passthrough_endpoint_router.get_vertex_credentials(project_id="test-project", location="us-central1")
    assert vertex_creds.vertex_project == "test-project"
    assert vertex_creds.vertex_location == "us-central1"
    assert vertex_creds.vertex_credentials == json.dumps({"type": "service_account", "project_id": "test"})


def test_initialize_deployment_for_pass_through_missing_params():
    """
    Test initialization fails when required Vertex AI parameters are missing
    """
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=LiteLLM_Params(
            model="vertex_ai/test-model",
            # Missing required parameters
            use_in_pass_through=True,
        ),
    )

    # Test that initialization raises ValueError
    with pytest.raises(
        ValueError,
        match="vertex_project, and vertex_location must be set in litellm_params for pass-through endpoints",
    ):
        router._initialize_deployment_for_pass_through(
            deployment=deployment,
            custom_llm_provider="vertex_ai",
        )


def test_initialize_deployment_when_pass_through_disabled():
    """
    Test that initialization simply exits when use_in_pass_through is False
    """
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=LiteLLM_Params(
            model="vertex_ai/test-model",
        ),
    )

    # This should exit without error, even with missing vertex parameters
    router._initialize_deployment_for_pass_through(
        deployment=deployment,
        custom_llm_provider="vertex_ai",
    )

    # If we reach this point, the test passes as the method exited without raising any errors
    assert True


@pytest.mark.usefixtures("isolate_passthrough_endpoint_router_state")
def test_add_vertex_pass_through_deployment():
    """
    Test adding a Vertex AI deployment with pass-through configuration
    """
    router = Router(model_list=[])

    # Create a deployment with Vertex AI pass-through settings
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=LiteLLM_Params(
            model="vertex_ai/test-model",
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials=json.dumps({"type": "service_account", "project_id": "test"}),
            use_in_pass_through=True,
        ),
    )

    # Add deployment to router
    router.add_deployment(deployment)

    # Get the vertex credentials from the router
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    # current state of pass-through vertex router

    vertex_creds = passthrough_endpoint_router.get_vertex_credentials(project_id="test-project", location="us-central1")

    # Verify the credentials were properly set
    assert vertex_creds.vertex_project == "test-project"
    assert vertex_creds.vertex_location == "us-central1"
    assert vertex_creds.vertex_credentials == json.dumps({"type": "service_account", "project_id": "test"})


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    yield
    loop.close()
    asyncio.set_event_loop(None)


class TestVertexAILivePassthroughIntegration:
    @pytest.fixture
    def mock_websocket(self):
        websocket = AsyncMock()
        websocket.headers = {"authorization": "Bearer test-token"}
        websocket.client_state = MagicMock()
        websocket.client_state.DISCONNECTED = "disconnected"
        return websocket

    @pytest.fixture
    def mock_user_api_key(self):
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            user_role="customer",
        )

    @pytest.fixture
    def mock_logging_obj(self):
        mock = MagicMock(spec=LiteLLMLoggingObj)
        mock.model_call_details = {}
        mock.response_cost_calculator.return_value = None
        return mock

    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async")
    @patch("litellm.proxy.proxy_server.proxy_logging_obj")
    @pytest.mark.asyncio
    async def test_vertex_ai_live_websocket_passthrough_route(
        self,
        mock_proxy_logging_obj,
        mock_ensure_access_token,
        mock_router,
        mock_websocket_passthrough,
        mock_websocket,
        mock_user_api_key,
        mock_logging_obj,
    ):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            vertex_ai_live_websocket_passthrough,
        )

        mock_router.get_vertex_credentials.return_value = MagicMock(
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials="test-credentials",
        )
        mock_router.set_default_vertex_config.return_value = None

        mock_ensure_access_token.return_value = ("test-access-token", "test-project")

        mock_websocket_passthrough.return_value = None

        result = await vertex_ai_live_websocket_passthrough(
            websocket=mock_websocket, user_api_key_dict=mock_user_api_key
        )

        mock_websocket_passthrough.assert_called_once()

        call_args = mock_websocket_passthrough.call_args
        assert call_args[1]["websocket"] == mock_websocket
        assert call_args[1]["user_api_key_dict"] == mock_user_api_key
        assert call_args[1]["endpoint"] == "/vertex_ai/live"

        assert result is None
