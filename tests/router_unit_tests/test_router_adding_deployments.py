import sys, os
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from unittest.mock import patch
import json

@pytest.mark.parametrize("reusable_credentials", [True, False])
def test_initialize_deployment_for_pass_through_success(reusable_credentials):
    """
    Test successful initialization of a Vertex AI pass-through deployment
    """
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.types.utils import CredentialItem

    vertex_project="test-project"
    vertex_location="us-central1"
    vertex_credentials=json.dumps({"type": "service_account", "project_id": "test"})

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
        CredentialAccessor.upsert_credentials([
            CredentialItem(
                credential_name="vertex_credentials",
                credential_values={
                    "vertex_project": vertex_project,
                    "vertex_location": vertex_location,
                    "vertex_credentials": vertex_credentials,
                },
                credential_info={}
            )
        ])
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
        model="vertex_ai/test-model",
    )

    # Verify the credentials were properly set
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    vertex_creds = passthrough_endpoint_router.get_vertex_credentials(
        project_id="test-project", location="us-central1"
    )
    assert vertex_creds.vertex_project == "test-project"
    assert vertex_creds.vertex_location == "us-central1"
    assert vertex_creds.vertex_credentials == json.dumps(
        {"type": "service_account", "project_id": "test"}
    )


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
            model="vertex_ai/test-model",
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
        model="vertex_ai/test-model",
    )

    # If we reach this point, the test passes as the method exited without raising any errors
    assert True


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
            vertex_credentials=json.dumps(
                {"type": "service_account", "project_id": "test"}
            ),
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
    print("\n vertex_pass_through_router.deployment_key_to_vertex_credentials\n\n")
    print(
        json.dumps(
            passthrough_endpoint_router.deployment_key_to_vertex_credentials,
            indent=4,
            default=str,
        )
    )

    vertex_creds = passthrough_endpoint_router.get_vertex_credentials(
        project_id="test-project", location="us-central1"
    )

    # Verify the credentials were properly set
    assert vertex_creds.vertex_project == "test-project"
    assert vertex_creds.vertex_location == "us-central1"
    assert vertex_creds.vertex_credentials == json.dumps(
        {"type": "service_account", "project_id": "test"}
    )
