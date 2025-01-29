import sys, os
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from unittest.mock import patch
import json


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
    from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
        vertex_pass_through_router,
    )

    # current state of pass-through vertex router
    print("\n vertex_pass_through_router.deployment_key_to_vertex_credentials\n\n")
    print(
        json.dumps(
            vertex_pass_through_router.deployment_key_to_vertex_credentials,
            indent=4,
            default=str,
        )
    )

    vertex_creds = vertex_pass_through_router.get_vertex_credentials(
        project_id="test-project", location="us-central1"
    )

    # Verify the credentials were properly set
    assert vertex_creds.vertex_project == "test-project"
    assert vertex_creds.vertex_location == "us-central1"
    assert vertex_creds.vertex_credentials == json.dumps(
        {"type": "service_account", "project_id": "test"}
    )
