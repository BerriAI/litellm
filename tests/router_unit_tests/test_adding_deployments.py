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

    # Mock the _set_vertex_pass_through_env_vars function
    with patch(
        "litellm.proxy.vertex_ai_endpoints.vertex_endpoints._set_vertex_pass_through_env_vars"
    ) as mock_set_vars:
        # Add the deployment
        router._add_deployment(deployment)

        # Verify _set_vertex_pass_through_env_vars was called with correct arguments
        mock_set_vars.assert_called_once()
        args = mock_set_vars.call_args[1]
        vertex_creds = args["vertex_pass_through_credentials"]

        assert vertex_creds.vertex_project == "test-project"
        assert vertex_creds.vertex_location == "us-central1"
        assert vertex_creds.vertex_credentials == json.dumps(
            {"type": "service_account", "project_id": "test"}
        )
