import hashlib
import json
import os

from dotenv import load_dotenv

load_dotenv()
import tempfile
from typing import Final

import pytest

import litellm
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.secret_managers.main import (
    get_secret,
)

_AWS_FIXTURE_MASTER_KEY_SHA256: Final = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


def test_aws_secret_manager():
    import json

    AWSSecretsManagerV2.load_aws_secret_manager(use_aws_secret_manager=True)

    secret_val = get_secret("litellm_master_key")

    print(f"secret_val: {secret_val}")

    # cast json to dict
    secret_val = json.loads(secret_val)

    assert (
        hashlib.sha256(secret_val["litellm_master_key"].encode()).hexdigest() == _AWS_FIXTURE_MASTER_KEY_SHA256
    ), "Expected the fixture value stored in the CI AWS account's litellm_master_key secret"


def redact_oidc_signature(secret_val):
    # remove the last part of `.` and replace it with "SIGNATURE_REMOVED"
    return secret_val.split(".")[:-1] + ["SIGNATURE_REMOVED"]


@pytest.mark.skipif(
    os.environ.get("K_SERVICE") is None,
    reason="Cannot run without being in GCP Cloud Run",
)
def test_oidc_google():
    secret_val = get_secret(
        "oidc/google/https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
    )

    print(f"secret_val: {redact_oidc_signature(secret_val)}")
@pytest.mark.skipif(
    os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN") is None,
    reason="Cannot run without being in GitHub Actions",
)
def test_oidc_github():
    secret_val = get_secret(
        "oidc/github/https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
    )

    print(f"secret_val: {redact_oidc_signature(secret_val)}")
