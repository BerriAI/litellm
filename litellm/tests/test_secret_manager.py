import sys, os, uuid
import time
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm import get_secret
from litellm.proxy.secret_managers.aws_secret_manager import load_aws_secret_manager


@pytest.mark.skip(reason="AWS Suspended Account")
def test_aws_secret_manager():
    load_aws_secret_manager(use_aws_secret_manager=True)

    secret_val = get_secret("litellm_master_key")

    print(f"secret_val: {secret_val}")

    assert secret_val == "sk-1234"


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


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circleci():
    secret_val = get_secret(
        "oidc/circleci/https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
    )

    print(f"secret_val: {redact_oidc_signature(secret_val)}")


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN_V2") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circleci_v2():
    secret_val = get_secret(
        "oidc/circleci_v2/https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
    )

    print(f"secret_val: {redact_oidc_signature(secret_val)}")
