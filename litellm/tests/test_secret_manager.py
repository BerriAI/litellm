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
from litellm.llms.azure import get_azure_ad_token_from_oidc
from litellm.llms.bedrock_httpx import BedrockLLM, BedrockConverseLLM


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
        "oidc/circleci/"
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


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circleci_with_azure():
    # TODO: Switch to our own Azure account, currently using ai.moda's account
    os.environ["AZURE_TENANT_ID"] = "17c0a27a-1246-4aa1-a3b6-d294e80e783c"
    os.environ["AZURE_CLIENT_ID"] = "4faf5422-b2bd-45e8-a6d7-46543a38acd0"
    azure_ad_token = get_azure_ad_token_from_oidc("oidc/circleci/")

    print(f"secret_val: {redact_oidc_signature(azure_ad_token)}")


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circle_v1_with_amazon():
    # The purpose of this test is to get logs using the older v1 of the CircleCI OIDC token

    # TODO: This is using ai.moda's IAM role, we should use LiteLLM's IAM role eventually
    aws_role_name = (
        "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci-v1-assume-only"
    )
    aws_web_identity_token = "oidc/circleci/"

    bllm = BedrockLLM()
    creds = bllm.get_credentials(
        aws_region_name="ca-west-1",
        aws_web_identity_token=aws_web_identity_token,
        aws_role_name=aws_role_name,
        aws_session_name="assume-v1-session",
    )

@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_oidc_circle_v1_with_amazon_fips():
    # The purpose of this test is to validate that we can assume a role in a FIPS region

    # TODO: This is using ai.moda's IAM role, we should use LiteLLM's IAM role eventually
    aws_role_name = (
        "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci-v1-assume-only"
    )
    aws_web_identity_token = "oidc/circleci/"

    bllm = BedrockConverseLLM()
    creds = bllm.get_credentials(
        aws_region_name="us-west-1",
        aws_web_identity_token=aws_web_identity_token,
        aws_role_name=aws_role_name,
        aws_session_name="assume-v1-session-fips",
        aws_sts_endpoint="https://sts-fips.us-west-1.amazonaws.com",
    )
