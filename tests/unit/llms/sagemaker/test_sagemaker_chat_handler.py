import pytest
import datetime
from typing import Final
from unittest.mock import patch

import boto3
from botocore.exceptions import ClientError

from litellm.llms.sagemaker.chat.handler import SagemakerChatHandler


def test_load_credentials_assumes_role_with_external_id(monkeypatch):
    """A trust policy requiring sts:ExternalId must be satisfied by the deployment's aws_external_id."""
    monkeypatch.delenv("AWS_EXTERNAL_ID", raising=False)

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if params.get("ExternalId") != "external-id-sm-chat":
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:AssumeRole"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIASMCHATROLEKEY",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
                }
            }

    optional_params = {
        "aws_access_key_id": "AKIASMCHATCALLERKEY",
        "aws_secret_access_key": "pod-caller-secret",
        "aws_region_name": "us-east-1",
        "aws_role_name": "arn:aws:iam::999999999999:role/litellm-sm-chat-role",
        "aws_session_name": "litellm-sm-chat-session",
        "aws_external_id": "external-id-sm-chat",
    }

    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        credentials, aws_region_name = SagemakerChatHandler()._load_credentials(optional_params)

    assert credentials.access_key == "ASIASMCHATROLEKEY"
    assert credentials.token == "assumed-session-token"
    assert aws_region_name == "us-east-1"
    assert "aws_external_id" not in optional_params


def test_load_credentials_assumes_role_with_session_tags(monkeypatch):
    """A trust policy gated on sts:TagSession only admits the session when the deployment's tags are sent."""
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    tags = [{"Key": "team", "Value": "genai"}]

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if list(params.get("Tags", ())) != tags:
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:TagSession"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIASMCHATTAGGED",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
                }
            }

    optional_params = {
        "aws_access_key_id": "AKIASMCHATCALLERKEY",
        "aws_secret_access_key": "pod-caller-secret",
        "aws_region_name": "us-east-1",
        "aws_role_name": "arn:aws:iam::999999999999:role/litellm-sm-chat-role",
        "aws_session_name": "litellm-sm-chat-session",
        "aws_session_tags": tags,
    }

    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        credentials, aws_region_name = SagemakerChatHandler()._load_credentials(optional_params)

    assert credentials.access_key == "ASIASMCHATTAGGED"
    assert aws_region_name == "us-east-1"
    assert "aws_session_tags" not in optional_params


@pytest.mark.parametrize("missing", ("boto3", "botocore"))
def test_missing_aws_extra_explains_installation(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from litellm.llms.sagemaker.chat.handler import SagemakerChatHandler

    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError, match=r"litellm\[aws\]"):
        SagemakerChatHandler()._load_credentials({})


@pytest.mark.parametrize("stream", (False, True))
def test_prepare_request_signs_payload_with_installed_aws_extra(stream: bool) -> None:
    import json
    from botocore.credentials import Credentials

    payload: Final = {"inputs": "héllo"}
    request: Final = SagemakerChatHandler()._prepare_request(
        Credentials("test-key", "test-secret", "test-session"), "test-endpoint", payload,
        {"stream": stream}, "us-east-1",
    )
    assert request.headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=test-key/")
    assert "/us-east-1/sagemaker/aws4_request" in request.headers["Authorization"]
    assert request.headers["X-Amz-Security-Token"] == "test-session"
    assert json.loads(request.body) == payload
    suffix: Final = "invocations-response-stream" if stream else "invocations"
    assert request.url == f"https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/test-endpoint/{suffix}"


@pytest.mark.parametrize("missing", ("boto3", "botocore"))
def test_prepare_request_without_aws_extra_explains_installation(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError, match=r'Install AWS support with pip install "litellm\[aws\]"') as caught:
        SagemakerChatHandler()._prepare_request(None, "test-endpoint", {"inputs": "hello"}, {}, "us-east-1")
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == missing
