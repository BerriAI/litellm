import datetime
import json
from typing import Final
from unittest.mock import AsyncMock, Mock, patch

import boto3
import httpx
import pytest
from botocore.exceptions import ClientError

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.sagemaker.chat.handler import SagemakerChatHandler
import logging
from litellm._logging import verbose_logger


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


@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
async def test_completion_sagemaker_messages_api(sync_mode):
    try:
        litellm.set_verbose = True
        verbose_logger.setLevel(logging.DEBUG)
        print("testing sagemaker")
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

        if sync_mode is True:
            client = HTTPHandler()
            with patch.object(client, "post") as mock_post:
                try:
                    resp = litellm.completion(
                        model="sagemaker_chat/huggingface-pytorch-tgi-inference-2024-08-23-15-48-59-245",
                        messages=[
                            {"role": "user", "content": "hi"},
                        ],
                        temperature=0.2,
                        max_tokens=80,
                        client=client,
                    )
                except Exception as e:
                    print(e)
                mock_post.assert_called_once()
                json_data = json.loads(mock_post.call_args.kwargs["data"])
                assert (
                    json_data["model"]
                    == "huggingface-pytorch-tgi-inference-2024-08-23-15-48-59-245"
                )
                assert json_data["messages"] == [{"role": "user", "content": "hi"}]
                assert json_data["temperature"] == 0.2
                assert json_data["max_tokens"] == 80

        else:
            client = AsyncHTTPHandler()
            with patch.object(client, "post") as mock_post:
                try:
                    resp = await litellm.acompletion(
                        model="sagemaker_chat/huggingface-pytorch-tgi-inference-2024-08-23-15-48-59-245",
                        messages=[
                            {"role": "user", "content": "hi"},
                        ],
                        temperature=0.2,
                        max_tokens=80,
                        num_retries=0,
                        client=client,
                    )
                except Exception as e:
                    print(e)
                mock_post.assert_called_once()
                json_data = json.loads(mock_post.call_args.kwargs["data"])
                assert (
                    json_data["model"]
                    == "huggingface-pytorch-tgi-inference-2024-08-23-15-48-59-245"
                )
                assert json_data["messages"] == [{"role": "user", "content": "hi"}]
                assert json_data["temperature"] == 0.2
                assert json_data["max_tokens"] == 80
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_missing_botocore_keeps_dependency_identity():
    import pytest

    with patch.dict("sys.modules", {"botocore": None}):
        with pytest.raises(ModuleNotFoundError, match="pip install boto3") as caught:
            SagemakerChatHandler()._load_credentials({})
    assert caught.value.name == "botocore"


def test_installed_botocore_signs_the_chat_request():
    from botocore.credentials import Credentials

    request = SagemakerChatHandler()._prepare_request(
        credentials=Credentials("test-key", "test-secret"), model="test-endpoint", data={"inputs": "ping"},
        optional_params={}, aws_region_name="us-west-2",
    )
    assert request.body == b'{"inputs": "ping"}'
    assert "/us-west-2/sagemaker/aws4_request" in request.headers["Authorization"]
