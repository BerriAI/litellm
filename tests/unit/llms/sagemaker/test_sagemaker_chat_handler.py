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


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_completion_sagemaker_messages_api(
    sync_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_body: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "unit-test-endpoint",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    response: Final = httpx.Response(
        200,
        json=response_body,
        request=httpx.Request(
            "POST",
            "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/unit-test-endpoint/invocations",
        ),
    )
    client: Final = HTTPHandler() if sync_mode else AsyncHTTPHandler()
    mock_post: Final = Mock(return_value=response) if sync_mode else AsyncMock(return_value=response)
    monkeypatch.setattr(client, "post", mock_post)
    model: Final = "sagemaker_chat/unit-test-endpoint"

    completion_response: Final = (
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=80,
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_region_name="us-east-1",
            client=client,
        )
        if sync_mode
        else await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=80,
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_region_name="us-east-1",
            client=client,
        )
    )

    assert mock_post.call_count == 1
    assert completion_response.choices[0].message.content == "response"
    request_body: Final = json.loads(mock_post.call_args.kwargs["data"])
    assert request_body["model"] == "unit-test-endpoint"
    assert request_body["messages"] == [{"role": "user", "content": "hi"}]
    assert request_body["temperature"] == 0.2
    assert request_body["max_tokens"] == 80
