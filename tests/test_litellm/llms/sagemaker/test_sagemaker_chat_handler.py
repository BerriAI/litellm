import datetime
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
