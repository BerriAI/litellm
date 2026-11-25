import base64
import os
from types import MappingProxyType
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm.files.main as files_main
from litellm.llms.bedrock.files.handler import BedrockFilesHandler
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import SpecialEnums


def _encode_unified_file_id(s3_uri: str) -> str:
    unified_file_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json",
        "unified-id",
        "",
        s3_uri,
        "model-id",
    )
    return base64.urlsafe_b64encode(unified_file_id.encode()).decode().rstrip("=")


class TestBedrockFilesHandler:
    def setup_method(self):
        self.handler = BedrockFilesHandler()

    def test_should_parse_direct_managed_s3_uri(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/litellm-bedrock-files-model-id-abc.jsonl",
            configured_bucket_name="safe-bucket",
        )

        assert bucket == "safe-bucket"
        assert key == "litellm-bedrock-files-model-id-abc.jsonl"

    def test_should_parse_managed_batch_output_uri(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/litellm-batch-outputs/job/",
            configured_bucket_name="safe-bucket",
        )

        assert bucket == "safe-bucket"
        assert key == "litellm-batch-outputs/job/"

    def test_should_reject_arbitrary_bucket(self):
        with pytest.raises(ValueError, match="configured storage bucket"):
            self.handler._parse_s3_uri(
                s3_uri="s3://other-bucket/litellm-bedrock-files-model-id-abc.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_reject_unmanaged_same_bucket_key(self):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/private/output.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_allow_legacy_same_bucket_key_when_server_flag_enabled(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/private/output.jsonl",
            configured_bucket_name="safe-bucket",
            allow_legacy_cloud_file_ids=True,
        )

        assert bucket == "safe-bucket"
        assert key == "private/output.jsonl"

    def test_should_keep_configured_prefix_for_legacy_keys(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/team-a/private/output.jsonl",
            configured_bucket_name="safe-bucket/team-a",
            allow_legacy_cloud_file_ids=True,
        )

        assert bucket == "safe-bucket"
        assert key == "team-a/private/output.jsonl"

    def test_should_reject_legacy_key_outside_configured_prefix(self):
        with pytest.raises(ValueError, match="configured storage prefix"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/team-b/private/output.jsonl",
                configured_bucket_name="safe-bucket/team-a",
                allow_legacy_cloud_file_ids=True,
            )

    def test_should_reject_dot_segment_key(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/litellm-bedrock-files/../secret.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_reject_empty_middle_path_segment(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/litellm-bedrock-files//secret.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_extract_unified_managed_s3_uri(self):
        file_id = _encode_unified_file_id(
            "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )

        assert (
            self.handler._extract_s3_uri_from_file_id(file_id)
            == "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )

    def test_should_reject_file_id_without_s3_scheme(self):
        with pytest.raises(ValueError, match="managed LiteLLM S3 file id"):
            self.handler._extract_s3_uri_from_file_id("safe-bucket/private.jsonl")

    def test_should_reject_unified_unmanaged_s3_uri(self):
        file_id = _encode_unified_file_id("s3://safe-bucket/private/output.jsonl")
        s3_uri = self.handler._extract_s3_uri_from_file_id(file_id)

        with pytest.raises(ValueError, match="LiteLLM-managed"):
            self.handler._parse_s3_uri(
                s3_uri=s3_uri,
                configured_bucket_name="safe-bucket",
            )

    def test_call_s3_bucket_name_wins_over_env_bucket(self):
        with patch.dict(os.environ, {"AWS_S3_BUCKET_NAME": "env-bucket"}):
            assert self.handler._get_configured_s3_bucket_name({"s3_bucket_name": "call-bucket"}) == "call-bucket"

    def test_env_bucket_is_used_when_the_call_names_none(self):
        with patch.dict(os.environ, {"AWS_S3_BUCKET_NAME": "env-bucket"}):
            assert self.handler._get_configured_s3_bucket_name({}) == "env-bucket"

    def test_missing_bucket_in_call_and_env_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="S3 bucket_name is required"):
                self.handler._get_configured_s3_bucket_name({})


def _echo_request_url_client() -> HTTPHandler:
    return HTTPHandler(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=str(request.url).encode()))
        )
    )


_SDK_AWS_CREDENTIALS: Final = MappingProxyType(
    {
        "aws_access_key_id": "AKIASDKCALLER",
        "aws_secret_access_key": "sdk-caller-secret",
        "aws_region_name": "us-west-2",
    }
)


def test_sdk_file_content_reads_from_the_bucket_passed_to_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("AWS_S3_OUTPUT_BUCKET_NAME", raising=False)

    response: Final = files_main.file_content(
        file_id="s3://sdk-bucket/litellm-bedrock-files/batch.jsonl",
        custom_llm_provider="bedrock",
        s3_bucket_name="sdk-bucket",
        client=_echo_request_url_client(),
        **_SDK_AWS_CREDENTIALS,
    )

    assert httpx.URL(response.content.decode()).path == "/sdk-bucket/litellm-bedrock-files/batch.jsonl"


def test_sdk_file_content_rejects_a_file_outside_the_bucket_passed_to_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("AWS_S3_OUTPUT_BUCKET_NAME", raising=False)

    with pytest.raises(Exception, match="configured storage bucket"):
        files_main.file_content(
            file_id="s3://other-bucket/litellm-bedrock-files/batch.jsonl",
            custom_llm_provider="bedrock",
            s3_bucket_name="sdk-bucket",
            client=_echo_request_url_client(),
            **_SDK_AWS_CREDENTIALS,
        )


@pytest.mark.asyncio
async def test_afile_content_assumes_role_with_external_id(monkeypatch):
    """A trust policy requiring sts:ExternalId must be satisfied by the deployment's aws_external_id."""
    import datetime

    import boto3
    from botocore.exceptions import ClientError

    monkeypatch.delenv("AWS_EXTERNAL_ID", raising=False)

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if params.get("ExternalId") != "external-id-files-download":
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:AssumeRole"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIAFILESDOWNLOADROLE",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
                }
            }

    class FakeS3Body:
        def read(self):
            return b'{"custom_id": "req-1"}'

    class FakeS3Client:
        def get_object(self, Bucket, Key):
            return {"Body": FakeS3Body()}

    def fake_boto3_client(service_name, **kwargs):
        if service_name == "sts":
            return FakeSTSClient()
        return FakeS3Client()

    optional_params = {
        "s3_bucket_name": "safe-bucket",
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "AKIAFILESDOWNLOADCALLER",
        "aws_secret_access_key": "pod-caller-secret",
        "aws_role_name": "arn:aws:iam::999999999999:role/litellm-files-download-role",
        "aws_session_name": "litellm-files-download-session",
        "aws_external_id": "external-id-files-download",
    }

    with patch.object(boto3, "client", side_effect=fake_boto3_client) as mock_boto3_client:
        response = await BedrockFilesHandler().afile_content(
            file_content_request={"file_id": "s3://safe-bucket/litellm-bedrock-files-model-id-abc.jsonl"},
            optional_params=optional_params,
            timeout=10.0,
            max_retries=None,
        )

    s3_client_kwargs = next(call.kwargs for call in mock_boto3_client.call_args_list if call.args[0] == "s3")
    assert s3_client_kwargs["aws_access_key_id"] == "ASIAFILESDOWNLOADROLE"
    assert s3_client_kwargs["aws_session_token"] == "assumed-session-token"
    assert response.content == b'{"custom_id": "req-1"}'


@pytest.mark.asyncio
async def test_afile_content_builds_the_s3_client_with_the_s3_pair_when_it_differs_from_the_aws_identity():
    import boto3

    class FakeS3Body:
        def read(self):
            return b'{"custom_id": "req-1"}'

    class FakeS3Client:
        def get_object(self, Bucket, Key):
            return {"Body": FakeS3Body()}

    optional_params = {
        "s3_bucket_name": "safe-bucket",
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "AKIABEDROCKONLY",
        "aws_secret_access_key": "bedrock-only-secret",
        "aws_session_token": "bedrock-only-token",
        "s3_access_key_id": "AKIAS3ONLY",
        "s3_secret_access_key": "s3-only-secret",
    }

    with patch.object(boto3, "client", return_value=FakeS3Client()) as mock_boto3_client:
        response = await BedrockFilesHandler().afile_content(
            file_content_request={"file_id": "s3://safe-bucket/litellm-bedrock-files-model-id-abc.jsonl"},
            optional_params=optional_params,
            timeout=10.0,
            max_retries=None,
        )

    s3_client_kwargs = mock_boto3_client.call_args.kwargs
    assert s3_client_kwargs["aws_access_key_id"] == "AKIAS3ONLY"
    assert s3_client_kwargs["aws_secret_access_key"] == "s3-only-secret"
    assert s3_client_kwargs["aws_session_token"] is None, "the aws_* session token belongs to the Bedrock identity"
    assert response.content == b'{"custom_id": "req-1"}'
