# What is this?
## Unit Tests for OpenAI Batches API
import asyncio
import json as json_module
import os
import sys
import traceback
import tempfile
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import pytest
from typing import Optional
import litellm
from unittest.mock import patch, MagicMock
import httpx
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


_BEDROCK_TEST_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "test-access-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    "AWS_REGION": "us-west-2",
    "AWS_DEFAULT_REGION": "us-west-2",
}


class _CaptureAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self):
        self.timeout = None
        self.event_hooks = None
        self.client_alias = "bedrock-test"
        self.put_calls = []
        self.post_calls = []
        self.batch_jobs = {}

    async def put(
        self,
        url: str,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream: bool = False,
        content=None,
    ):
        self.put_calls.append(
            {
                "url": url,
                "data": data,
                "json": json,
                "params": params,
                "headers": headers or {},
                "timeout": timeout,
                "stream": stream,
                "content": content,
            }
        )
        body = data if data is not None else content
        content_bytes = body.encode("utf-8") if isinstance(body, str) else body or b""
        content_length = len(content_bytes)
        return httpx.Response(
            status_code=200,
            headers={"Content-Length": str(content_length)},
            request=httpx.Request("PUT", url),
        )

    async def post(
        self,
        url: str,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream: bool = False,
        logging_obj=None,
        files=None,
        content=None,
    ):
        self.post_calls.append(
            {
                "url": url,
                "data": data,
                "json": json,
                "params": params,
                "headers": headers or {},
                "timeout": timeout,
                "stream": stream,
                "content": content,
            }
        )
        raw = json if json is not None else (data if data is not None else content)
        payload = raw if isinstance(raw, dict) else json_module.loads(raw)
        job_name = payload["jobName"]
        job_arn = f"arn:aws:bedrock:us-west-2:941277531214:model-invocation-job/{job_name}"
        self.batch_jobs[job_arn] = {
            "jobArn": job_arn,
            "jobName": job_name,
            "modelId": payload["modelId"],
            "roleArn": payload["roleArn"],
            "status": "InProgress",
            "submitTime": "2026-06-02T03:50:00Z",
            "lastModifiedTime": "2026-06-02T03:55:00Z",
            "inputDataConfig": payload["inputDataConfig"],
            "outputDataConfig": payload["outputDataConfig"],
        }
        return httpx.Response(
            status_code=200,
            json={"jobArn": job_arn, "jobName": job_name, "status": "Submitted"},
            request=httpx.Request("POST", url),
        )


@pytest.mark.asyncio()
async def test_async_create_file():
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    litellm._turn_on_debug()
    print("Testing async create batch")

    file_name = "bedrock_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    capture_client = _CaptureAsyncHTTPHandler()
    with (
        patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV),
        open(file_path, "rb") as batch_file,
    ):
        file_obj = await litellm.acreate_file(
            file=batch_file,
            purpose="batch",
            custom_llm_provider="bedrock",
            s3_bucket_name="litellm-proxy-941277531214",
            client=capture_client,
        )

    assert len(capture_client.put_calls) == 1
    put_call = capture_client.put_calls[0]
    assert put_call["url"].startswith(
        "https://s3.us-west-2.amazonaws.com/litellm-proxy-941277531214/"
    )
    assert "/litellm-bedrock-files-us.anthropic.claude-haiku-4-5-20251001-v1-0-" in (
        put_call["url"]
    )
    assert put_call["url"].endswith(".jsonl")
    assert put_call["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert "recordId" in put_call["data"]
    assert file_obj.id.startswith(
        "s3://litellm-proxy-941277531214/litellm-bedrock-files-"
    )
    assert file_obj.filename.endswith(".jsonl")


@pytest.mark.asyncio()
async def test_async_file_and_batch():
    """
    Test file retrieval
    """
    litellm._turn_on_debug()
    file_name = "bedrock_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    capture_client = _CaptureAsyncHTTPHandler()
    with patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV):
        with open(file_path, "rb") as batch_file:
            file_obj = await litellm.acreate_file(
                file=batch_file,
                purpose="batch",
                custom_llm_provider="bedrock",
                s3_bucket_name="litellm-proxy-941277531214",
                client=capture_client,
            )
        assert len(capture_client.put_calls) == 1
        print("CREATED FILE RESPONSE=", file_obj)

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
            return_value=capture_client,
        ):
            # create batch
            create_batch_response = await litellm.acreate_batch(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_obj.id,
                metadata={"key1": "value1", "key2": "value2"},
                custom_llm_provider="bedrock",
                #########################################################
                # bedrock specific params
                #########################################################
                model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                aws_batch_role_arn="arn:aws:iam::941277531214:role/service-role/AmazonBedrockExecutionRoleForAgents_BB9HNW6V4CV",
            )
            assert len(capture_client.post_calls) == 1
            print("CREATED BATCH RESPONSE=", create_batch_response)

            # retrieve batch
            mock_bedrock_client = MagicMock()
            mock_bedrock_client.get_model_invocation_job.side_effect = (
                lambda jobIdentifier: capture_client.batch_jobs[jobIdentifier]
            )
            with patch("boto3.client", return_value=mock_bedrock_client):
                retrieve_batch_response = await litellm.aretrieve_batch(
                    batch_id=create_batch_response.id,
                    custom_llm_provider="bedrock",
                    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                )
            mock_bedrock_client.get_model_invocation_job.assert_called_once_with(
                jobIdentifier=create_batch_response.id
            )
            print("RETRIEVED BATCH RESPONSE=", retrieve_batch_response)

    # Validate the response
    assert retrieve_batch_response.id == create_batch_response.id
    assert retrieve_batch_response.object == "batch"
    assert retrieve_batch_response.status in [
        "validating",
        "in_progress",
        "completed",
        "failed",
        "cancelled",
    ]


@pytest.mark.asyncio()
async def test_mock_bedrock_file_url_mapping():
    """
    Simple test to capture PUT URL and validate mapping to file ID.
    """
    print("Testing Bedrock file URL mapping")

    capture_client = _CaptureAsyncHTTPHandler()
    with (
        patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV),
        open(
            os.path.join(os.path.dirname(__file__), "bedrock_batch_completions.jsonl"),
            "rb",
        ) as batch_file,
    ):
        file_obj = await litellm.acreate_file(
            file=batch_file,
            purpose="batch",
            custom_llm_provider="bedrock",
            s3_bucket_name="litellm-proxy-941277531214",
            client=capture_client,
        )

    captured_put_url = capture_client.put_calls[0]["url"]
    print(f"PUT URL: {captured_put_url}")
    print(f"File ID: {file_obj.id}")

    # Validate URL was captured and response is correct
    assert captured_put_url is not None
    assert file_obj.id.startswith("s3://")

    # Verify mapping
    from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

    bedrock_config = BedrockFilesConfig()
    expected_s3_uri, _ = bedrock_config._convert_https_url_to_s3_uri(captured_put_url)
    assert file_obj.id == expected_s3_uri


@pytest.mark.asyncio()
async def test_bedrock_retrieve_batch():
    """
    Test bedrock batch retrieval functionality, validating that input and output file IDs
    are correctly extracted from the Bedrock response and included in the final transformed response.
    """
    print("Testing bedrock batch retrieval")

    mock_bedrock_response = {
        "jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123",
        "jobName": "test-job-123",
        "modelId": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "roleArn": "arn:aws:iam::123456789012:role/service-role/AmazonBedrockExecutionRoleForAgents_TEST",
        "status": "Completed",
        "message": "",
        "submitTime": "2024-01-01T12:00:00Z",
        "lastModifiedTime": "2024-01-01T12:30:00Z",
        "endTime": "2024-01-01T13:00:00Z",
        "inputDataConfig": {
            "s3InputDataConfig": {"s3Uri": "s3://test-bucket/input/test-input.jsonl"}
        },
        "outputDataConfig": {
            "s3OutputDataConfig": {"s3Uri": "s3://test-bucket/output/"}
        },
    }

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.get_model_invocation_job.return_value = mock_bedrock_response
    mock_creds = MagicMock(access_key="ak", secret_key="sk", token="tok")

    with (
        patch("boto3.client", return_value=mock_bedrock_client),
        patch(
            "litellm.llms.bedrock.batches.transformation.BedrockBatchesConfig.get_credentials",
            return_value=mock_creds,
        ),
    ):
        batch_response = await litellm.aretrieve_batch(
            batch_id="arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

        assert (
            batch_response.id
            == "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-123"
        )
        assert batch_response.object == "batch"
        assert batch_response.status == "completed"
        assert batch_response.endpoint == "/v1/chat/completions"

        assert batch_response.input_file_id == "s3://test-bucket/input/test-input.jsonl"
        # Bedrock returns only the output *prefix*; the handler predicts the
        # actual output object as <prefix>/<job-id>/<basename(input)>.out.
        assert (
            batch_response.output_file_id
            == "s3://test-bucket/output/test-job-123/test-input.jsonl.out"
        )


def test_bedrock_batch_with_encryption_key_in_post_request():
    """
    Test that s3_encryption_key_id is included in the AWS POST request payload.
    """
    import json
    import litellm

    test_kms_key_id = (
        "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
    )

    captured_request_body = None

    def mock_post(*args, **kwargs):
        nonlocal captured_request_body
        if "data" in kwargs:
            captured_request_body = kwargs["data"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job",
            "jobName": "test-job",
            "status": "Submitted",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        return mock_response

    with (
        patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV),
        patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            side_effect=mock_post,
        ),
    ):
        response = litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="s3://test-bucket/input/test.jsonl",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            s3_encryption_key_id=test_kms_key_id,
            aws_batch_role_arn="arn:aws:iam::123456789012:role/test-role",
        )

    assert captured_request_body is not None, "Request body was not captured"

    request_data = json.loads(captured_request_body)
    print("REQUEST DATA to bedrock batch creation", json.dumps(request_data, indent=4))

    assert "outputDataConfig" in request_data
    assert "s3OutputDataConfig" in request_data["outputDataConfig"]
    assert "s3EncryptionKeyId" in request_data["outputDataConfig"]["s3OutputDataConfig"]
    assert (
        request_data["outputDataConfig"]["s3OutputDataConfig"]["s3EncryptionKeyId"]
        == test_kms_key_id
    )

    print("SUCCESS: s3_encryption_key_id properly included in AWS POST request")
