"""Bedrock batch: request-level S3 buckets flow through create_batch kwargs into litellm_params."""

import json
from unittest.mock import MagicMock, patch

import litellm


def test_bedrock_batch_s3_output_bucket_name_kwarg_in_payload():
    captured_request_body = None

    def mock_post(*args, **kwargs):
        nonlocal captured_request_body
        if "data" in kwargs:
            captured_request_body = kwargs["data"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jobArn": (
                "arn:aws:bedrock:us-west-2:123456789012:"
                "model-invocation-job/test-job"
            ),
            "jobName": "test-job",
            "status": "Submitted",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        return mock_response

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        side_effect=mock_post,
    ):
        litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="s3://input-bucket/input/test.jsonl",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_batch_role_arn="arn:aws:iam::123456789012:role/test-role",
            s3_output_bucket_name="dedicated-output-bucket",
        )

    assert captured_request_body is not None
    request_data = json.loads(captured_request_body)
    s3_uri = request_data["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    assert s3_uri.startswith("s3://dedicated-output-bucket/")


def test_bedrock_batch_s3_bucket_override_dict_wins_in_litellm_params():
    captured_request_body = None

    def mock_post(*args, **kwargs):
        nonlocal captured_request_body
        if "data" in kwargs:
            captured_request_body = kwargs["data"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jobArn": (
                "arn:aws:bedrock:us-west-2:123456789012:"
                "model-invocation-job/test-job"
            ),
            "jobName": "test-job",
            "status": "Submitted",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        return mock_response

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        side_effect=mock_post,
    ):
        litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="s3://input-bucket/input/test.jsonl",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_batch_role_arn="arn:aws:iam::123456789012:role/test-role",
            s3_output_bucket_name="model-default-output-bucket",
            _litellm_batch_s3_bucket_overrides={
                "s3_output_bucket_name": "request-level-output-bucket"
            },
        )

    assert captured_request_body is not None
    request_data = json.loads(captured_request_body)
    s3_uri = request_data["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    assert s3_uri.startswith("s3://request-level-output-bucket/")
