from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
from litellm.llms.bedrock.files.transformation import BedrockFilesConfig


def test_bedrock_file_upload_signing_uses_deployment_credentials(monkeypatch):
    config = BedrockFilesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, ""

    monkeypatch.setattr(config, "_sign_s3_request", capture_signing)

    result = config.transform_create_file_request(
        model="",
        create_file_data={
            "file": (
                "batch.jsonl",
                b'{"custom_id":"req-1","body":{"model":"bedrock/model"}}\n',
                "application/jsonl",
            ),
            "purpose": "batch",
        },
        optional_params={},
        litellm_params={
            "s3_bucket_name": "deployment-bucket",
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "eu-west-1",
        },
    )

    assert "eu-west-1" in result["url"]
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"
    assert captured["optional_params"]["aws_secret_access_key"] == "deployment-secret"
    assert captured["optional_params"]["aws_region_name"] == "eu-west-1"


def test_bedrock_batch_signing_uses_deployment_credentials(monkeypatch):
    config = BedrockBatchesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, b"{}"

    monkeypatch.setattr(config.common_utils, "sign_aws_request", capture_signing)

    result = config.transform_create_batch_request(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        create_batch_data={
            "input_file_id": "s3://deployment-bucket/input.jsonl",
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
        },
        optional_params={},
        litellm_params={
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "eu-west-1",
            "aws_batch_role_arn": "arn:aws:iam::123456789012:role/bedrock-batch",
        },
    )

    assert result["url"].startswith("https://bedrock.eu-west-1.amazonaws.com/")
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"
    assert captured["optional_params"]["aws_secret_access_key"] == "deployment-secret"
    assert captured["optional_params"]["aws_region_name"] == "eu-west-1"


def test_bedrock_batch_retrieval_signing_uses_deployment_credentials(monkeypatch):
    config = BedrockBatchesConfig()
    captured = {}

    def capture_signing(**kwargs):
        captured.update(kwargs)
        return {}, b""

    monkeypatch.setattr(config.common_utils, "sign_aws_request", capture_signing)

    result = config.transform_retrieve_batch_request(
        batch_id="arn:aws:bedrock:eu-west-1:123456789012:model-invocation-job/job-1",
        optional_params={},
        litellm_params={
            "aws_access_key_id": "deployment-access-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "eu-west-1",
        },
    )

    assert result["url"].startswith("https://bedrock.eu-west-1.amazonaws.com/")
    assert captured["optional_params"]["aws_access_key_id"] == "deployment-access-key"
    assert captured["optional_params"]["aws_secret_access_key"] == "deployment-secret"
    assert captured["optional_params"]["aws_region_name"] == "eu-west-1"
