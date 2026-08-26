"""
Test bedrock files transformation functionality
"""

import json
import os
from collections.abc import Mapping
from unittest.mock import MagicMock
from urllib.parse import unquote, urlparse

import pytest
from botocore.auth import S3SigV4Auth, SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from litellm.llms.bedrock.files.transformation import BedrockJsonlFilesTransformation


class TestBedrockFilesTransformation:
    """Test bedrock files transformation"""

    def test_transform_openai_jsonl_content_to_bedrock_jsonl_content(self):
        """
        Test transformation of OpenAI JSONL format to Bedrock batch format.

        Validates that the transformation correctly converts OpenAI batch completion
        format to Bedrock's expected batch format with proper recordId and modelInput structure.
        """
        # Initialize the transformation class
        transformation = BedrockJsonlFilesTransformation()

        # Load input JSONL file
        input_file_path = os.path.join(
            os.path.dirname(__file__), "input_batch_completions.jsonl"
        )

        # Read and parse the JSONL content
        openai_jsonl_content = []
        with open(input_file_path, "r") as f:
            for line in f:
                if line.strip():
                    openai_jsonl_content.append(json.loads(line))

        # Transform the content
        bedrock_jsonl_content = (
            transformation._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                openai_jsonl_content=openai_jsonl_content
            )
        )

        # Basic validation
        assert len(bedrock_jsonl_content) == len(
            openai_jsonl_content
        ), "Should have same number of records"

        # Check structure of transformed records
        for i, record in enumerate(bedrock_jsonl_content):
            assert "recordId" in record, f"Record {i+1} should have recordId"
            assert "modelInput" in record, f"Record {i+1} should have modelInput"

            # Check recordId matches custom_id from input
            expected_custom_id = openai_jsonl_content[i].get("custom_id")
            assert (
                record["recordId"] == expected_custom_id
            ), f"Record {i+1} recordId should match custom_id"

            # Check modelInput has expected structure
            model_input = record["modelInput"]
            assert isinstance(
                model_input, dict
            ), f"Record {i+1} modelInput should be a dictionary"

            # For Anthropic models, should have anthropic_version and messages
            if "anthropic.claude" in openai_jsonl_content[i]["body"]["model"]:
                assert (
                    "anthropic_version" in model_input
                ), f"Record {i+1} should have anthropic_version"
                assert "messages" in model_input, f"Record {i+1} should have messages"
                assert (
                    "max_tokens" in model_input
                ), f"Record {i+1} should have max_tokens"

    def test_nova_text_only_uses_converse_format(self):
        """
        Test that Nova models produce Converse API format in batch modelInput.

        Verifies that:
        - max_tokens is wrapped inside inferenceConfig.maxTokens
        - messages use Converse content block format
        - No raw OpenAI keys (max_tokens, temperature) at the top level
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        openai_jsonl_content = [
            {
                "custom_id": "nova-text-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "us.amazon.nova-pro-v1:0",
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.7,
                },
            }
        ]

        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            openai_jsonl_content
        )

        assert len(result) == 1
        record = result[0]
        assert record["recordId"] == "nova-text-1"

        model_input = record["modelInput"]

        # Must have inferenceConfig with maxTokens, NOT top-level max_tokens
        assert (
            "inferenceConfig" in model_input
        ), "Nova modelInput must contain inferenceConfig"
        assert model_input["inferenceConfig"]["maxTokens"] == 50
        assert model_input["inferenceConfig"]["temperature"] == 0.7
        assert (
            "max_tokens" not in model_input
        ), "max_tokens must NOT be at the top level for Nova"
        assert (
            "temperature" not in model_input
        ), "temperature must NOT be at the top level for Nova"

        # Must have messages
        assert "messages" in model_input

        # Nova Pro rejects empty additionalModelRequestFields / system — they must be absent
        assert (
            "additionalModelRequestFields" not in model_input
        ), "Nova: empty additionalModelRequestFields must be omitted, not serialized as {}"
        assert (
            "system" not in model_input
        ), "Nova: empty system must be omitted, not serialized as []"

    def test_nova_batch_jsonl_omits_empty_converse_fields(self):
        """
        Regression test: Amazon Nova Pro returns 400 Malformed input request when
        additionalModelRequestFields or system are present but empty in the Converse
        API payload.  The proxy must strip these keys when they carry no data.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        openai_jsonl_content = [
            {
                "custom_id": "req-0",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "us.amazon.nova-pro-v1:0",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is 1 + 1? Answer with just the number.",
                        }
                    ],
                    "max_tokens": 16,
                },
            }
        ]

        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            openai_jsonl_content
        )

        assert len(result) == 1
        model_input = result[0]["modelInput"]

        assert (
            "additionalModelRequestFields" not in model_input
            or model_input["additionalModelRequestFields"]
        ), "additionalModelRequestFields must be absent or non-empty — Nova rejects {}"
        assert (
            "system" not in model_input or model_input["system"]
        ), "system must be absent or non-empty — Nova rejects []"

        # Validate the exact shape AWS accepts
        assert model_input == {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "What is 1 + 1? Answer with just the number."}
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": 16},
        }

    def test_nova_image_content_uses_converse_image_blocks(self):
        """
        Test that image_url content blocks are converted to Bedrock Converse
        image format for Nova models in batch.

        Verifies that:
        - image_url blocks are converted to {"image": {"format": ..., "source": {"bytes": ...}}}
        - text blocks are converted to {"text": "..."}
        - No raw OpenAI image_url type remains
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        # 1x1 transparent PNG
        img_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )

        openai_jsonl_content = [
            {
                "custom_id": "nova-img-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "us.amazon.nova-pro-v1:0",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64," + img_b64
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 100,
                },
            }
        ]

        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            openai_jsonl_content
        )

        assert len(result) == 1
        model_input = result[0]["modelInput"]

        # Check inferenceConfig
        assert "inferenceConfig" in model_input
        assert model_input["inferenceConfig"]["maxTokens"] == 100
        assert "max_tokens" not in model_input

        # Check messages structure
        messages = model_input["messages"]
        assert len(messages) == 1
        content_blocks = messages[0]["content"]

        # Should have text block and image block in Converse format
        has_text = False
        has_image = False
        for block in content_blocks:
            if "text" in block:
                has_text = True
            if "image" in block:
                has_image = True
                # Verify Converse image format
                assert "format" in block["image"], "Image block must have format field"
                assert "source" in block["image"], "Image block must have source field"
                assert (
                    "bytes" in block["image"]["source"]
                ), "Image source must have bytes field"
            # Must NOT have OpenAI-style image_url
            assert (
                "image_url" not in block
            ), "image_url must not appear in Converse format"
            assert (
                block.get("type") != "image_url"
            ), "type=image_url must not appear in Converse format"

        assert has_text, "Should have a text content block"
        assert has_image, "Should have an image content block"

    def test_anthropic_still_works_after_nova_fix(self):
        """
        Regression test: ensure Anthropic models are still correctly
        transformed after the Converse API provider changes.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        openai_jsonl_content = [
            {
                "custom_id": "claude-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "max_tokens": 10,
                },
            }
        ]

        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            openai_jsonl_content
        )

        assert len(result) == 1
        model_input = result[0]["modelInput"]

        # Anthropic should have anthropic_version
        assert "anthropic_version" in model_input
        assert "messages" in model_input
        assert "max_tokens" in model_input

    def test_get_complete_file_url_respects_s3_region_name(self):
        """
        s3_region_name in litellm_params must be used when building the S3 URL.
        Previously the code fell back to us-west-2 even when s3_region_name was set,
        breaking GovCloud (us-gov-west-1) deployments.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        jsonl_content = json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "bedrock/amazon.nova-pro-v1:0",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ).encode()

        create_file_data = {
            "file": ("batch.jsonl", jsonl_content, "application/jsonl"),
            "purpose": "batch",
        }

        litellm_params = {
            "s3_bucket_name": "litellm-batch-352026",
            "s3_region_name": "us-gov-west-1",
        }

        url = config.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="amazon.nova-pro-v1:0",
            optional_params={},
            litellm_params=litellm_params,
            data=create_file_data,
        )

        assert "us-gov-west-1" in url, f"Expected us-gov-west-1 in URL but got: {url}"
        assert (
            "us-west-2" not in url
        ), f"us-west-2 must not appear when s3_region_name is set, got: {url}"
        assert "litellm-batch-352026" in url

    def test_get_complete_file_url_sanitizes_untrusted_filename(self):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        create_file_data = {
            "file": ("../../owned.jsonl?acl=public", b"hello", "application/jsonl"),
            "purpose": "assistants",
        }

        url = config.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="amazon.nova-pro-v1:0",
            optional_params={"aws_region_name": "us-west-2"},
            litellm_params={"s3_bucket_name": "safe-bucket"},
            data=create_file_data,
        )

        parsed_url = urlparse(url)
        object_key = unquote(parsed_url.path).split("/safe-bucket/", 1)[1]
        assert object_key.startswith("litellm-bedrock-files/")
        assert object_key.endswith("-owned.jsonl_acl_public")
        assert ".." not in object_key
        assert parsed_url.query == ""

    def test_batch_object_name_sanitizes_model_path(self):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        object_name = config._get_s3_object_name_from_batch_jsonl(
            [{"body": {"model": "bedrock/../../secret:model"}}]
        )

        assert object_name.startswith("litellm-bedrock-files-")
        assert object_name.endswith(".jsonl")
        assert "/" not in object_name
        assert ".." not in object_name

    def test_transform_create_file_request_injects_s3_region_for_signing(self):
        """
        When s3_region_name is provided, transform_create_file_request must pass
        that region to _sign_s3_request so SigV4 signatures use the correct region.
        """
        from unittest.mock import patch

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        jsonl_content = json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "bedrock/amazon.nova-pro-v1:0",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ).encode()

        create_file_data = {
            "file": ("batch.jsonl", jsonl_content, "application/jsonl"),
            "purpose": "batch",
        }

        litellm_params = {
            "s3_bucket_name": "litellm-batch-352026",
            "s3_region_name": "us-gov-west-1",
        }

        captured_optional_params: dict = {}

        def fake_sign(content, api_base, optional_params, s3_encryption_key_id=None):
            captured_optional_params.update(optional_params)
            return {"Authorization": "fake"}, content

        with patch.object(config, "_sign_s3_request", side_effect=fake_sign):
            config.transform_create_file_request(
                model="amazon.nova-pro-v1:0",
                create_file_data=create_file_data,
                optional_params={},
                litellm_params=litellm_params,
            )

        assert (
            captured_optional_params.get("aws_region_name") == "us-gov-west-1"
        ), "s3_region_name must be forwarded as aws_region_name for SigV4 signing"

    def test_s3_region_name_wins_over_aws_region_name_for_signing(self):
        """
        When both s3_region_name and aws_region_name are set to different values,
        s3_region_name must win for signing (same as for the URL). Otherwise the
        SigV4 signature would be computed against a different region than the URL,
        causing SignatureDoesNotMatch from AWS.
        """
        from unittest.mock import patch

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        jsonl_content = json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "bedrock/amazon.nova-pro-v1:0",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ).encode()

        create_file_data = {
            "file": ("batch.jsonl", jsonl_content, "application/jsonl"),
            "purpose": "batch",
        }

        litellm_params = {
            "s3_bucket_name": "litellm-batch-352026",
            "s3_region_name": "us-gov-west-1",
        }
        # aws_region_name set to something different - s3_region_name must still win
        optional_params = {"aws_region_name": "us-east-1"}

        captured_optional_params: dict = {}

        def fake_sign(content, api_base, optional_params, s3_encryption_key_id=None):
            captured_optional_params.update(optional_params)
            return {"Authorization": "fake"}, content

        with patch.object(config, "_sign_s3_request", side_effect=fake_sign):
            config.transform_create_file_request(
                model="amazon.nova-pro-v1:0",
                create_file_data=create_file_data,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )

        assert (
            captured_optional_params.get("aws_region_name") == "us-gov-west-1"
        ), "s3_region_name must override aws_region_name for SigV4 signing"

    def _signed_upload_request(self, litellm_params: dict) -> dict:
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        jsonl_content = json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "bedrock/amazon.nova-pro-v1:0",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ).encode()

        request = config.transform_create_file_request(
            model="amazon.nova-pro-v1:0",
            create_file_data={
                "file": ("batch.jsonl", jsonl_content, "application/jsonl"),
                "purpose": "batch",
            },
            optional_params={
                "aws_access_key_id": "test-key-id",
                "aws_secret_access_key": "test-secret",
                "aws_region_name": "us-west-2",
            },
            litellm_params={"s3_bucket_name": "litellm-batch-bucket", **litellm_params},
        )
        assert isinstance(request, dict)
        return request

    def test_upload_signs_sse_kms_headers_when_key_configured(self, monkeypatch):
        """
        Buckets whose policy requires SSE-KMS reject the batch input-file PutObject
        unless the upload carries the aws:kms encryption headers; they must also be
        covered by SigV4 SignedHeaders or S3 answers SignatureDoesNotMatch.
        """
        monkeypatch.delenv("AWS_S3_ENCRYPTION_KEY_ID", raising=False)
        kms_key = "arn:aws:kms:us-west-2:1234:key/abcd"

        request = self._signed_upload_request({"s3_encryption_key_id": kms_key})

        headers = {key.lower(): value for key, value in request["headers"].items()}
        assert headers["x-amz-server-side-encryption"] == "aws:kms"
        assert headers["x-amz-server-side-encryption-aws-kms-key-id"] == kms_key
        signed_headers = headers["authorization"].split("SignedHeaders=")[1].split(",")[0]
        assert "x-amz-server-side-encryption" in signed_headers
        assert "x-amz-server-side-encryption-aws-kms-key-id" in signed_headers

    def test_upload_reads_sse_kms_key_from_env(self, monkeypatch):
        monkeypatch.setenv("AWS_S3_ENCRYPTION_KEY_ID", "env-kms-key")

        request = self._signed_upload_request({})

        headers = {key.lower(): value for key, value in request["headers"].items()}
        assert headers["x-amz-server-side-encryption-aws-kms-key-id"] == "env-kms-key"

    def test_upload_omits_sse_headers_when_no_key_configured(self, monkeypatch):
        monkeypatch.delenv("AWS_S3_ENCRYPTION_KEY_ID", raising=False)

        request = self._signed_upload_request({})

        headers = {key.lower() for key in request["headers"]}
        assert "x-amz-server-side-encryption" not in headers
        assert "x-amz-server-side-encryption-aws-kms-key-id" not in headers

    def test_create_file_response_reports_uploaded_object_size(self):
        """
        S3 answers PutObject with an empty body, so the returned FileObject must report the
        size of the body that was uploaded instead of the response's Content-Length (always 0).
        """
        import httpx

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        litellm_params: dict = {"s3_bucket_name": "litellm-batch-bucket"}
        jsonl_content = json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "bedrock/amazon.nova-pro-v1:0",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ).encode()

        request = config.transform_create_file_request(
            model="amazon.nova-pro-v1:0",
            create_file_data={
                "file": ("batch.jsonl", jsonl_content, "application/jsonl"),
                "purpose": "batch",
            },
            optional_params={
                "aws_access_key_id": "test-key-id",
                "aws_secret_access_key": "test-secret",
                "aws_region_name": "us-west-2",
            },
            litellm_params=litellm_params,
        )
        assert isinstance(request, dict)
        uploaded_size = len(request["data"].encode("utf-8"))
        assert uploaded_size > 0

        file_object = config.transform_create_file_response(
            model=None,
            raw_response=httpx.Response(
                status_code=200,
                headers={"Content-Length": "0", "ETag": '"abc123"'},
                content=b"",
            ),
            logging_obj=MagicMock(),
            litellm_params=litellm_params,
        )

        assert file_object.bytes == uploaded_size

    def test_openai_passthrough_still_works(self):
        """
        Regression test: ensure OpenAI-compatible models (e.g. gpt-oss)
        still use passthrough format.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()

        openai_jsonl_content = [
            {
                "custom_id": "openai-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "openai.gpt-oss-120b-1:0",
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                    ],
                    "max_tokens": 10,
                },
            }
        ]

        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            openai_jsonl_content
        )

        assert len(result) == 1
        model_input = result[0]["modelInput"]

        # OpenAI-compatible should use passthrough: max_tokens at top level
        assert "messages" in model_input
        assert "max_tokens" in model_input
        assert model_input["max_tokens"] == 10

    def test_resolves_model_alias_before_provider_mapping(self, monkeypatch):
        import litellm
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.setitem(
            litellm.model_alias_map,
            "bedrock-batch",
            "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
        )

        result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "req-1",
                    "body": {
                        "model": "bedrock-batch",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 16,
                    },
                }
            ]
        )

        assert result == [
            {
                "recordId": "req-1",
                "modelInput": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                    "max_tokens": 16,
                    "anthropic_version": "bedrock-2023-05-31",
                },
            }
        ]

    def test_resolves_model_alias_before_embedding_mapping(self, monkeypatch):
        import litellm
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.setitem(
            litellm.model_alias_map,
            "bedrock-embedding-batch",
            "bedrock/amazon.titan-embed-text-v2:0",
        )

        result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "embedding-1",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "bedrock-embedding-batch",
                        "input": "hello",
                    },
                }
            ]
        )

        assert result == [
            {
                "recordId": "embedding-1",
                "modelInput": {"inputText": "hello"},
            }
        ]

    def test_unmapped_alias_falls_back_to_target_model(self):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "req-1",
                    "body": {
                        "model": "bedrock-batch",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 16,
                    },
                },
                {
                    "custom_id": "req-2",
                    "body": {
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 16,
                    },
                },
            ],
            target_model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

        expected_model_input = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 16,
            "anthropic_version": "bedrock-2023-05-31",
        }
        assert result == [
            {"recordId": "req-1", "modelInput": expected_model_input},
            {"recordId": "req-2", "modelInput": expected_model_input},
        ]

    def test_record_provider_wins_over_target_model(self):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "openai-1",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "openai.gpt-oss-120b-1:0",
                        "messages": [{"role": "user", "content": "Hello!"}],
                        "max_tokens": 10,
                    },
                }
            ],
            target_model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

        assert result == [
            {
                "recordId": "openai-1",
                "modelInput": {
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 10,
                },
            }
        ]

    def test_embedding_alias_falls_back_to_target_model(self):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "embedding-1",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "bedrock-embedding-batch",
                        "input": "hello",
                    },
                }
            ],
            target_model="bedrock/amazon.titan-embed-text-v2:0",
        )

        assert result == [
            {
                "recordId": "embedding-1",
                "modelInput": {"inputText": "hello"},
            }
        ]

    def test_create_file_request_threads_deployment_model_to_alias_records(self):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        class CapturingSignConfig(BedrockFilesConfig):
            def __init__(self):
                super().__init__()
                self.signed_content: str | None = None

            def _sign_s3_request(self, content, api_base, optional_params, s3_encryption_key_id=None):
                self.signed_content = content
                return {"Authorization": "fake"}, content

        config = CapturingSignConfig()
        jsonl_content = json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "bedrock-batch",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ).encode()

        config.transform_create_file_request(
            model="",
            create_file_data={
                "file": ("batch.jsonl", jsonl_content, "application/jsonl"),
                "purpose": "batch",
            },
            optional_params={},
            litellm_params={
                "s3_bucket_name": "litellm-batch-352026",
                "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            },
        )

        assert config.signed_content is not None
        record = json.loads(config.signed_content)
        assert record["modelInput"]["anthropic_version"] == "bedrock-2023-05-31"
        assert "model" not in record["modelInput"]


class TestBedrockFilesEmbeddingTransformation:
    """
    Tests for routing OpenAI /v1/embeddings batch JSONL records through the
    Titan v2 transformer so AWS Bedrock's CreateModelInvocationJob receives
    a valid modelInput body.

    Scope is intentionally Titan v2 only - other embedding models will get
    their own follow-up PRs/tests so each schema is exercised in isolation.
    """

    def test_titan_v2_embedding_jsonl_matches_fixture(self):
        """Round-trip the input fixture against the expected Bedrock output."""
        import json
        import os

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "input_batch_embeddings.jsonl")) as f:
            openai_jsonl = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(here, "expected_bedrock_batch_embeddings.jsonl")) as f:
            expected = [json.loads(line) for line in f if line.strip()]

        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            openai_jsonl
        )

        assert result == expected

    def test_titan_v2_simple_string_input(self):
        """Single string `input` maps to `{"inputText": <str>}` with no extras."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "e1",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "bedrock/amazon.titan-embed-text-v2:0",
                        "input": "Hello",
                    },
                }
            ]
        )

        assert result == [{"recordId": "e1", "modelInput": {"inputText": "Hello"}}]

    def test_titan_v2_dimensions_and_encoding_format(self):
        """OpenAI `dimensions` / `encoding_format` map to Titan v2 schema."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "e1",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "bedrock/amazon.titan-embed-text-v2:0",
                        "input": "Hi",
                        "dimensions": 256,
                        "encoding_format": "float",
                    },
                }
            ]
        )

        model_input = result[0]["modelInput"]
        assert model_input["inputText"] == "Hi"
        assert model_input["dimensions"] == 256
        assert model_input["embeddingTypes"] == ["float"]

    def test_embedding_routing_falls_back_to_body_shape(self):
        """Records without `url` still route via `input` presence."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "e1",
                    "body": {
                        "model": "bedrock/amazon.titan-embed-text-v2:0",
                        "input": "Hello",
                    },
                }
            ]
        )

        assert result[0]["modelInput"] == {"inputText": "Hello"}

    def test_embedding_single_element_list_input_is_accepted(self):
        """A single-element list maps to the same shape as a bare string."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "e1",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "bedrock/amazon.titan-embed-text-v2:0",
                        "input": ["only one"],
                    },
                }
            ]
        )

        assert result[0]["modelInput"]["inputText"] == "only one"

    def test_embedding_multi_input_list_raises(self):
        """Multi-element `input` lists are rejected with a clear message."""
        import pytest

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        with pytest.raises(ValueError, match="one input per JSONL record"):
            config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                [
                    {
                        "custom_id": "e1",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "model": "bedrock/amazon.titan-embed-text-v2:0",
                            "input": ["a", "b"],
                        },
                    }
                ]
            )

    def test_embedding_missing_input_raises(self):
        """A record routed to /v1/embeddings without `input` is an error."""
        import pytest

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        with pytest.raises(ValueError, match="missing required `input`"):
            config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                [
                    {
                        "custom_id": "e1",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {"model": "bedrock/amazon.titan-embed-text-v2:0"},
                    }
                ]
            )

    def test_mixed_chat_and_embedding_in_same_batch(self):
        """Chat and embedding records in the same JSONL each take their path."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "chat-1",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                },
                {
                    "custom_id": "embed-1",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "bedrock/amazon.titan-embed-text-v2:0",
                        "input": "Hi",
                    },
                },
            ]
        )

        assert result[0]["recordId"] == "chat-1"
        assert "messages" in result[0]["modelInput"]
        assert result[0]["modelInput"]["anthropic_version"] == "bedrock-2023-05-31"

        assert result[1]["recordId"] == "embed-1"
        assert result[1]["modelInput"] == {"inputText": "Hi"}

    def test_unsupported_embedding_model_raises_not_implemented(self):
        """Cohere/Nova/Titan-G1 embed get a clear NotImplementedError, not a corrupt body."""
        import pytest

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        for unsupported_model in (
            "bedrock/cohere.embed-english-v3",
            "bedrock/amazon.titan-embed-text-v1",
            "bedrock/amazon.titan-embed-image-v1",
            "bedrock/amazon.nova-2-multimodal-embeddings-v1:0",
        ):
            with pytest.raises(NotImplementedError, match="titan-embed-text-v2"):
                config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                    [
                        {
                            "custom_id": "e1",
                            "method": "POST",
                            "url": "/v1/embeddings",
                            "body": {"model": unsupported_model, "input": "Hi"},
                        }
                    ]
                )

    def test_titan_v2_model_name_variants_route_correctly(self):
        """All common Titan v2 model id shapes route through the embedding path."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        for model_id in (
            "amazon.titan-embed-text-v2:0",
            "bedrock/amazon.titan-embed-text-v2:0",
            "us.amazon.titan-embed-text-v2:0",
            "bedrock/us.amazon.titan-embed-text-v2:0",
        ):
            result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                [
                    {
                        "custom_id": "e1",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {"model": model_id, "input": "Hi"},
                    }
                ]
            )
            assert result[0]["modelInput"] == {
                "inputText": "Hi"
            }, f"model id {model_id} did not route to Titan v2 embedding path"

    def test_pretokenized_input_list_of_ints_raises(self):
        """`input: List[int]` (pre-tokenized) is rejected, not silently mis-shaped."""
        import pytest

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        with pytest.raises(
            (NotImplementedError, ValueError), match=r"pre-tokenized|one input per"
        ):
            config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                [
                    {
                        "custom_id": "e1",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "model": "bedrock/amazon.titan-embed-text-v2:0",
                            "input": [1, 2, 3],
                        },
                    }
                ]
            )

    def test_pretokenized_single_wrapped_list_raises(self):
        """`input: List[List[int]]` with one element is rejected as pre-tokenized."""
        import pytest

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        with pytest.raises(NotImplementedError, match="pre-tokenized"):
            config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
                [
                    {
                        "custom_id": "e1",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "model": "bedrock/amazon.titan-embed-text-v2:0",
                            "input": [[1, 2, 3]],
                        },
                    }
                ]
            )

    def test_record_with_both_input_and_messages_routes_to_chat(self):
        """If a record has both fields, chat wins (safer default - see helper docstring)."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        config = BedrockFilesConfig()
        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "ambiguous-1",
                    "body": {
                        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "input": "this should be ignored by chat path",
                        "max_tokens": 5,
                    },
                }
            ]
        )

        assert "messages" in result[0]["modelInput"]
        assert "inputText" not in result[0]["modelInput"]


    def test_titan_v2_marker_boundary_rejects_lookalikes(self):
        """The marker must end at `:`, `/`, or end-of-string to avoid false positives."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        # Look-alikes that must NOT route through the Titan v2 path
        for model in (
            "bedrock/amazon.titan-embed-text-v20:0",
            "bedrock/amazon.titan-embed-text-v2-experimental:0",
            "bedrock/amazon.titan-embed-text-v2foo",
        ):
            assert not BedrockFilesConfig._is_titan_v2_embed_model(
                model
            ), f"{model} unexpectedly matched the Titan v2 marker"

        # Real Titan v2 ids that MUST match
        for model in (
            "amazon.titan-embed-text-v2:0",
            "bedrock/amazon.titan-embed-text-v2:0",
            "us.amazon.titan-embed-text-v2:0",
            "arn:aws:bedrock:us-east-1:123:foundation-model/amazon.titan-embed-text-v2:0",
        ):
            assert BedrockFilesConfig._is_titan_v2_embed_model(
                model
            ), f"{model} unexpectedly missed the Titan v2 marker"

    def test_titan_v2_accepted_when_registry_schema_field_matches(self, mocker):
        """Registry-driven happy path: nested
        `provider_specific_entry.bedrock_invocation_schema == "titan_v2"`
        is the authoritative signal."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        mocker.patch(
            "litellm.get_model_info",
            return_value={
                "provider_specific_entry": {"bedrock_invocation_schema": "titan_v2"}
            },
        )
        assert BedrockFilesConfig._is_titan_v2_embed_model(
            "amazon.titan-embed-text-v2:0"
        )

    def test_titan_v2_rejected_when_registry_schema_field_differs(self, mocker):
        """Registry resolves with a different schema value (e.g. a hypothetical
        Cohere Embed entry) -> reject. Registry is authoritative; no substring
        second-chance for ids the registry knows."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        mocker.patch(
            "litellm.get_model_info",
            return_value={
                "provider_specific_entry": {"bedrock_invocation_schema": "cohere_v3"}
            },
        )
        # Even though the model id looks like Titan v2, the registry says
        # otherwise and we trust it.
        assert not BedrockFilesConfig._is_titan_v2_embed_model(
            "amazon.titan-embed-text-v2:0"
        )

    def test_titan_v2_falls_back_to_marker_when_registry_lacks_schema_field(
        self, mocker
    ):
        """Registry resolves but the entry has no
        `provider_specific_entry.bedrock_invocation_schema` field yet (e.g.
        a stale local registry) -> fall through to substring."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        # No provider_specific_entry at all
        mocker.patch(
            "litellm.get_model_info",
            return_value={"mode": "embedding"},
        )
        assert BedrockFilesConfig._is_titan_v2_embed_model(
            "amazon.titan-embed-text-v2:0"
        )

        # provider_specific_entry present but missing the schema key
        mocker.patch(
            "litellm.get_model_info",
            return_value={
                "mode": "embedding",
                "provider_specific_entry": {"unrelated": "value"},
            },
        )
        assert BedrockFilesConfig._is_titan_v2_embed_model(
            "amazon.titan-embed-text-v2:0"
        )

    def test_titan_v2_accepted_when_registry_silent(self, mocker):
        """Marker-only match is fine for ids the registry can't resolve
        (cross-region profile prefixes, ARN forms)."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        mocker.patch("litellm.get_model_info", side_effect=Exception("not mapped"))
        assert BedrockFilesConfig._is_titan_v2_embed_model(
            "us.amazon.titan-embed-text-v2:0"
        )
        assert BedrockFilesConfig._is_titan_v2_embed_model(
            "arn:aws:bedrock:us-east-1:123:foundation-model/amazon.titan-embed-text-v2:0"
        )

    def test_lookup_provider_specific_field_helper(self, mocker):
        """Direct coverage of the nested registry field helper."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        # Happy path: returns the nested field's string value
        mocker.patch(
            "litellm.get_model_info",
            return_value={
                "provider_specific_entry": {"bedrock_invocation_schema": "titan_v2"}
            },
        )
        assert (
            BedrockFilesConfig._lookup_provider_specific_field(
                "anything", "bedrock_invocation_schema"
            )
            == "titan_v2"
        )

        # Registry raises -> None
        mocker.patch("litellm.get_model_info", side_effect=Exception("not mapped"))
        assert (
            BedrockFilesConfig._lookup_provider_specific_field("anything", "any")
            is None
        )

        # Registry returns non-dict -> None
        mocker.patch("litellm.get_model_info", return_value="not a dict")
        assert (
            BedrockFilesConfig._lookup_provider_specific_field("anything", "any")
            is None
        )

        # Registry returns dict without provider_specific_entry -> None
        mocker.patch("litellm.get_model_info", return_value={"mode": "embedding"})
        assert (
            BedrockFilesConfig._lookup_provider_specific_field(
                "anything", "bedrock_invocation_schema"
            )
            is None
        )

        # provider_specific_entry exists but isn't a dict -> None
        mocker.patch(
            "litellm.get_model_info",
            return_value={"provider_specific_entry": "not a dict"},
        )
        assert (
            BedrockFilesConfig._lookup_provider_specific_field(
                "anything", "bedrock_invocation_schema"
            )
            is None
        )

        # provider_specific_entry dict missing the requested field -> None
        mocker.patch(
            "litellm.get_model_info",
            return_value={"provider_specific_entry": {"unrelated": "x"}},
        )
        assert (
            BedrockFilesConfig._lookup_provider_specific_field(
                "anything", "bedrock_invocation_schema"
            )
            is None
        )

        # Non-string nested value -> None
        mocker.patch(
            "litellm.get_model_info",
            return_value={"provider_specific_entry": {"bedrock_invocation_schema": 42}},
        )
        assert (
            BedrockFilesConfig._lookup_provider_specific_field(
                "anything", "bedrock_invocation_schema"
            )
            is None
        )

        # Empty-string nested value -> None
        mocker.patch(
            "litellm.get_model_info",
            return_value={"provider_specific_entry": {"bedrock_invocation_schema": ""}},
        )
        assert (
            BedrockFilesConfig._lookup_provider_specific_field(
                "anything", "bedrock_invocation_schema"
            )
            is None
        )

    def test_classify_batch_record_helper(self):
        """Helper classifies by `url` first, then by body shape."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
        from litellm.types.llms.bedrock import BedrockBatchRecordKind

        assert (
            BedrockFilesConfig._classify_batch_record(
                {"url": "/v1/embeddings", "body": {"input": "x"}}
            )
            is BedrockBatchRecordKind.EMBEDDING
        )
        # body-only fallback
        assert (
            BedrockFilesConfig._classify_batch_record({"body": {"input": "x"}})
            is BedrockBatchRecordKind.EMBEDDING
        )
        # chat shape
        assert (
            BedrockFilesConfig._classify_batch_record(
                {"url": "/v1/chat/completions", "body": {"messages": []}}
            )
            is BedrockBatchRecordKind.CHAT
        )
        # ambiguous body without any recognized key is treated as chat
        assert (
            BedrockFilesConfig._classify_batch_record({"body": {}})
            is BedrockBatchRecordKind.CHAT
        )

    @pytest.mark.parametrize("body", ["not a mapping", ["messages"], 7, None], ids=["str", "list", "int", "missing"])
    def test_classify_batch_record_falls_back_to_chat_for_non_mapping_body(self, body):
        """A malformed body must not crash the whole upload during classification.

        Chat is the only kind whose transformer tolerates an unexpected shape and
        raises a readable error; routing a non-mapping body anywhere else would
        blow up on attribute access before the caller sees which record is bad.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
        from litellm.types.llms.bedrock import BedrockBatchRecordKind

        record = {"body": body} if body is not None else {}
        assert BedrockFilesConfig._classify_batch_record(record) is BedrockBatchRecordKind.CHAT

    def test_embedding_kind_is_rejected_by_the_chat_normalizer(self):
        """Embeddings have no chat equivalent, so the normalizer refuses them outright.

        The caller routes embeddings to the Titan transformer before ever getting
        here; this guard is what keeps a future caller from quietly shipping an
        embedding body through the chat path.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
        from litellm.types.llms.bedrock import BedrockBatchRecordKind

        with pytest.raises(ValueError, match="do not have a chat-completion equivalent"):
            BedrockFilesConfig._transform_batch_body_to_chat_body(
                {"model": "bedrock/amazon.titan-embed-text-v2:0", "input": "hi"},
                BedrockBatchRecordKind.EMBEDDING,
            )

    def test_explicit_chat_url_with_input_body_short_circuits_to_chat(self):
        """Explicit url=/v1/chat/completions wins even if body looks like embedding.

        Without this short-circuit, a chat record whose body happens to carry
        `input` (and no `messages`) would be mis-routed to the embedding
        transformer, corrupting the modelInput.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        from litellm.types.llms.bedrock import BedrockBatchRecordKind

        # Direct helper assertion
        assert (
            BedrockFilesConfig._classify_batch_record(
                {
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "input": "this would mis-route under the old precedence",
                    },
                }
            )
            is BedrockBatchRecordKind.CHAT
        )

        # End-to-end: a record like this routes through the chat path. We
        # just need to make sure we DON'T silently produce an inputText
        # body and call it a chat completion.
        config = BedrockFilesConfig()
        result = config._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "explicit-chat-with-input",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "input": "should not become inputText",
                        "max_tokens": 5,
                    },
                }
            ]
        )

        model_input = result[0]["modelInput"]
        assert (
            "inputText" not in model_input
        ), "explicit chat URL must not produce an embedding-shaped modelInput"

    def test_coerce_embedding_input_helper_isolated(self):
        """Direct coverage of the extracted input-normalization helper."""
        import pytest

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        # Happy paths
        assert BedrockFilesConfig._coerce_embedding_input_to_string("hello") == "hello"
        assert (
            BedrockFilesConfig._coerce_embedding_input_to_string(["hello"]) == "hello"
        )

        # Error paths
        with pytest.raises(ValueError, match="missing required `input`"):
            BedrockFilesConfig._coerce_embedding_input_to_string(None, model="m")
        with pytest.raises(ValueError, match="one input per JSONL record"):
            BedrockFilesConfig._coerce_embedding_input_to_string(["a", "b"])
        # A multi-element list of ints is rejected as "one input per JSONL
        # record" too - we can't tell if it's pre-tokenized or "3 strings"
        # without more context, so the most-actionable error wins.
        with pytest.raises(ValueError, match="one input per JSONL record"):
            BedrockFilesConfig._coerce_embedding_input_to_string([1, 2, 3])
        # Single-element list wrapping a token list -> pre-tokenized error.
        with pytest.raises(NotImplementedError, match="pre-tokenized"):
            BedrockFilesConfig._coerce_embedding_input_to_string([[1, 2, 3]])
        # Single-element list wrapping a bare int -> pre-tokenized error.
        with pytest.raises(NotImplementedError, match="pre-tokenized"):
            BedrockFilesConfig._coerce_embedding_input_to_string([42])
        with pytest.raises(ValueError, match="must be a string"):
            BedrockFilesConfig._coerce_embedding_input_to_string({"unsupported": True})

    def test_other_non_embedding_urls_do_not_route_to_embeddings(self):
        """An `input` body only means "embedding" when the url says so."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
        from litellm.types.llms.bedrock import BedrockBatchRecordKind

        # /v1/completions (legacy completions endpoint)
        assert (
            BedrockFilesConfig._classify_batch_record(
                {"url": "/v1/completions", "body": {"input": "x"}}
            )
            is BedrockBatchRecordKind.TEXT_COMPLETION
        )
        assert (
            BedrockFilesConfig._classify_batch_record(
                {"url": "/v1/responses", "body": {"input": "x"}}
            )
            is BedrockBatchRecordKind.RESPONSES
        )
        # Arbitrary unknown url - caller's explicit signal still wins
        assert (
            BedrockFilesConfig._classify_batch_record(
                {"url": "/v1/moderations", "body": {"input": "x"}}
            )
            is BedrockBatchRecordKind.CHAT
        )


class TestBedrockBatchNonChatEndpointRecords:
    """`/v1/completions` and `/v1/responses` JSONL records (issue #35639).

    Bedrock batch `modelInput` is always the model's InvokeModel/Converse body,
    so a record shaped for another OpenAI endpoint has to be normalized to chat
    completions first. Before this normalization every record below either
    raised `BadRequestError` at `POST /v1/files` (Anthropic, Nova) or silently
    shipped an empty `messages` list to AWS (passthrough providers).
    """

    ANTHROPIC_MODEL = "bedrock/us.anthropic.claude-sonnet-4-6"
    NOVA_MODEL = "bedrock/us.amazon.nova-pro-v1:0"
    PASSTHROUGH_MODEL = "bedrock/openai.gpt-oss-120b-1:0"

    def _transform(self, record: dict) -> dict:
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content([record])
        assert len(result) == 1
        assert result[0]["recordId"] == record["custom_id"]
        return result[0]["modelInput"]

    def test_anthropic_text_completion_record_wraps_prompt(self):
        model_input = self._transform(
            {
                "custom_id": "1",
                "method": "POST",
                "url": "/v1/completions",
                "body": {
                    "model": self.ANTHROPIC_MODEL,
                    "prompt": "Summarize the following call transcript",
                    "max_tokens": 64,
                },
            }
        )

        assert model_input == {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Summarize the following call transcript"}],
                }
            ],
            "max_tokens": 64,
            "anthropic_version": "bedrock-2023-05-31",
        }

    def test_anthropic_text_completion_record_keeps_every_prompt_in_a_list(self):
        model_input = self._transform(
            {
                "custom_id": "2",
                "method": "POST",
                "url": "/v1/completions",
                "body": {
                    "model": self.ANTHROPIC_MODEL,
                    "prompt": ["first prompt", "second prompt"],
                    "max_tokens": 8,
                },
            }
        )

        # Consecutive user messages are merged by the Anthropic transform, the
        # same way they are on the real-time path.
        assert model_input["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first prompt"},
                    {"type": "text", "text": "second prompt"},
                ],
            }
        ]
        assert "prompt" not in model_input

    def test_anthropic_responses_record_wraps_string_input(self):
        model_input = self._transform(
            {
                "custom_id": "3",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": self.ANTHROPIC_MODEL,
                    "input": "hi",
                    "max_output_tokens": 16,
                },
            }
        )

        assert model_input == {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_tokens": 16,
            "anthropic_version": "bedrock-2023-05-31",
        }
        assert "tools" not in model_input, "an empty tools array must not be shipped to Bedrock"

    def test_anthropic_responses_record_maps_instructions_and_input_items(self):
        """The Responses-specific params go through the same bridge as real time."""
        model_input = self._transform(
            {
                "custom_id": "4",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": self.ANTHROPIC_MODEL,
                    "instructions": "be terse",
                    "input": [
                        {"role": "user", "content": "what is 2+2?"},
                        {"role": "assistant", "content": "4"},
                        {"role": "user", "content": "and 3+3?"},
                    ],
                    "max_output_tokens": 32,
                    "temperature": 0.2,
                },
            }
        )

        assert model_input["system"] == [{"type": "text", "text": "be terse"}]
        assert model_input["max_tokens"] == 32
        assert model_input["temperature"] == 0.2
        assert [message["role"] for message in model_input["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert model_input["messages"][-1]["content"] == [{"type": "text", "text": "and 3+3?"}]
        assert "input" not in model_input
        assert "max_output_tokens" not in model_input

    def test_responses_record_keeps_metadata(self):
        """`metadata` reaches the bridge, which reads it as its own kwarg."""
        model_input = self._transform(
            {
                "custom_id": "4b",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": self.PASSTHROUGH_MODEL,
                    "input": "hi",
                    "metadata": {"tenant": "acct-1"},
                },
            }
        )

        assert model_input["metadata"] == {"tenant": "acct-1"}

    @pytest.mark.parametrize("model_attr", ["ANTHROPIC_MODEL", "NOVA_MODEL"], ids=["anthropic", "nova"])
    def test_modelled_providers_do_not_smuggle_metadata_into_the_bedrock_body(self, model_attr):
        """Providers with a real InvokeModel schema leave `metadata` out of `modelInput`.

        Batch `modelInput` has to match the model's own InvokeModel body, and
        neither the Anthropic messages body nor the Nova body has a field for
        arbitrary caller labels. Nova in particular answers `400 Malformed input
        request` for any key it does not recognize, so translating `metadata`
        into the Converse-level `requestMetadata` would fail the record rather
        than preserve the labels. The passthrough providers keep it because
        their body is the OpenAI request itself.
        """
        model_input = self._transform(
            {
                "custom_id": "4c",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": getattr(self, model_attr),
                    "input": "hi",
                    "max_output_tokens": 8,
                    "metadata": {"tenant": "acct-1"},
                },
            }
        )

        assert "metadata" not in model_input
        assert "requestMetadata" not in model_input

    @pytest.mark.parametrize(
        "body",
        [
            {"prompt": "hi"},
            {"input": "hi"},
        ],
        ids=["prompt", "input"],
    )
    def test_nova_converse_record_wraps_prompt_and_input(self, body):
        url = "/v1/completions" if "prompt" in body else "/v1/responses"
        model_input = self._transform(
            {
                "custom_id": "5",
                "method": "POST",
                "url": url,
                "body": {"model": self.NOVA_MODEL, **body},
            }
        )

        assert model_input["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]

    @pytest.mark.parametrize(
        "body",
        [
            {"prompt": "hi"},
            {"input": "hi"},
        ],
        ids=["prompt", "input"],
    )
    def test_passthrough_provider_record_no_longer_emits_empty_messages(self, body):
        """The passthrough branch used to emit `{"messages": [], "prompt": ...}`.

        That shape is accepted by `POST /v1/files`, so the whole batch job was
        submitted to AWS and only failed there.
        """
        url = "/v1/completions" if "prompt" in body else "/v1/responses"
        model_input = self._transform(
            {
                "custom_id": "6",
                "method": "POST",
                "url": url,
                "body": {"model": self.PASSTHROUGH_MODEL, **body},
            }
        )

        # Asserted on the serialized form, since the passthrough branch hands
        # `messages` straight to S3 without a per-provider transform.
        assert json.loads(json.dumps(model_input)) == {"messages": [{"role": "user", "content": "hi"}]}

    def test_mixed_endpoints_in_one_file_keep_their_own_shapes(self):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            [
                {
                    "custom_id": "chat",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.ANTHROPIC_MODEL,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4,
                    },
                },
                {
                    "custom_id": "text",
                    "url": "/v1/completions",
                    "body": {"model": self.ANTHROPIC_MODEL, "prompt": "hi", "max_tokens": 4},
                },
                {
                    "custom_id": "responses",
                    "url": "/v1/responses",
                    "body": {"model": self.ANTHROPIC_MODEL, "input": "hi", "max_output_tokens": 4},
                },
                {
                    "custom_id": "embedding",
                    "url": "/v1/embeddings",
                    "body": {"model": "bedrock/amazon.titan-embed-text-v2:0", "input": "hi"},
                },
            ]
        )

        assert [record["recordId"] for record in result] == [
            "chat",
            "text",
            "responses",
            "embedding",
        ]
        for record in result[:3]:
            assert record["modelInput"]["messages"] == [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ]
        assert result[3]["modelInput"] == {"inputText": "hi"}

    @pytest.mark.parametrize(
        ("url", "expected_message"),
        [
            ("/v1/completions", "missing required `prompt` field"),
            ("/v1/responses", "missing required `input` field"),
        ],
    )
    def test_missing_required_field_raises_actionable_error(self, url, expected_message):
        with pytest.raises(ValueError, match=expected_message):
            self._transform(
                {
                    "custom_id": "7",
                    "method": "POST",
                    "url": url,
                    "body": {"model": self.ANTHROPIC_MODEL, "max_tokens": 4},
                }
            )

    def test_prompt_body_without_url_is_still_wrapped(self):
        """A record can omit `url`; the body shape then decides."""
        model_input = self._transform(
            {
                "custom_id": "8",
                "body": {"model": self.ANTHROPIC_MODEL, "prompt": "hi", "max_tokens": 4},
            }
        )

        assert model_input["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def test_messages_win_over_prompt_when_url_is_absent(self):
        model_input = self._transform(
            {
                "custom_id": "9",
                "body": {
                    "model": self.ANTHROPIC_MODEL,
                    "messages": [{"role": "user", "content": "from messages"}],
                    "prompt": "from prompt",
                    "max_tokens": 4,
                },
            }
        )

        assert model_input["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "from messages"}]}
        ]


class TestBedrockFileContentTransformation:
    """SigV4-signed S3 GetObject retrieval of Bedrock batch output files."""

    S3_URI = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    EXPECTED_URL = "https://s3.us-west-2.amazonaws.com/my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"

    def _litellm_params(self) -> dict:
        return {
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_secret_access_key": "secret",
            "aws_region_name": "us-west-2",
        }

    def test_transform_file_content_request_signs_s3_get(self, monkeypatch):
        """The request transform must produce the S3 object URL plus SigV4 GET headers."""
        import hashlib

        from litellm.llms.bedrock.files.transformation import (
            S3_SIGNED_GET_HEADERS_PARAM,
            BedrockFilesConfig,
        )

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")
        litellm_params = self._litellm_params()

        url, params = BedrockFilesConfig().transform_file_content_request(
            file_content_request={"file_id": self.S3_URI},
            optional_params={},
            litellm_params=litellm_params,
        )

        assert url == self.EXPECTED_URL
        assert params == {}

        signed_headers = litellm_params[S3_SIGNED_GET_HEADERS_PARAM]
        content_hashes = {
            value
            for name, value in signed_headers.items()
            if name.lower() == "x-amz-content-sha256"
        }
        assert content_hashes == {hashlib.sha256(b"").hexdigest()}, (
            "GET has no payload, so the content hash must be the empty-body hash."
            " The header name is matched case-insensitively because botocore picks"
            " its own casing and HTTP header names are case-insensitive"
        )
        authorization = signed_headers["Authorization"]
        assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/")
        assert "/us-west-2/s3/aws4_request" in authorization
        assert "x-amz-content-sha256" in authorization
        assert "X-Amz-Date" in signed_headers

    def test_transform_file_content_request_decodes_unified_file_id(self, monkeypatch):
        """Base64 unified ids carrying llm_output_file_id must resolve to their S3 object."""
        import base64

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
        from litellm.types.utils import SpecialEnums

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")
        unified_file_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
            "application/json", "unified-id", "", self.S3_URI, "model-id"
        )
        encoded_file_id = (
            base64.urlsafe_b64encode(unified_file_id.encode()).decode().rstrip("=")
        )

        url, _ = BedrockFilesConfig().transform_file_content_request(
            file_content_request={"file_id": encoded_file_id},
            optional_params={},
            litellm_params=self._litellm_params(),
        )

        assert url == self.EXPECTED_URL

    def test_transform_file_content_request_rejects_foreign_bucket(self, monkeypatch):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")

        with pytest.raises(ValueError, match="configured storage bucket"):
            BedrockFilesConfig().transform_file_content_request(
                file_content_request={
                    "file_id": "s3://other-bucket/litellm-batch-outputs/job/x.jsonl.out"
                },
                optional_params={},
                litellm_params=self._litellm_params(),
            )

    def test_transform_file_content_request_rejects_unmanaged_key(self, monkeypatch):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")

        with pytest.raises(ValueError, match="LiteLLM-managed"):
            BedrockFilesConfig().transform_file_content_request(
                file_content_request={"file_id": "s3://my-bucket/private/x.jsonl"},
                optional_params={},
                litellm_params=self._litellm_params(),
            )

    def test_extract_s3_uri_rejects_non_managed_file_id(self):
        """A file id that is neither an s3:// URI nor a unified id must be rejected."""
        from litellm.llms.bedrock.files.transformation import (
            extract_s3_uri_from_file_id,
        )

        with pytest.raises(ValueError, match="managed LiteLLM S3 file id"):
            extract_s3_uri_from_file_id("file-1234567890")

    def test_transform_file_content_request_requires_configured_bucket(
        self, monkeypatch
    ):
        """Without a server-configured bucket (env or snapshot), the request must fail
        before any S3 call rather than guessing a bucket from the file id."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)

        with pytest.raises(ValueError, match="S3 bucket_name is required"):
            BedrockFilesConfig().transform_file_content_request(
                file_content_request={"file_id": self.S3_URI},
                optional_params={},
                litellm_params=self._litellm_params(),
            )

    def test_transform_file_content_request_requires_file_id(self, monkeypatch):
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")

        with pytest.raises(ValueError, match="file_id is required"):
            BedrockFilesConfig().transform_file_content_request(
                file_content_request={},
                optional_params={},
                litellm_params=self._litellm_params(),
            )

    def _trusted(self, **deployment_litellm_params) -> dict:
        """Build the trusted snapshot the way the proxy does: deployment
        litellm_params funneled through ``CredentialLiteLLMParams`` (the strict
        allowlist ``get_deployment_credentials_with_provider`` applies) before
        retrieval ever sees them. Injecting a raw ``MappingProxyType`` would
        bypass that filter and hide whether a bucket field actually survives
        into the snapshot in production."""
        from types import MappingProxyType

        from litellm.types.router import CredentialLiteLLMParams

        snapshot = CredentialLiteLLMParams(**deployment_litellm_params).model_dump(
            exclude_none=True
        )
        params = self._litellm_params()
        params["_litellm_internal_model_credentials"] = MappingProxyType(snapshot)
        return params

    def test_retrieves_from_distinct_output_bucket(self, monkeypatch):
        """Batch outputs can land in a separate s3_output_bucket_name. Retrieval
        must validate the file id against the output bucket too, not just the
        input bucket, or the very outputs the feature serves are unreachable.
        The snapshot is built through the production credential filter, so this
        fails if s3_output_bucket_name is dropped from that allowlist."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
        monkeypatch.delenv("AWS_S3_OUTPUT_BUCKET_NAME", raising=False)

        url, _ = BedrockFilesConfig().transform_file_content_request(
            file_content_request={
                "file_id": "s3://out-bucket/litellm-batch-outputs/job/in.jsonl.out"
            },
            optional_params={},
            litellm_params=self._trusted(
                s3_bucket_name="in-bucket", s3_output_bucket_name="out-bucket"
            ),
        )

        assert (
            url
            == "https://s3.us-west-2.amazonaws.com/out-bucket/litellm-batch-outputs/job/in.jsonl.out"
        )

    def test_output_bucket_falls_back_to_env(self, monkeypatch):
        """The output bucket resolves from AWS_S3_OUTPUT_BUCKET_NAME when not in
        the trusted snapshot, mirroring the input-bucket env fallback."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "in-bucket")
        monkeypatch.setenv("AWS_S3_OUTPUT_BUCKET_NAME", "env-out-bucket")

        url, _ = BedrockFilesConfig().transform_file_content_request(
            file_content_request={
                "file_id": "s3://env-out-bucket/litellm-batch-outputs/job/in.jsonl.out"
            },
            optional_params={},
            litellm_params=self._litellm_params(),
        )

        assert (
            url
            == "https://s3.us-west-2.amazonaws.com/env-out-bucket/litellm-batch-outputs/job/in.jsonl.out"
        )

    def test_input_bucket_still_validates_when_output_bucket_set(self, monkeypatch):
        """Adding output-bucket support must not break retrieval of input-bucket
        objects when both buckets are configured."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
        monkeypatch.delenv("AWS_S3_OUTPUT_BUCKET_NAME", raising=False)

        url, _ = BedrockFilesConfig().transform_file_content_request(
            file_content_request={
                "file_id": "s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out"
            },
            optional_params={},
            litellm_params=self._trusted(
                s3_bucket_name="in-bucket", s3_output_bucket_name="out-bucket"
            ),
        )

        assert (
            url
            == "https://s3.us-west-2.amazonaws.com/in-bucket/litellm-batch-outputs/job/in.jsonl.out"
        )

    def test_rejects_bucket_outside_input_and_output(self, monkeypatch):
        """A file id whose bucket is neither the input nor the output bucket is
        still rejected (SSRF / bucket-confusion guard)."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
        monkeypatch.delenv("AWS_S3_OUTPUT_BUCKET_NAME", raising=False)

        with pytest.raises(ValueError, match="configured storage bucket"):
            BedrockFilesConfig().transform_file_content_request(
                file_content_request={
                    "file_id": "s3://other-bucket/litellm-batch-outputs/job/x.jsonl.out"
                },
                optional_params={},
                litellm_params=self._trusted(
                    s3_bucket_name="in-bucket", s3_output_bucket_name="out-bucket"
                ),
            )

    def test_sign_request_without_botocore_raises_helpful_error(self, monkeypatch):
        """A missing botocore must surface an actionable 'install boto3' error
        rather than a raw import failure."""
        import sys

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")
        monkeypatch.setitem(sys.modules, "botocore.auth", None)

        with pytest.raises(ImportError, match="boto3"):
            BedrockFilesConfig().transform_file_content_request(
                file_content_request={"file_id": self.S3_URI},
                optional_params={},
                litellm_params=self._litellm_params(),
            )

    def test_bucket_resolved_from_trusted_model_credentials(self, monkeypatch):
        """Per-model s3_bucket_name must be honored via the server-side credential snapshot."""
        from types import MappingProxyType

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
        litellm_params = self._litellm_params()
        litellm_params["_litellm_internal_model_credentials"] = MappingProxyType(
            {"s3_bucket_name": "my-bucket"}
        )

        url, _ = BedrockFilesConfig().transform_file_content_request(
            file_content_request={"file_id": self.S3_URI},
            optional_params={},
            litellm_params=litellm_params,
        )

        assert url == self.EXPECTED_URL

    def test_s3_region_name_wins_for_content_signing(self, monkeypatch):
        """s3_region_name must override aws_region_name for both the URL and the signature."""
        from litellm.llms.bedrock.files.transformation import (
            S3_SIGNED_GET_HEADERS_PARAM,
            BedrockFilesConfig,
        )

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")
        litellm_params = self._litellm_params()
        litellm_params["s3_region_name"] = "eu-west-1"

        url, _ = BedrockFilesConfig().transform_file_content_request(
            file_content_request={"file_id": self.S3_URI},
            optional_params={},
            litellm_params=litellm_params,
        )

        assert url.startswith("https://s3.eu-west-1.amazonaws.com/")
        authorization = litellm_params[S3_SIGNED_GET_HEADERS_PARAM]["Authorization"]
        assert "/eu-west-1/s3/aws4_request" in authorization

    def test_validate_environment_merges_and_pops_signed_get_headers(self):
        from litellm.llms.bedrock.files.transformation import (
            S3_SIGNED_GET_HEADERS_PARAM,
            BedrockFilesConfig,
        )

        litellm_params = {
            S3_SIGNED_GET_HEADERS_PARAM: {"Authorization": "AWS4-HMAC-SHA256 test"}
        }

        headers = BedrockFilesConfig().validate_environment(
            headers={"x-custom": "kept"},
            model="",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
        )

        assert headers == {
            "x-custom": "kept",
            "Authorization": "AWS4-HMAC-SHA256 test",
        }
        assert S3_SIGNED_GET_HEADERS_PARAM not in litellm_params

    def test_transform_file_content_response_wraps_binary_content(self):
        import httpx

        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        raw_response = httpx.Response(
            status_code=200,
            content=b'{"recordId": "CALL0000001"}',
            request=httpx.Request("GET", self.EXPECTED_URL),
        )

        result = BedrockFilesConfig().transform_file_content_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.response.content == b'{"recordId": "CALL0000001"}'

    def test_transform_file_content_response_raises_on_s3_error(self):
        import httpx

        from litellm.llms.bedrock.common_utils import BedrockError
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        raw_response = httpx.Response(
            status_code=403,
            content=b"<Error><Code>AccessDenied</Code></Error>",
            request=httpx.Request("GET", self.EXPECTED_URL),
        )

        with pytest.raises(BedrockError, match="AccessDenied"):
            BedrockFilesConfig().transform_file_content_response(
                raw_response=raw_response,
                logging_obj=MagicMock(),
                litellm_params={},
            )

    def test_file_content_end_to_end_sends_signed_get(self, monkeypatch):
        """litellm.file_content must issue a SigV4-signed GET and return the S3 object bytes."""
        import httpx
        import respx

        import litellm

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")

        with respx.mock:
            route = respx.get(self.EXPECTED_URL).mock(
                return_value=httpx.Response(200, content=b'{"recordId": "x"}')
            )

            response = litellm.file_content(
                file_id=self.S3_URI,
                custom_llm_provider="bedrock",
                **self._litellm_params(),
            )

        assert route.called
        request = route.calls[0].request
        assert request.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "x-amz-content-sha256" in request.headers
        assert response.content == b'{"recordId": "x"}'

    @pytest.mark.asyncio
    async def test_afile_content_end_to_end_sends_signed_get(self, monkeypatch):
        """Async variant: litellm.afile_content over the same signed GET path."""
        import httpx
        import respx

        import litellm

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", "my-bucket")
        # respx can only intercept httpx transports
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.in_memory_llm_clients_cache.flush_cache()

        with respx.mock:
            route = respx.get(self.EXPECTED_URL).mock(
                return_value=httpx.Response(200, content=b'{"recordId": "x"}')
            )

            response = await litellm.afile_content(
                file_id=self.S3_URI,
                custom_llm_provider="bedrock",
                **self._litellm_params(),
            )

        assert route.called
        assert (
            route.calls[0]
            .request.headers["Authorization"]
            .startswith("AWS4-HMAC-SHA256")
        )
        assert response.content == b'{"recordId": "x"}'


class TestBedrockFilesS3SignatureEncoding:
    """
    S3 rebuilds the canonical request from the wire path with single percent-encoding,
    which botocore models as S3SigV4Auth. Plain SigV4Auth quotes the already encoded
    path a second time, so an object key holding any character that percent-encodes
    (a configured bucket prefix with a space) is signed over %2520 while the request
    carries %20, and S3 answers 403 SignatureDoesNotMatch.
    """

    ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
    SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    BUCKET_WITH_SPACED_PREFIX = "my-bucket/LLM AI Projects"
    REGION = "us-west-2"

    def _credential_params(self) -> dict[str, str]:
        return {
            "aws_access_key_id": self.ACCESS_KEY,
            "aws_secret_access_key": self.SECRET_KEY,
            "aws_region_name": self.REGION,
        }

    def _signature_under(
        self,
        signer_cls: type[SigV4Auth],
        method: str,
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> str:
        sent = {name.lower(): value for name, value in headers.items()}
        signed_names = (
            sent["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")
        )
        request = AWSRequest(
            method=method,
            url=url,
            data=body,
            headers={name: sent[name] for name in signed_names if name in sent},
        )
        request.context["timestamp"] = sent["x-amz-date"]
        signer = signer_cls(
            Credentials(self.ACCESS_KEY, self.SECRET_KEY), "s3", self.REGION
        )
        return signer.signature(
            signer.string_to_sign(request, signer.canonical_request(request)), request
        )

    def _assert_signed_the_way_s3_reads_it(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
    ) -> None:
        assert "%20" in url, "the object key must reach the wire percent-encoded"
        sent_signature = headers["Authorization"].split("Signature=")[1].strip()
        assert sent_signature == self._signature_under(
            S3SigV4Auth, method, url, body, headers
        )
        assert sent_signature != self._signature_under(
            SigV4Auth, method, url, body, headers
        )

    def test_create_file_signs_spaced_object_key_the_way_s3_does(self) -> None:
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        content = json.dumps(
            {
                "custom_id": "1",
                "body": {
                    "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            }
        )
        signed = BedrockFilesConfig().transform_create_file_request(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            create_file_data={
                "file": ("batch.jsonl", content.encode("utf-8"), "application/jsonl"),
                "purpose": "batch",
            },
            optional_params=self._credential_params(),
            litellm_params={
                "s3_bucket_name": self.BUCKET_WITH_SPACED_PREFIX,
                "s3_region_name": self.REGION,
            },
        )

        self._assert_signed_the_way_s3_reads_it(
            method="PUT",
            url=signed["url"],
            body=signed["data"].encode("utf-8"),
            headers=signed["headers"],
        )

    def test_file_content_signs_spaced_object_key_the_way_s3_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.llms.bedrock.files.transformation import (
            S3_SIGNED_GET_HEADERS_PARAM,
            BedrockFilesConfig,
        )

        monkeypatch.setenv("AWS_S3_BUCKET_NAME", self.BUCKET_WITH_SPACED_PREFIX)
        litellm_params = {
            "s3_bucket_name": self.BUCKET_WITH_SPACED_PREFIX,
            "s3_region_name": self.REGION,
            **self._credential_params(),
        }

        url, _ = BedrockFilesConfig().transform_file_content_request(
            file_content_request={
                "file_id": "s3://my-bucket/LLM AI Projects/litellm-bedrock-files-model-abc.jsonl"
            },
            optional_params={},
            litellm_params=litellm_params,
        )

        self._assert_signed_the_way_s3_reads_it(
            method="GET",
            url=url,
            body=None,
            headers=litellm_params[S3_SIGNED_GET_HEADERS_PARAM],
        )
