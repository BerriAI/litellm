"""
Test bedrock files transformation functionality
"""

import json
import os
from urllib.parse import unquote, urlparse

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

        def fake_sign(content, api_base, optional_params):
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

        def fake_sign(content, api_base, optional_params):
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

    def test_url_embeddings_with_missing_input_raises_not_chat_error(self):
        """url says embed, body lacks input → embedding-path error, not chat-path crash."""
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

    def test_is_embedding_record_helper(self):
        """Helper detects embeddings via `url` first, then by body shape."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        assert BedrockFilesConfig._is_embedding_record(
            {"url": "/v1/embeddings", "body": {"input": "x"}}
        )
        # body-only fallback
        assert BedrockFilesConfig._is_embedding_record({"body": {"input": "x"}})
        # chat shape
        assert not BedrockFilesConfig._is_embedding_record(
            {"url": "/v1/chat/completions", "body": {"messages": []}}
        )
        # ambiguous body without `input` is treated as not-embedding
        assert not BedrockFilesConfig._is_embedding_record({"body": {}})

    def test_explicit_chat_url_with_input_body_short_circuits_to_chat(self):
        """Explicit url=/v1/chat/completions wins even if body looks like embedding.

        Without this short-circuit, a chat record whose body happens to carry
        `input` (and no `messages`) would be mis-routed to the embedding
        transformer, corrupting the modelInput.
        """
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        # Direct helper assertion
        assert not BedrockFilesConfig._is_embedding_record(
            {
                "url": "/v1/chat/completions",
                "body": {
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "input": "this would mis-route under the old precedence",
                },
            }
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

    def test_other_non_embedding_urls_route_to_chat(self):
        """Any non-/v1/embeddings url short-circuits to chat path."""
        from litellm.llms.bedrock.files.transformation import BedrockFilesConfig

        # /v1/completions (legacy completions endpoint)
        assert not BedrockFilesConfig._is_embedding_record(
            {"url": "/v1/completions", "body": {"input": "x"}}
        )
        # Arbitrary unknown url - caller's explicit signal still wins
        assert not BedrockFilesConfig._is_embedding_record(
            {"url": "/v1/responses", "body": {"input": "x"}}
        )
