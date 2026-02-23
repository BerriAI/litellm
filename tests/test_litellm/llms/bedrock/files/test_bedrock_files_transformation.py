"""
Test bedrock files transformation functionality
"""
import json
import os
from typing import Any, Dict, List

import pytest

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
            os.path.dirname(__file__), 
            "input_batch_completions.jsonl"
        )
        
        # Read and parse the JSONL content
        openai_jsonl_content = []
        with open(input_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    openai_jsonl_content.append(json.loads(line))
        
        # Transform the content
        bedrock_jsonl_content = transformation._transform_openai_jsonl_content_to_bedrock_jsonl_content(
            openai_jsonl_content=openai_jsonl_content
        )
        
        # Print the transformation results for validation
        print("\n=== INPUT (OpenAI format) ===")
        for i, content in enumerate(openai_jsonl_content):
            print(f"Record {i+1}:")
            print(json.dumps(content, indent=2))
            print()
        
        print("\n=== OUTPUT (Bedrock format) ===")
        for i, content in enumerate(bedrock_jsonl_content):
            print(f"Record {i+1}:")
            print(json.dumps(content, indent=2))
            print()
        
        # Basic validation
        assert len(bedrock_jsonl_content) == len(openai_jsonl_content), "Should have same number of records"
        
        # Check structure of transformed records
        for i, record in enumerate(bedrock_jsonl_content):
            assert "recordId" in record, f"Record {i+1} should have recordId"
            assert "modelInput" in record, f"Record {i+1} should have modelInput"
            
            # Check recordId matches custom_id from input
            expected_custom_id = openai_jsonl_content[i].get("custom_id")
            assert record["recordId"] == expected_custom_id, f"Record {i+1} recordId should match custom_id"
            
            # Check modelInput has expected structure
            model_input = record["modelInput"]
            assert isinstance(model_input, dict), f"Record {i+1} modelInput should be a dictionary"
            
            # For Anthropic models, should have anthropic_version and messages
            if "anthropic.claude" in openai_jsonl_content[i]["body"]["model"]:
                assert "anthropic_version" in model_input, f"Record {i+1} should have anthropic_version"
                assert "messages" in model_input, f"Record {i+1} should have messages"
                assert "max_tokens" in model_input, f"Record {i+1} should have max_tokens"
        
        # Write expected output to file for reference
        expected_output_path = os.path.join(
            os.path.dirname(__file__), 
            "expected_bedrock_batch_completions.jsonl"
        )
        
        with open(expected_output_path, 'w') as f:
            for record in bedrock_jsonl_content:
                f.write(json.dumps(record) + '\n')
        
        print(f"\n=== Expected output written to: {expected_output_path} ===")

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
        assert "inferenceConfig" in model_input, (
            "Nova modelInput must contain inferenceConfig"
        )
        assert model_input["inferenceConfig"]["maxTokens"] == 50
        assert model_input["inferenceConfig"]["temperature"] == 0.7
        assert "max_tokens" not in model_input, (
            "max_tokens must NOT be at the top level for Nova"
        )
        assert "temperature" not in model_input, (
            "temperature must NOT be at the top level for Nova"
        )

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
                assert "format" in block["image"], (
                    "Image block must have format field"
                )
                assert "source" in block["image"], (
                    "Image block must have source field"
                )
                assert "bytes" in block["image"]["source"], (
                    "Image source must have bytes field"
                )
            # Must NOT have OpenAI-style image_url
            assert "image_url" not in block, (
                "image_url must not appear in Converse format"
            )
            assert block.get("type") != "image_url", (
                "type=image_url must not appear in Converse format"
            )

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
                    "model": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
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

