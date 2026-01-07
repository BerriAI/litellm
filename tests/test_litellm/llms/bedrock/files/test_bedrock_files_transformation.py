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

