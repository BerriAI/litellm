import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

class TestPassThroughContentLength:
    """Tests for the Content-Length mismatch fix in pass-through endpoints."""

    def test_content_length_consistency(self):
        """Test that the Content-Length is consistent when using pre-serialized JSON."""
        # Test data
        test_data = {
            "prompt": "\n\nHuman: Tell me a short joke\n\nAssistant:",
            "max_tokens_to_sample": 50,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # Method 1: Using json parameter (what causes the issue)
        request1 = httpx.Request(
            method="POST",
            url="https://example.com",
            json=test_data
        )
        
        # Method 2: Using data parameter with pre-serialized JSON (our fix)
        json_str = json.dumps(test_data)
        request2 = httpx.Request(
            method="POST",
            url="https://example.com",
            content=json_str.encode(),
            headers={"Content-Type": "application/json"}
        )
        
        # Print the actual differences for verification
        print(f"Method 1 (json): Content-Length={request1.headers.get('content-length')}, Actual={len(request1.content)}")
        print(f"Method 2 (data): Content-Length={request2.headers.get('content-length')}, Actual={len(request2.content)}")
        print(f"Method 1 body: {request1.content}")
        print(f"Method 2 body: {request2.content}")
        
        # Assert that the Content-Length header matches the actual body length for our fix
        assert len(request2.content) == int(request2.headers.get("content-length", 0))
        
        # Demonstrate the potential mismatch with the json parameter
        # Note: This might not always fail depending on how httpx serializes JSON,
        # but it demonstrates the potential issue
        json_str_manual = json.dumps(test_data)
        assert len(json_str_manual.encode()) != len(request1.content), "JSON serialization should be different"
        
    @pytest.mark.parametrize("use_data", [True, False])
    def test_aws_sigv4_content_length_consistency(self, use_data):
        """
        Test that demonstrates how using data with pre-serialized JSON ensures
        Content-Length consistency for AWS SigV4 authentication.
        """
        # Test data
        test_data = {
            "prompt": "\n\nHuman: Tell me a short joke\n\nAssistant:",
            "max_tokens_to_sample": 50,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # Simulate SigV4 authentication process
        # 1. Pre-serialize JSON for signing
        json_str = json.dumps(test_data)
        content_length_for_signing = len(json_str.encode())
        
        # 2. Create the actual request
        if use_data:
            # Our fix: Use pre-serialized JSON with data parameter
            request = httpx.Request(
                method="POST",
                url="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke",
                content=json_str.encode(),
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(content_length_for_signing)
                }
            )
        else:
            # Original approach: Use json parameter (which causes the issue)
            request = httpx.Request(
                method="POST",
                url="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke",
                json=test_data,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(content_length_for_signing)
                }
            )
        
        # Check if Content-Length matches actual content length
        actual_content_length = len(request.content)
        expected_content_length = int(request.headers.get("content-length", 0))
        
        print(f"Use data: {use_data}")
        print(f"Expected Content-Length: {expected_content_length}")
        print(f"Actual content length: {actual_content_length}")
        print(f"Content: {request.content}")
        
        if use_data:
            # Our fix should ensure Content-Length matches
            assert actual_content_length == expected_content_length, "Content-Length mismatch with data parameter"
        else:
            # The original approach might cause a mismatch
            # Note: This might not always fail depending on how httpx serializes JSON
            if actual_content_length != expected_content_length:
                print("Content-Length mismatch detected with json parameter!")
                print(f"This demonstrates the issue fixed by our PR.")
