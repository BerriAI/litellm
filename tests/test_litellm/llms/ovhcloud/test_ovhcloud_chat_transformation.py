"""
Unit tests for OVHCloud AI Endpoints chat integration.
"""

import os
import sys

import pytest

from litellm.llms.ovhcloud.utils import OVHCloudException

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.ovhcloud.chat.transformation import (
    OVHCloudChatCompletionStreamingHandler,
    OVHCloudChatConfig,
)

config = OVHCloudChatConfig()
model = "ovhcloud/Mistral-7B-Instruct-v0.3"

class TestOvhCloudChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        chunk = {
            "id": "test_id",
            "created": 1234567890,
            "model": "gpt-oss-20b",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [
                {"delta": {"content": "test content", "reasoning": "test reasoning"}}
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result.id == "test_id"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "gpt-oss-20b"
        assert result.usage.prompt_tokens == chunk["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == chunk["usage"]["completion_tokens"]
        assert result.usage.total_tokens == chunk["usage"]["total_tokens"]
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["reasoning_content"] == "test reasoning"

    def test_chunk_parser_error_response(self):
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        error_chunk = {
            "error": {
                "message": "test error",
                "code": 400,
            } 
        }

        with pytest.raises(OVHCloudException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert "OVHCloud Error: test error" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_chunk_parser_key_error(self):
        handler = OVHCloudChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        invalid_chunk = {"incomplete": "data"}

        with pytest.raises(OVHCloudException) as exc_info:
            handler.chunk_parser(invalid_chunk)

        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


class TestOVHCloudConfig:
    def test_transform_request_basic(self):
        """Test basic request transformation"""        
        transformed_request = config.transform_request(
            model,
            messages=[
                {"role": "user", "content": "Hello, world!"}
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert transformed_request["model"] == model
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_transform_request_with_extra_body(self):
        """Test request transformation with extra_body parameters"""        
        transformed_request = config.transform_request(
            model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={"extra_body": {"custom_param": "custom_value"}},
            litellm_params={},
            headers={},
        )

        assert transformed_request["custom_param"] == "custom_value"
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping"""        
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }
        
        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )
        
        assert mapped_params["temperature"] == 0.7
        assert mapped_params["max_tokens"] == 100
        assert mapped_params["top_p"] == 0.9

    def test_get_error_class(self):
        """Test error class creation"""        
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"}
        )
        
        assert isinstance(error, OVHCloudException)
        assert error.message == "Test error"
        assert error.status_code == 400


def test_ovhcloud_integration():
    import os
    from litellm import completion
    
    api_key = os.getenv("OVHCLOUD_API_KEY") 
    
    if not api_key:
        pytest.skip("OVHCLOUD_API_KEY not set, skipping test")
    
    response = completion(
        model,
        messages=[{"role": "user", "content": "Say hello in one word"}],
        api_key=api_key,
        max_tokens=10,
        temperature=0.7
    )
    
    assert response.choices[0].message.content
    assert len(response.choices[0].message.content.strip()) > 0
    assert response.model
    assert response.usage
    assert response.usage.total_tokens > 0

def test_OVHCloud_streaming_integration():
    """
    Integration test for streaming - requires real API key
    Run with: pytest -k test_OVHCloud_streaming_integration -s
    """
    import os
    from litellm import completion
    
    api_key = os.getenv("OVHCLOUD_API_KEY") 
    
    if not api_key:
        pytest.skip("OVHCLOUD_API_KEY not set, skipping test")
    
    try:
        print(f"🔍 Testing streaming with API key: {api_key[:6]}...{api_key[-4:]} (length: {len(api_key)})")
        print(f"🔍 API base URL: {os.getenv('OVHCLOUD_API_BASE')}")
        
        response = completion(
            model,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            api_key=api_key,
            max_tokens=50,
            stream=True
        )

        chunks = []
        content_parts = []

        for chunk in response:
            chunks.append(chunk)
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0, "Should receive at least one chunk"
        assert len(content_parts) > 0, "Should receive content in chunks"

        full_content = "".join(content_parts)
        assert len(full_content.strip()) > 0, "Should have non-empty content"

        print(f"✅ Received {len(chunks)} chunks")
        print(f"✅ Full content: {full_content}")

    except Exception as e:
        print(f"❌ Streaming integration test error details:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        if hasattr(e, 'status_code'):
            print(f"   Status code: {e.status_code}")
        if hasattr(e, 'response'):
            print(f"   Response: {e.response}")
            
        pytest.fail(f"Streaming integration test failed: {type(e).__name__}: {str(e)}")

def test_ovhcloud_with_custom_base_url():
    """
    Test OVHCloud with custom base URL
    """
    import os
    from litellm import completion
    
    api_key = os.getenv("OVHCLOUD_API_KEY") 
    
    if not api_key:
        pytest.skip("OVHCLOUD_API_KEY not set, skipping test")

    custom_base_url = os.getenv("OVHCLOUD_API_BASE", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1")
        
    try:
        response = completion(
            model,
            messages=[{"role": "user", "content": "Hello"}],
            api_key=api_key,
            api_base=custom_base_url,
            max_tokens=5
        )
        
        assert response.choices[0].message.content
        print(f"✅ Custom base URL test passed: {response.choices[0].message.content}")
        
    except Exception as e:
        pytest.fail(f"Custom base URL test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])