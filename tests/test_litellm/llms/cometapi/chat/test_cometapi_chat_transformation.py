"""
Unit tests for CometAPI Chat Configuration

Tests the CometAPIChatConfig class methods using mocks
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.cometapi.chat.transformation import (
    CometAPIChatCompletionStreamingHandler,
    CometAPIConfig,
)
from litellm.llms.cometapi.common_utils import CometAPIException


class TestCometAPIChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = CometAPIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test input chunk
        chunk = {
            "id": "test_id",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [
                {"delta": {"content": "test content", "reasoning": "test reasoning"}}
            ],
        }

        # Parse chunk
        result = handler.chunk_parser(chunk)

        # Verify response
        assert result.id == "test_id"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "gpt-3.5-turbo"
        assert result.usage.prompt_tokens == chunk["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == chunk["usage"]["completion_tokens"]
        assert result.usage.total_tokens == chunk["usage"]["total_tokens"]
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["reasoning_content"] == "test reasoning"

    def test_chunk_parser_error_response(self):
        handler = CometAPIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test error chunk
        error_chunk = {
            "error": {
                "message": "test error",
                "code": 400,
            }
        }

        # Verify error handling
        with pytest.raises(CometAPIException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert "CometAPI Error: test error" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_chunk_parser_key_error(self):
        handler = CometAPIChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )

        # Test invalid chunk missing required fields
        invalid_chunk = {"incomplete": "data"}

        # Verify KeyError handling
        with pytest.raises(CometAPIException) as exc_info:
            handler.chunk_parser(invalid_chunk)

        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


class TestCometAPIConfig:
    def test_transform_request_basic(self):
        """Test basic request transformation"""
        config = CometAPIConfig()
        
        transformed_request = config.transform_request(
            model="cometapi/gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, world!"}
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert transformed_request["model"] == "cometapi/gpt-3.5-turbo"
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_transform_request_with_extra_body(self):
        """Test request transformation with extra_body parameters"""
        config = CometAPIConfig()
        
        transformed_request = config.transform_request(
            model="cometapi/gpt-4",
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={"extra_body": {"custom_param": "custom_value"}},
            litellm_params={},
            headers={},
        )

        # Validate that extra_body parameters are merged into the request
        assert transformed_request["custom_param"] == "custom_value"
        assert transformed_request["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ]

    def test_cache_control_flag_removal(self):
        """Test cache control flag removal from messages"""
        config = CometAPIConfig()
        
        transformed_request = config.transform_request(
            model="cometapi/gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, world!",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )
        
        # CometAPI should remove cache_control flags by default
        assert transformed_request["messages"][0].get("cache_control") is None

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping"""
        config = CometAPIConfig()
        
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }
        
        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="cometapi/gpt-3.5-turbo",
            drop_params=False,
        )
        
        assert mapped_params["temperature"] == 0.7
        assert mapped_params["max_tokens"] == 100
        assert mapped_params["top_p"] == 0.9

    def test_get_error_class(self):
        """Test error class creation"""
        config = CometAPIConfig()
        
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"}
        )
        
        assert isinstance(error, CometAPIException)
        assert error.message == "Test error"
        assert error.status_code == 400


# Integration test example (requires real API key)
@pytest.mark.skip(reason="Skipping integration test")
def test_cometapi_integration():
    """
    Integration test - requires real API key
    Run with: pytest -k test_cometapi_integration -s
    """
    import os
    from litellm import completion
    
    # Try to get API key from multiple environment variables
    api_key = (
        os.getenv("COMETAPI_API_KEY") 
        or os.getenv("COMETAPI_KEY")
        or os.getenv("COMET_API_KEY")
    )
    
    if not api_key:
        pytest.skip("COMETAPI_API_KEY not set - skipping integration test")
    
    response = completion(
        model="cometapi/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        api_key=api_key,
        max_tokens=10,
        temperature=0.7
    )
    
    # Verify response structure
    assert response.choices[0].message.content
    assert len(response.choices[0].message.content.strip()) > 0
    assert response.model
    assert response.usage
    assert response.usage.total_tokens > 0


def test_cometapi_streaming_integration():
    """
    Integration test for streaming - requires real API key
    Run with: pytest -k test_cometapi_streaming_integration -s
    """
    import os
    from litellm import completion
    
    # Try to get API key from multiple environment variables
    api_key = (
        os.getenv("COMETAPI_API_KEY") 
        or os.getenv("COMETAPI_KEY")
        or os.getenv("COMET_API_KEY")
    )
    
    if not api_key:
        pytest.skip("COMETAPI_API_KEY not set - skipping streaming integration test")
    
    try:
        print(f"🔍 Testing streaming with API key: {api_key[:6]}...{api_key[-4:]} (length: {len(api_key)})")
        print(f"🔍 API base URL: {os.getenv('COMETAPI_API_BASE', 'default')}")
        
        # test streaming API call
        response = completion(
            model="cometapi/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            api_key=api_key,
            max_tokens=50,
            stream=True
        )

        # collect streaming response
        chunks = []
        content_parts = []

        for chunk in response:
            chunks.append(chunk)
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        # Verify we received at least one chunk and content
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
            
        # Re-raise with more context for pytest
        pytest.fail(f"Streaming integration test failed: {type(e).__name__}: {str(e)}")
def test_cometapi_with_custom_base_url():
    """
    Test CometAPI with custom base URL
    """
    import os
    from litellm import completion
    
    api_key = (
        os.getenv("COMETAPI_API_KEY") 
        or os.getenv("COMETAPI_KEY")
        or os.getenv("COMET_API_KEY")
    )
    
    custom_base_url = os.getenv("COMETAPI_API_BASE", "https://api.cometapi.com/v1")
    
    if not api_key:
        pytest.skip("COMETAPI_API_KEY not set - skipping custom base URL test")
    
    try:
        response = completion(
            model="cometapi/gpt-3.5-turbo",
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
    # Quick test runner
    pytest.main([__file__, "-v"])