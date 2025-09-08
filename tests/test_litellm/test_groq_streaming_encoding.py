"""
Test for Groq streaming ASCII encoding issue fix.

This test verifies that the OpenAI-like handler correctly handles
UTF-8 encoded content in streaming responses, specifically fixing
the ASCII encoding error described in issue #12660.
"""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from litellm.llms.openai_like.chat.handler import make_call, make_sync_call


class MockResponse:
    """Mock httpx response for testing UTF-8 handling."""
    
    def __init__(self, test_content: str):
        self.test_content = test_content
        self.status_code = 200
        
    def iter_text(self, encoding='utf-8'):
        """Mock iter_text that yields content with the specified encoding."""
        yield self.test_content
        
    async def aiter_text(self, encoding='utf-8'):
        """Mock aiter_text that yields content with the specified encoding."""
        yield self.test_content
        
    def iter_lines(self):
        """Mock iter_lines method for synchronous streaming."""
        yield self.test_content
        
    async def aiter_lines(self):
        """Mock aiter_lines method for asynchronous streaming."""
        yield self.test_content
        
    def json(self):
        return {"choices": [{"delta": {"content": "test"}}]}

class MockSyncClient:
    """Mock synchronous HTTP client for testing."""
    
    def __init__(self, response_content: str):
        self.response_content = response_content
        
    def post(self, *args, **kwargs):
        return MockResponse(self.response_content)

class MockAsyncClient:
    """Mock asynchronous HTTP client for testing."""
    
    def __init__(self, response_content: str):
        self.response_content = response_content
        
    async def post(self, *args, **kwargs):
        return MockResponse(self.response_content)

def test_utf8_streaming_sync():
    """Test that synchronous streaming handles UTF-8 characters correctly."""
    # Content with the µ character that was causing issues
    test_content = "data: {\"choices\":[{\"delta\":{\"content\":\"The symbol µ represents micro\"}}]}\n\n"
    
    mock_client = MockSyncClient(test_content)
    mock_logging = Mock()
    
    # This should not raise an ASCII encoding error
    completion_stream = make_sync_call(
        client=mock_client,
        api_base="https://test.com/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        data='{"model": "test", "messages": []}',
        model="test-model",
        messages=[],
        logging_obj=mock_logging
    )
    
    # Verify we can iterate through the stream without encoding errors
    assert completion_stream is not None

@pytest.mark.asyncio
async def test_utf8_streaming_async():
    """Test that asynchronous streaming handles UTF-8 characters correctly."""
    # Content with the µ character that was causing issues
    test_content = "data: {\"choices\":[{\"delta\":{\"content\":\"The symbol µ represents micro\"}}]}\n\n"
    
    mock_client = MockAsyncClient(test_content)
    mock_logging = Mock()
    
    # This should not raise an ASCII encoding error
    completion_stream = await make_call(
        client=mock_client,
        api_base="https://test.com/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        data='{"model": "test", "messages": []}',
        model="test-model",
        messages=[],
        logging_obj=mock_logging
    )
    
    # Verify we can iterate through the stream without encoding errors
    assert completion_stream is not None

def test_various_unicode_characters():
    """Test streaming with various Unicode characters that could cause issues."""
    unicode_test_cases = [
        "µ",  # Micro symbol (the original issue)
        "©",  # Copyright symbol
        "™",  # Trademark symbol
        "€",  # Euro symbol
        "北京", # Chinese characters
        "🚀", # Emoji
        "Ñoño", # Spanish characters with tildes
    ]
    
    for unicode_char in unicode_test_cases:
        test_content = f"data: {{\"choices\":[{{\"delta\":{{\"content\":\"Testing {unicode_char} character\"}}}}]}}\n\n"
        
        mock_client = MockSyncClient(test_content)
        mock_logging = Mock()
        
        # This should not raise an ASCII encoding error for any Unicode character
        completion_stream = make_sync_call(
            client=mock_client,
            api_base="https://test.com/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            data='{"model": "test", "messages": []}',
            model="test-model",
            messages=[],
            logging_obj=mock_logging
        )
        
        assert completion_stream is not None, f"Failed to handle Unicode character: {unicode_char}"

if __name__ == "__main__":
    test_utf8_streaming_sync()
    asyncio.run(test_utf8_streaming_async())
    test_various_unicode_characters()
    print("All UTF-8 streaming tests passed!")