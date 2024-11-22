import os
import pytest
import litellm
import asyncio
from litellm import completion, acompletion

# Test fixtures
@pytest.fixture
def setup_keywordsai():
    """Setup common test parameters"""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hi 👋 - i'm openai"}],
        "tool_messages": [{"role": "user", "content": "Get the current weather in San Francisco, CA"}],
        "extra_body": {
            "keywordsai_params": {
                "customer_params": {
                    "customer_identifier": "test_litellm_logging",
                    "email": "test@test.com",
                    "name": "test user"
                },
                "thread_identifier": "test_litellm_thread",
                "metadata": {"key": "value"},
                "evaluation_identifier": "test_litellm_evaluation",
                "prompt_id": "test_litellm_prompt",
            }
        },
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "get_current_weather"}}
    }

@pytest.fixture(autouse=True)
def setup_litellm():
    """Setup LiteLLM configuration before each test"""
    litellm.api_base = None
    litellm.success_callback = ["keywordsai"]
    yield
    # Cleanup after tests if needed
    litellm.success_callback = []

def test_keywordsai_proxy():
    """Test KeywordsAI as a proxy"""
    litellm.api_base = "https://api.keywordsai.co/api/"
    api_key = os.getenv("KEYWORDSAI_API_KEY")
    
    response = litellm.completion(
        api_key=api_key,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hi, I am logging from litellm with KeywordsAI!"}]
    )
    
    assert response is not None

def test_basic_completion(setup_keywordsai):
    """Test basic completion without streaming or tools"""
    response = completion(
        model=setup_keywordsai["model"], 
        messages=setup_keywordsai["messages"], 
        metadata=setup_keywordsai["extra_body"]
    )
    assert response is not None

def test_streaming_completion(setup_keywordsai):
    """Test streaming completion"""
    response = completion(
        model=setup_keywordsai["model"], 
        messages=setup_keywordsai["messages"], 
        metadata=setup_keywordsai["extra_body"],
        stream=True
    )
    chunks = [chunk for chunk in response]
    assert len(chunks) > 0

def test_completion_with_tools(setup_keywordsai):
    """Test completion with tools"""
    response = completion(
        model=setup_keywordsai["model"],
        messages=setup_keywordsai["tool_messages"],
        tools=setup_keywordsai["tools"],
        tool_choice=setup_keywordsai["tool_choice"],
        metadata=setup_keywordsai["extra_body"]
    )
    assert response is not None

def test_streaming_completion_with_tools(setup_keywordsai):
    """Test streaming completion with tools"""
    response = completion(
        model=setup_keywordsai["model"],
        messages=setup_keywordsai["tool_messages"],
        tools=setup_keywordsai["tools"],
        tool_choice=setup_keywordsai["tool_choice"],
        metadata=setup_keywordsai["extra_body"],
        stream=True
    )
    chunks = [chunk for chunk in response]
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_async_completion(setup_keywordsai):
    """Test async completion"""
    response = await acompletion(
        model=setup_keywordsai["model"],
        messages=setup_keywordsai["messages"],
        metadata=setup_keywordsai["extra_body"]
    )
    assert response is not None

@pytest.mark.asyncio
async def test_async_streaming_completion(setup_keywordsai):
    """Test async streaming completion"""
    response = await acompletion(
        model=setup_keywordsai["model"],
        messages=setup_keywordsai["messages"],
        metadata=setup_keywordsai["extra_body"],
        stream=True
    )
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_async_completion_with_tools(setup_keywordsai):
    """Test async completion with tools"""
    response = await acompletion(
        model=setup_keywordsai["model"],
        messages=setup_keywordsai["tool_messages"],
        tools=setup_keywordsai["tools"],
        tool_choice=setup_keywordsai["tool_choice"],
        metadata=setup_keywordsai["extra_body"]
    )
    assert response is not None

