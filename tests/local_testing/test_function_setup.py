# What is this?
## Unit tests for the 'function_setup()' function
import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, uuid
from litellm.utils import function_setup, Rules
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR
from datetime import datetime


def test_empty_content():
    """
    Make a chat completions request with empty content -> expect this to work
    """
    rules_obj = Rules()

    def completion():
        pass

    function_setup(
        original_function="completion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        messages=[],
        litellm_call_id=str(uuid.uuid4()),
    )


def test_thought_signature_removal_for_non_gemini():
    """
    Test that thought signatures are removed from tool call IDs when sending to non-Gemini models
    """
    rules_obj = Rules()
    
    # Create messages with thought signatures (as would come from Gemini)
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_123{THOUGHT_SIGNATURE_SEPARATOR}sig1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": f"call_123{THOUGHT_SIGNATURE_SEPARATOR}sig1",
            "content": "Sunny, 72°F"
        }
    ]
    
    # Call function_setup with OpenAI model (non-Gemini)
    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="gpt-4",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="openai"
    )
    
    # Verify thought signatures were removed
    processed_messages = kwargs["messages"]
    assert processed_messages[1]["tool_calls"][0]["id"] == "call_123"
    assert processed_messages[2]["tool_call_id"] == "call_123"
    assert THOUGHT_SIGNATURE_SEPARATOR not in processed_messages[1]["tool_calls"][0]["id"]
    assert THOUGHT_SIGNATURE_SEPARATOR not in processed_messages[2]["tool_call_id"]


def test_thought_signature_preserved_for_gemini():
    """
    Test that thought signatures are preserved when sending to Gemini models
    """
    rules_obj = Rules()
    
    # Create messages with thought signatures
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_456{THOUGHT_SIGNATURE_SEPARATOR}sig2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": f"call_456{THOUGHT_SIGNATURE_SEPARATOR}sig2",
            "content": "Rainy, 65°F"
        }
    ]
    
    # Call function_setup with Gemini model
    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="gemini-1.5-pro",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="vertex_ai"
    )
    
    # Verify thought signatures were preserved (messages should be unchanged)
    processed_messages = kwargs["messages"]
    assert THOUGHT_SIGNATURE_SEPARATOR in processed_messages[1]["tool_calls"][0]["id"]
    assert THOUGHT_SIGNATURE_SEPARATOR in processed_messages[2]["tool_call_id"]


def test_thought_signature_removal_with_multiple_tool_calls():
    """
    Test that thought signatures are removed from multiple tool calls
    """
    rules_obj = Rules()
    
    messages = [
        {"role": "user", "content": "Get weather and time"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_1{THOUGHT_SIGNATURE_SEPARATOR}sig1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                },
                {
                    "id": f"call_2{THOUGHT_SIGNATURE_SEPARATOR}sig2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": f"call_1{THOUGHT_SIGNATURE_SEPARATOR}sig1",
            "content": "Sunny"
        },
        {
            "role": "tool",
            "tool_call_id": f"call_2{THOUGHT_SIGNATURE_SEPARATOR}sig2",
            "content": "3:00 PM"
        }
    ]
    
    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="claude-3-opus",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="anthropic"
    )
    
    processed_messages = kwargs["messages"]
    
    # Check all tool call IDs are cleaned
    assert processed_messages[1]["tool_calls"][0]["id"] == "call_1"
    assert processed_messages[1]["tool_calls"][1]["id"] == "call_2"
    assert processed_messages[2]["tool_call_id"] == "call_1"
    assert processed_messages[3]["tool_call_id"] == "call_2"


def test_messages_without_tool_calls_unchanged():
    """
    Test that messages without tool calls pass through unchanged
    """
    rules_obj = Rules()
    
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    logging_obj, kwargs = function_setup(
        original_function="acompletion",
        rules_obj=rules_obj,
        start_time=datetime.now(),
        model="gpt-4",
        messages=messages,
        litellm_call_id=str(uuid.uuid4()),
        custom_llm_provider="openai"
    )
    
    # Messages should be unchanged
    assert kwargs["messages"] == messages
