import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import completion_with_config

config = {
    "default_fallback_models": ["gpt-3.5-turbo", "claude-instant-1", "j2-ultra"],
    "model": {
        "claude-instant-1": {
            "needs_moderation": True
        },
        "gpt-3.5-turbo": {
            "error_handling": {
                "ContextWindowExceededError": {"fallback_model": "gpt-3.5-turbo-16k"} 
            }
        }
    }
}

def test_config_context_window_exceeded():
    try:
        sample_text = "how does a court case get to the Supreme Court?" * 1000
        messages = [{"content": sample_text, "role": "user"}]
        response = completion_with_config(model="gpt-3.5-turbo", messages=messages, config=config)
        print(response)
    except Exception as e:
        print(f"Exception: {e}")
        pytest.fail(f"An exception occurred: {e}")

# test_config_context_window_exceeded() 

def test_config_context_moderation():
    try:
        messages=[{"role": "user", "content": "I want to kill them."}]
        response = completion_with_config(model="claude-instant-1", messages=messages, config=config)
        print(response)
    except Exception as e:
        print(f"Exception: {e}")
        pytest.fail(f"An exception occurred: {e}")

# test_config_context_moderation() 

def test_config_context_default_fallback():
    try:
        messages=[{"role": "user", "content": "Hey, how's it going?"}]
        response = completion_with_config(model="claude-instant-1", messages=messages, config=config, api_key="bad-key")
        print(response)
    except Exception as e:
        print(f"Exception: {e}")
        pytest.fail(f"An exception occurred: {e}")

# test_config_context_default_fallback() 


config = {
    "default_fallback_models": ["gpt-3.5-turbo", "claude-instant-1", "j2-ultra"],
    "available_models": ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-0314", "gpt-4-0613", 
                         "j2-ultra", "command-nightly", "togethercomputer/llama-2-70b-chat", "chat-bison", "chat-bison@001", "claude-2"],
    "adapt_to_prompt_size": True, # type: ignore
    "model": {
        "claude-instant-1": {
            "needs_moderation": True
        },
        "gpt-3.5-turbo": {
            "error_handling": {
                "ContextWindowExceededError": {"fallback_model": "gpt-3.5-turbo-16k"} 
            }
        }
    }
}

def test_config_context_adapt_to_prompt():
    try:
        sample_text = "how does a court case get to the Supreme Court?" * 1000
        messages = [{"content": sample_text, "role": "user"}]
        response = completion_with_config(model="gpt-3.5-turbo", messages=messages, config=config)
        print(response)
    except Exception as e:
        print(f"Exception: {e}")
        pytest.fail(f"An exception occurred: {e}")

test_config_context_adapt_to_prompt() 