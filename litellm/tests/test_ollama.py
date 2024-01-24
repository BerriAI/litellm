import sys, os
import traceback
from typing import List
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm.llms.ollama import OllamaUsage, ResponseJSON
from litellm.llms.ollama_chat import ChatResponseJSON, MessageJSON, OllamaChatUsage

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

## for ollama we can't test making the completion call
from litellm.utils import get_optional_params, get_llm_provider


def test_get_ollama_params():
    try:
        converted_params = get_optional_params(
            custom_llm_provider="ollama",
            model="llama2",
            max_tokens=20,
            temperature=0.5,
            stream=True,
        )
        print("Converted params", converted_params)
        assert converted_params == {
            "num_predict": 20,
            "stream": True,
            "temperature": 0.5,
        }, f"{converted_params} != {'num_predict': 20, 'stream': True, 'temperature': 0.5}"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_get_ollama_params()


def test_get_ollama_model():
    try:
        model, custom_llm_provider, _, _ = get_llm_provider("ollama/code-llama-22")
        print("Model", "custom_llm_provider", model, custom_llm_provider)
        assert custom_llm_provider == "ollama", f"{custom_llm_provider} != ollama"
        assert model == "code-llama-22", f"{model} != code-llama-22"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_get_ollama_model()

def test_ollama_json_mode():
    # assert that format: json gets passed as is to ollama
    try:
        converted_params = get_optional_params(custom_llm_provider="ollama", model="llama2", format = "json", temperature=0.5)
        print("Converted params", converted_params)
        assert converted_params == {'temperature': 0.5, 'format': 'json'}, f"{converted_params} != {'temperature': 0.5, 'format': 'json'}"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_ollama_json_mode()


def test_ollama_usage():
    # stubbed mid-streaming response
    response: ResponseJSON = {
        "response": "The",
        "done": False
    }
    prompt = "Why is the sky blue?"

    usage = OllamaUsage(prompt, response, encoding).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert 'prompt_tokens' not in usage.__dict__
    assert 'completion_tokens' not in usage.__dict__
    assert 'total_tokens' not in usage.__dict__

    # stubbed final stream response
    response: ResponseJSON = {
        "response": "",
        "done": True,
        "prompt_eval_count": 26,
        "eval_count": 290
    }

    usage = OllamaUsage(prompt, response, encoding).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 26
    assert usage.completion_tokens == 290
    assert usage.total_tokens == 316

    # stubbed final stream response, with missing metrics keys
    response: ResponseJSON = {
        "response": "",
        "done": True,
    }

    usage = OllamaUsage(prompt, response, encoding).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 6
    assert 'completion_tokens' not in usage.__dict__
    assert usage.total_tokens == 6

    # stubbed non-streamed response
    # see: https://github.com/jmorganca/ollama/blob/main/docs/api.md#response-1
    response: ResponseJSON = {
        "response": "The sky is blue because it is the color of the sky.",
        "done": True,
        "prompt_eval_count": 26,
        "eval_count": 290
    }

    usage = OllamaUsage(prompt, response, encoding).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 26
    assert usage.completion_tokens == 290
    assert usage.total_tokens == 316

    # stubbed non-streamed response, with missing metrics keys
    response: ResponseJSON = {
        "response": "The sky is blue because it is the color of the sky.",
        "done": True,
    }

    usage = OllamaUsage(prompt, response, encoding).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 6
    assert usage.completion_tokens == 13
    assert usage.total_tokens == 19
# test_ollama_usage()


def test_ollama_chat_usage():
    chat_request: Dict[str, List[MessageJSON]] = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI",
            },
            {
                "role": "user",
                "content": "Why is the sky blue?"
            },
            {
                "role": "assistant",
                "content": "The sky is blue because it is the color of the sky.",
            },
            {
                "role": "user",
                "content": "What about the ocean?"
            }
        ]
    }

    # stubbed mid-streaming response
    response: ChatResponseJSON = {
        "message": {
            "role": "assistant",
            "content": "The",
        },
        "done": False
    }

    usage = OllamaChatUsage(chat_request["messages"], response).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert 'prompt_tokens' not in usage.__dict__
    assert 'completion_tokens' not in usage.__dict__
    assert 'total_tokens' not in usage.__dict__

    # stubbed final stream response
    response: ChatResponseJSON = {
        "done": True,
        "prompt_eval_count": 26,
        "eval_count": 290
    }

    usage = OllamaChatUsage(chat_request["messages"], response).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 26
    assert usage.completion_tokens == 290
    assert usage.total_tokens == 316

    # stubbed final stream response, missing metrics keys
    response: ChatResponseJSON = {
        "done": True,
    }

    usage = OllamaChatUsage(chat_request["messages"], response).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 48
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 55


    # stubbed non-streamed response
    # see: https://github.com/jmorganca/ollama/blob/main/docs/api.md#response-8
    response: ChatResponseJSON = {
        "message": {
            "role": "assistant",
            "content": "The sky is blue because it is the color of the sky.",
        },
        "done": True,
        "prompt_eval_count": 26,
        "eval_count": 290,
    }

    usage = OllamaChatUsage(chat_request["messages"], response).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 26
    assert usage.completion_tokens == 290
    assert usage.total_tokens == 316

    # stubbed non-streamed response, missing metrics keys
    response: ChatResponseJSON = {
        "message": {
            "role": "assistant",
            "content": "The sky is blue because it is the color of the sky.",
        },
        "done": True,
    }

    usage = OllamaChatUsage(chat_request["messages"], response).get_usage()

    assert isinstance(usage, litellm.Usage)
    assert usage.prompt_tokens == 48
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 68
# test_ollama_chat_usage()
