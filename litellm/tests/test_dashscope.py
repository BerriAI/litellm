import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion, completion_cost, Timeout, ModelResponse
from litellm import RateLimitError


from litellm.utils import get_optional_params, get_llm_provider


def test_get_dashscope_params():
    try:
        converted_params = get_optional_params(
            custom_llm_provider="dashscope",
            model="qwen-plus",
            max_tokens=20,
            temperature=0.5,
            stream=True,
        )
        print("Converted params", converted_params)
        assert converted_params == {
            "stream": True,
            "temperature": 0.5,
            "max_tokens": 20,
        }, f"{converted_params} != {'stream': True, 'temperature': 0.5, 'max_tokens': 20}"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


#test_get_dashscope_params()


def test_get_dashscope_model():
    try:
        model, custom_llm_provider, _, _ = get_llm_provider("dashscope/qwen-plus")
        print("Model", "custom_llm_provider", model, custom_llm_provider)
        assert custom_llm_provider == "dashscope", f"{custom_llm_provider} != dashscope"
        assert model == "qwen-plus", f"{model} != qwen-plus"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

#test_get_dashscope_model()
        
def test_dashscope_json_mode():
    # assert that format: json gets passed as is to ollama 
    try:
        converted_params = get_optional_params(custom_llm_provider="dashscope", model="qwen-plus", format = "json", temperature=0.5)
        print("Converted params", converted_params)
        assert converted_params == {'temperature': 0.5, 'stream': False, 'format': 'json'}, f"{converted_params} != {'temperature': 0.5, 'stream': False, 'format': 'json'}"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

#test_dashscope_json_mode()
