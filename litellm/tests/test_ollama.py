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


## for ollama we can't test making the completion call
from litellm.utils import get_optional_params, get_llm_provider

def test_get_ollama_params():
    converted_params = get_optional_params(custom_llm_provider="ollama", model="llama2", max_tokens=20, temperature=0.5, stream=True)
    print("Converted params", converted_params)
    assert converted_params == {'num_predict': 20, 'stream': True, 'temperature': 0.5}, f"{converted_params} != {'num_predict': 20, 'stream': True, 'temperature': 0.5}"
# test_get_ollama_params()


def test_get_ollama_model():
    model, custom_llm_provider, _, _ = get_llm_provider("ollama/code-llama-22")
    print("Model", "custom_llm_provider", model, custom_llm_provider)
    assert custom_llm_provider == "ollama", f"{custom_llm_provider} != ollama"
    assert model == "code-llama-22", f"{model} != code-llama-22"
# test_get_ollama_model()