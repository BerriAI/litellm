import asyncio
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
from unittest import mock

## for ollama we can't test making the completion call
from litellm.utils import EmbeddingResponse, get_optional_params, get_llm_provider


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


mock_ollama_embedding_response = EmbeddingResponse(model="ollama/nomic-embed-text")

@mock.patch(
    "litellm.llms.ollama.ollama_embeddings",
    return_value=mock_ollama_embedding_response,
)
def test_ollama_embeddings(mock_embeddings):
    # assert that ollama_embeddings is called with the right parameters
    try:
        embeddings = litellm.embedding(model="ollama/nomic-embed-text", input=["hello world"])
        print(embeddings)
        mock_embeddings.assert_called_once_with(
            api_base="http://localhost:11434",
            model="nomic-embed-text",
            prompts=["hello world"],
            optional_params=mock.ANY,
            logging_obj=mock.ANY,
            model_response=mock.ANY,
            encoding=mock.ANY,
        )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_ollama_embeddings()

@mock.patch(
    "litellm.llms.ollama.ollama_aembeddings",
    return_value=mock_ollama_embedding_response,
)
def test_ollama_aembeddings(mock_aembeddings):
    # assert that ollama_aembeddings is called with the right parameters
    try:
        embeddings = asyncio.run(litellm.aembedding(model="ollama/nomic-embed-text", input=["hello world"]))
        print(embeddings)
        mock_aembeddings.assert_called_once_with(
            api_base="http://localhost:11434",
            model="nomic-embed-text",
            prompts=["hello world"],
            optional_params=mock.ANY,
            logging_obj=mock.ANY,
            model_response=mock.ANY,
            encoding=mock.ANY,
        )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_ollama_aembeddings()
