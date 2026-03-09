# What is this?
## This tests the `get_optional_params_embeddings` function
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
from litellm import embedding
from litellm.utils import get_optional_params_embeddings, get_llm_provider


def test_vertex_projects():
    litellm.drop_params = True
    model, custom_llm_provider, _, _ = get_llm_provider(
        model="vertex_ai/textembedding-gecko"
    )
    optional_params = get_optional_params_embeddings(
        model=model,
        user="test-litellm-user-5",
        dimensions=None,
        encoding_format="base64",
        custom_llm_provider=custom_llm_provider,
        **{
            "vertex_ai_project": "my-test-project",
            "vertex_ai_location": "us-east-1",
        },
    )

    print(f"received optional_params: {optional_params}")

    assert "vertex_ai_project" in optional_params
    assert "vertex_ai_location" in optional_params


# test_vertex_projects()


def test_bedrock_embed_v2_regular():
    model, custom_llm_provider, _, _ = get_llm_provider(
        model="bedrock/amazon.titan-embed-text-v2:0"
    )
    optional_params = get_optional_params_embeddings(
        model=model,
        dimensions=512,
        custom_llm_provider=custom_llm_provider,
    )
    print(f"received optional_params: {optional_params}")
    assert optional_params == {"dimensions": 512}


def test_bedrock_embed_v2_with_drop_params():
    litellm.drop_params = True
    model, custom_llm_provider, _, _ = get_llm_provider(
        model="bedrock/amazon.titan-embed-text-v2:0"
    )
    optional_params = get_optional_params_embeddings(
        model=model,
        dimensions=512,
        user="test-litellm-user-5",
        encoding_format="base64",
        custom_llm_provider=custom_llm_provider,
    )
    print(f"received optional_params: {optional_params}")
    assert optional_params == {"dimensions": 512, "embeddingTypes": ["binary"]}


def test_openai_non_text_embedding_3_with_allowed_openai_params():
    """
    Test that `dimensions` is allowed for non-text-embedding-3 OpenAI models
    when `allowed_openai_params=["dimensions"]` is passed. Without this flag,
    an UnsupportedParamsError would be raised.
    """
    model, custom_llm_provider, _, _ = get_llm_provider(
        model="openai/nvidia/llama-3.2-nv-embedqa-1b-v2"
    )
    optional_params = get_optional_params_embeddings(
        model=model,
        dimensions=1024,
        custom_llm_provider=custom_llm_provider,
        allowed_openai_params=["dimensions"],
    )
    print(f"received optional_params: {optional_params}")
    assert optional_params.get("dimensions") == 1024


def test_openai_non_text_embedding_3_without_allowed_openai_params_raises():
    """
    Test that passing `dimensions` to a non-text-embedding-3 OpenAI model
    without `allowed_openai_params` still raises UnsupportedParamsError.
    """
    from litellm.exceptions import UnsupportedParamsError

    model, custom_llm_provider, _, _ = get_llm_provider(
        model="openai/nvidia/llama-3.2-nv-embedqa-1b-v2"
    )
    with pytest.raises(UnsupportedParamsError):
        get_optional_params_embeddings(
            model=model,
            dimensions=1024,
            custom_llm_provider=custom_llm_provider,
        )
