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
