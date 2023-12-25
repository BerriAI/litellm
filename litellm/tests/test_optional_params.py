#### What this tests ####
#    This tests if get_optional_params works as expected
import sys, os, time, inspect, asyncio, traceback
import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.utils import get_optional_params_embeddings

## get_optional_params_embeddings
### Models: OpenAI, Azure, Bedrock
### Scenarios: w/ optional params + litellm.drop_params = True


def test_bedrock_optional_params_embeddings():
    litellm.drop_params = True
    optional_params = get_optional_params_embeddings(
        user="John", encoding_format=None, custom_llm_provider="bedrock"
    )
    assert len(optional_params) == 0


def test_openai_optional_params_embeddings():
    litellm.drop_params = True
    optional_params = get_optional_params_embeddings(
        user="John", encoding_format=None, custom_llm_provider="openai"
    )
    assert len(optional_params) == 1
    assert optional_params["user"] == "John"


def test_azure_optional_params_embeddings():
    litellm.drop_params = True
    optional_params = get_optional_params_embeddings(
        user="John", encoding_format=None, custom_llm_provider="azure"
    )
    assert len(optional_params) == 1
    assert optional_params["user"] == "John"
