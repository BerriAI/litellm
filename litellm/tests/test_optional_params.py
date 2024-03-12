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


def test_azure_gpt_optional_params_gpt_vision():
    # for OpenAI, Azure all extra params need to get passed as extra_body to OpenAI python. We assert we actually set extra_body here
    optional_params = litellm.utils.get_optional_params(
        user="John",
        custom_llm_provider="azure",
        max_tokens=10,
        temperature=0.2,
        enhancements={"ocr": {"enabled": True}, "grounding": {"enabled": True}},
        dataSources=[
            {
                "type": "AzureComputerVision",
                "parameters": {
                    "endpoint": "<your_computer_vision_endpoint>",
                    "key": "<your_computer_vision_key>",
                },
            }
        ],
    )

    print(optional_params)
    assert optional_params["max_tokens"] == 10
    assert optional_params["temperature"] == 0.2
    assert optional_params["extra_body"] == {
        "enhancements": {"ocr": {"enabled": True}, "grounding": {"enabled": True}},
        "dataSources": [
            {
                "type": "AzureComputerVision",
                "parameters": {
                    "endpoint": "<your_computer_vision_endpoint>",
                    "key": "<your_computer_vision_key>",
                },
            }
        ],
    }


# test_azure_gpt_optional_params_gpt_vision()


def test_azure_gpt_optional_params_gpt_vision_with_extra_body():
    # if user passes extra_body, we should not over write it, we should pass it along to OpenAI python
    optional_params = litellm.utils.get_optional_params(
        user="John",
        custom_llm_provider="azure",
        max_tokens=10,
        temperature=0.2,
        extra_body={
            "meta": "hi",
        },
        enhancements={"ocr": {"enabled": True}, "grounding": {"enabled": True}},
        dataSources=[
            {
                "type": "AzureComputerVision",
                "parameters": {
                    "endpoint": "<your_computer_vision_endpoint>",
                    "key": "<your_computer_vision_key>",
                },
            }
        ],
    )

    print(optional_params)
    assert optional_params["max_tokens"] == 10
    assert optional_params["temperature"] == 0.2
    assert optional_params["extra_body"] == {
        "enhancements": {"ocr": {"enabled": True}, "grounding": {"enabled": True}},
        "dataSources": [
            {
                "type": "AzureComputerVision",
                "parameters": {
                    "endpoint": "<your_computer_vision_endpoint>",
                    "key": "<your_computer_vision_key>",
                },
            }
        ],
        "meta": "hi",
    }


# test_azure_gpt_optional_params_gpt_vision_with_extra_body()


def test_openai_extra_headers():
    optional_params = litellm.utils.get_optional_params(
        user="John",
        custom_llm_provider="openai",
        max_tokens=10,
        temperature=0.2,
        extra_headers={"AI-Resource Group": "ishaan-resource"},
    )

    print(optional_params)
    assert optional_params["max_tokens"] == 10
    assert optional_params["temperature"] == 0.2
    assert optional_params["extra_headers"] == {"AI-Resource Group": "ishaan-resource"}
