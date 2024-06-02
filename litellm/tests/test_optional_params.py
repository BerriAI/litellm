#### What this tests ####
#    This tests if get_optional_params works as expected
import sys, os, time, inspect, asyncio, traceback
import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.utils import get_optional_params_embeddings, get_optional_params
from litellm.llms.prompt_templates.factory import (
    map_system_message_pt,
)
from litellm.types.completion import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageParam,
)

## get_optional_params_embeddings
### Models: OpenAI, Azure, Bedrock
### Scenarios: w/ optional params + litellm.drop_params = True


def test_supports_system_message():
    """
    Check if litellm.completion(...,supports_system_message=False)
    """
    messages = [
        ChatCompletionSystemMessageParam(role="system", content="Listen here!"),
        ChatCompletionUserMessageParam(role="user", content="Hello there!"),
    ]

    new_messages = map_system_message_pt(messages=messages)

    assert len(new_messages) == 1
    assert new_messages[0]["role"] == "user"

    ## confirm you can make a openai call with this param

    response = litellm.completion(
        model="gpt-3.5-turbo", messages=new_messages, supports_system_message=False
    )

    assert isinstance(response, litellm.ModelResponse)


@pytest.mark.parametrize(
    "stop_sequence, expected_count", [("\n", 0), (["\n"], 0), (["finish_reason"], 1)]
)
def test_anthropic_optional_params(stop_sequence, expected_count):
    """
    Test if whitespace character optional param is dropped by anthropic
    """
    litellm.drop_params = True
    optional_params = get_optional_params(
        model="claude-3", custom_llm_provider="anthropic", stop=stop_sequence
    )
    assert len(optional_params) == expected_count


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


def test_databricks_optional_params():
    litellm.drop_params = True
    optional_params = get_optional_params(
        model="",
        user="John",
        custom_llm_provider="databricks",
        max_tokens=10,
        temperature=0.2,
    )
    print(f"optional_params: {optional_params}")
    assert len(optional_params) == 2
    assert "user" not in optional_params


def test_azure_ai_mistral_optional_params():
    litellm.drop_params = True
    optional_params = get_optional_params(
        model="mistral-large-latest",
        user="John",
        custom_llm_provider="openai",
        max_tokens=10,
        temperature=0.2,
    )
    assert "user" not in optional_params


def test_azure_gpt_optional_params_gpt_vision():
    # for OpenAI, Azure all extra params need to get passed as extra_body to OpenAI python. We assert we actually set extra_body here
    optional_params = litellm.utils.get_optional_params(
        model="",
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
        model="",
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
        model="",
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


@pytest.mark.parametrize(
    "api_version",
    [
        "2024-02-01",
        "2024-07-01",  # potential future version with tool_choice="required" supported
        "2023-07-01-preview",
        "2024-03-01-preview",
    ],
)
def test_azure_tool_choice(api_version):
    """
    Test azure tool choice on older + new version
    """
    litellm.drop_params = True
    optional_params = litellm.utils.get_optional_params(
        model="chatgpt-v-2",
        user="John",
        custom_llm_provider="azure",
        max_tokens=10,
        temperature=0.2,
        extra_headers={"AI-Resource Group": "ishaan-resource"},
        tool_choice="required",
        api_version=api_version,
    )

    print(f"{optional_params}")
    if api_version == "2024-07-01":
        assert optional_params["tool_choice"] == "required"
    else:
        assert (
            "tool_choice" not in optional_params
        ), "tool_choice={} for api version={}".format(
            optional_params["tool_choice"], api_version
        )
