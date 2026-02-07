#### What this tests ####
#    This tests if get_optional_params works as expected
import asyncio
import inspect
import os
import sys
import time
import traceback

import pytest

sys.path.insert(0, os.path.abspath("../.."))
from unittest.mock import MagicMock, patch

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import map_system_message_pt
from litellm.types.completion import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from litellm.utils import (
    get_optional_params,
    get_optional_params_embeddings,
    get_optional_params_image_gen,
    get_requester_metadata,
    validate_openai_optional_params,
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


def test_get_requester_metadata_returns_none_for_empty():
    metadata = {"requester_metadata": {}}
    assert get_requester_metadata(metadata) is None


@patch("litellm.main.openai_chat_completions.completion")
def test_requester_metadata_forwarded_to_openai(mock_completion):
    mock_completion.return_value = MagicMock()
    metadata = {
        "requester_metadata": {
            "custom_meta_key": "value",
            "hidden_params": "secret",
            "int_value": 123,
        }
    }

    original_api_key = litellm.api_key
    litellm.api_key = "sk-test"
    original_preview_flag = litellm.enable_preview_features
    litellm.enable_preview_features = True

    try:
        litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            metadata=metadata,
        )
    finally:
        litellm.api_key = original_api_key
        litellm.enable_preview_features = original_preview_flag

    sent_metadata = mock_completion.call_args.kwargs["optional_params"]["metadata"]
    assert sent_metadata == {"custom_meta_key": "value"}


def test_get_optional_params_with_allowed_openai_params():
    """
    Test if use can dynamically pass in allowed_openai_params to override default behavior
    """
    litellm.drop_params = True
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response_format = {"type": "json"}
    reasoning_effort = "low"
    optional_params = get_optional_params(
        model="cf/llama-3.1-70b-instruct",
        custom_llm_provider="cloudflare",
        allowed_openai_params=["tools", "reasoning_effort", "response_format"],
        tools=tools,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )
    print(f"optional_params: {optional_params}")
    assert optional_params["tools"] == tools
    assert optional_params["response_format"] == response_format
    assert optional_params["reasoning_effort"] == reasoning_effort


def test_bedrock_optional_params_embeddings():
    litellm.drop_params = True
    optional_params = get_optional_params_embeddings(
        model="", user="John", encoding_format=None, custom_llm_provider="bedrock"
    )
    assert len(optional_params) == 0


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        "us.meta.llama3-2-11b-instruct-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ],
)
def test_bedrock_optional_params_completions(model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "structure_output",
                "description": "Send structured output back to the user",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "sentiment": {"type": "string"},
                    },
                    "required": ["reasoning", "sentiment"],
                    "additionalProperties": False,
                },
                "additionalProperties": False,
            },
        }
    ]
    optional_params = get_optional_params(
        model=model,
        max_tokens=10,
        temperature=0.1,
        tools=tools,
        custom_llm_provider="bedrock",
    )
    print(f"optional_params: {optional_params}")
    assert len(optional_params) == 4
    assert optional_params == {
        "maxTokens": 10,
        "stream": False,
        "temperature": 0.1,
        "tools": tools,
    }


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.titan-large",
        "bedrock/meta.llama3-2-11b-instruct-v1:0",
        "bedrock/ai21.j2-ultra-v1",
        "bedrock/cohere.command-nightly",
        "bedrock/mistral.mistral-7b",
    ],
)
def test_bedrock_optional_params_simple(model):
    litellm.drop_params = True
    get_optional_params(
        model=model,
        max_tokens=10,
        temperature=0.1,
        custom_llm_provider="bedrock",
    )


@pytest.mark.parametrize(
    "model, expected_dimensions, dimensions_kwarg",
    [
        ("bedrock/amazon.titan-embed-text-v1", False, None),
        ("bedrock/amazon.titan-embed-image-v1", True, "embeddingConfig"),
        ("bedrock/amazon.titan-embed-text-v2:0", True, "dimensions"),
        ("bedrock/cohere.embed-multilingual-v3", True, None),
    ],
)
def test_bedrock_optional_params_embeddings_dimension(
    model, expected_dimensions, dimensions_kwarg
):
    litellm.drop_params = True
    optional_params = get_optional_params_embeddings(
        model=model,
        user="John",
        encoding_format=None,
        dimensions=20,
        custom_llm_provider="bedrock",
    )
    if expected_dimensions:
        assert len(optional_params) == 1
    else:
        assert len(optional_params) == 0

    if dimensions_kwarg is not None:
        assert dimensions_kwarg in optional_params


def test_google_ai_studio_optional_params_embeddings():
    optional_params = get_optional_params_embeddings(
        model="",
        user="John",
        encoding_format=None,
        custom_llm_provider="gemini",
        drop_params=True,
    )
    assert len(optional_params) == 0


def test_openai_optional_params_embeddings():
    litellm.drop_params = True
    optional_params = get_optional_params_embeddings(
        model="", user="John", encoding_format=None, custom_llm_provider="openai"
    )
    assert len(optional_params) == 1
    assert optional_params["user"] == "John"


def test_azure_optional_params_embeddings():
    litellm.drop_params = True
    optional_params = get_optional_params_embeddings(
        model="chatgpt-v-3",
        user="John",
        encoding_format=None,
        custom_llm_provider="azure",
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
        stream=True,
    )
    print(f"optional_params: {optional_params}")
    assert len(optional_params) == 3
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


def test_vertex_ai_llama_3_optional_params():
    litellm.vertex_llama3_models = ["meta/llama3-405b-instruct-maas"]
    litellm.drop_params = True
    optional_params = get_optional_params(
        model="meta/llama3-405b-instruct-maas",
        user="John",
        custom_llm_provider="vertex_ai",
        max_tokens=10,
        temperature=0.2,
    )
    assert "user" not in optional_params


def test_vertex_ai_mistral_optional_params():
    litellm.vertex_mistral_models = ["mistral-large@2407"]
    litellm.drop_params = True
    optional_params = get_optional_params(
        model="mistral-large@2407",
        user="John",
        custom_llm_provider="vertex_ai",
        max_tokens=10,
        temperature=0.2,
    )
    assert "user" not in optional_params
    assert "max_tokens" in optional_params
    assert "temperature" in optional_params


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
        model="chatgpt-v-3",
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
        ), "tool choice should not be present. Got - tool_choice={} for api version={}".format(
            optional_params["tool_choice"], api_version
        )


@pytest.mark.parametrize("drop_params", [True, False, None])
def test_dynamic_drop_params(drop_params):
    """
    Make a call to cohere w/ drop params = True vs. false.
    """
    if drop_params is True:
        optional_params = litellm.utils.get_optional_params(
            model="command-r",
            custom_llm_provider="cohere",
            response_format={"type": "json"},
            drop_params=drop_params,
        )
    else:
        try:
            optional_params = litellm.utils.get_optional_params(
                model="command-r",
                custom_llm_provider="cohere",
                response_format={"type": "json"},
                drop_params=drop_params,
            )
            pytest.fail("Expected to fail")
        except Exception as e:
            pass


def test_dynamic_drop_params_e2e():
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()
    ) as mock_response:
        try:
            response = litellm.completion(
                model="command-r",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                response_format={"key": "value"},
                drop_params=True,
            )
        except Exception as e:
            pass

        mock_response.assert_called_once()
        print(mock_response.call_args.kwargs["data"])
        assert "response_format" not in mock_response.call_args.kwargs["data"]


def test_dynamic_pass_additional_params():
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()
    ) as mock_response:
        try:
            response = litellm.completion(
                model="command-r",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                custom_param="test",
                api_key="my-custom-key",
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            pass

        mock_response.assert_called_once()
        print(mock_response.call_args.kwargs["data"])
        assert "custom_param" in mock_response.call_args.kwargs["data"]
        assert "api_key" not in mock_response.call_args.kwargs["data"]


@pytest.mark.parametrize(
    "model, provider, should_drop",
    [("command-r", "cohere", True), ("gpt-3.5-turbo", "openai", False)],
)
def test_drop_params_parallel_tool_calls(model, provider, should_drop):
    """
    https://github.com/BerriAI/litellm/issues/4584
    """
    response = litellm.utils.get_optional_params(
        model=model,
        custom_llm_provider=provider,
        response_format={"type": "json"},
        parallel_tool_calls=True,
        drop_params=True,
    )

    print(response)

    if should_drop:
        assert "response_format" not in response
        assert "parallel_tool_calls" not in response
    else:
        assert "response_format" in response
        assert "parallel_tool_calls" in response


def test_dynamic_drop_params_parallel_tool_calls():
    """
    https://github.com/BerriAI/litellm/issues/4584
    """
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()
    ) as mock_response:
        try:
            response = litellm.completion(
                model="command-r",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                parallel_tool_calls=True,
                drop_params=True,
            )
        except Exception as e:
            pass

        mock_response.assert_called_once()
        print(mock_response.call_args.kwargs["data"])
        assert "parallel_tool_calls" not in mock_response.call_args.kwargs["data"]


@pytest.mark.parametrize("drop_params", [True, False, None])
def test_dynamic_drop_additional_params(drop_params):
    """
    Make a call to cohere, dropping 'response_format' specifically
    """
    if drop_params is True:
        optional_params = litellm.utils.get_optional_params(
            model="command-r",
            custom_llm_provider="cohere",
            response_format={"type": "json"},
            additional_drop_params=["response_format"],
        )
    else:
        try:
            optional_params = litellm.utils.get_optional_params(
                model="command-r",
                custom_llm_provider="cohere",
                response_format={"type": "json"},
            )
            pytest.fail("Expected to fail")
        except Exception as e:
            pass


def test_dynamic_drop_additional_params_stream_options():
    """
    Make a call to vertex ai, dropping 'stream_options' specifically
    """
    optional_params = litellm.utils.get_optional_params(
        model="mistral-large-2411@001",
        custom_llm_provider="vertex_ai",
        stream_options={"include_usage": True},
        additional_drop_params=["stream_options"],
    )

    assert "stream_options" not in optional_params


def test_dynamic_drop_additional_params_e2e():
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()
    ) as mock_response:
        try:
            response = litellm.completion(
                model="command-r",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                response_format={"key": "value"},
                additional_drop_params=["response_format"],
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            pass

        mock_response.assert_called_once()
        print(mock_response.call_args.kwargs["data"])
        assert "response_format" not in mock_response.call_args.kwargs["data"]
        assert "additional_drop_params" not in mock_response.call_args.kwargs["data"]


def test_get_optional_params_image_gen():
    response = litellm.utils.get_optional_params_image_gen(
        aws_region_name="us-east-1", custom_llm_provider="openai"
    )

    print(response)

    assert "aws_region_name" not in response
    response = litellm.utils.get_optional_params_image_gen(
        aws_region_name="us-east-1", custom_llm_provider="bedrock"
    )

    print(response)

    assert "aws_region_name" in response


def test_bedrock_optional_params_embeddings_provider_specific_params():
    optional_params = get_optional_params_embeddings(
        model="my-custom-model",
        custom_llm_provider="huggingface",
        wait_for_model=True,
    )
    assert len(optional_params) == 1


def test_get_optional_params_num_retries():
    """
    Relevant issue - https://github.com/BerriAI/litellm/issues/5124
    """
    with patch(
        "litellm.main.get_optional_params",
        new=MagicMock(return_value={"max_retries": 0}),
    ) as mock_client:
        _ = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            num_retries=10,
        )

        mock_client.assert_called()

        print(f"mock_client.call_args: {mock_client.call_args}")
        assert mock_client.call_args.kwargs["max_retries"] == 10


@pytest.mark.parametrize(
    "provider",
    [
        "vertex_ai",
        "vertex_ai_beta",
    ],
)
def test_vertex_safety_settings(provider):
    litellm.vertex_ai_safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    optional_params = get_optional_params(
        model="gemini-1.5-pro", custom_llm_provider=provider
    )
    assert len(optional_params) == 1


@pytest.mark.parametrize(
    "model, provider, expectedAddProp",
    [("gemini-1.5-pro", "vertex_ai_beta", False), ("gpt-3.5-turbo", "openai", True)],
)
def test_parse_additional_properties_json_schema(model, provider, expectedAddProp):
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=provider,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_reasoning",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "explanation": {"type": "string"},
                                    "output": {"type": "string"},
                                },
                                "required": ["explanation", "output"],
                                "additionalProperties": False,
                            },
                        },
                        "final_answer": {"type": "string"},
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )

    print(optional_params)

    if provider == "vertex_ai_beta":
        schema = optional_params["response_schema"]
    elif provider == "openai":
        schema = optional_params["response_format"]["json_schema"]["schema"]
    assert ("additionalProperties" in schema) == expectedAddProp


def test_o1_model_params():
    optional_params = get_optional_params(
        model="o1-preview-2024-09-12",
        custom_llm_provider="openai",
        seed=10,
        user="John",
    )
    assert optional_params["seed"] == 10
    assert optional_params["user"] == "John"


def test_azure_o1_model_params():
    optional_params = get_optional_params(
        model="o1-preview",
        custom_llm_provider="azure",
        seed=10,
        user="John",
    )
    assert optional_params["seed"] == 10
    assert optional_params["user"] == "John"


@pytest.mark.parametrize(
    "temperature, expected_error",
    [(0.2, True), (1, False), (0, True)],
)
@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_o1_model_temperature_params(provider, temperature, expected_error):
    if expected_error:
        with pytest.raises(litellm.UnsupportedParamsError):
            get_optional_params(
                model="o1-preview",
                custom_llm_provider=provider,
                temperature=temperature,
            )
    else:
        get_optional_params(
            model="o1-preview-2024-09-12",
            custom_llm_provider="openai",
            temperature=temperature,
        )


def test_unmapped_gemini_model_params():
    """
    Test if unmapped gemini model optional params are translated correctly
    """
    optional_params = get_optional_params(
        model="gemini-new-model",
        custom_llm_provider="vertex_ai",
        stop="stop_word",
    )
    assert optional_params["stop_sequences"] == ["stop_word"]


def _check_additional_properties(schema):
    if isinstance(schema, dict):
        # Remove the 'additionalProperties' key if it exists and is set to False
        if "additionalProperties" in schema or "strict" in schema:
            raise ValueError(
                "additionalProperties and strict should not be in the schema"
            )

        # Recursively process all dictionary values
        for key, value in schema.items():
            _check_additional_properties(value)

    elif isinstance(schema, list):
        # Recursively process all items in the list
        for item in schema:
            _check_additional_properties(item)

    return schema


@pytest.mark.parametrize(
    "provider, model",
    [
        ("hosted_vllm", "my-vllm-model"),
        ("gemini", "gemini-1.5-pro"),
        ("vertex_ai", "gemini-1.5-pro"),
    ],
)
def test_drop_nested_params_add_prop_and_strict(provider, model):
    """
    Relevant issue - https://github.com/BerriAI/litellm/issues/5288

    Relevant issue - https://github.com/BerriAI/litellm/issues/6136
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "structure_output",
                "description": "Send structured output back to the user",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "sentiment": {"type": "string"},
                    },
                    "required": ["reasoning", "sentiment"],
                    "additionalProperties": False,
                },
                "additionalProperties": False,
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "structure_output"}}
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=provider,
        temperature=0.2,
        tools=tools,
        tool_choice=tool_choice,
        additional_drop_params=[
            ["tools", "function", "strict"],
            ["tools", "function", "additionalProperties"],
        ],
    )

    _check_additional_properties(optional_params["tools"])


def test_hosted_vllm_tool_param():
    """
    Relevant issue - https://github.com/BerriAI/litellm/issues/6228
    """
    optional_params = get_optional_params(
        model="my-vllm-model",
        custom_llm_provider="hosted_vllm",
        temperature=0.2,
        tools=None,
        tool_choice=None,
    )
    assert "tools" not in optional_params
    assert "tool_choice" not in optional_params


def test_unmapped_vertex_anthropic_model():
    optional_params = get_optional_params(
        model="claude-3-5-sonnet-v250@20241022",
        custom_llm_provider="vertex_ai",
        max_retries=10,
    )
    assert "max_retries" not in optional_params


@pytest.mark.parametrize("provider", ["anthropic", "vertex_ai"])
def test_anthropic_parallel_tool_calls(provider):
    optional_params = get_optional_params(
        model="claude-3-5-sonnet-v250@20241022",
        custom_llm_provider=provider,
        parallel_tool_calls=True,
    )
    print(f"optional_params: {optional_params}")
    assert optional_params["tool_choice"]["disable_parallel_tool_use"] is False


def test_anthropic_computer_tool_use():
    tools = [
        {
            "type": "computer_20241022",
            "function": {
                "name": "computer",
                "parameters": {
                    "display_height_px": 100,
                    "display_width_px": 100,
                    "display_number": 1,
                },
            },
        }
    ]

    optional_params = get_optional_params(
        model="claude-3-5-sonnet-v250@20241022",
        custom_llm_provider="anthropic",
        tools=tools,
    )
    assert optional_params["tools"][0]["type"] == "computer_20241022"
    assert optional_params["tools"][0]["display_height_px"] == 100
    assert optional_params["tools"][0]["display_width_px"] == 100
    assert optional_params["tools"][0]["display_number"] == 1


def test_vertex_schema_field():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "json",
                "description": "Respond with a JSON object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thinking": {
                            "type": "string",
                            "description": "Your internal thoughts on different problem details given the guidance.",
                        },
                        "problems": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "icon": {
                                        "type": "string",
                                        "enum": [
                                            "BarChart2",
                                            "Bell",
                                        ],
                                        "description": "The name of a Lucide icon to display",
                                    },
                                    "color": {
                                        "type": "string",
                                        "description": "A Tailwind color class for the icon, e.g., 'text-red-500'",
                                    },
                                    "problem": {
                                        "type": "string",
                                        "description": "The title of the problem being addressed, approximately 3-5 words.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "A brief explanation of the problem, approximately 20 words.",
                                    },
                                    "impacts": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "A list of potential impacts or consequences of the problem, approximately 3 words each.",
                                    },
                                    "automations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "A list of potential automations to address the problem, approximately 3-5 words each.",
                                    },
                                },
                                "required": [
                                    "icon",
                                    "color",
                                    "problem",
                                    "description",
                                    "impacts",
                                    "automations",
                                ],
                                "additionalProperties": False,
                            },
                            "description": "Please generate problem cards that match this guidance.",
                        },
                    },
                    "required": ["thinking", "problems"],
                    "additionalProperties": False,
                    "$schema": "http://json-schema.org/draft-07/schema#",
                },
            },
        }
    ]

    optional_params = get_optional_params(
        model="gemini-1.5-flash",
        custom_llm_provider="vertex_ai",
        tools=tools,
    )
    print(optional_params)
    print(optional_params["tools"][0]["function_declarations"][0])
    assert (
        "$schema"
        not in optional_params["tools"][0]["function_declarations"][0]["parameters"]
    )


def test_watsonx_tool_choice():
    optional_params = get_optional_params(
        model="gemini-1.5-pro", custom_llm_provider="watsonx", tool_choice="auto"
    )
    print(optional_params)
    assert optional_params["tool_choice_option"] == "auto"


def test_watsonx_text_top_k():
    optional_params = get_optional_params(
        model="gemini-1.5-pro", custom_llm_provider="watsonx_text", top_k=10
    )
    print(optional_params)
    assert optional_params["top_k"] == 10


def test_together_ai_model_params():
    optional_params = get_optional_params(
        model="together_ai", custom_llm_provider="together_ai", logprobs=1
    )
    print(optional_params)
    assert optional_params["logprobs"] == 1


def test_forward_user_param():
    from litellm.utils import get_supported_openai_params, get_optional_params

    model = "claude-3-5-sonnet-20240620"
    optional_params = get_optional_params(
        model=model,
        user="test_user",
        custom_llm_provider="anthropic",
    )

    assert optional_params["metadata"]["user_id"] == "test_user"


def test_lm_studio_embedding_params():
    optional_params = get_optional_params_embeddings(
        model="lm_studio/gemma2-9b-it",
        custom_llm_provider="lm_studio",
        dimensions=1024,
        drop_params=True,
    )
    assert len(optional_params) == 0


def test_ollama_pydantic_obj():
    from pydantic import BaseModel

    class ResponseFormat(BaseModel):
        x: str
        y: str

    get_optional_params(
        model="qwen2:0.5b",
        custom_llm_provider="ollama",
        response_format=ResponseFormat,
    )


def test_gemini_frequency_penalty():
    from litellm.utils import get_supported_openai_params

    optional_params = get_supported_openai_params(
        model="gemini-1.5-flash",
        custom_llm_provider="vertex_ai",
        request_type="chat_completion",
    )
    assert optional_params is not None
    assert "frequency_penalty" in optional_params


def test_litellm_proxy_claude_3_5_sonnet():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    tool_choice = "auto"

    optional_params = get_optional_params(
        model="claude-3-5-sonnet",
        custom_llm_provider="litellm_proxy",
        tools=tools,
        tool_choice=tool_choice,
    )
    assert optional_params["tools"] == tools
    assert optional_params["tool_choice"] == tool_choice


def test_is_vertex_anthropic_model():
    assert (
        litellm.VertexAIAnthropicConfig().is_supported_model(
            model="claude-3-5-sonnet", custom_llm_provider="litellm_proxy"
        )
        is False
    )


def test_groq_response_format_json_schema():
    optional_params = get_optional_params(
        model="llama-3.1-70b-versatile",
        custom_llm_provider="groq",
        response_format={"type": "json_object"},
    )
    assert optional_params is not None
    assert "response_format" in optional_params
    assert optional_params["response_format"]["type"] == "json_object"


def test_gemini_frequency_penalty():
    optional_params = get_optional_params(
        model="gemini-1.5-flash", custom_llm_provider="gemini", frequency_penalty=0.5
    )
    assert optional_params["frequency_penalty"] == 0.5


def test_azure_prediction_param():
    optional_params = get_optional_params(
        model="chatgpt-v2",
        custom_llm_provider="azure",
        prediction={
            "type": "content",
            "content": "LiteLLM is a very useful way to connect to a variety of LLMs.",
        },
    )
    assert optional_params["prediction"] == {
        "type": "content",
        "content": "LiteLLM is a very useful way to connect to a variety of LLMs.",
    }


def test_vertex_ai_ft_llama():
    optional_params = get_optional_params(
        model="1984786713414729728",
        custom_llm_provider="vertex_ai",
        frequency_penalty=0.5,
        max_retries=10,
    )
    assert optional_params["frequency_penalty"] == 0.5
    assert "max_retries" not in optional_params


@pytest.mark.parametrize(
    "model, expected_thinking",
    [
        ("claude-3-5-sonnet", False),
        ("claude-3-7-sonnet", True),
        ("gpt-3.5-turbo", False),
    ],
)
def test_anthropic_thinking_param(model, expected_thinking):
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider="anthropic",
        thinking={"type": "enabled", "budget_tokens": 1024},
        drop_params=True,
    )
    if expected_thinking:
        assert "thinking" in optional_params
    else:
        assert "thinking" not in optional_params


def test_bedrock_invoke_anthropic_max_tokens():
    passed_params = {
        "model": "invoke/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "functions": None,
        "function_call": None,
        "temperature": 0.8,
        "top_p": None,
        "n": 1,
        "stream": False,
        "stream_options": None,
        "stop": None,
        "max_tokens": None,
        "max_completion_tokens": 1024,
        "modalities": None,
        "prediction": None,
        "audio": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "logit_bias": None,
        "user": None,
        "custom_llm_provider": "bedrock",
        "response_format": {"type": "text"},
        "seed": None,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "generate_plan",
                    "description": "Generate a plan to execute the task using only the tools outlined in your context.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "The type of step to execute",
                                        },
                                        "tool_name": {
                                            "type": "string",
                                            "description": "The name of the tool to use for this step",
                                        },
                                        "tool_input": {
                                            "type": "object",
                                            "description": "The input to pass to the tool. Make sure this complies with the schema for the tool.",
                                        },
                                        "tool_output": {
                                            "type": "object",
                                            "description": "(Optional) The output from the tool if needed for future steps. Make sure this complies with the schema for the tool.",
                                        },
                                    },
                                    "required": ["type"],
                                },
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_wire_tool",
                    "description": "Create a wire transfer with complete wire instructions",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "company_id": {
                                "type": "integer",
                                "description": "The ID of the company receiving the investment",
                            },
                            "investment_id": {
                                "type": "integer",
                                "description": "The ID of the investment memo",
                            },
                            "dollar_amount": {
                                "type": "number",
                                "description": "The amount to wire in USD",
                            },
                            "wiring_instructions": {
                                "type": "object",
                                "description": "Complete bank account and routing information for the wire",
                                "properties": {
                                    "account_name": {
                                        "type": "string",
                                        "description": "Name on the bank account",
                                    },
                                    "address_1": {
                                        "type": "string",
                                        "description": "Primary address line",
                                    },
                                    "address_2": {
                                        "type": "string",
                                        "description": "Secondary address line (optional)",
                                    },
                                    "city": {"type": "string"},
                                    "state": {"type": "string"},
                                    "zip": {"type": "string"},
                                    "country": {"type": "string", "default": "US"},
                                    "bank_name": {"type": "string"},
                                    "account_number": {"type": "string"},
                                    "routing_number": {"type": "string"},
                                    "account_type": {
                                        "type": "string",
                                        "enum": ["checking", "savings"],
                                        "default": "checking",
                                    },
                                    "swift_code": {
                                        "type": "string",
                                        "description": "Required for international wires",
                                    },
                                    "iban": {
                                        "type": "string",
                                        "description": "Required for some international wires",
                                    },
                                    "bank_city": {"type": "string"},
                                    "bank_state": {"type": "string"},
                                    "bank_country": {"type": "string", "default": "US"},
                                    "bank_to_bank_instructions": {
                                        "type": "string",
                                        "description": "Additional instructions for the bank (optional)",
                                    },
                                    "intermediary_bank_name": {
                                        "type": "string",
                                        "description": "Name of intermediary bank if required (optional)",
                                    },
                                },
                                "required": [
                                    "account_name",
                                    "address_1",
                                    "country",
                                    "bank_name",
                                    "account_number",
                                    "routing_number",
                                    "account_type",
                                    "bank_country",
                                ],
                            },
                        },
                        "required": [
                            "company_id",
                            "investment_id",
                            "dollar_amount",
                            "wiring_instructions",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_companies",
                    "description": "Search for companies by name or other criteria to get their IDs",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Name or part of name to search for",
                            },
                            "batch": {
                                "type": "string",
                                "description": 'Optional batch filter (e.g., "W21", "S22")',
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "live",
                                    "dead",
                                    "adrift",
                                    "exited",
                                    "went_public",
                                    "all",
                                ],
                                "description": "Filter by company status",
                                "default": "live",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": "Success or error status",
                            },
                            "results": {
                                "type": "array",
                                "description": "List of companies matching the search criteria",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "integer",
                                            "description": "Company ID to use in other API calls",
                                        },
                                        "name": {"type": "string"},
                                        "batch": {"type": "string"},
                                        "status": {"type": "string"},
                                        "valuation": {"type": "string"},
                                        "url": {"type": "string"},
                                        "description": {"type": "string"},
                                        "founders": {"type": "string"},
                                    },
                                },
                            },
                            "results_count": {
                                "type": "integer",
                                "description": "Number of companies returned",
                            },
                            "total_matches": {
                                "type": "integer",
                                "description": "Total number of matches found",
                            },
                        },
                    },
                },
            },
        ],
        "tool_choice": None,
        "max_retries": 0,
        "logprobs": None,
        "top_logprobs": None,
        "extra_headers": None,
        "api_version": None,
        "parallel_tool_calls": None,
        "drop_params": True,
        "reasoning_effort": None,
        "additional_drop_params": None,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that helps prepare a wire for a pro rata investment.",
            },
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
        "thinking": None,
        "kwargs": {},
    }
    optional_params = get_optional_params(**passed_params)
    print(f"optional_params: {optional_params}")

    assert "max_tokens_to_sample" not in optional_params
    assert optional_params["max_tokens"] == 1024


def test_bedrock_invoke_claude_4_anthropic_max_tokens():
    passed_params = {
        "model": "invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "functions": None,
        "function_call": None,
        "temperature": 0.8,
        "top_p": None,
        "n": 1,
        "stream": False,
        "stream_options": None,
        "stop": None,
        "max_tokens": None,
        "max_completion_tokens": 1024,
        "modalities": None,
        "prediction": None,
        "audio": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "logit_bias": None,
        "user": None,
        "custom_llm_provider": "bedrock",
        "response_format": {"type": "text"},
        "seed": None,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "generate_plan",
                    "description": "Generate a plan to execute the task using only the tools outlined in your context.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "The type of step to execute",
                                        },
                                        "tool_name": {
                                            "type": "string",
                                            "description": "The name of the tool to use for this step",
                                        },
                                        "tool_input": {
                                            "type": "object",
                                            "description": "The input to pass to the tool. Make sure this complies with the schema for the tool.",
                                        },
                                        "tool_output": {
                                            "type": "object",
                                            "description": "(Optional) The output from the tool if needed for future steps. Make sure this complies with the schema for the tool.",
                                        },
                                    },
                                    "required": ["type"],
                                },
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_wire_tool",
                    "description": "Create a wire transfer with complete wire instructions",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "company_id": {
                                "type": "integer",
                                "description": "The ID of the company receiving the investment",
                            },
                            "investment_id": {
                                "type": "integer",
                                "description": "The ID of the investment memo",
                            },
                            "dollar_amount": {
                                "type": "number",
                                "description": "The amount to wire in USD",
                            },
                            "wiring_instructions": {
                                "type": "object",
                                "description": "Complete bank account and routing information for the wire",
                                "properties": {
                                    "account_name": {
                                        "type": "string",
                                        "description": "Name on the bank account",
                                    },
                                    "address_1": {
                                        "type": "string",
                                        "description": "Primary address line",
                                    },
                                    "address_2": {
                                        "type": "string",
                                        "description": "Secondary address line (optional)",
                                    },
                                    "city": {"type": "string"},
                                    "state": {"type": "string"},
                                    "zip": {"type": "string"},
                                    "country": {"type": "string", "default": "US"},
                                    "bank_name": {"type": "string"},
                                    "account_number": {"type": "string"},
                                    "routing_number": {"type": "string"},
                                    "account_type": {
                                        "type": "string",
                                        "enum": ["checking", "savings"],
                                        "default": "checking",
                                    },
                                    "swift_code": {
                                        "type": "string",
                                        "description": "Required for international wires",
                                    },
                                    "iban": {
                                        "type": "string",
                                        "description": "Required for some international wires",
                                    },
                                    "bank_city": {"type": "string"},
                                    "bank_state": {"type": "string"},
                                    "bank_country": {"type": "string", "default": "US"},
                                    "bank_to_bank_instructions": {
                                        "type": "string",
                                        "description": "Additional instructions for the bank (optional)",
                                    },
                                    "intermediary_bank_name": {
                                        "type": "string",
                                        "description": "Name of intermediary bank if required (optional)",
                                    },
                                },
                                "required": [
                                    "account_name",
                                    "address_1",
                                    "country",
                                    "bank_name",
                                    "account_number",
                                    "routing_number",
                                    "account_type",
                                    "bank_country",
                                ],
                            },
                        },
                        "required": [
                            "company_id",
                            "investment_id",
                            "dollar_amount",
                            "wiring_instructions",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_companies",
                    "description": "Search for companies by name or other criteria to get their IDs",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Name or part of name to search for",
                            },
                            "batch": {
                                "type": "string",
                                "description": 'Optional batch filter (e.g., "W21", "S22")',
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "live",
                                    "dead",
                                    "adrift",
                                    "exited",
                                    "went_public",
                                    "all",
                                ],
                                "description": "Filter by company status",
                                "default": "live",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": "Success or error status",
                            },
                            "results": {
                                "type": "array",
                                "description": "List of companies matching the search criteria",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {
                                            "type": "integer",
                                            "description": "Company ID to use in other API calls",
                                        },
                                        "name": {"type": "string"},
                                        "batch": {"type": "string"},
                                        "status": {"type": "string"},
                                        "valuation": {"type": "string"},
                                        "url": {"type": "string"},
                                        "description": {"type": "string"},
                                        "founders": {"type": "string"},
                                    },
                                },
                            },
                            "results_count": {
                                "type": "integer",
                                "description": "Number of companies returned",
                            },
                            "total_matches": {
                                "type": "integer",
                                "description": "Total number of matches found",
                            },
                        },
                    },
                },
            },
        ],
        "tool_choice": None,
        "max_retries": 0,
        "logprobs": None,
        "top_logprobs": None,
        "extra_headers": None,
        "api_version": None,
        "parallel_tool_calls": None,
        "drop_params": True,
        "reasoning_effort": None,
        "additional_drop_params": None,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that helps prepare a wire for a pro rata investment.",
            },
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
        "thinking": None,
        "kwargs": {},
    }
    optional_params = get_optional_params(**passed_params)
    print(f"optional_params: {optional_params}")

    assert "max_tokens_to_sample" not in optional_params
    assert optional_params["max_tokens"] == 1024


def test_azure_modalities_param():
    optional_params = get_optional_params(
        model="chatgpt-v2",
        custom_llm_provider="azure",
        modalities=["text", "audio"],
        audio={"type": "audio_input", "input": "test.wav"},
    )
    assert optional_params["modalities"] == ["text", "audio"]
    assert optional_params["audio"] == {"type": "audio_input", "input": "test.wav"}


def test_litellm_proxy_thinking_param():
    optional_params = get_optional_params(
        model="gpt-4o",
        custom_llm_provider="litellm_proxy",
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    assert optional_params["extra_body"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }


def test_gemini_modalities_param():
    optional_params = get_optional_params(
        model="gemini-1.5-pro",
        custom_llm_provider="gemini",
        modalities=["text", "image"],
    )

    assert optional_params["responseModalities"] == ["TEXT", "IMAGE"]


def test_azure_response_format_param():
    optional_params = litellm.get_optional_params(
        model="azure/o_series/test-o3-mini",
        custom_llm_provider="azure/o_series",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time in a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. San Francisco",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    )


@pytest.mark.parametrize(
    "model, provider",
    [
        ("claude-3-7-sonnet-20240620-v1:0", "anthropic"),
        ("anthropic.claude-3-7-sonnet-20250219-v1:0", "bedrock"),
        ("invoke/anthropic.claude-3-7-sonnet-20240620-v1:0", "bedrock"),
        ("claude-3-7-sonnet@20250219", "vertex_ai"),
    ],
)
def test_anthropic_unified_reasoning_content(model, provider):
    optional_params = get_optional_params(
        model=model,
        custom_llm_provider=provider,
        reasoning_effort="high",
    )
    assert optional_params["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_azure_response_format(monkeypatch):
    monkeypatch.setenv("AZURE_API_VERSION", "2025-02-01")
    optional_params = get_optional_params(
        model="azure/gpt-4o-mini",
        custom_llm_provider="azure",
        response_format={"type": "json_object"},
    )
    assert optional_params["response_format"] == {"type": "json_object"}


def test_cohere_embed_dimensions_param():
    optional_params = get_optional_params_embeddings(
        model="embed-multilingual-v3.0",
        custom_llm_provider="cohere",
        encoding_format="float",
    )
    assert optional_params["embedding_types"] == ["float"]


def test_optional_params_with_additional_drop_params():
    optional_params = get_optional_params(
        model="gpt-4o",
        custom_llm_provider="openai",
        additional_drop_params=["red"],
        drop_params=True,
        red="blue",
    )
    print(f"optional_params: {optional_params}")
    assert "red" not in optional_params
    assert "red" not in optional_params["extra_body"]


def test_azure_ai_cohere_embed_input_type_param():
    optional_params = get_optional_params_embeddings(
        model="embed-v-4-0",
        custom_llm_provider="azure_ai",
        input_type="text",
        dimensions=1536,
    )
    assert optional_params["dimensions"] == 1536
    assert optional_params["extra_body"]["input_type"] == "text"


def test_optional_params_image_gen_with_aspect_ratio():
    optional_params = get_optional_params_image_gen(
        model="imagen-4.0-ultra-generate-001",
        custom_llm_provider="vertex_ai",
        aspect_ratio="16:9",
    )
    assert optional_params["aspect_ratio"] == "16:9"


def test_optional_params_responses_api_allowed_openai_params():
    from litellm import responses
    from unittest.mock import patch, MagicMock
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            response = litellm.responses(
                model="openai/o1-pro",
                input="Tell me a three sentence bedtime story about a unicorn.",
                max_output_tokens=100,
                top_logprobs=10,
                allowed_openai_params=["top_logprobs"],
                client=client,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("error: ", e)

        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs
        print("request_body: ", request_body)
        assert "top_logprobs" in request_body["json"]


def test_validate_openai_optional_params_stop_truncation():
    """
    Test that validate_openai_optional_params truncates stop sequences to 4 elements
    when more than 4 are provided, as OpenAI only supports up to 4 stop sequences.
    """
    # Test with more than 4 stop sequences - should truncate to 4
    stop_sequences = ["stop1", "stop2", "stop3", "stop4", "stop5", "stop6"]
    result = validate_openai_optional_params(stop=stop_sequences)
    assert result == ["stop1", "stop2", "stop3", "stop4"]
    assert len(result) == 4
    
    # Test with exactly 4 stop sequences - should not truncate
    stop_sequences_4 = ["stop1", "stop2", "stop3", "stop4"]
    result = validate_openai_optional_params(stop=stop_sequences_4)
    assert result == ["stop1", "stop2", "stop3", "stop4"]
    assert len(result) == 4
    
    # Test with less than 4 stop sequences - should not truncate
    stop_sequences_2 = ["stop1", "stop2"]
    result = validate_openai_optional_params(stop=stop_sequences_2)
    assert result == ["stop1", "stop2"]
    assert len(result) == 2
    
    # Test with single stop sequence as string - should return as is
    stop_string = "stop1"
    result = validate_openai_optional_params(stop=stop_string)
    assert result == "stop1"
    
    # Test with None - should return None
    result = validate_openai_optional_params(stop=None)
    assert result is None
    
    # Test with empty list - should return empty list
    result = validate_openai_optional_params(stop=[])
    assert result == []


def test_validate_openai_optional_params_disable_stop_sequence_limit():
    """
    Test that validate_openai_optional_params respects the disable_stop_sequence_limit flag.
    When litellm.disable_stop_sequence_limit is True, stop sequences should not be truncated.
    """
    # Save original value
    original_value = litellm.disable_stop_sequence_limit
    
    try:
        # Test with disable_stop_sequence_limit = True - should NOT truncate
        litellm.disable_stop_sequence_limit = True
        stop_sequences = ["stop1", "stop2", "stop3", "stop4", "stop5", "stop6"]
        result = validate_openai_optional_params(stop=stop_sequences)
        assert result == ["stop1", "stop2", "stop3", "stop4", "stop5", "stop6"]
        assert len(result) == 6
        
        # Test with disable_stop_sequence_limit = False - should truncate to 4
        litellm.disable_stop_sequence_limit = False
        stop_sequences = ["stop1", "stop2", "stop3", "stop4", "stop5", "stop6"]
        result = validate_openai_optional_params(stop=stop_sequences)
        assert result == ["stop1", "stop2", "stop3", "stop4"]
        assert len(result) == 4
    finally:
        # Restore original value
        litellm.disable_stop_sequence_limit = original_value


def test_validate_openai_optional_params_integration():
    """
    Test that validate_openai_optional_params is properly integrated in the completion flow.
    """
    # Test that completion with more than 4 stop sequences works without error
    try:
        with patch("litellm.llms.openai.openai.OpenAI") as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.model = "gpt-3.5-turbo"
            mock_response.id = "test-id"
            mock_response.created = 1234567890
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            
            mock_client.return_value.chat.completions.create.return_value = mock_response
            
            # Call completion with more than 4 stop sequences
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                stop=["stop1", "stop2", "stop3", "stop4", "stop5", "stop6"],
                mock_response="Test response"  # This will use mock
            )
            
            # Verify the call was made (stop sequences should be truncated internally)
            assert response is not None
    except Exception as e:
        # Should not raise an exception
        pytest.fail(f"validate_openai_optional_params integration failed: {e}")
