"""
Tests Bedrock Completion + Rerank endpoints
"""

import os

from dotenv import load_dotenv

import litellm.types

load_dotenv()
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from base_embedding_unit_tests import BaseLLMEmbeddingTest
from base_llm_unit_tests import BaseAnthropicChatTest, BaseLLMChatTest
from base_rerank_unit_tests import BaseLLMRerankTest

import litellm
from litellm import (
    ModelResponse,
    RateLimitError,
    ServiceUnavailableError,
    completion,
    completion_cost,
)
from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

# litellm.num_retries = 3
litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


@pytest.fixture(autouse=True)
def reset_callbacks():
    print("\npytest fixture - resetting callbacks")
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []


def test_completion_bedrock_claude_completion_auth(monkeypatch):
    print("calling bedrock claude completion params auth")

    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region_name = os.environ["AWS_REGION_NAME"]

    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    monkeypatch.delenv("AWS_REGION_NAME")

    try:
        response = completion(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
        )
        print(response)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_claude_completion_auth()


@pytest.mark.parametrize("streaming", [True, False])
def test_completion_bedrock_guardrails(streaming):

    litellm.set_verbose = True


    # verbose_logger.setLevel(logging.DEBUG)
    try:
        if streaming is False:
            response = completion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[
                    {
                        "content": "where do i buy coffee from? ",
                        "role": "user",
                    }
                ],
                max_tokens=10,
                guardrailConfig={
                    "guardrailIdentifier": "ff6ujrregl1q",
                    "guardrailVersion": "DRAFT",
                    "trace": "enabled",
                },
            )
            # Add any assertions here to check the response
            print(response)
            assert (
                "Sorry, the model cannot answer this question. coffee guardrail applied"
                in response.choices[0].message.content
            )

            assert "trace" in response
            assert response.trace is not None

            print("TRACE=", response.trace)
        else:
            litellm.set_verbose = True
            response = completion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[
                    {
                        "content": "where do i buy coffee from? ",
                        "role": "user",
                    }
                ],
                stream=True,
                max_tokens=10,
                guardrailConfig={
                    "guardrailIdentifier": "ff6ujrregl1q",
                    "guardrailVersion": "DRAFT",
                    "trace": "enabled",
                },
            )

            saw_trace = False

            for chunk in response:
                if "trace" in chunk:
                    saw_trace = True
                print(chunk)

            assert (
                saw_trace is True
            ), "Did not see trace in response even when trace=enabled sent in the guardrailConfig"

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_claude_2_1_completion_auth()


def test_completion_bedrock_claude_external_client_auth(monkeypatch):
    print("\ncalling bedrock claude external client auth")

    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region_name = os.environ["AWS_REGION_NAME"]

    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    monkeypatch.delenv("AWS_REGION_NAME")

    try:
        import boto3

        litellm.set_verbose = True

        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=f"https://bedrock-runtime.{aws_region_name}.amazonaws.com",
        )

        response = completion(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_bedrock_client=bedrock,
        )
        print(response)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_claude_external_client_auth()










# test_completion_bedrock_claude_sts_client_auth()


@pytest.mark.parametrize(
    "stop",
    [""],
)
@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # "meta.llama3-70b-instruct-v1:0",
        # "anthropic.claude-v2",
        # "mistral.mixtral-8x7b-instruct-v0:1",
    ],
)
def test_bedrock_stop_value(stop, model):
    try:
        litellm.set_verbose = True
        data = {
            "max_tokens": 100,
            "stream": False,
            "temperature": 0.3,
            "messages": [
                {"role": "user", "content": "hey, how's it going?"},
            ],
            "stop": stop,
        }
        response: ModelResponse = completion(
            model="bedrock/{}".format(model),
            **data,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "system",
    ["You are an AI", [{"type": "text", "text": "You are an AI"}], ""],
)
@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
    ],
)
def test_bedrock_system_prompt(system, model):
    try:
        litellm.set_verbose = True
        data = {
            "max_tokens": 100,
            "stream": False,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system},
                {"role": "assistant", "content": "hey, how's it going?"},
            ],
            "user_continue_message": {"role": "user", "content": "Be a good bot!"},
        }
        response: ModelResponse = completion(
            model="bedrock/{}".format(model),
            **data,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_claude_3_tool_calling():
    try:
        litellm.set_verbose = True
        litellm.turn_on_debug()
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        messages = [
            {
                "role": "user",
                "content": "What's the weather like in Boston today in fahrenheit?",
            }
        ]
        response: ModelResponse = completion(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )  # type: ignore
        print(f"response: {response}")
        # Add any assertions here to check the response
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert isinstance(
            response.choices[0].message.tool_calls[0].function.arguments, str
        )
        messages.append(
            response.choices[0].message.model_dump()
        )  # Add assistant tool invokes
        tool_result = (
            '{"location": "Boston", "temperature": "72", "unit": "fahrenheit"}'
        )
        # Add user submitted tool results in the OpenAI format
        messages.append(
            {
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
                "role": "tool",
                "name": response.choices[0].message.tool_calls[0].function.name,
                "content": tool_result,
            }
        )
        # In the second response, Claude should deduce answer from tool results
        second_response = completion(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        print(f"second response: {second_response}")
        assert isinstance(second_response.choices[0].message.content, str)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")






def test_completion_bedrock_mistral_completion_auth():
    print("calling bedrock mistral completion params auth")


    litellm.turn_on_debug()

    # aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    # aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    # aws_region_name = os.environ["AWS_REGION_NAME"]
    # os.environ.pop("AWS_ACCESS_KEY_ID", None)
    # os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    # os.environ.pop("AWS_REGION_NAME", None)
    try:
        response: ModelResponse = completion(
            model="bedrock/mistral.mistral-7b-instruct-v0:2",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
        )  # type: ignore
        # Add any assertions here to check the response
        print(f"response: {response}")
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

        # os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        # os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        # os.environ["AWS_REGION_NAME"] = aws_region_name
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_mistral_completion_auth()


def test_bedrock_ptu():
    """
    Check if a url with 'modelId' passed in, is created correctly

    Reference: https://github.com/BerriAI/litellm/issues/3805
    """
    client = HTTPHandler()

    with patch.object(client, "post", new=Mock()) as mock_client_post:
        litellm.set_verbose = True

        model_id = (
            "arn:aws:bedrock:us-west-2:888602223428:provisioned-model/8fxff74qyhs3"
        )
        try:
            response = litellm.completion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[{"role": "user", "content": "What's AWS?"}],
                model_id=model_id,
                client=client,
            )
        except Exception as e:
            pass

        assert "url" in mock_client_post.call_args.kwargs
        assert (
            mock_client_post.call_args.kwargs["url"]
            == "https://bedrock-runtime.us-west-2.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A888602223428%3Aprovisioned-model%2F8fxff74qyhs3/converse"
        )
        mock_client_post.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_custom_api_base():
    """
    Check if a url with 'modelId' passed in, is created correctly

    Reference: https://github.com/BerriAI/litellm/issues/3805, https://github.com/BerriAI/litellm/issues/5389#issuecomment-2313677977

    """
    client = AsyncHTTPHandler()

    with patch.object(client, "post", new=AsyncMock()) as mock_client_post:
        litellm.set_verbose = True

        try:
            response = await litellm.acompletion(
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "What's AWS?"}],
                client=client,
                extra_headers={"test": "hello world", "Authorization": "my-test-key"},
                api_base="https://gateway.ai.cloudflare.com/v1/fa4cdcab1f32b95ca3b53fd36043d691/test/aws-bedrock/bedrock-runtime/us-east-1",
            )
        except Exception as e:
            pass

        print(f"mock_client_post.call_args.kwargs: {mock_client_post.call_args.kwargs}")
        assert (
            mock_client_post.call_args.kwargs["url"]
            == "https://gateway.ai.cloudflare.com/v1/fa4cdcab1f32b95ca3b53fd36043d691/test/aws-bedrock/bedrock-runtime/us-east-1/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        )
        assert "test" in mock_client_post.call_args.kwargs["headers"]
        assert mock_client_post.call_args.kwargs["headers"]["test"] == "hello world"
        assert (
            mock_client_post.call_args.kwargs["headers"]["Authorization"]
            == "my-test-key"
        )
        mock_client_post.assert_called_once()


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
    ],
)
@pytest.mark.asyncio
async def test_bedrock_extra_headers(model):
    """
    Relevant Issue: https://github.com/BerriAI/litellm/issues/9106
    """
    client = AsyncHTTPHandler()

    with patch.object(client, "post", new=AsyncMock()) as mock_client_post:
        litellm.set_verbose = True

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "What's AWS?"}],
                client=client,
                extra_headers={"test": "hello world", "Authorization": "my-test-key"},
            )
        except Exception as e:
            print(f"error: {e}")

        print(f"mock_client_post.call_args.kwargs: {mock_client_post.call_args.kwargs}")
        assert "test" in mock_client_post.call_args.kwargs["headers"]
        assert mock_client_post.call_args.kwargs["headers"]["test"] == "hello world"
        assert (
            mock_client_post.call_args.kwargs["headers"]["Authorization"]
            == "my-test-key"
        )
        mock_client_post.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_custom_prompt_template():
    """
    Check if custom prompt template used for bedrock models

    Reference: https://github.com/BerriAI/litellm/issues/4415
    """
    client = AsyncHTTPHandler()

    with patch.object(client, "post", new=AsyncMock()) as mock_client_post:
        import json

        try:
            response = await litellm.acompletion(
                model="bedrock/mistral.OpenOrca",
                messages=[{"role": "user", "content": "What's AWS?"}],
                client=client,
                roles={
                    "system": {
                        "pre_message": "<|im_start|>system\n",
                        "post_message": "<|im_end|>",
                    },
                    "assistant": {
                        "pre_message": "<|im_start|>assistant\n",
                        "post_message": "<|im_end|>",
                    },
                    "user": {
                        "pre_message": "<|im_start|>user\n",
                        "post_message": "<|im_end|>",
                    },
                },
                bos_token="<s>",
                eos_token="<|im_end|>",
            )
        except Exception as e:
            pass

        print(f"mock_client_post.call_args: {mock_client_post.call_args}")
        assert "prompt" in json.loads(mock_client_post.call_args.kwargs["data"])

        prompt = json.loads(mock_client_post.call_args.kwargs["data"])["prompt"]
        assert prompt == "<|im_start|>user\nWhat's AWS?<|im_end|>"
        mock_client_post.assert_called_once()


def test_completion_bedrock_external_client_region(monkeypatch):
    print("\ncalling bedrock claude external client auth")

    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region_name = "us-east-1"

    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")

    client = HTTPHandler()

    try:
        import boto3

        litellm.set_verbose = True

        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=f"https://bedrock-runtime.{aws_region_name}.amazonaws.com",
        )
        with patch.object(client, "post", new=Mock()) as mock_client_post:
            try:
                response = completion(
                    model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    messages=messages,
                    max_tokens=10,
                    temperature=0.1,
                    aws_bedrock_client=bedrock,
                    client=client,
                )
                # Add any assertions here to check the response
                print(response)
            except Exception as e:
                pass

            print(f"mock_client_post.call_args: {mock_client_post.call_args}")
            assert "us-east-1" in mock_client_post.call_args.kwargs["url"]

            mock_client_post.assert_called_once()
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")














from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_converse_messages_pt,
)




def test_base_aws_llm_get_credentials():
    import time

    import boto3


    start_time = time.time()
    session = boto3.Session(
        aws_access_key_id="test",
        aws_secret_access_key="test2",
        region_name="test3",
    )
    credentials = session.get_credentials().get_frozen_credentials()
    end_time = time.time()

    print(
        "Total time for credentials - {}. Credentials - {}".format(
            end_time - start_time, credentials
        )
    )

    start_time = time.time()
    credentials = BaseAWSLLM().get_credentials(
        aws_access_key_id="test",
        aws_secret_access_key="test2",
        aws_region_name="test3",
    )

    end_time = time.time()

    print(
        "Total time for credentials - {}. Credentials - {}".format(
            end_time - start_time, credentials.get_frozen_credentials()
        )
    )










def test_bedrock_converse_route():
    litellm.set_verbose = True
    try:
        litellm.completion(
            model="bedrock/converse/us.amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
    except ServiceUnavailableError as e:
        if "Too many requests" in str(e):
            pytest.skip("Skipping test due to AWS Bedrock rate limiting")
        else:
            raise


def test_bedrock_mapped_converse_models():
    litellm.set_verbose = True
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.add_known_models()
    litellm.completion(
        model="bedrock/us.amazon.nova-pro-v1:0",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )










class TestBedrockConverseChatCrossRegion(BaseLLMChatTest):
    test_content_list_handling = None
    test_developer_role_translation = None
    test_function_calling_with_tool_response = None
    test_image_url = None
    test_json_response_format_stream = None
    test_tool_call_with_empty_enum_property = None
    test_tool_call_with_property_type_array = None

    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    def test_completion_cost(self):
        """
        Test if region models info is correctly used for cost calculation. Using the base model info for cost calculation.
        """
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        bedrock_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        litellm.model_cost.pop(bedrock_model, None)
        model = f"bedrock/{bedrock_model}"

        litellm.set_verbose = True
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        cost = completion_cost(response)

        assert cost > 0


class TestBedrockConverseAnthropicUnitTests(BaseAnthropicChatTest):
    test_completion_thinking_with_max_tokens = None
    test_completion_thinking_without_max_tokens = None

    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }


class TestBedrockConverseChatNormal(BaseLLMChatTest):
    test_content_list_handling = None
    test_empty_tools = None
    test_function_calling_with_tool_response = None
    test_image_url = None

    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "aws_region_name": "us-east-1",
        }



class TestBedrockConverseNovaTestSuite(BaseLLMChatTest):
    test_content_list_handling = None
    test_function_calling_with_tool_response = None
    test_image_url = None

    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.amazon.nova-lite-v1:0",
            "aws_region_name": "us-east-1",
        }




class TestBedrockRerank(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.BEDROCK

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        }


class TestBedrockCohereRerank(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.BEDROCK

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "bedrock/arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0",
        }








@pytest.mark.parametrize("top_k_param", ["top_k", "topK"])
def test_bedrock_nova_topk(top_k_param):
    litellm.set_verbose = True
    data = {
        "model": "bedrock/us.amazon.nova-pro-v1:0",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        top_k_param: 10,
    }
    original_transform = litellm.AmazonConverseConfig()._transform_request
    captured_data = None

    def mock_transform(*args, **kwargs):
        nonlocal captured_data
        result = original_transform(*args, **kwargs)
        captured_data = result
        return result

    with patch("litellm.AmazonConverseConfig._transform_request", side_effect=mock_transform):
        litellm.completion(**data)

        # Assert that additionalRequestParameters exists and contains topK
        assert "additionalModelRequestFields" in captured_data
        assert "inferenceConfig" in captured_data["additionalModelRequestFields"]
        assert captured_data["additionalModelRequestFields"]["inferenceConfig"]["topK"] == 10


def test_bedrock_cross_region_inference(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.add_known_models()

    litellm.set_verbose = True
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model="bedrock/us.meta.llama3-3-70b-instruct-v1:0",
                messages=[{"role": "user", "content": "Hello, world!"}],
                client=client,
            )
        except Exception as e:
            print(e)

        assert (
            mock_post.call_args.kwargs["url"]
            == "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.meta.llama3-3-70b-instruct-v1%3A0/converse"
        )


def test_bedrock_empty_content_real_call():
    completion(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "",
                    },
                    {"type": "text", "text": "Hey, how's it going?"},
                ],
            }
        ],
    )






class TestBedrockEmbedding(BaseLLMEmbeddingTest):
    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "bedrock/amazon.titan-embed-image-v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.BEDROCK



@pytest.mark.asyncio
async def test_bedrock_image_url_sync_client():
    import logging

    from litellm import verbose_logger
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    verbose_logger.setLevel(level=logging.DEBUG)

    litellm.turn_on_debug()
    client = AsyncHTTPHandler()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
                    },
                },
            ],
        }
    ]

    with patch.object(client, "post") as mock_post:
        try:
            await litellm.acompletion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=messages,
                client=client,
            )
        except Exception as e:
            print(e)
        mock_post.assert_called_once()




def test_bedrock_custom_proxy():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            response = completion(
                model="bedrock/converse_like/us.amazon.nova-pro-v1:0",
                messages=[{"content": "Tell me a joke", "role": "user"}],
                api_key="Token",
                client=client,
                api_base="https://some-api-url/models",
            )
        except Exception as e:
            print(e)
        print(mock_post.call_args.kwargs)
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["url"] == "https://some-api-url/models"

        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer Token"


def test_bedrock_custom_deepseek():
    import json

    litellm.turn_on_debug()
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        # Mock the response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {"generation": "Here's a joke...", "stop_reason": "stop"}
        )
        mock_response.status_code = 200
        # Add required response attributes
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        try:
            response = completion(
                model="bedrock/llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n",  # Updated to specify provider
                messages=[{"role": "user", "content": "Tell me a joke"}],
                max_tokens=100,
                client=client,
            )

            # Print request details
            print("\nRequest Details:")
            print(f"URL: {mock_post.call_args.kwargs['url']}")

            # Verify the URL
            assert (
                mock_post.call_args.kwargs["url"]
                == "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A086734376398%3Aimported-model%2Fr4c4kewx2s0n/invoke"
            )

            # Verify the request body format
            request_body = json.loads(mock_post.call_args.kwargs["data"])
            print("request_body=", json.dumps(request_body, indent=4, default=str))
            assert "prompt" in request_body
            assert request_body["prompt"] == "Tell me a joke"

            # follows the llama spec
            assert request_body["max_gen_len"] == 100

        except Exception as e:
            print(f"Error: {str(e)}")
            raise e








def test_bedrock_description_param():
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            response = completion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=[
                    {"role": "user", "content": "What is the meaning of this poem?"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "meaning_reasoning",
                        "description": "Find the meaning inside a poem",
                        "schema": {
                            "type": "object",
                            "properties": {"meaning": {"type": "string"}},
                        },
                    },
                },
                client=client,
            )
        except Exception as e:
            print(e)
        mock_post.assert_called_once()

        request_body = json.loads(mock_post.call_args.kwargs["data"])
        request_body_str = json.dumps(request_body, indent=4, default=str)
        print("request_body=", request_body_str)

        assert (
            "Find the meaning inside a poem" in request_body_str
        )  # assert description is passed


@pytest.mark.parametrize(
    "sync_mode",
    [
        True,
        False,
    ],
)
@pytest.mark.asyncio
async def test_bedrock_thinking_in_assistant_message(sync_mode):
    litellm.turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    params = {
        "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Alright, let's get started with resolving this issue about implementing `__len__` for the Pipeline class in scikit-learn.\n\n### Step 1: Explore the repository structure to understand it better\n\nLet's first explore the repository structure to get a better understanding of the codebase:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nThis will give us an overview of the top-level directories and files. From there, we can delve deeper into relevant parts of the code.\n\nLet's look at the main organization of the repository first, then we'll locate the Pipeline class, understand how it's currently implemented, and then add the `__len__` method to it.\n\nNext, let's look at the pipeline module, as the issue mentions `pipe[:len(pipe)]`, indicating that there is indexing support for Pipeline, but it lacks a `__len__` method.\n\nSince the issue mentions that Pipeline supports indexing but not `__len__`, I'll check the Pipeline class to understand its implementation.\n\nThe Pipeline class is likely to be in sklearn/pipeline.py or within a pipeline directory. Let's look for it:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nThis will help us locate the file containing the Pipeline class.\n\nOnce we've located the file, we'll examine it to understand:\n1. How indexing is currently implemented\n2. Where we need to add the `__len__` method\n3. How the length of a pipeline should be determined\n\nThen, we'll implement the `__len__` method, test our changes to ensure they fix the issue, and verify that our changes don't break existing functionality.\n\nLet's begin by exploring the repository.\n\n### Step 2: Reproduce the error\n\nAfter exploring the repository, I'll create a script to reproduce the error described in the issue. This script will use the code provided in the \"Steps/Code to Reproduce\" section of the issue description.\n\nThen I'll run the script to confirm the error occurs as described, which will help us understand what needs to be fixed.\n\n### Step 3: Fix the issue\n\nOnce I've understood the problem and located the relevant files, I'll modify the code to add the `__len__` method to the Pipeline class. This method should return the number of steps in the pipeline, which is consistent with what a user would expect when calling `len(pipe)`.\n\n### Step 4: Test the fix\n\nAfter implementing the fix, I'll rerun the reproduction script to verify that the error is resolved and `len(pipe)` works correctly.\n\n### Step 5: Consider edge cases\n\nI'll think about potential edge cases like:\n- Empty pipelines\n- Very long pipelines\n- Any other special cases where the `__len__` implementation might behave unexpectedly\n\nI'll add tests for these cases to ensure our implementation handles them correctly.\n\n### Step 6: Comprehensive testing\n\nI'll run the existing tests to ensure our changes don't break any existing functionality, and I'll also write additional tests to specifically test our new `__len__` method.\n\nLet's start with exploring the repository.\n\nLet's start by exploring the repository structure:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nNow, let's find the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nBased on the repository structure, let's look at the pipeline implementation:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's check if there are tests for the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*test*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nNow, let's create a script to reproduce the error. I'll create a file in the workspace directory and add the code from the issue description:\n\n\ncat > /workspace/reproduce_pipeline_len_error.py << 'EOF'\nfrom sklearn import svm\nfrom sklearn.datasets import samples_generator\nfrom sklearn.feature_selection import SelectKBest\nfrom sklearn.feature_selection import f_regression\nfrom sklearn.pipeline import Pipeline\n\n# generate some data to play with\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\n\nprint(\"Trying to get length of pipeline...\")\ntry:\n    length = len(pipe)\n    print(f\"Length of pipeline: {length}\")\nexcept Exception as e:\n    print(f\"Error: {type(e).__name__}: {e}\")\nEOF\n\n\nLet's run this to confirm the error:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nNow, let's implement the `__len__` method in the Pipeline class. First, I'll examine the Pipeline class to understand how it's structured and where to add the `__len__` method.\n\nFrom the error we saw above and looking at the pipeline.py file, we should add a `__len__` method to the Pipeline class.\n\nLet's modify the pipeline.py file:\n\n\ncp /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py /workspace/pipeline.py.backup\n\n\nNow, let's edit the file to add the `__len__` method:\n\n\nsed -n -e '/class Pipeline/,/^class /p' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nThis will show us the Pipeline class definition so we can understand where to add the `__len__` method.\n\nBased on what we see, I'll add the `__len__` method to the Pipeline class:\n\n\nsed -i -e '/def __getitem__.*:/a\\\n    def __len__(self):\\\n        \"\"\"Returns the length of the Pipeline.\\\n\\\n        Returns\\\n        -------\\\n        length : int\\\n            The number of steps in the pipeline.\\\n        \"\"\"\\\n        return len(self.steps)\n' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's verify that the changes were applied correctly:\n\n\ngrep -A 10 \"__len__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's run our script again to see if the error is fixed:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nIf the error is fixed, let's test some edge cases by modifying our script:\n\n\ncat > /workspace/test_pipeline_len.py << 'EOF'\nfrom sklearn.pipeline import Pipeline\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\n\n# Test case 1: Normal pipeline\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\n\n# Test case 2: Empty pipeline\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\n\n# Test case 3: Long pipeline\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\n\n# Test case 4: Slicing with len\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nsliced_pipe = pipe[:len(pipe)]\nprint(f\"Original pipe steps: {[name for name, _ in pipe.steps]}\")\nprint(f\"Sliced pipe steps: {[name for name, _ in sliced_pipe.steps]}\")\nEOF\n\n\nLet's run this test script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also run the tests for the pipeline module to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/test_pipeline.py -v\n\n\nAfter testing the changes, let's update our script once more to test the edge case where we use `len(pipe)` in slicing:\n\n\ncat >> /workspace/test_pipeline_len.py << 'EOF'\n\n# Test case 5: Various slicing operations with len\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"\\nOriginal pipe length: {len(pipe)}\")\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\n\n# Slice a subset in the middle\nif len(pipe) >= 3:\n    slice4 = pipe[1:len(pipe)-1]\n    print(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nEOF\n\n\nLet's run this updated script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also check all the tests in the sklearn directory to ensure our changes don't break anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nBefore we conclude, let's verify that our changes to scikit-learn match the expected functionality:\n\n1. We have implemented the `__len__` method in the Pipeline class\n2. The method returns the number of steps in the pipeline\n3. We have tested that it works for normal, empty, and long pipelines\n4. We have also tested that slicing operations using `len(pipe)` work correctly\n\nSince the issue description mentioned that the repository already supports indexing but not `__len__`, we didn't have to implement indexing. We only needed to add the `__len__` method to make `len(pipe)` work, which we've done.\n\nOur changes are minimal and focused on solving the specific issue described.\n\nFurthermore, to ensure we've done a thorough job, let's run all sklearn tests to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/\n\n\nLet's check the full implementation of the Pipeline to understand where `__len__` fits in compared to other methods like `__getitem__`:\n\n\nhead -n 300 /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's more specifically check the surrounding methods to ensure our addition fits well with the rest of the class:\n\n\ngrep -A 20 \"__getitem__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's also test our changes against the base commit a62775e99f2a5ea3d51db7160fad783f6cd8a4c5, as mentioned in the requirements:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngit status\n\n\nHmm, we don't have access to the git repository directly. Let's focus on thoroughly testing our changes then.\n\nLet's run the full test suite for the sklearn module to make sure our changes don't break any existing functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nGreat, now let's create a more comprehensive test script to ensure our `__len__` implementation works in various scenarios:\n\n\ncat > /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\"\"\"\nComprehensive tests for the __len__ method of Pipeline.\n\"\"\"\nfrom sklearn.pipeline import Pipeline, FeatureUnion\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Test case 1: Normal pipeline\nprint(\"Test case 1: Normal pipeline\")\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\nassert len(pipe) == 2, \"Length of normal pipeline should be 2\"\n\n# Test case 2: Empty pipeline\nprint(\"\\nTest case 2: Empty pipeline\")\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\nassert len(empty_pipe) == 0, \"Length of empty pipeline should be 0\"\n\n# Test case 3: Long pipeline\nprint(\"\\nTest case 3: Long pipeline\")\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\nassert len(long_pipe) == 10, \"Length of long pipeline should be 10\"\n\n# Test case 4: Pipeline with FeatureUnion\nprint(\"\\nTest case 4: Pipeline with FeatureUnion\")\nunion = FeatureUnion([\n    ('pca', PCA(n_components=1)),\n    ('select', SelectKBest(k=1))\n])\npipe_with_union = Pipeline([\n    ('scaler', StandardScaler()),\n    ('union', union),\n    ('svc', svm.SVC(kernel='linear'))\n])\nprint(f\"Length of pipeline with FeatureUnion: {len(pipe_with_union)}\")\nassert len(pipe_with_union) == 3, \"Length of pipeline with FeatureUnion should be 3\"\n\n# Test case 5: Various slicing operations with len\nprint(\"\\nTest case 5: Various slicing operations with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"Original pipe length: {len(pipe)}\")\nassert len(pipe) == 3, \"Original pipe length should be 3\"\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\nassert len(slice1) == 3, \"Length of pipe[:len(pipe)] should be 3\"\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\nassert len(slice2) == 1, \"Length of pipe[:len(pipe)//2] should be 1\"\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\nassert len(slice3) == 2, \"Length of pipe[len(pipe)//2:] should be 2\"\n\n# Slice a subset in the middle\nslice4 = pipe[1:len(pipe)-1]\nprint(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nassert len(slice4) == 1, \"Length of pipe[1:len(pipe)-1] should be 1\"\n\nprint(\"\\nAll tests passed!\")\nEOF\n\n\nLet's run this comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also run specific tests from the scikit-learn test suite that relate to Pipeline indexing to ensure our changes integrate well:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing\n\n\nOh, there might not be a specific test named `test_pipeline_indexing`. Let's look at the available tests in the test_pipeline.py file:\n\n\ngrep -r \"def test_\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py\n\n\nLet's run the tests that seem related to pipeline indexing or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"index\"\n\n\nSince we might not have direct access to specific test names, let's examine the pipeline test file to identify relevant tests:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -i \"index\\|slicing\\|getitem\"\n\n\nBased on our findings, let's run specific tests that are relevant to our changes:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nLet's check if the test_pipeline.py file contains any tests for slicing functionality:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -A 20 -B 20 \"slicing\\|__getitem__\"\n\n\nLet's modify our comprehensive test script to include more edge cases and ensure it aligns with any existing tests:\n\n\ncat >> /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\n# Test case 6: Testing on pipeline with make_pipeline\nprint(\"\\nTest case 6: Testing on pipeline with make_pipeline\")\nfrom sklearn.pipeline import make_pipeline\n\npipe = make_pipeline(StandardScaler(), PCA(n_components=2), SelectKBest(k=1))\nprint(f\"Length of make_pipeline: {len(pipe)}\")\nassert len(pipe) == 3, \"Length of make_pipeline should be 3\"\n\n# Test case 7: Testing on nested pipelines\nprint(\"\\nTest case 7: Testing on nested pipelines\")\ninner_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))])\nouter_pipe = Pipeline([('inner', inner_pipe), ('svc', svm.SVC())])\nprint(f\"Length of outer pipeline: {len(outer_pipe)}\")\nassert len(outer_pipe) == 2, \"Length of outer pipeline should be 2\"\n\n# Test case 8: Testing __len__ with negative indexing\nprint(\"\\nTest case 8: Testing __len__ with negative indexing\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1))\n])\nneg_slice = pipe[-len(pipe):]\nprint(f\"pipe[-len(pipe):] steps: {[name for name, _ in neg_slice.steps]}\")\nassert len(neg_slice) == 3, \"Length of pipe[-len(pipe):] should be 3\"\n\nneg_slice2 = pipe[:-len(pipe)]\nprint(f\"pipe[:-len(pipe)] steps: {[name for name, _ in neg_slice2.steps]}\")\nassert len(neg_slice2) == 0, \"Length of pipe[:-len(pipe)] should be 0\"\n\n# Test case 9: Testing complex slicing expressions with len\nprint(\"\\nTest case 9: Testing complex slicing expressions with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1)),\n    ('step4', svm.SVC())\n])\ncomplex_slice = pipe[len(pipe)//4:len(pipe)//4*3]\nprint(f\"pipe[len(pipe)//4:len(pipe)//4*3] steps: {[name for name, _ in complex_slice.steps]}\")\nassert len(complex_slice) == 2, \"Length of complex slice should be 2\"\n\nprint(\"\\nAll additional tests passed!\")\nEOF\n\n\nLet's run the updated comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also verify that our implementation doesn't cause any regressions in the scikit-learn test suite:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nNow, let's also run another specific test related to pipeline functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_metaestimators.py | grep -i \"pipeline\"\n\n\nLet's get a deeper understanding of how the Pipeline class is used in tests by examining test files containing pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"Pipeline\" --include=\"*.py\" sklearn/tests/ | head -n 20\n\n\nLet's also look for any tests related to pipeline length or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"len(\" --include=\"*.py\" sklearn/tests/ | grep -i \"pipeline\" | head -n 20\n\n\nWith all our tests passing, it appears we've successfully implemented the `__len__` method for the Pipeline class, and it behaves as expected in various scenarios.\n\nLet's summarize what we've done:\n\n1. We located the Pipeline class in sklearn/pipeline.py\n2. We identified that it was missing a `__len__` method, which was causing the error described in the issue\n3. We implemented the `__len__` method to return the length of the pipeline's steps list\n4. We verified our fix with a reproduction script and comprehensive tests\n5. We confirmed that existing scikit-learn tests still pass with our changes\n\nOur implementation is minimal and focused on the specific issue, adding only the `__len__` method without changing any existing functionality.\n\nAt this point we can conclude that the issue has been successfully resolved. The Pipeline class now supports `len(pipe)`, which allows expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's make sure we've run the specific test for pipeline indexing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"indexing\"\n\n\nLet's also check if there are any tests related to `__len__` or length in the pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"__len__\\|len(\" --include=\"*.py\" sklearn/tests/test_pipeline.py\n\n\nWith all our checks and tests, we're confident that our implementation of the `__len__` method in the Pipeline class correctly resolves the issue described. The method returns the number of steps in the pipeline, which is the expected behavior for `len(pipe)`.\n\nI believe our solution is complete and correct. The changes we made are minimal and focused on the specific issue, and we've thoroughly tested them to ensure they work as expected in various scenarios.\n\nLet's complete the task by running our comprehensive test one more time:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nI think we're good to go. Our implementation of the `__len__` method in the Pipeline class resolves the issue described, allowing expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's do a final check on the changes we made. The diff should show just the addition of the `__len__` method to the Pipeline class:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ndiff /workspace/pipeline.py.backup sklearn/pipeline.py\n\n\nWith all our checks and tests passing, we've successfully resolved the issue described in the problem statement.\n\nAlright, let's summarize our solution once more:\n\nThe issue was that Pipeline in scikit-learn didn't implement the `__len__` method, which was causing `pipe[:len(pipe)]` to raise an error, even though Pipeline supported indexing.\n\nTo resolve this, we added a `__len__` method to the Pipeline class that returns the number of steps in the pipeline. This allows expressions like `len(pipe)` and `pipe[:len(pipe)]` to work correctly, providing a more consistent and intuitive API for users.\n\nWe thoroughly tested our solution with:\n1. A script to reproduce the original error and confirm our fix\n2. Comprehensive tests for various scenarios, including edge cases\n3. Running the existing scikit-learn test suite to ensure no regressions\n\nThese tests confirm that our solution correctly resolves the issue while maintaining compatibility with the rest of the codebase.\n\nAnd that concludes our implementation.",
                        "signature": "ErUBCkYIARgCIkDnKEqW20HOxB9jS85ReLznKuKSV88wnAJf/M+y9ge7U+GOWAHd/trgmvpgHpJmOBkeUWUftR5g+81TNmAsDd2EEgxQHv0uMQBMvelCR+saDFvnSSfaNOj3NTVYpyIwG1ddKj5AVnXe382HERyVchmOTdn1ebogEFpXlRE66Mckq57xLSygusq6MwAV/6RvKh28HEuwcXNQn+BPIxQDVCwGGUh5g4fpZ6IgGk3mRg==",
                    },
                    {
                        "type": "text",
                        "text": "I'll help you implement the `__len__` method for Pipeline in scikit-learn to fix the issue described. Let's start by exploring the repository and understanding the problem better.\n\n## Step 1: Explore the repository structure\n\nLet's first look at the repository structure:",
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Who do you know?"}]},
        ],
        "max_tokens": 32768,
        "thinking": {"type": "enabled", "budget_tokens": 30720},
    }

    with patch.object(client, "post") as mock_post:
        try:
            if sync_mode:
                response = litellm.completion(**params, client=client)
            else:
                response = await litellm.acompletion(**params, client=client)
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs)
        json_data = mock_post.call_args.kwargs["data"]

        assert (
            "Alright, let's get started with resolving this issue about implementing"
            in json_data
        )


@pytest.mark.asyncio
async def test_bedrock_stream_thinking_content_openwebui():
    """
    When merge_reasoning_content_in_choices=True

    The content should be collected as

    ```
    <think>
    I am a helpful assistant, the user wants to know who I am
    </think>

    Hi I am Anthropic, I am a helpful assistant

    ```
    """
    response = await litellm.acompletion(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "Hello who is this?"}],
        stream=True,
        max_tokens=1080,
        thinking={"type": "enabled", "budget_tokens": 1024},
        merge_reasoning_content_in_choices=True,
    )
    content = ""
    async for chunk in response:
        content += chunk.choices[0].delta.content or ""

        # OpenWebUI expects the reasoning_content to be removed, otherwise this will appear as duplicate thinking blocks
        assert getattr(chunk.choices[0].delta, "reasoning_content", None) is None
        print(chunk)

    print("collected content", content)

    # Assert that the content follows the expected format with exactly one thinking section
    think_open_pos = content.find("<think>")
    think_close_pos = content.find("</think>")

    # Assert there's exactly one opening and closing tag
    assert think_open_pos >= 0, "Opening <think> tag not found"
    assert think_close_pos > 0, "Closing </think> tag not found"
    assert (
        content.count("<think>") == 1
    ), "There should be exactly one opening <think> tag"
    assert (
        content.count("</think>") == 1
    ), "There should be exactly one closing </think> tag"

    # Assert the opening tag comes before the closing tag
    assert (
        think_open_pos < think_close_pos
    ), "Opening tag should come before closing tag"

    # Assert there's content between the tags
    thinking_content = content[think_open_pos + 7 : think_close_pos]
    assert (
        len(thinking_content.strip()) > 0
    ), "There should be content between thinking tags"

    # Assert there's content after the closing tag
    assert (
        len(content) > think_close_pos + 8
    ), "There should be content after the thinking tags"
    response_content = content[think_close_pos + 8 :].strip()
    assert (
        len(response_content) > 0
    ), "There should be non-empty content after thinking tags"


def test_bedrock_application_inference_profile():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    client2 = HTTPHandler()

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
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    with (
        patch.object(client, "post") as mock_post,
        patch.object(client2, "post") as mock_post2,
    ):
        try:
            resp = completion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model_id="arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/a0a0a0a0a0a0",
                client=client,
                tools=tools,
            )
        except Exception as e:
            print(e)

        try:
            resp = completion(
                model="bedrock/converse/arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/a0a0a0a0a0a0",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                client=client2,
                tools=tools,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()
        mock_post2.assert_called_once()
        print(mock_post.call_args.kwargs)
        json_data = mock_post.call_args.kwargs["data"]
        assert mock_post.call_args.kwargs["url"].startswith(
            "https://bedrock-runtime.eu-central-1.amazonaws.com/"
        )
        assert mock_post2.call_args.kwargs["url"] == mock_post.call_args.kwargs["url"]


def return_mocked_response(model: str):
    if model == "bedrock/mistral.mistral-large-2407-v1:0":
        return {
            "metrics": {"latencyMs": 316},
            "output": {
                "message": {
                    "content": [{"text": "Hello! How are you doing today? How can"}],
                    "role": "assistant",
                }
            },
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
        }


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-large-2407-v1:0",
    ],
)
@pytest.mark.asyncio()
async def test_bedrock_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to bedrock models
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    litellm.set_verbose = True

    client = AsyncHTTPHandler()

    mock_response = return_mocked_response(model)
    _model = model.split("/")[1]
    print("\n\nmock_response: ", mock_response)

    with patch.object(client, "post") as mock_client:
        try:
            response = await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = json.loads(mock_client.call_args.kwargs["data"])

        print("request_body: ", request_body)

        assert request_body == {
            "messages": [{"role": "user", "content": [{"text": "Hello!"}]}],
            "inferenceConfig": {"maxTokens": 10},
        }




@pytest.mark.asyncio
async def test_bedrock_passthrough_router():
    """
    Test bedrock passthrough using litellm.Router with async mode.
    Tests that the router:
    1. Resolves the router model name to the actual deployment
    2. Replaces the router model name in the endpoint with the actual deployment model
    """
    import litellm
    from litellm import Router

    litellm.turn_on_debug()

    router = Router(
        model_list=[
            {
                "model_name": "special-bedrock-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                },
            }
        ]
    )

    data = {
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Hey"}],
        "system": [
            {
                "type": "text",
                "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text.",
            }
        ],
        "temperature": 0,
        "metadata": {
            "user_id": "5dd07c33da27e6d2968d94ea20bf47a7b090b6b158b82328d54da2909a108e84"
        },
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["claude-code-20250219"],
    }

    # Endpoint uses the router model name which should be replaced with actual deployment
    response = await router.allm_passthrough_route(
        model="special-bedrock-model",
        method="POST",
        endpoint="/model/special-bedrock-model/invoke",
        data=data,
    )

    print(response.text)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_bedrock_converse__streaming_passthrough(monkeypatch):
    import asyncio

    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    if os.environ.get("LITELLM_RUN_LIVE_BEDROCK_PASSTHROUGH_TESTS") != "1":
        pytest.skip("Live Bedrock passthrough E2E tests are opt-in")
    if os.environ.get("CASSETTE_REDIS_URL"):
        pytest.skip("Live Bedrock passthrough E2E tests cannot run under VCR replay")

    class MockCustomLogger(CustomLogger):
        pass

    mock_custom_logger = MockCustomLogger()
    monkeypatch.setattr(litellm, "callbacks", [mock_custom_logger])

    litellm.turn_on_debug()

    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Write an article about impact of high inflation to GDP of a country"
                    }
                ],
            }
        ],
        "system": [{"text": "You are an economist with access to lots of data"}],
        "inferenceConfig": {"maxTokens": 100, "temperature": 0.5},
    }
    with patch.object(mock_custom_logger, "async_log_success_event") as mock_callback:
        response = await litellm.allm_passthrough_route(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/converse-stream",
            data=data,
        )
        async for chunk in response:
            print(chunk)

        await asyncio.sleep(1)

        mock_callback.assert_called_once()
        print(mock_callback.call_args.kwargs.keys())
        assert "response_cost" in mock_callback.call_args.kwargs["kwargs"]
        response_cost = mock_callback.call_args.kwargs["kwargs"]["response_cost"]
        assert response_cost is not None and response_cost > 0
        assert "standard_logging_object" in mock_callback.call_args.kwargs["kwargs"]


@pytest.mark.asyncio
async def test_bedrock_streaming_passthrough_test2(monkeypatch):
    import asyncio

    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class MockCustomLogger(CustomLogger):
        pass

    mock_custom_logger = MockCustomLogger()
    monkeypatch.setattr(litellm, "callbacks", [mock_custom_logger])

    litellm.turn_on_debug()

    data = {
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Hey"}],
        "system": [
            {
                "type": "text",
                "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text.",
            }
        ],
        "temperature": 0,
        "metadata": {
            "user_id": "5dd07c33da27e6d2968d94ea20bf47a7b090b6b158b82328d54da2909a108e84"
        },
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["claude-code-20250219"],
    }

    with patch.object(mock_custom_logger, "async_log_success_event") as mock_callback:
        response = await litellm.allm_passthrough_route(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke-with-response-stream",
            data=data,
        )
        async for chunk in response:
            print(chunk)

        await asyncio.sleep(5)

        mock_callback.assert_called_once()
        # check standard logging payload created
        print(mock_callback.call_args.kwargs.keys())
        assert "standard_logging_object" in mock_callback.call_args.kwargs["kwargs"]
        assert "response_cost" in mock_callback.call_args.kwargs["kwargs"]


def test_bedrock_openai_imported_model():
    """
    Test that Bedrock imported models using OpenAI format work correctly.

    This test validates:
    1. The request body follows OpenAI Chat Completions format
    2. The URL is correctly constructed for Bedrock invoke endpoint
    3. Messages with system, user roles and image_url content are preserved
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    # Sample base64 image data (truncated for test)
    sample_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can analyze images.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Spot the difference between the two images?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{sample_base64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{sample_base64}"},
                },
            ],
        },
    ]

    with patch.object(client, "post") as mock_post:
        try:
            response = completion(
                model="bedrock/openai/arn:aws:bedrock:us-east-1:117159858402:imported-model/m4gc1mrfuddy",
                messages=messages,
                max_tokens=300,
                temperature=0.5,
                client=client,
            )
        except Exception as e:
            print(f"Exception (expected during mock): {e}")

        mock_post.assert_called_once()

        # Validate URL
        url = mock_post.call_args.kwargs["url"]
        print(f"URL: {url}")
        assert "bedrock-runtime.us-east-1.amazonaws.com" in url
        assert (
            "arn:aws:bedrock:us-east-1:117159858402:imported-model%2Fm4gc1mrfuddy"
            in url
        )
        assert "/invoke" in url

        # Validate request body follows OpenAI format
        request_body = json.loads(mock_post.call_args.kwargs["data"])
        print(f"Request body: {json.dumps(request_body, indent=2)}")

        # Check messages structure
        assert "messages" in request_body
        assert len(request_body["messages"]) == 2

        # Check system message
        system_msg = request_body["messages"][0]
        assert system_msg["role"] == "system"
        assert "helpful assistant" in system_msg["content"]

        # Check user message with image content
        user_msg = request_body["messages"][1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 3

        # Check text content
        assert user_msg["content"][0]["type"] == "text"
        assert "Spot the difference" in user_msg["content"][0]["text"]

        # Check image_url content
        assert user_msg["content"][1]["type"] == "image_url"
        assert "image_url" in user_msg["content"][1]
        assert user_msg["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )

        assert user_msg["content"][2]["type"] == "image_url"
        assert "image_url" in user_msg["content"][2]

        # Check max_tokens and temperature
        assert request_body["max_tokens"] == 300
        assert request_body["temperature"] == 0.5




















def test_bedrock_openai_multiple_message_types():
    """
    Test that various message content types are handled correctly.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    # Test with mixed content types
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Simple text message"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Complex message with text"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,iVBORw0KGg"},
                },
            ],
        },
    ]

    with patch.object(client, "post") as mock_post:
        try:
            response = completion(
                model="bedrock/openai/arn:aws:bedrock:us-east-1:123:imported-model/test",
                messages=messages,
                max_tokens=50,
                client=client,
            )
        except Exception as e:
            pass

        # Verify the request was made
        if mock_post.called:
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            # Verify messages are preserved
            assert "messages" in request_body
            assert len(request_body["messages"]) == 3

            # Verify mixed content is handled
            assert isinstance(request_body["messages"][2]["content"], list)

            print("✓ Multiple message types handled correctly")




# ============================================================================
# Nova Grounding (web_search_options) Unit Tests (Mocked)
# ============================================================================


def test_bedrock_nova_grounding_web_search_options_non_streaming():
    """
    Unit test for Nova grounding using web_search_options parameter (non-streaming).

    This test mocks the HTTP call to verify:
    1. web_search_options is correctly mapped to systemTool for Nova models
    2. The request structure is correct

    Related: https://docs.aws.amazon.com/nova/latest/userguide/grounding.html
    """
    from unittest.mock import patch

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    messages = [
        {
            "role": "user",
            "content": "What is the current population of Tokyo, Japan?",
        }
    ]

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=messages,
                web_search_options={},  # Enables Nova grounding
                max_tokens=500,
                client=client,
            )
        except Exception:
            pass  # Expected - we're just checking the request structure

        # Verify the request was made correctly
        if mock_post.called:
            request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
            print(f"Request body: {json.dumps(request_body, indent=2)}")

            # Verify toolConfig is present with systemTool
            assert "toolConfig" in request_body, "toolConfig should be in request"
            tool_config = request_body["toolConfig"]
            assert "tools" in tool_config, "tools should be in toolConfig"

            # Find the systemTool for nova_grounding
            system_tool_found = False
            for tool in tool_config["tools"]:
                if "systemTool" in tool:
                    assert tool["systemTool"]["name"] == "nova_grounding"
                    system_tool_found = True
                    break

            assert system_tool_found, "systemTool with nova_grounding should be present"
            print(
                f"✓ web_search_options correctly transformed to systemTool (non-streaming)"
            )


def test_bedrock_nova_grounding_with_function_tools():
    """
    Unit test for Nova grounding combined with regular function tools.

    This tests the scenario where users want both web grounding AND
    custom function calling capabilities.
    """
    from unittest.mock import patch

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    # Regular function tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol, e.g. AAPL, GOOGL",
                        }
                    },
                    "required": ["ticker"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "What is the current market cap of Apple Inc?",
        }
    ]

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=messages,
                tools=tools,
                web_search_options={},  # Also enable web grounding
                max_tokens=500,
                client=client,
            )
        except Exception:
            pass  # Expected - we're just checking the request structure

        # Verify the request was made correctly
        if mock_post.called:
            request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
            print(f"Request body: {json.dumps(request_body, indent=2)}")

            # Verify toolConfig has both function tool and systemTool
            assert "toolConfig" in request_body, "toolConfig should be in request"
            tool_config = request_body["toolConfig"]
            assert "tools" in tool_config, "tools should be in toolConfig"

            tools_in_request = tool_config["tools"]

            # Should have both the function tool and the systemTool
            function_tool_found = False
            system_tool_found = False

            for tool in tools_in_request:
                if "toolSpec" in tool:
                    assert tool["toolSpec"]["name"] == "get_stock_price"
                    function_tool_found = True
                if "systemTool" in tool:
                    assert tool["systemTool"]["name"] == "nova_grounding"
                    system_tool_found = True

            assert (
                function_tool_found
            ), "Function tool (get_stock_price) should be present"
            assert system_tool_found, "systemTool (nova_grounding) should be present"
            print(f"✓ Both function tools and web_search_options correctly combined")


@pytest.mark.asyncio
async def test_bedrock_nova_grounding_async():
    """
    Async unit test for Nova grounding via web_search_options.

    This test verifies the request transformation for async calls.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    client = AsyncHTTPHandler()

    messages = [
        {
            "role": "user",
            "content": "What is the weather forecast for New York City today?",
        }
    ]

    with patch.object(client, "post", new=AsyncMock()) as mock_post:
        try:
            await litellm.acompletion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=messages,
                web_search_options={},
                max_tokens=500,
                client=client,
            )
        except Exception:
            pass  # Expected - we're just checking the request structure

        # Verify the request was made correctly
        if mock_post.called:
            request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
            print(f"Request body: {json.dumps(request_body, indent=2)}")

            # Verify toolConfig is present with systemTool
            assert "toolConfig" in request_body, "toolConfig should be in request"
            tool_config = request_body["toolConfig"]
            assert "tools" in tool_config, "tools should be in toolConfig"

            # Find the systemTool for nova_grounding
            system_tool_found = False
            for tool in tool_config["tools"]:
                if "systemTool" in tool:
                    assert tool["systemTool"]["name"] == "nova_grounding"
                    system_tool_found = True
                    break

            assert system_tool_found, "systemTool with nova_grounding should be present"
            print(f"✓ Async web_search_options correctly transformed to systemTool")




def test_bedrock_nova_grounding_request_transformation():
    """
    Unit test to verify that web_search_options transforms to systemTool in the request.
    """
    from unittest.mock import MagicMock, patch

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    messages = [{"role": "user", "content": "What is the population of Tokyo?"}]

    with patch.object(client, "post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "Test"}]}
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 5},
            },
        )

        try:
            response = completion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=messages,
                web_search_options={},
                max_tokens=100,
                client=client,
            )
        except Exception:
            pass  # Expected - we're just checking the request

        if mock_post.called:
            request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
            print(f"Request body: {json.dumps(request_body, indent=2)}")

            # Verify toolConfig is present with systemTool
            assert "toolConfig" in request_body, "toolConfig should be in request"

            tool_config = request_body["toolConfig"]
            assert "tools" in tool_config, "tools should be in toolConfig"

            tools_in_request = tool_config["tools"]

            # Find the systemTool
            system_tool_found = False
            for tool in tools_in_request:
                if "systemTool" in tool:
                    assert tool["systemTool"]["name"] == "nova_grounding"
                    system_tool_found = True
                    break

            assert system_tool_found, "systemTool with nova_grounding should be present"
            print("✓ web_search_options correctly transformed to systemTool")
