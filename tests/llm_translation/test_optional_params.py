#### What this tests ####
#    This tests if get_optional_params works as expected
import asyncio
import inspect
import time
import traceback

import pytest

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

    response = litellm.completion(model="gpt-3.5-turbo", messages=new_messages, supports_system_message=False)

    assert isinstance(response, litellm.ModelResponse)


# test_azure_gpt_optional_params_gpt_vision()


# test_azure_gpt_optional_params_gpt_vision_with_extra_body()


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
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()) as mock_response:
        try:
            response = litellm.completion(
                model="command-r-08-2024",
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
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()) as mock_response:
        try:
            response = litellm.completion(
                model="command-r-08-2024",
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


def test_dynamic_drop_params_parallel_tool_calls():
    """
    https://github.com/BerriAI/litellm/issues/4584
    """
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()) as mock_response:
        try:
            response = litellm.completion(
                model="command-r-08-2024",
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


def test_dynamic_drop_additional_params_e2e():
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=MagicMock()) as mock_response:
        try:
            response = litellm.completion(
                model="command-r-08-2024",
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
