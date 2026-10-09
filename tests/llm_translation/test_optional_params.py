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


def _drop_warnings(mock_warning):
    """The formatted drop warnings only.

    `verbose_logger.warning` is shared, so asserting on a raw call count would
    make these tests fail the day something unrelated in `get_optional_params`
    logs a warning. Filter to this message and format it, since the assertions
    are about what the message tells the caller.
    """
    messages = []
    for call in mock_warning.call_args_list:
        template = call.args[0]
        if "is not sending" not in template:
            continue
        messages.append(template % call.args[1:])
    return messages


@pytest.fixture
def reset_unvalidated_param_warnings():
    """The drop warning is deduped for the life of the process.

    A test that asserts it fires has to start from an empty set, and must not
    leave its own entries behind for the next one.
    """
    from litellm.utils import _unvalidated_param_warned

    _unvalidated_param_warned.clear()
    yield
    _unvalidated_param_warned.clear()


def test_user_param_dropped_warns_for_non_catalog_model(reset_unvalidated_param_warnings):
    """A param that is exempt from the unsupported-param error is still dropped.

    `user` is only a supported param for models in OpenAI's catalog, and it is in
    PROVIDER_UNVALIDATED_PARAMS, so for any other model it is neither sent nor
    refused. Reported as #45015, where per-customer spend attribution recorded
    nothing and the caller had no way to see why.
    """
    with patch("litellm.utils.verbose_logger.warning") as mock_warning:
        optional_params = get_optional_params(
            model="Qwen3.6-35B-A3B",
            custom_llm_provider="openai",
            user="customer-1",
        )

    assert "user" not in optional_params

    messages = _drop_warnings(mock_warning)
    assert len(messages) == 1
    assert "`user`" in messages[0]
    assert "Qwen3.6-35B-A3B" in messages[0]
    # The workaround is the point of the message; without it the warning only
    # tells the caller that something was lost, not what to do instead.
    assert "x-litellm-end-user-id" in messages[0]


def test_user_param_drop_warning_is_deduped(reset_unvalidated_param_warnings):
    """Once per (param, provider, model), not once per request."""
    with patch("litellm.utils.verbose_logger.warning") as mock_warning:
        for _ in range(3):
            get_optional_params(
                model="Qwen3.6-35B-A3B",
                custom_llm_provider="openai",
                user="customer-1",
            )

    assert len(_drop_warnings(mock_warning)) == 1


def test_user_param_not_warned_when_the_model_supports_it(reset_unvalidated_param_warnings):
    """A catalog model sends `user`, so there is nothing to warn about."""
    with patch("litellm.utils.verbose_logger.warning") as mock_warning:
        optional_params = get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            user="customer-1",
        )

    assert optional_params["user"] == "customer-1"
    assert _drop_warnings(mock_warning) == []


def test_exempt_transport_param_without_a_hint_does_not_warn(reset_unvalidated_param_warnings):
    """`stream` is exempt from the same check but must stay silent.

    It is a transport control rather than something a caller tracks the effect
    of, so it is deliberately absent from UNVALIDATED_PARAM_DROP_HINTS. Without
    this test the warning could be widened to every exempt param and nothing
    would fail.
    """
    with patch("litellm.utils.verbose_logger.warning") as mock_warning:
        get_optional_params(
            model="Qwen3.6-35B-A3B",
            custom_llm_provider="openai",
            stream=True,
        )

    assert _drop_warnings(mock_warning) == []
