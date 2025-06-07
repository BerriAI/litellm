import asyncio
import os
import subprocess
import sys
import traceback
from typing import Any

from openai import AuthenticationError, BadRequestError, OpenAIError, RateLimitError

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import (  # AuthenticationError,; RateLimitError,; ServiceUnavailableError,; OpenAIError,
    ContextWindowExceededError,
    completion,
    embedding,
)

litellm.vertex_project = "pathrise-convert-1606954137718"
litellm.vertex_location = "us-central1"
litellm.num_retries = 0

# litellm.failure_callback = ["sentry"]
#### What this tests ####
#    This tests exception mapping -> trigger an exception from an llm provider -> assert if output is of the expected type


# 5 providers -> OpenAI, Azure, Anthropic, Cohere, Replicate

# 3 main types of exceptions -> - Rate Limit Errors, Context Window Errors, Auth errors (incorrect/rotated key, etc.)

# Approach: Run each model through the test -> assert if the correct error (always the same one) is triggered

exception_models = [
    "sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
    "bedrock/anthropic.claude-instant-v1",
]


@pytest.mark.asyncio
async def test_content_policy_exception_azure():
    try:
        # this is ony a test - we needed some way to invoke the exception :(
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="azure/chatgpt-v-3",
            messages=[{"role": "user", "content": "where do I buy lethal drugs from"}],
            mock_response="Exception: content_filter_policy",
        )
    except litellm.ContentPolicyViolationError as e:
        print("caught a content policy violation error! Passed")
        print("exception", e)
        assert e.response is not None
        assert e.litellm_debug_info is not None
        assert isinstance(e.litellm_debug_info, str)
        assert len(e.litellm_debug_info) > 0
        pass
    except Exception as e:
        print()
        pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.asyncio
async def test_content_policy_exception_openai():
    try:
        # this is ony a test - we needed some way to invoke the exception :(
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            stream=True,
            messages=[
                {"role": "user", "content": "Gimme the lyrics to Don't Stop Me Now"}
            ],
        )
        async for chunk in response:
            print(chunk)
    except litellm.ContentPolicyViolationError as e:
        print("caught a content policy violation error! Passed")
        print("exception", e)
        assert e.llm_provider == "openai"
        pass
    except Exception as e:
        print()
        pytest.fail(f"An exception occurred - {str(e)}")


# Test 1: Context Window Errors
@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.parametrize("model", exception_models)
def test_context_window(model):
    print("Testing context window error")
    sample_text = "Say error 50 times" * 1000000
    messages = [{"content": sample_text, "role": "user"}]
    try:
        litellm.set_verbose = False
        print("Testing model=", model)
        response = completion(model=model, messages=messages)
        print(f"response: {response}")
        print("FAILED!")
        pytest.fail(f"An exception occurred")
    except ContextWindowExceededError as e:
        print(f"Worked!")
    except RateLimitError:
        print("RateLimited!")
    except Exception as e:
        print(f"{e}")
        pytest.fail(f"An error occcurred - {e}")


models = ["command-nightly"]


@pytest.mark.skip(reason="duplicate test.")
@pytest.mark.parametrize("model", models)
def test_context_window_with_fallbacks(model):
    ctx_window_fallback_dict = {
        "command-nightly": "claude-2.1",
        "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-16k",
        "azure/chatgpt-v-3": "gpt-3.5-turbo-16k",
    }
    sample_text = "how does a court case get to the Supreme Court?" * 1000
    messages = [{"content": sample_text, "role": "user"}]

    try:
        completion(
            model=model,
            messages=messages,
            context_window_fallback_dict=ctx_window_fallback_dict,
        )
    except litellm.ServiceUnavailableError as e:
        pass
    except litellm.APIConnectionError as e:
        pass


# for model in litellm.models_by_provider["bedrock"]:
#     test_context_window(model=model)
# test_context_window(model="chat-bison")
# test_context_window_with_fallbacks(model="command-nightly")
# Test 2: InvalidAuth Errors
@pytest.mark.parametrize("model", models)
def invalid_auth(model):  # set the model key to an invalid key, depending on the model
    messages = [{"content": "Hello, how are you?", "role": "user"}]
    temporary_key = None
    try:
        if model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-instruct":
            temporary_key = os.environ["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = "bad-key"
        elif "bedrock" in model:
            temporary_aws_access_key = os.environ["AWS_ACCESS_KEY_ID"]
            os.environ["AWS_ACCESS_KEY_ID"] = "bad-key"
            temporary_aws_region_name = os.environ["AWS_REGION_NAME"]
            os.environ["AWS_REGION_NAME"] = "bad-key"
            temporary_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = "bad-key"
        elif model == "azure/chatgpt-v-3":
            temporary_key = os.environ["AZURE_API_KEY"]
            os.environ["AZURE_API_KEY"] = "bad-key"
        elif model == "claude-3-5-haiku-20241022":
            temporary_key = os.environ["ANTHROPIC_API_KEY"]
            os.environ["ANTHROPIC_API_KEY"] = "bad-key"
        elif model == "command-nightly":
            temporary_key = os.environ["COHERE_API_KEY"]
            os.environ["COHERE_API_KEY"] = "bad-key"
        elif "j2" in model:
            temporary_key = os.environ["AI21_API_KEY"]
            os.environ["AI21_API_KEY"] = "bad-key"
        elif "togethercomputer" in model:
            temporary_key = os.environ["TOGETHERAI_API_KEY"]
            os.environ["TOGETHERAI_API_KEY"] = (
                "84060c79880fc49df126d3e87b53f8a463ff6e1c6d27fe64207cde25cdfcd1f24a"
            )
        elif model in litellm.openrouter_models:
            temporary_key = os.environ["OPENROUTER_API_KEY"]
            os.environ["OPENROUTER_API_KEY"] = "bad-key"
        elif model in litellm.aleph_alpha_models:
            temporary_key = os.environ["ALEPH_ALPHA_API_KEY"]
            os.environ["ALEPH_ALPHA_API_KEY"] = "bad-key"
        elif model in litellm.nlp_cloud_models:
            temporary_key = os.environ["NLP_CLOUD_API_KEY"]
            os.environ["NLP_CLOUD_API_KEY"] = "bad-key"
        elif (
            model
            == "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
        ):
            temporary_key = os.environ["REPLICATE_API_KEY"]
            os.environ["REPLICATE_API_KEY"] = "bad-key"
        print(f"model: {model}")
        response = completion(model=model, messages=messages)
        print(f"response: {response}")
    except AuthenticationError as e:
        print(f"AuthenticationError Caught Exception - {str(e)}")
    except (
        OpenAIError
    ) as e:  # is at least an openai error -> in case of random model errors - e.g. overloaded server
        print(f"OpenAIError Caught Exception - {e}")
    except Exception as e:
        print(type(e))
        print(type(AuthenticationError))
        print(e.__class__.__name__)
        print(f"Uncaught Exception - {e}")
        pytest.fail(f"Error occurred: {e}")
    if temporary_key != None:  # reset the key
        if model == "gpt-3.5-turbo":
            os.environ["OPENAI_API_KEY"] = temporary_key
        elif model == "chatgpt-test":
            os.environ["AZURE_API_KEY"] = temporary_key
            azure = True
        elif model == "claude-3-5-haiku-20241022":
            os.environ["ANTHROPIC_API_KEY"] = temporary_key
        elif model == "command-nightly":
            os.environ["COHERE_API_KEY"] = temporary_key
        elif (
            model
            == "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
        ):
            os.environ["REPLICATE_API_KEY"] = temporary_key
        elif "j2" in model:
            os.environ["AI21_API_KEY"] = temporary_key
        elif "togethercomputer" in model:
            os.environ["TOGETHERAI_API_KEY"] = temporary_key
        elif model in litellm.aleph_alpha_models:
            os.environ["ALEPH_ALPHA_API_KEY"] = temporary_key
        elif model in litellm.nlp_cloud_models:
            os.environ["NLP_CLOUD_API_KEY"] = temporary_key
        elif "bedrock" in model:
            os.environ["AWS_ACCESS_KEY_ID"] = temporary_aws_access_key
            os.environ["AWS_REGION_NAME"] = temporary_aws_region_name
            os.environ["AWS_SECRET_ACCESS_KEY"] = temporary_secret_key
    return


# for model in litellm.models_by_provider["bedrock"]:
#     invalid_auth(model=model)
# invalid_auth(model="command-nightly")


# Test 3: Invalid Request Error
@pytest.mark.parametrize("model", models)
def test_invalid_request_error(model):
    messages = [{"content": "hey, how's it going?", "role": "user"}]

    with pytest.raises(BadRequestError):
        completion(model=model, messages=messages, max_tokens="hello world")


@pytest.mark.parametrize(
    "provider", ["predibase", "vertex_ai_beta", "anthropic", "databricks", "watsonx", "fireworks_ai"]
)
def test_exception_mapping(provider):
    """
    For predibase, run through a set of mock exceptions

    assert that they are being mapped correctly
    """
    litellm.set_verbose = True
    error_map = {
        400: litellm.BadRequestError,
        401: litellm.AuthenticationError,
        404: litellm.NotFoundError,
        408: litellm.Timeout,
        429: litellm.RateLimitError,
        500: litellm.InternalServerError,
        503: litellm.ServiceUnavailableError,
    }

    for code, expected_exception in error_map.items():
        mock_response = Exception()
        setattr(mock_response, "text", "This is an error message")
        setattr(mock_response, "llm_provider", provider)
        setattr(mock_response, "status_code", code)

        response: Any = None
        try:
            response = completion(
                model="{}/test-model".format(provider),
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response=mock_response,
            )
        except expected_exception:
            continue
        except Exception as e:
            traceback.print_exc()
            response = "{}".format(str(e))
        pytest.fail(
            "Did not raise expected exception. Expected={}, Return={},".format(
                expected_exception, response
            )
        )

    pass


def test_fireworks_ai_exception_mapping():
    """
    Comprehensive test for Fireworks AI exception mapping, including:
    1. Standard 429 rate limit errors
    2. Text-based rate limit detection (the main issue fixed)
    3. Generic 400 errors that should NOT be rate limits
    4. ExceptionCheckers utility function
    
    Related to: https://github.com/BerriAI/litellm/pull/11455
    Based on Fireworks AI documentation: https://docs.fireworks.ai/tools-sdks/python-client/api-reference
    """
    import litellm
    from litellm.llms.fireworks_ai.common_utils import FireworksAIException
    from litellm.litellm_core_utils.exception_mapping_utils import ExceptionCheckers
    
    # Test scenarios covering all important cases
    test_scenarios = [
        {
            "name": "Standard 429 rate limit with proper status code",
            "status_code": 429,
            "message": "Rate limit exceeded. Please try again in 60 seconds.",
            "expected_exception": litellm.RateLimitError,
        },
        {
            "name": "Status 400 with rate limit text (the main issue fixed)",
            "status_code": 400,
            "message": '{"error":{"object":"error","type":"invalid_request_error","message":"rate limit exceeded, please try again later"}}',
            "expected_exception": litellm.RateLimitError,
        },
        {
            "name": "Status 400 with generic invalid request (should NOT be rate limit)",
            "status_code": 400,
            "message": '{"error":{"type":"invalid_request_error","message":"Invalid parameter value"}}',
            "expected_exception": litellm.BadRequestError,
        },
    ]
    
    # Test each scenario
    for scenario in test_scenarios:
        mock_exception = FireworksAIException(
            status_code=scenario["status_code"],
            message=scenario["message"],
            headers={}
        )
        
        try:
            response = litellm.completion(
                model="fireworks_ai/llama-v3p1-70b-instruct",
                messages=[{"role": "user", "content": "Hello"}],
                mock_response=mock_exception,
            )
            pytest.fail(f"Expected {scenario['expected_exception'].__name__} to be raised")
        except scenario["expected_exception"] as e:
            if scenario["expected_exception"] == litellm.RateLimitError:
                assert "rate limit" in str(e).lower() or "429" in str(e)
        except Exception as e:
            pytest.fail(f"Expected {scenario['expected_exception'].__name__} but got {type(e).__name__}: {e}")
    
    # Test ExceptionCheckers.is_error_str_rate_limit() method directly
    
    # Test cases that should return True (rate limit detected)
    rate_limit_strings = [
        "429 rate limit exceeded",
        "Rate limit exceeded, please try again later", 
        "RATE LIMIT ERROR",
        "Error 429: rate limit",
        '{"error":{"type":"invalid_request_error","message":"rate limit exceeded, please try again later"}}',
        "HTTP 429 Too Many Requests",
    ]
    
    for error_str in rate_limit_strings:
        assert ExceptionCheckers.is_error_str_rate_limit(error_str), f"Should detect rate limit in: {error_str}"
    
    # Test cases that should return False (not rate limit)
    non_rate_limit_strings = [
        "400 Bad Request",
        "Authentication failed", 
        "Invalid model specified",
        "Context window exceeded",
        "Internal server error",
        "",
        "Some other error message",
    ]
    
    for error_str in non_rate_limit_strings:
        assert not ExceptionCheckers.is_error_str_rate_limit(error_str), f"Should NOT detect rate limit in: {error_str}"
    
    # Test edge cases
    assert not ExceptionCheckers.is_error_str_rate_limit(None)  # type: ignore
    assert not ExceptionCheckers.is_error_str_rate_limit(42)  # type: ignore


from typing import Optional, Union

from openai import AsyncOpenAI, OpenAI


def _pre_call_utils(
    call_type: str,
    data: dict,
    client: Union[OpenAI, AsyncOpenAI],
    sync_mode: bool,
    streaming: Optional[bool],
):
    if call_type == "embedding":
        data["input"] = "Hello world!"
        mapped_target: Any = client.embeddings.with_raw_response
        if sync_mode:
            original_function = litellm.embedding
        else:
            original_function = litellm.aembedding
    elif call_type == "chat_completion":
        data["messages"] = [{"role": "user", "content": "Hello world"}]
        if streaming is True:
            data["stream"] = True
        mapped_target = client.chat.completions.with_raw_response  # type: ignore
        if sync_mode:
            original_function = litellm.completion
        else:
            original_function = litellm.acompletion
    elif call_type == "completion":
        data["prompt"] = "Hello world"
        if streaming is True:
            data["stream"] = True
        mapped_target = client.completions.with_raw_response  # type: ignore
        if sync_mode:
            original_function = litellm.text_completion
        else:
            original_function = litellm.atext_completion

    return data, original_function, mapped_target


def _pre_call_utils_httpx(
    call_type: str,
    data: dict,
    client: Union[HTTPHandler, AsyncHTTPHandler],
    sync_mode: bool,
    streaming: Optional[bool],
):
    mapped_target: Any = client.client
    if call_type == "embedding":
        data["input"] = "Hello world!"

        if sync_mode:
            original_function = litellm.embedding
        else:
            original_function = litellm.aembedding
    elif call_type == "chat_completion":
        data["messages"] = [{"role": "user", "content": "Hello world"}]
        if streaming is True:
            data["stream"] = True

        if sync_mode:
            original_function = litellm.completion
        else:
            original_function = litellm.acompletion
    elif call_type == "completion":
        data["prompt"] = "Hello world"
        if streaming is True:
            data["stream"] = True
        if sync_mode:
            original_function = litellm.text_completion
        else:
            original_function = litellm.atext_completion

    return data, original_function, mapped_target


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.parametrize(
    "provider, model, call_type, streaming",
    [
        ("openai", "text-embedding-ada-002", "embedding", None),
        ("openai", "gpt-3.5-turbo", "chat_completion", False),
        ("openai", "gpt-3.5-turbo", "chat_completion", True),
        ("openai", "gpt-3.5-turbo-instruct", "completion", True),
        ("azure", "azure/chatgpt-v-3", "chat_completion", True),
        ("azure", "azure/text-embedding-ada-002", "embedding", True),
        ("azure", "azure_text/gpt-3.5-turbo-instruct", "completion", True),
    ],
)
@pytest.mark.asyncio
async def test_exception_with_headers(sync_mode, provider, model, call_type, streaming):
    """
    User feedback: litellm says "No deployments available for selected model, Try again in 60 seconds"
    but Azure says to retry in at most 9s

    ```
    {"message": "litellm.proxy.proxy_server.embeddings(): Exception occured - No deployments available for selected model, Try again in 60 seconds. Passed model=text-embedding-ada-002. pre-call-checks=False, allowed_model_region=n/a, cooldown_list=[('b49cbc9314273db7181fe69b1b19993f04efb88f2c1819947c538bac08097e4c', {'Exception Received': 'litellm.RateLimitError: AzureException RateLimitError - Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-09-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please retry after 9 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.', 'Status Code': '429'})]", "level": "ERROR", "timestamp": "2024-08-22T03:25:36.900476"}
    ```
    """
    print(f"Received args: {locals()}")
    import openai

    if sync_mode:
        if provider == "openai":
            openai_client = openai.OpenAI(api_key="")
        elif provider == "azure":
            openai_client = openai.AzureOpenAI(
                api_key="", base_url="", api_version=litellm.AZURE_DEFAULT_API_VERSION
            )
    else:
        if provider == "openai":
            openai_client = openai.AsyncOpenAI(api_key="")
        elif provider == "azure":
            openai_client = openai.AsyncAzureOpenAI(
                api_key="", base_url="", api_version=litellm.AZURE_DEFAULT_API_VERSION
            )

    data = {"model": model}
    data, original_function, mapped_target = _pre_call_utils(
        call_type=call_type,
        data=data,
        client=openai_client,
        sync_mode=sync_mode,
        streaming=streaming,
    )

    cooldown_time = 30.0

    def _return_exception(*args, **kwargs):
        import datetime

        from httpx import Headers, Request, Response

        kwargs = {
            "request": Request("POST", "https://www.google.com"),
            "message": "Error code: 429 - Rate Limit Error!",
            "body": {"detail": "Rate Limit Error!"},
            "code": None,
            "param": None,
            "type": None,
            "response": Response(
                status_code=429,
                headers=Headers(
                    {
                        "date": "Sat, 21 Sep 2024 22:56:53 GMT",
                        "server": "uvicorn",
                        "retry-after": "30",
                        "content-length": "30",
                        "content-type": "application/json",
                    }
                ),
                request=Request("POST", "http://0.0.0.0:9000/chat/completions"),
            ),
            "status_code": 429,
            "request_id": None,
        }

        exception = Exception()
        for k, v in kwargs.items():
            setattr(exception, k, v)
        raise exception

    with patch.object(
        mapped_target,
        "create",
        side_effect=_return_exception,
    ):
        new_retry_after_mock_client = MagicMock(return_value=-1)

        litellm.utils._get_retry_after_from_exception_header = (
            new_retry_after_mock_client
        )

        exception_raised = False
        try:
            if sync_mode:
                resp = original_function(**data, client=openai_client)
                if streaming:
                    for chunk in resp:
                        continue
            else:
                resp = await original_function(**data, client=openai_client)

                if streaming:
                    async for chunk in resp:
                        continue

        except litellm.RateLimitError as e:
            exception_raised = True
            assert e.litellm_response_headers is not None
            assert int(e.litellm_response_headers["retry-after"]) == cooldown_time

        if exception_raised is False:
            print(resp)
        assert exception_raised


def test_openai_gateway_timeout_error():
    """
    Test that the OpenAI gateway timeout error is raised
    """
    openai_client = OpenAI()
    mapped_target = openai_client.chat.completions.with_raw_response  # type: ignore
    def _return_exception(*args, **kwargs):
        import datetime

        from httpx import Headers, Request, Response

        kwargs = {
            "request": Request("POST", "https://www.google.com"),
            "message": "Error code: 504 - Gateway Timeout Error!",
            "body": {"detail": "Gateway Timeout Error!"},
            "code": None,
            "param": None,
            "type": None,
            "response": Response(
                status_code=504,
                headers=Headers(
                    {
                        "date": "Sat, 21 Sep 2024 22:56:53 GMT",
                        "server": "uvicorn",
                        "content-length": "30",
                        "content-type": "application/json",
                    }
                ),
                request=Request("POST", "http://0.0.0.0:9000/chat/completions"),
            ),
            "status_code": 504,
            "request_id": None,
        }

        exception = Exception()
        for k, v in kwargs.items():
            setattr(exception, k, v)
        raise exception

    try: 
        with patch.object(
            mapped_target,
            "create",
            side_effect=_return_exception,
        ):
            litellm.completion(model="openai/gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}], client=openai_client)
        pytest.fail("Expected to raise Timeout")
    except litellm.Timeout as e:
        assert e.status_code == 504


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "provider, model, call_type",
    [
        ("anthropic", "claude-3-haiku-20240307", "chat_completion"),
    ],
)
@pytest.mark.asyncio
async def test_exception_with_headers_httpx(
    sync_mode, provider, model, call_type, streaming
):
    """
    User feedback: litellm says "No deployments available for selected model, Try again in 60 seconds"
    but Azure says to retry in at most 9s

    ```
    {"message": "litellm.proxy.proxy_server.embeddings(): Exception occured - No deployments available for selected model, Try again in 60 seconds. Passed model=text-embedding-ada-002. pre-call-checks=False, allowed_model_region=n/a, cooldown_list=[('b49cbc9314273db7181fe69b1b19993f04efb88f2c1819947c538bac08097e4c', {'Exception Received': 'litellm.RateLimitError: AzureException RateLimitError - Requests to the Embeddings_Create Operation under Azure OpenAI API version 2023-09-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please retry after 9 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.', 'Status Code': '429'})]", "level": "ERROR", "timestamp": "2024-08-22T03:25:36.900476"}
    ```
    """
    print(f"Received args: {locals()}")
    import openai

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    data = {"model": model}
    data, original_function, mapped_target = _pre_call_utils_httpx(
        call_type=call_type,
        data=data,
        client=client,
        sync_mode=sync_mode,
        streaming=streaming,
    )

    cooldown_time = 30.0

    def _return_exception(*args, **kwargs):
        import datetime

        from httpx import Headers, HTTPStatusError, Request, Response

        # Create the Request object
        request = Request("POST", "http://0.0.0.0:9000/chat/completions")

        # Create the Response object with the necessary headers and status code
        response = Response(
            status_code=429,
            headers=Headers(
                {
                    "date": "Sat, 21 Sep 2024 22:56:53 GMT",
                    "server": "uvicorn",
                    "retry-after": "30",
                    "content-length": "30",
                    "content-type": "application/json",
                }
            ),
            request=request,
        )

        # Create and raise the HTTPStatusError exception
        raise HTTPStatusError(
            message="Error code: 429 - Rate Limit Error!",
            request=request,
            response=response,
        )

    with patch.object(
        mapped_target,
        "send",
        side_effect=_return_exception,
    ):
        new_retry_after_mock_client = MagicMock(return_value=-1)

        litellm.utils._get_retry_after_from_exception_header = (
            new_retry_after_mock_client
        )

        exception_raised = False
        try:
            if sync_mode:
                resp = original_function(**data, client=client)
                if streaming:
                    for chunk in resp:
                        continue
            else:
                resp = await original_function(**data, client=client)

                if streaming:
                    async for chunk in resp:
                        continue

        except litellm.RateLimitError as e:
            exception_raised = True
            assert (
                e.litellm_response_headers is not None
            ), "litellm_response_headers is None"
            print("e.litellm_response_headers", e.litellm_response_headers)
            assert int(e.litellm_response_headers["retry-after"]) == cooldown_time

        if exception_raised is False:
            print(resp)
        assert exception_raised


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["azure/chatgpt-v-3", "openai/gpt-3.5-turbo"])
async def test_bad_request_error_contains_httpx_response(model):
    """
    Test that the BadRequestError contains the httpx response

    Relevant issue: https://github.com/BerriAI/litellm/issues/6732
    """
    try:
        await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Hello world"}],
            bad_arg="bad_arg",
        )
        pytest.fail("Expected to raise BadRequestError")
    except litellm.BadRequestError as e:
        print("e.response", e.response)
        print("vars(e.response)", vars(e.response))
        assert e.response is not None


def test_exceptions_base_class():
    try:
        raise litellm.RateLimitError(
            message="BedrockException: Rate Limit Error",
            model="model",
            llm_provider="bedrock",
        )
    except litellm.RateLimitError as e:
        assert isinstance(e, litellm.RateLimitError)
        assert e.code == "429"
        assert e.type == "throttling_error"


def test_context_window_exceeded_error_from_litellm_proxy():
    from httpx import Response
    from litellm.litellm_core_utils.exception_mapping_utils import (
        extract_and_raise_litellm_exception,
    )

    args = {
        "response": Response(status_code=400, text="Bad Request"),
        "error_str": "Error code: 400 - {'error': {'message': \"litellm.ContextWindowExceededError: litellm.BadRequestError: this is a mock context window exceeded error\\nmodel=gpt-3.5-turbo. context_window_fallbacks=None. fallbacks=None.\\n\\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks\\nReceived Model Group=gpt-3.5-turbo\\nAvailable Model Group Fallbacks=None\", 'type': None, 'param': None, 'code': '400'}}",
        "model": "gpt-3.5-turbo",
        "custom_llm_provider": "litellm_proxy",
    }
    with pytest.raises(litellm.ContextWindowExceededError):
        extract_and_raise_litellm_exception(**args)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("stream_mode", [True, False])
@pytest.mark.parametrize("model", ["azure/gpt-4o-new-test"])  # "gpt-4o-mini",
@pytest.mark.asyncio
async def test_exception_bubbling_up(sync_mode, stream_mode, model):
    """
    make sure code, param, and type are bubbled up
    """
    import litellm

    litellm.set_verbose = True
    with pytest.raises(Exception) as exc_info:
        if sync_mode:
            litellm.completion(
                model=model,
                messages=[{"role": "usera", "content": "hi"}],
                stream=stream_mode,
                sync_stream=sync_mode,
            )
        else:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "usera", "content": "hi"}],
                stream=stream_mode,
                sync_stream=sync_mode,
            )

    assert exc_info.value.code == "invalid_value"
    assert exc_info.value.param is not None
    assert exc_info.value.type == "invalid_request_error"