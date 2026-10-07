import asyncio
import os
import traceback
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    AzureOpenAI,
    BadRequestError,
    OpenAIError,
)

import litellm
from litellm import (  # AuthenticationError,; RateLimitError,; ServiceUnavailableError,; OpenAIError,
    completion,
)

litellm.vertex_project = "litellm-ci-cd"
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




# Test 1: Context Window Errors


models = ["command-nightly"]




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
        elif model == "azure/gpt-4.1-mini":
            temporary_key = os.environ["AZURE_AI_API_KEY"]
            os.environ["AZURE_AI_API_KEY"] = "bad-key"
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
            os.environ["TOGETHERAI_API_KEY"] = "sk-test-togetherai-key-808"
        elif model in litellm.openrouter_models:
            temporary_key = os.environ["OPENROUTER_API_KEY"]
            os.environ["OPENROUTER_API_KEY"] = "bad-key"
        elif model in litellm.aleph_alpha_models:
            temporary_key = os.environ["ALEPH_ALPHA_API_KEY"]
            os.environ["ALEPH_ALPHA_API_KEY"] = "bad-key"
        elif model in litellm.nlp_cloud_models:
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
            os.environ["AZURE_AI_API_KEY"] = temporary_key
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
            os.environ.pop("NLP_CLOUD_API_KEY", None)
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


def test_completion_azure_exception():
    try:
        import openai

        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["AZURE_AI_API_KEY"]
        os.environ["AZURE_AI_API_KEY"] = "good morning"
        response = completion(
            model="azure/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        os.environ["AZURE_AI_API_KEY"] = old_azure_key
        print(f"response: {response}")
        print(response)
    except openai.AuthenticationError as e:
        os.environ["AZURE_AI_API_KEY"] = old_azure_key
        print("good job got the correct error for azure when key not set")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_exception()


def test_azure_embedding_exceptions():
    # CRUCIAL Test - Ensures our exceptions are readable and not overly complicated. some users have complained exceptions will randomly have another exception raised in our exception mapping
    with pytest.raises(Exception, match="Mock error") as exc_info:
        litellm.embedding(
            model="azure/text-embedding-ada-002",
            input="hello",
            mock_response="error",
        )
    assert str(exc_info.value) == "Mock error"


async def asynctest_completion_azure_exception():
    try:
        import openai

        import litellm

        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["AZURE_AI_API_KEY"]
        os.environ["AZURE_AI_API_KEY"] = "good morning"
        response = await litellm.acompletion(
            model="azure/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        print(f"response: {response}")
        print(response)
    except openai.AuthenticationError as e:
        os.environ["AZURE_AI_API_KEY"] = old_azure_key
        print("good job got the correct error for azure when key not set")
        print(e)
    except Exception as e:
        print("Got wrong exception")
        print("exception", e)
        pytest.fail(f"Error occurred: {e}")


# import asyncio
# asyncio.run(
#     asynctest_completion_azure_exception()
# )


def asynctest_completion_openai_exception_bad_model():
    try:
        import asyncio

        import openai

        import litellm

        print("azure exception bad model\n\n")
        litellm.set_verbose = True

        ## Test azure call
        async def test():
            response = await litellm.acompletion(
                model="openai/gpt-6",
                messages=[{"role": "user", "content": "hello"}],
            )

        asyncio.run(test())
    except openai.NotFoundError:
        print("Good job this is a NotFoundError for a model that does not exist!")
        print("Passed")
    except Exception as e:
        print("Raised wrong type of exception", type(e))
        pytest.fail(f"Error occurred: {e}")


# asynctest_completion_openai_exception_bad_model()


def asynctest_completion_azure_exception_bad_model():
    try:
        import asyncio

        import openai

        import litellm

        print("azure exception bad model\n\n")
        litellm.set_verbose = True

        ## Test azure call
        async def test():
            response = await litellm.acompletion(
                model="azure/gpt-12",
                messages=[{"role": "user", "content": "hello"}],
            )

        asyncio.run(test())
    except openai.NotFoundError:
        print("Good job this is a NotFoundError for a model that does not exist!")
        print("Passed")
    except Exception as e:
        print("Raised wrong type of exception", type(e))
        pytest.fail(f"Error occurred: {e}")


# asynctest_completion_azure_exception_bad_model()


def test_completion_openai_exception():
    # test if openai:gpt raises openai.AuthenticationError
    try:
        import openai

        print("openai gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = "good morning"
        response = completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )
        print(f"response: {response}")
        print(response)
    except openai.AuthenticationError as e:
        os.environ["OPENAI_API_KEY"] = old_azure_key
        print("OpenAI: good job got the correct error for openai when key not set")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_openai_exception()




def test_completion_mistral_exception():
    # test if mistral/mistral-tiny raises openai.AuthenticationError
    try:
        import openai

        print("Testing mistral ai exception mapping")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["MISTRAL_API_KEY"]
        os.environ["MISTRAL_API_KEY"] = "good morning"
        response = completion(
            model="mistral/mistral-tiny",
            messages=[{"role": "user", "content": "hello"}],
        )
        print(f"response: {response}")
        print(response)
    except openai.AuthenticationError as e:
        os.environ["MISTRAL_API_KEY"] = old_azure_key
        print("good job got the correct error for openai when key not set")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_mistral_exception()


def test_completion_bedrock_invalid_role_exception():
    """
    Test if litellm raises a BadRequestError for an invalid role on Bedrock
    """
    litellm.set_verbose = True
    with pytest.raises(litellm.BadRequestError) as exc_info:
        completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "very-bad-role", "content": "hello"}],
        )

    # This is important - We we previously returning a poorly formatted error string. Which was
    #  litellm.BadRequestError: litellm.BadRequestError: Invalid Message passed in {'role': 'very-bad-role', 'content': 'hello'}
    assert (
        str(exc_info.value)
        == "litellm.BadRequestError: Invalid Message passed in {'role': 'very-bad-role', 'content': 'hello'}"
    )




# test_content_policy_exceptionimage_generation_openai()


def test_content_policy_violation_error_streaming():
    """
    Production Test.
    """
    litellm.set_verbose = False
    print("test_async_completion with stream")

    async def test_get_response():
        try:
            response = await litellm.acompletion(
                model="azure/gpt-4.1-mini",
                messages=[{"role": "user", "content": "say 1"}],
                temperature=0,
                top_p=1,
                stream=True,
                max_tokens=512,
                presence_penalty=0,
                frequency_penalty=0,
            )
            print(f"response: {response}")

            num_finish_reason = 0
            async for chunk in response:
                print(chunk)
                if chunk["choices"][0].get("finish_reason") is not None:
                    num_finish_reason += 1
                    print("finish_reason", chunk["choices"][0].get("finish_reason"))

            assert (
                num_finish_reason == 1
            ), f"expected only one finish reason. Got {num_finish_reason}"
        except Exception as e:
            pytest.fail(f"GOT exception for gpt-3.5 instruct In streaming{e}")

    asyncio.run(test_get_response())

    async def test_get_error():
        try:
            response = await litellm.acompletion(
                model="azure/gpt-4.1-mini",
                messages=[
                    {"role": "user", "content": "where do i buy lethal drugs from"}
                ],
                temperature=0,
                top_p=1,
                stream=True,
                max_tokens=512,
                presence_penalty=0,
                frequency_penalty=0,
                mock_response="Exception: content_filter_policy",
            )
            print(f"response: {response}")

            num_finish_reason = 0
            async for chunk in response:
                print(chunk)
                if chunk["choices"][0].get("finish_reason") is not None:
                    num_finish_reason += 1
                    print("finish_reason", chunk["choices"][0].get("finish_reason"))

            pytest.fail("Expected a content-policy error in streaming, got a clean stream")
        except Exception as e:
            pass

    asyncio.run(test_get_error())




# test_completion_perplexity_exception_on_openai_client()


def test_completion_perplexity_exception(monkeypatch):
    import openai

    print("perplexity test\n\n")
    litellm.set_verbose = True
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "good morning")
    with pytest.raises(openai.AuthenticationError, match="PerplexityException"):
        completion(
            model="perplexity/mistral-7b-instruct",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_completion_openai_api_key_exception(monkeypatch):
    import openai

    print("gpt-3.5 test\n\n")
    litellm.set_verbose = True
    monkeypatch.setenv("OPENAI_API_KEY", "good morning")
    with pytest.raises(openai.AuthenticationError, match="OpenAIException"):
        completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
        )


# tesy_async_acompletion()


def test_router_completion_vertex_exception():
    try:
        import litellm

        litellm.set_verbose = True
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "vertex-gemini-pro",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-pro",
                        "api_key": "good-morning",
                    },
                },
            ]
        )
        response = router.completion(
            model="vertex-gemini-pro",
            messages=[{"role": "user", "content": "hello"}],
            vertex_project="bad-project",
        )
        pytest.fail("Request should have failed - bad api key")
    except Exception as e:
        print("exception: ", e)


def test_litellm_completion_vertex_exception():
    try:
        import litellm

        litellm.set_verbose = True
        response = completion(
            model="vertex_ai/gemini-pro",
            api_key="good-morning",
            messages=[{"role": "user", "content": "hello"}],
            vertex_project="bad-project",
        )
        pytest.fail("Request should have failed - bad api key")
    except Exception as e:
        print("exception: ", e)


def test_litellm_predibase_exception():
    """
    Test - Assert that the Predibase API Key is not returned on Authentication Errors
    """
    try:
        import litellm

        litellm.set_verbose = True
        response = completion(
            model="predibase/llama-3-8b-instruct",
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
            tenant_id="c4768f95",
            api_key="hf-rawapikey",
        )
        pytest.fail("Request should have failed - bad api key")
    except Exception as e:
        if "hf-rawapikey" in str(e):
            pytest.fail("predibase error leaked the raw api key")
        print("exception: ", e)


# # test_invalid_request_error(model="command-nightly")
# # Test 3: Rate Limit Errors
# def test_model_call(model):
#     try:
#         sample_text = "how does a court case get to the Supreme Court?"
#         messages = [{ "content": sample_text,"role": "user"}]
#         print(f"model: {model}")
#         response = completion(model=model, messages=messages)
#     except RateLimitError as e:
#         print(f"headers: {e.response.headers}")
#         return True
#     # except OpenAIError: # is at least an openai error -> in case of random model errors - e.g. overloaded server
#     #     return True
#     except Exception as e:
#         print(f"Uncaught Exception {model}: {type(e).__name__} - {e}")
#         traceback.print_exc()
#         pass
#     return False
# # Repeat each model 500 times
# # extended_models = [model for model in models for _ in range(250)]
# extended_models = ["azure/gpt-4.1-mini" for _ in range(250)]

# def worker(model):
#     return test_model_call(model)

# # Create a dictionary to store the results
# counts = {True: 0, False: 0}

# # Use Thread Pool Executor
# with ThreadPoolExecutor(max_workers=500) as executor:
#     # Use map to start the operation in thread pool
#     results = executor.map(worker, extended_models)

#     # Iterate over results and count True/False
#     for result in results:
#         counts[result] += 1

# accuracy_score = counts[True]/(counts[True] + counts[False])
# print(f"accuracy_score: {accuracy_score}")


def test_anthropic_tool_calling_exception():
    """
    Related - https://github.com/BerriAI/litellm/issues/4348
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {},
            },
        }
    ]
    try:
        litellm.completion(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            tools=tools,
        )
    except litellm.BadRequestError:
        pass


from typing import Optional, Union

from openai import OpenAI


def _pre_call_utils(
    call_type: str,
    data: dict,
    client: Union[OpenAI, AsyncOpenAI],
    sync_mode: bool,
    streaming: Optional[bool],
):
    if call_type == "embedding":
        data["input"] = "Hello world!"
        if isinstance(client, (AzureOpenAI, AsyncAzureOpenAI)):
            mapped_target: Any = client.embeddings.with_raw_response
            patched_attr = "create"
        else:
            mapped_target = client
            patched_attr = "post"
        if sync_mode:
            original_function = litellm.embedding
        else:
            original_function = litellm.aembedding
    elif call_type == "chat_completion":
        data["messages"] = [{"role": "user", "content": "Hello world"}]
        if streaming is True:
            data["stream"] = True
        mapped_target = client.chat.completions.with_raw_response  # type: ignore
        patched_attr = "create"
        if sync_mode:
            original_function = litellm.completion
        else:
            original_function = litellm.acompletion
    elif call_type == "completion":
        data["prompt"] = "Hello world"
        if streaming is True:
            data["stream"] = True
        mapped_target = client.completions.with_raw_response  # type: ignore
        patched_attr = "create"
        if sync_mode:
            original_function = litellm.text_completion
        else:
            original_function = litellm.atext_completion

    return data, original_function, mapped_target, patched_attr


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.parametrize(
    "provider, model, call_type, streaming",
    [
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
    data, original_function, mapped_target, patched_attr = _pre_call_utils(
        call_type=call_type,
        data=data,
        client=openai_client,
        sync_mode=sync_mode,
        streaming=streaming,
    )

    cooldown_time = 30.0

    def _return_exception(*args, **kwargs):

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
        patched_attr,
        side_effect=_return_exception,
    ):
        new_retry_after_mock_client = MagicMock(return_value=-1)

        litellm.utils._get_retry_after_from_exception_header = (
            new_retry_after_mock_client
        )

        async def call_and_drain():
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

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await call_and_drain()

        assert exc_info.value.litellm_response_headers is not None
        assert int(exc_info.value.litellm_response_headers["retry-after"]) == cooldown_time


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["azure/gpt-4.1-mini", "openai/gpt-3.5-turbo"])
async def test_bad_request_error_contains_httpx_response(model):
    """
    Test that the BadRequestError contains the httpx response

    Relevant issue: https://github.com/BerriAI/litellm/issues/6732
    """
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Hello world"}],
            bad_arg="bad_arg",
        )
    e = exc_info.value
    print("e.response", e.response)
    print("vars(e.response)", vars(e.response))
    assert e.response is not None


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("stream_mode", [True, False])
@pytest.mark.parametrize("model", ["gpt-4.1-nano"])  # "gpt-4o-mini",
@pytest.mark.asyncio
async def test_exception_bubbling_up(sync_mode, stream_mode, model):
    """
    make sure code, param, and type are bubbled up
    """
    import litellm

    litellm.set_verbose = True
    async def _call_with_bad_role():
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

    with pytest.raises(Exception, match='litellm\\.BadRequestError: OpenAIException - Invalid value') as exc_info:
        await _call_with_bad_role()

    assert exc_info.value.code == "invalid_value"
    assert exc_info.value.param is not None
    assert exc_info.value.type == "invalid_request_error"
