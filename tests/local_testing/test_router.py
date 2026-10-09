#### What this tests ####
# This tests litellm router

import asyncio
import os
import time
import traceback
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

import litellm
import litellm.types
import litellm.types.router
from litellm import Router
from litellm.types.router import DeploymentTypedDict
from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE

load_dotenv()






def test_router_deployment_typing():
    deployment_typed_dict = DeploymentTypedDict(
        model_name="hi", litellm_params={"model": "hello-world"}
    )
    for value in deployment_typed_dict.items():
        assert not isinstance(value, BaseModel)


@pytest.mark.asyncio()
async def test_router_provider_wildcard_routing_regex():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/fo::*:static::*",
                "litellm_params": {
                    "model": "openai/fo::*:static::*",
                    "api_base": FAKE_OPENAI_API_BASE,
                },
            },
            {
                "model_name": "openai/foo3::hello::*",
                "litellm_params": {
                    "model": "openai/foo3::hello::*",
                    "api_base": FAKE_OPENAI_API_BASE,
                },
            },
        ]
    )

    print("router model list = ", router.get_model_list())

    response1 = await router.acompletion(
        model="openai/fo::anything-can-be-here::static::anything-can-be-here",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 1 = ", response1)

    response2 = await router.acompletion(
        model="openai/foo3::hello::static::anything-can-be-here",
        messages=[{"role": "user", "content": "hello"}],
    )

    print("response 2 = ", response2)








def test_router_sensitive_keys():
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",  # openai model name
                    "litellm_params": {  # params for litellm completion/embedding call
                        "model": "azure/gpt-4.1-mini",
                        "api_key": "special-key",
                    },
                    "model_info": {"id": 12345},
                },
            ],
        )
    except Exception as e:
        print(f"error msg - {str(e)}")
        if "special-key" in str(e):
            pytest.fail("router error leaked the api key")




@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_router_retries(sync_mode):
    """
    - make sure retries work as expected
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "bad-key"},
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-4.1-nano",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ]

    router = Router(model_list=model_list, num_retries=2)

    if sync_mode:
        router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )
    else:
        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )

        print(response.choices[0].message)


def test_exception_raising():
    # this tests if the router raises an exception when invalid params are set
    # in this test both deployments have bad keys - Keep this test. It validates if the router raises the most recent exception
    litellm.set_verbose = True
    import openai

    try:
        print("testing if router raises an exception")
        old_api_key = os.environ["AZURE_AI_API_KEY"]
        os.environ["AZURE_AI_API_KEY"] = ""
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  #
                    "model": "gpt-3.5-turbo",
                    "api_key": "bad-key",
                },
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
            routing_strategy="simple-shuffle",
            set_verbose=False,
            num_retries=1,
        )  # type: ignore
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello this request will fail"}],
        )
        os.environ["AZURE_AI_API_KEY"] = old_api_key
        pytest.fail(f"Should have raised an Auth Error")
    except openai.AuthenticationError:
        print(
            "Test Passed: Caught an OPENAI AUTH Error, Good job. This is what we needed!"
        )
        os.environ["AZURE_AI_API_KEY"] = old_api_key
        router.reset()
    except Exception as e:
        os.environ["AZURE_AI_API_KEY"] = old_api_key
        print("Got unexpected exception on router!", e)


# test_exception_raising()


def test_reading_key_from_model_list():
    # [PROD TEST CASE]
    # this tests if the router can read key from model list and make completion call, and completion + stream call. This is 90% of the router use case
    # DO NOT REMOVE THIS TEST. It's an IMP ONE. Speak to Ishaan, if you are tring to remove this
    litellm.set_verbose = False
    import openai

    try:
        print("testing if router raises an exception")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 240000,
                "rpm": 1800,
            }
        ]

        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
            routing_strategy="simple-shuffle",
            set_verbose=True,
            num_retries=1,
        )  # type: ignore
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello this request will fail"}],
        )
        print("\n response", response)
        str_response = response.choices[0].message.content
        print("\n str_response", str_response)
        assert len(str_response) > 0

        print("\n Testing streaming response")
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello this request will fail"}],
            stream=True,
        )
        completed_response = ""
        for chunk in response:
            if chunk is not None:
                print(chunk)
                completed_response += chunk.choices[0].delta.content or ""
        print("\n completed_response", completed_response)
        assert len(completed_response) > 0
        print("\n Passed Streaming")
        router.reset()
    except Exception as e:
        print(f"FAILED TEST")
        pytest.fail(f"Got unexpected exception on router! - {e}")


# test_reading_key_from_model_list()


def test_call_one_endpoint():
    # [PROD TEST CASE]
    # user passes one deployment they want to call on the router, we call the specified one
    # this test makes a completion calls azure/gpt-4.1-mini, it should work
    try:
        print("Testing calling a specific deployment")
        old_api_key = os.environ["AZURE_AI_API_KEY"]

        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": old_api_key,
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "azure/text-embedding-ada-002",
                    "api_key": os.environ["AZURE_AI_API_KEY"],
                    "api_base": os.environ["AZURE_AI_API_BASE"],
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]
        litellm.set_verbose = True
        router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            set_verbose=True,
            num_retries=1,
        )  # type: ignore
        old_api_base = os.environ.pop("AZURE_AI_API_BASE", None)

        async def call_azure_completion():
            response = await router.acompletion(
                model="azure/gpt-4.1-mini",
                messages=[{"role": "user", "content": "hello this request will pass"}],
                specific_deployment=True,
            )
            print("\n response", response)

        async def call_azure_embedding():
            response = await router.aembedding(
                model="azure/text-embedding-ada-002",
                input=["good morning from litellm"],
                specific_deployment=True,
            )

            print("\n response", response)

        asyncio.run(call_azure_completion())
        asyncio.run(call_azure_embedding())

        os.environ["AZURE_AI_API_BASE"] = old_api_base
        os.environ["AZURE_AI_API_KEY"] = old_api_key
    except Exception as e:
        print(f"FAILED TEST")
        pytest.fail(f"Got unexpected exception on router! - {e}")


# test_call_one_endpoint()


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_async_router_context_window_fallback(sync_mode):
    """
    - Give a gpt-4 model group with different context windows (8192k vs. 128k)
    - Send a 10k prompt
    - Assert it works
    """
    import os

    from large_text import text

    litellm.set_verbose = False
    litellm.turn_on_debug()

    print(f"len(text): {len(text)}")
    try:
        model_list = [
            {
                "model_name": "gpt-4",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-4",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "api_base": os.getenv("OPENAI_API_BASE"),
                },
            },
            {
                "model_name": "gpt-4-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "gpt-4-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]

        router = Router(model_list=model_list, set_verbose=True, context_window_fallbacks=[{"gpt-4": ["gpt-4-turbo"]}], num_retries=0)  # type: ignore
        if sync_mode is False:
            response = await router.acompletion(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": text * 2},
                    {"role": "user", "content": "Who was Alexander?"},
                ],
            )

            print(f"response: {response}")
            assert "gpt-4-turbo" in response.model
        else:
            response = router.completion(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": text * 2},
                    {"role": "user", "content": "Who was Alexander?"},
                ],
            )
            assert "gpt-4-turbo" in response.model
    except Exception as e:
        pytest.fail(f"Got unexpected exception on router! - {str(e)}")














### FUNCTION CALLING


def test_function_calling():
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "tpm": 100000,
            "rpm": 10000,
        },
    ]

    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    functions = [
        {
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
        }
    ]

    router = Router(model_list=model_list)
    response = router.completion(
        model="gpt-3.5-turbo", messages=messages, functions=functions
    )
    router.reset()
    print(response)


# test_acompletion_on_router()




# test_function_calling_on_router()


### IMAGE GENERATION
# asyncio.run(test_aimg_gen_on_router())


# test_img_gen_on_router()
###


# test_aembedding_on_router()


# test_azure_embedding_on_router()


# test_bedrock_on_router()


# test openai-compatible endpoint
# asyncio.run(test_mistral_on_router())


def test_openai_completion_on_router():
    # [PROD Use Case] - Makes an acompletion call + async acompletion call, and sync acompletion call, sync completion + stream
    # 4 LLM API calls made here. If it fails, add retries. Do not remove this test.
    litellm.set_verbose = True
    print("\n Testing OpenAI on router\n")
    try:
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
        ]
        router = Router(model_list=model_list)

        async def test():
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "hello from litellm test",
                    }
                ],
            )
            print(response)
            assert len(response.choices[0].message.content) > 0

            print("\n streaming + acompletion test")
            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"hello from litellm test {time.time()}",
                    }
                ],
                stream=True,
            )
            complete_response = ""
            print(response)
            # if you want to see all the attributes and methods
            async for chunk in response:
                print(chunk)
                complete_response += chunk.choices[0].delta.content or ""
            print("\n complete response: ", complete_response)
            assert len(complete_response) > 0

        asyncio.run(test())
        print("\n Testing Sync completion calls \n")
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "hello from litellm test2",
                }
            ],
        )
        print(response)
        assert len(response.choices[0].message.content) > 0

        print("\n streaming + completion test")
        response = router.completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "hello from litellm test3",
                }
            ],
            stream=True,
        )
        complete_response = ""
        print(response)
        for chunk in response:
            print(chunk)
            complete_response += chunk.choices[0].delta.content or ""
        print("\n complete response: ", complete_response)
        assert len(complete_response) > 0
        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_openai_completion_on_router()








# test_reading_keys_os_environ()




# test_reading_openai_keys_os_environ()


def test_router_anthropic_key_dynamic():
    anthropic_api_key = os.environ.pop("ANTHROPIC_API_KEY")
    model_list = [
        {
            "model_name": "anthropic-claude",
            "litellm_params": {
                "model": os.environ.get(
                    "CI_CD_DEFAULT_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
                ),
                "api_key": anthropic_api_key,
            },
        }
    ]

    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    router.completion(model="anthropic-claude", messages=messages)
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key


def test_router_timeout():
    litellm.set_verbose = True
    import logging

    from litellm._logging import verbose_logger

    verbose_logger.setLevel(logging.DEBUG)
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "openai/slow-endpoint",
                "api_base": FAKE_OPENAI_API_BASE,
                "api_key": "fake-key",
            },
        }
    ]
    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    start_time = time.time()
    try:
        res = router.completion(
            model="gpt-3.5-turbo", messages=messages, timeout=0.5
        )
        print(res)
        pytest.fail("this should have timed out")
    except litellm.exceptions.Timeout as e:
        print("got timeout exception")
        print(e)
        print(vars(e))
        pass


@pytest.mark.asyncio
async def test_router_amoderation():
    model_list = [
        {
            "model_name": "openai-moderations",
            "litellm_params": {
                "model": "omni-moderation-latest",
                "api_key": os.getenv("OPENAI_API_KEY", None),
            },
        }
    ]

    router = Router(model_list=model_list)
    ## Test 1: user facing function
    result = await router.amoderation(
        model="omni-moderation-latest", input="this is valid good text"
    )




@pytest.mark.asyncio
async def test_router_text_completion_client():
    # This tests if we re-use the Async OpenAI client
    # This test fails when we create a new Async OpenAI client per request
    try:
        model_list = [
            {
                "model_name": "fake-openai-endpoint",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": os.getenv("OPENAI_API_KEY", None),
                    "api_base": FAKE_OPENAI_API_BASE,
                },
            }
        ]
        router = Router(model_list=model_list, debug_level="DEBUG", set_verbose=True)
        tasks = []
        for _ in range(300):
            tasks.append(
                router.atext_completion(
                    model="fake-openai-endpoint",
                    prompt="hello from litellm test",
                )
            )

        # Execute all coroutines concurrently
        responses = await asyncio.gather(*tasks)
        print(responses)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.fixture
def mock_response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        **{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1699896916,
            "model": "gpt-3.5-turbo-0125",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{\n"location": "Boston, MA"\n}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )


@pytest.mark.asyncio
async def test_router_model_usage(mock_response):
    """
    Test if tracking used model tpm works as expected
    """
    model = "my-fake-model"
    model_tpm = 100
    setattr(
        mock_response,
        "usage",
        litellm.Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )

    print(f"mock_response: {mock_response}")
    model_tpm = 100
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "tpm": model_tpm,
                    "mock_response": mock_response,
                },
            }
        ]
    )

    allowed_fails = 1  # allow for changing b/w minutes

    for _ in range(2):
        try:
            _ = await llm_router.acompletion(
                model=model, messages=[{"role": "user", "content": "Hey!"}]
            )
            await asyncio.sleep(3)

            initial_usage_tuple = await llm_router.get_model_group_usage(
                model_group=model
            )
            initial_usage = initial_usage_tuple[0]

            # completion call - 10 tokens
            _ = await llm_router.acompletion(
                model=model, messages=[{"role": "user", "content": "Hey!"}]
            )

            await asyncio.sleep(3)
            updated_usage_tuple = await llm_router.get_model_group_usage(
                model_group=model
            )
            updated_usage = updated_usage_tuple[0]

            assert updated_usage == initial_usage + 10  # type: ignore
            break
        except Exception as e:
            if allowed_fails > 0:
                print(
                    f"Decrementing allowed_fails: {allowed_fails}.\nReceived error - {str(e)}"
                )
                allowed_fails -= 1
            else:
                print(f"allowed_fails: {allowed_fails}")
                raise e








def test_router_cooldown_api_connection_error():
    from litellm.router_utils.cooldown_handlers import _is_cooldown_required

    with pytest.raises(litellm.APIConnectionError) as exc_info:
        _ = litellm.completion(
            model="vertex_ai/gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    e = exc_info.value
    assert (
        _is_cooldown_required(
            litellm_router_instance=Router(),
            model_id="",
            exception_status=e.code,
            exception_str=str(e),
        )
        is False
    )

    router = Router(
        model_list=[
            {
                "model_name": "gemini-1.5-pro",
                "litellm_params": {"model": "vertex_ai/gemini-1.5-pro"},
            }
        ]
    )

    try:
        router.completion(
            model="gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.APIConnectionError:
        pass


def test_router_correctly_reraise_error():
    """
    User feedback: There is a problem with my messages array, but the error exception thrown is a Rate Limit error.
    ```
    Rate Limit: Error code: 429 - {'error': {'message': 'No deployments available for selected model, Try again in 60 seconds. Passed model=gemini-2.5-flash-lite..
    ```
    What they want? Propagation of the real error.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gemini-1.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-1.5-pro",
                    "mock_response": "litellm.RateLimitError",
                },
            }
        ]
    )

    try:
        router.completion(
            model="gemini-1.5-pro",
            messages=[{"role": "admin", "content": "Fail on this!"}],
        )
    except litellm.RateLimitError:
        pass


















# @pytest.mark.parametrize("on_error", [True, False])
# @pytest.mark.asyncio
# async def test_router_response_headers(on_error):
#     router = Router(
#         model_list=[
#             {
#                 "model_name": "gpt-3.5-turbo",
#                 "litellm_params": {
#                     "model": "azure/gpt-4.1-mini",
#                     "api_key": os.getenv("AZURE_AI_API_KEY"),
#                     "api_base": os.getenv("AZURE_AI_API_BASE"),
#                     "tpm": 100000,
#                     "rpm": 100000,
#                 },
#             },
#             {
#                 "model_name": "gpt-3.5-turbo",
#                 "litellm_params": {
#                     "model": "azure/gpt-4.1-mini",
#                     "api_key": os.getenv("AZURE_AI_API_KEY"),
#                     "api_base": os.getenv("AZURE_AI_API_BASE"),
#                     "tpm": 500,
#                     "rpm": 500,
#                 },
#             },
#         ]
#     )

#     response = await router.acompletion(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": "Hello world!"}],
#         mock_testing_rate_limit_error=on_error,
#     )

#     response_headers = response._hidden_params["additional_headers"]

#     print(response_headers)

#     assert response_headers["x-ratelimit-limit-requests"] == 100500
#     assert int(response_headers["x-ratelimit-remaining-requests"]) > 0
#     assert response_headers["x-ratelimit-limit-tokens"] == 100500
#     assert int(response_headers["x-ratelimit-remaining-tokens"]) > 0


def test_router_completion_with_model_id():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "123"},
            }
        ]
    )

    with patch.object(
        router, "routing_strategy_pre_call_checks"
    ) as mock_pre_call_checks:
        router.completion(model="123", messages=[{"role": "user", "content": "hi"}])
        mock_pre_call_checks.assert_not_called()
