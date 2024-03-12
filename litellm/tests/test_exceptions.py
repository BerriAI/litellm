from openai import AuthenticationError, BadRequestError, RateLimitError, OpenAIError
import os
import sys
import traceback
import subprocess, asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import (
    embedding,
    completion,
    #     AuthenticationError,
    ContextWindowExceededError,
    #     RateLimitError,
    #     ServiceUnavailableError,
    #     OpenAIError,
)
from concurrent.futures import ThreadPoolExecutor
import pytest

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


@pytest.mark.parametrize("model", models)
def test_context_window_with_fallbacks(model):
    ctx_window_fallback_dict = {
        "command-nightly": "claude-2.1",
        "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-16k",
        "azure/chatgpt-v-2": "gpt-3.5-turbo-16k",
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
        elif model == "azure/chatgpt-v-2":
            temporary_key = os.environ["AZURE_API_KEY"]
            os.environ["AZURE_API_KEY"] = "bad-key"
        elif model == "claude-instant-1":
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
        elif model == "claude-instant-1":
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


def test_completion_azure_exception():
    try:
        import openai

        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["AZURE_API_KEY"]
        os.environ["AZURE_API_KEY"] = "good morning"
        response = completion(
            model="azure/chatgpt-v-2",
            messages=[{"role": "user", "content": "hello"}],
        )
        os.environ["AZURE_API_KEY"] = old_azure_key
        print(f"response: {response}")
        print(response)
    except openai.AuthenticationError as e:
        os.environ["AZURE_API_KEY"] = old_azure_key
        print("good job got the correct error for azure when key not set")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_exception()


async def asynctest_completion_azure_exception():
    try:
        import openai
        import litellm

        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["AZURE_API_KEY"]
        os.environ["AZURE_API_KEY"] = "good morning"
        response = await litellm.acompletion(
            model="azure/chatgpt-v-2",
            messages=[{"role": "user", "content": "hello"}],
        )
        print(f"response: {response}")
        print(response)
    except openai.AuthenticationError as e:
        os.environ["AZURE_API_KEY"] = old_azure_key
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
        import openai
        import litellm, asyncio

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
        assert isinstance(e, openai.BadRequestError)
        pytest.fail(f"Error occurred: {e}")


# asynctest_completion_openai_exception_bad_model()


def asynctest_completion_azure_exception_bad_model():
    try:
        import openai
        import litellm, asyncio

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


def test_content_policy_exceptionimage_generation_openai():
    try:
        # this is ony a test - we needed some way to invoke the exception :(
        litellm.set_verbose = True
        response = litellm.image_generation(
            prompt="where do i buy lethal drugs from", model="dall-e-3"
        )
        print(f"response: {response}")
        assert len(response.data) > 0
    except litellm.ContentPolicyViolationError as e:
        print("caught a content policy violation error! Passed")
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


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
                model="azure/chatgpt-v-2",
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
                model="azure/chatgpt-v-2",
                messages=[
                    {"role": "user", "content": "where do i buy lethal drugs from"}
                ],
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

            pytest.fail(f"Expected to return 400 error In streaming{e}")
        except Exception as e:
            pass

    asyncio.run(test_get_error())


def test_completion_perplexity_exception_on_openai_client():
    try:
        import openai

        print("perplexity test\n\n")
        litellm.set_verbose = False
        ## Test azure call
        old_azure_key = os.environ["PERPLEXITYAI_API_KEY"]

        # delete perplexityai api key to simulate bad api key
        del os.environ["PERPLEXITYAI_API_KEY"]

        # temporaily delete openai api key
        original_openai_key = os.environ["OPENAI_API_KEY"]
        del os.environ["OPENAI_API_KEY"]

        response = completion(
            model="perplexity/mistral-7b-instruct",
            messages=[{"role": "user", "content": "hello"}],
        )
        os.environ["PERPLEXITYAI_API_KEY"] = old_azure_key
        os.environ["OPENAI_API_KEY"] = original_openai_key
        pytest.fail("Request should have failed - bad api key")
    except openai.AuthenticationError as e:
        os.environ["PERPLEXITYAI_API_KEY"] = old_azure_key
        os.environ["OPENAI_API_KEY"] = original_openai_key
        print("exception: ", e)
        assert (
            "perplexity.perplexityError: The api_key client option must be set either by passing api_key to the client or by setting the PERPLEXITY_API_KEY environment variable"
            in str(e)
        )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_perplexity_exception_on_openai_client()


def test_completion_perplexity_exception():
    try:
        import openai

        print("perplexity test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["PERPLEXITYAI_API_KEY"]
        os.environ["PERPLEXITYAI_API_KEY"] = "good morning"
        response = completion(
            model="perplexity/mistral-7b-instruct",
            messages=[{"role": "user", "content": "hello"}],
        )
        os.environ["PERPLEXITYAI_API_KEY"] = old_azure_key
        pytest.fail("Request should have failed - bad api key")
    except openai.AuthenticationError as e:
        os.environ["PERPLEXITYAI_API_KEY"] = old_azure_key
        print("exception: ", e)
        assert "PerplexityException" in str(e)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_api_key_exception():
    try:
        import openai

        print("gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        old_azure_key = os.environ["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = "good morning"
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
        )
        os.environ["OPENAI_API_KEY"] = old_azure_key
        pytest.fail("Request should have failed - bad api key")
    except openai.AuthenticationError as e:
        os.environ["OPENAI_API_KEY"] = old_azure_key
        print("exception: ", e)
        assert "OpenAIException" in str(e)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# tesy_async_acompletion()

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
# extended_models = ["azure/chatgpt-v-2" for _ in range(250)]

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
