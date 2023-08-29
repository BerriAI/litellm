from openai.error import AuthenticationError, InvalidRequestError, RateLimitError, OpenAIError
import os
import sys
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import (
    embedding,
    completion,
#     AuthenticationError,
#     InvalidRequestError,
    ContextWindowExceededError,
#     RateLimitError,
#     ServiceUnavailableError,
#     OpenAIError,
)
from concurrent.futures import ThreadPoolExecutor
import pytest
litellm.vertex_project = "pathrise-convert-1606954137718"
litellm.vertex_location = "us-central1"

litellm.failure_callback = ["sentry"]
#### What this tests ####
#    This tests exception mapping -> trigger an exception from an llm provider -> assert if output is of the expected type


# 5 providers -> OpenAI, Azure, Anthropic, Cohere, Replicate

# 3 main types of exceptions -> - Rate Limit Errors, Context Window Errors, Auth errors (incorrect/rotated key, etc.)

# Approach: Run each model through the test -> assert if the correct error (always the same one) is triggered

# models = ["gpt-3.5-turbo", "chatgpt-test",  "claude-instant-1", "command-nightly"]
test_model = "claude-instant-1"
models = ["claude-instant-1"]


def logging_fn(model_call_dict):
    return
    if "model" in model_call_dict:
        print(f"model_call_dict: {model_call_dict['model']}")
    else:
        print(f"model_call_dict: {model_call_dict}")


# Test 1: Context Window Errors
@pytest.mark.parametrize("model", models)
def test_context_window(model):
    sample_text = "how does a court case get to the Supreme Court?" * 50000
    messages = [{"content": sample_text, "role": "user"}]
    try:
        print(f"model: {model}")
        response = completion(
            model=model,
            messages=messages,
            logger_fn=logging_fn,
        )
    except ContextWindowExceededError as e:
        print(f"ContextWindowExceededError: {e.llm_provider}")
        return
    except InvalidRequestError as e:
        print(f"InvalidRequestError: {e.llm_provider}")
        print(f"InvalidRequestError message: {e.message}")
        return
    except OpenAIError as e:
        print(f"OpenAIError: {e.llm_provider}")
        return
    except Exception as e:
        print("Uncaught Error in test_context_window")
        print(f"Error Type: {type(e).__name__}")
        print(f"Uncaught Exception - {e}")
        pytest.fail(f"Error occurred: {e}")
    return


test_context_window(test_model)


# Test 2: InvalidAuth Errors
@pytest.mark.parametrize("model", models)
def invalid_auth(model):  # set the model key to an invalid key, depending on the model
    messages = [{"content": "Hello, how are you?", "role": "user"}]
    temporary_key = None
    try:
        if model == "gpt-3.5-turbo":
            temporary_key = os.environ["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = "bad-key"
        elif model == "chatgpt-test":
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
            os.environ["TOGETHERAI_API_KEY"] = "84060c79880fc49df126d3e87b53f8a463ff6e1c6d27fe64207cde25cdfcd1f24a"
        elif model in litellm.openrouter_models:
            temporary_key = os.environ["OPENROUTER_API_KEY"]
            os.environ["OPENROUTER_API_KEY"] = "bad-key"
        elif (
            model
            == "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"
        ):
            temporary_key = os.environ["REPLICATE_API_KEY"]
            os.environ["REPLICATE_API_KEY"] = "bad-key"
        print(f"model: {model}")
        response = completion(
            model=model, messages=messages
        )
        print(f"response: {response}")
    except AuthenticationError as e:
        print(f"AuthenticationError Caught Exception - {e.llm_provider}")
    except (
        OpenAIError
    ):  # is at least an openai error -> in case of random model errors - e.g. overloaded server
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
        elif ("togethercomputer" in model):
            os.environ["TOGETHERAI_API_KEY"] = temporary_key
    return


invalid_auth(test_model)

# Test 3: Rate Limit Errors
# def test_model_call(model):
#     try:
#         sample_text = "how does a court case get to the Supreme Court?"
#         messages = [{ "content": sample_text,"role": "user"}]
#         print(f"model: {model}")
#         response = completion(model=model, messages=messages)
#     except RateLimitError:
#         return True
#     except OpenAIError: # is at least an openai error -> in case of random model errors - e.g. overloaded server
#         return True
#     except Exception as e:
#         print(f"Uncaught Exception {model}: {type(e).__name__} - {e}")
#         traceback.print_exc()
#         pass
#     return False
# # Repeat each model 500 times
# extended_models = [model for model in models for _ in range(250)]

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
