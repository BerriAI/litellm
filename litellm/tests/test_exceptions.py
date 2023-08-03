from openai.error import AuthenticationError, InvalidRequestError, RateLimitError, OpenAIError
import os 
import sys
import traceback
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion
from concurrent.futures import ThreadPoolExecutor
import pytest

# litellm.set_verbose = True
#### What this tests ####
#    This tests exception mapping -> trigger an exception from an llm provider -> assert if output is of the expected type


# 5 providers -> OpenAI, Azure, Anthropic, Cohere, Replicate

# 3 main types of exceptions -> - Rate Limit Errors, Context Window Errors, Auth errors (incorrect/rotated key, etc.)

# Approach: Run each model through the test -> assert if the correct error (always the same one) is triggered

# models = ["gpt-3.5-turbo", "chatgpt-test",  "claude-instant-1", "command-nightly"]
models = ["command-nightly"]
def logging_fn(model_call_dict):
    print(f"model_call_dict: {model_call_dict['model']}")
# Test 1: Context Window Errors
@pytest.mark.parametrize("model", models)
def test_context_window(model):
    sample_text = "how does a court case get to the Supreme Court?" * 100000
    messages = [{"content": sample_text, "role": "user"}]
    try:
        azure = model == "chatgpt-test"
        print(f"model: {model}")
        response = completion(model=model, messages=messages, azure=azure, logger_fn=logging_fn)
        print(f"response: {response}")
    except InvalidRequestError:
        print("InvalidRequestError")
        return
    except OpenAIError:
        print("OpenAIError")
        return
    except Exception as e:
        print("Uncaught Error in test_context_window")
        # print(f"Error Type: {type(e).__name__}")
        print(f"Uncaught Exception - {e}")
        pytest.fail(f"Error occurred: {e}")
    return
test_context_window("command-nightly")
# # Test 2: InvalidAuth Errors
# def logger_fn(model_call_object: dict):
#     print(f"model call details: {model_call_object}")

# @pytest.mark.parametrize("model", models)
# def invalid_auth(model): # set the model key to an invalid key, depending on the model 
#     messages = [{ "content": "Hello, how are you?","role": "user"}]
#     try: 
#         azure = False
#         if model == "gpt-3.5-turbo":
#             os.environ["OPENAI_API_KEY"] = "bad-key"
#         elif model == "chatgpt-test":
#             os.environ["AZURE_API_KEY"] = "bad-key"
#             azure = True
#         elif model == "claude-instant-1":
#             os.environ["ANTHROPIC_API_KEY"] = "bad-key"
#         elif model == "command-nightly":
#             os.environ["COHERE_API_KEY"] = "bad-key"
#         elif model == "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1":
#             os.environ["REPLICATE_API_KEY"] = "bad-key"
#             os.environ["REPLICATE_API_TOKEN"] = "bad-key"
#         print(f"model: {model}")
#         response = completion(model=model, messages=messages, azure=azure)
#         print(f"response: {response}")
#     except AuthenticationError as e:
#         return
#     except OpenAIError: # is at least an openai error -> in case of random model errors - e.g. overloaded server
#         return
#     except Exception as e:
#         print(f"Uncaught Exception - {e}")
#         pytest.fail(f"Error occurred: {e}")
#     return

# # Test 3: Rate Limit Errors 
# def test_model(model):
#     try: 
#         sample_text = "how does a court case get to the Supreme Court?" * 50000
#         messages = [{ "content": sample_text,"role": "user"}]
#         azure = False
#         if model == "chatgpt-test":
#             azure = True
#         print(f"model: {model}")
#         response = completion(model=model, messages=messages, azure=azure)
#     except RateLimitError:
#         return True
#     except OpenAIError: # is at least an openai error -> in case of random model errors - e.g. overloaded server
#         return True
#     except Exception as e:
#         print(f"Uncaught Exception {model}: {type(e).__name__} - {e}")
#         pass
#     return False

# # Repeat each model 500 times
# extended_models = [model for model in models for _ in range(250)]

# def worker(model):
#     return test_model(model)

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


