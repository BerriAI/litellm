# import sys, os
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest
# import litellm
# from litellm import completion_with_retries
# from litellm import (
#     AuthenticationError,
#     InvalidRequestError,
#     RateLimitError,
#     ServiceUnavailableError,
#     OpenAIError,
# )

# user_message = "Hello, whats the weather in San Francisco??"
# messages = [{"content": user_message, "role": "user"}]


# def logger_fn(user_model_dict):
#     # print(f"user_model_dict: {user_model_dict}")
#     pass

# # normal call
# def test_completion_custom_provider_model_name():
#     try:
#         response = completion_with_retries(
#             model="together_ai/togethercomputer/llama-2-70b-chat",
#             messages=messages,
#             logger_fn=logger_fn,
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# # bad call
# # def test_completion_custom_provider_model_name():
# #     try:
# #         response = completion_with_retries(
# #             model="bad-model",
# #             messages=messages,
# #             logger_fn=logger_fn,
# #         )
# #         # Add any assertions here to check the response
# #         print(response)
# #     except Exception as e:
# #         pytest.fail(f"Error occurred: {e}")

# # impact on exception mapping
# def test_context_window():
#     sample_text = "how does a court case get to the Supreme Court?" * 5000
#     messages = [{"content": sample_text, "role": "user"}]
#     try:
#         model = "chatgpt-test"
#         response = completion_with_retries(
#             model=model,
#             messages=messages,
#             custom_llm_provider="azure",
#             logger_fn=logger_fn,
#         )
#         print(f"response: {response}")
#     except InvalidRequestError as e:
#         print(f"InvalidRequestError: {e.llm_provider}")
#         return
#     except OpenAIError as e:
#         print(f"OpenAIError: {e.llm_provider}")
#         return
#     except Exception as e:
#         print("Uncaught Error in test_context_window")
#         print(f"Error Type: {type(e).__name__}")
#         print(f"Uncaught Exception - {e}")
#         pytest.fail(f"Error occurred: {e}")
#     return


# test_context_window()

# test_completion_custom_provider_model_name()
