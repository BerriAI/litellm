#### What this tests ####
#    This tests error logging (with custom user functions) for the `completion` + `embedding` endpoints without callbacks (i.e. slack, posthog, etc. not set)
#    Requirements: Remove any env keys you have related to slack/posthog/etc. + anthropic api key (cause an exception)

# import sys, os
# import traceback

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# from litellm import embedding, completion
# from infisical import InfisicalClient
# import pytest

# infisical_token = os.environ["INFISICAL_TOKEN"]

# litellm.secret_manager_client = InfisicalClient(token=infisical_token)

# user_message = "Hello, whats the weather in San Francisco??"
# messages = [{"content": user_message, "role": "user"}]


# def test_completion_openai():
#     try:
#         response = completion(model="gpt-3.5-turbo", messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         litellm.secret_manager_client = None
#         pytest.fail(f"Error occurred: {e}")
#     litellm.secret_manager_client = None


# test_completion_openai()
