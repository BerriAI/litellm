#### What this tests ####
#    This tests error logging (with custom user functions) for the `completion` + `embedding` endpoints without callbacks (i.e. slack, posthog, etc. not set)
#    Requirements: Remove any env keys you have related to slack/posthog/etc. + anthropic api key (cause an exception)

import sys, os
import traceback
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion
from infisical import InfisicalClient
import pytest

infisical_token = os.environ["INFISICAL_TOKEN"]

litellm.secret_manager_client = InfisicalClient(token=infisical_token)

user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]

def test_completion_azure():
    try:
        response = completion(model="gpt-3.5-turbo", deployment_id="chatgpt-test", messages=messages, custom_llm_provider="azure")
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_openai():
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_completion_openai_with_optional_params():
    try:
        response = completion(model="gpt-3.5-turbo", messages=messages, temperature=0.5, top_p=0.1, user="ishaan_dev@berri.ai")
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")