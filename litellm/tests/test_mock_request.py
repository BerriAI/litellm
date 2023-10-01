#### What this tests ####
#    This tests mock request calls to litellm

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


def test_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, stream=False)
        print(response)
        print(type(response))
    except:
        traceback.print_exc()


# test_mock_request()
def test_streaming_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, stream=True)
        complete_response = ""
        for chunk in response:
            complete_response += chunk["choices"][0]["delta"]["content"]
        if complete_response == "":
            raise Exception("Empty response received")
    except:
        traceback.print_exc()


test_streaming_mock_request()
