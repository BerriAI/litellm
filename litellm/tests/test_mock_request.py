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
        response = litellm.completion(model=model, messages=messages, mock_request=True)
        print(response)
    except:
        traceback.print_exc()

test_mock_request()