#### What this tests ####
#    This tests chaos monkeys - if random parts of the system are broken / things aren't sent correctly - what happens.
#    Expect to add more edge cases to this over time.

import sys, os
import traceback
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion


litellm.set_verbose = True
user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]
model_val = None

def test_completion_with_no_model():
    # test on empty
    with pytest.raises(ValueError):
        response = completion(messages=messages)


def test_completion_with_empty_model():
    # test on empty
    try:
        response = completion(model=model_val, messages=messages)
    except Exception as e:
        print(f"error occurred: {e}")
        pass


def test_completion_with_no_provider():
    # test on empty
    try:
        model = "cerebras/btlm-3b-8k-base"
        response = completion(model=model, messages=messages)
    except Exception as e:
        print(f"error occurred: {e}")
        pass

test_completion_with_no_provider()
# # bad key
# temp_key = os.environ.get("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = "bad-key"
# # test on openai completion call
# try:
#     response = completion(model="gpt-3.5-turbo", messages=messages)
#     print(f"response: {response}")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass
# os.environ["OPENAI_API_KEY"] = str(temp_key)  # this passes linting#5
