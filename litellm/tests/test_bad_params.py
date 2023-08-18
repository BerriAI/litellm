#### What this tests ####
#    This tests chaos monkeys - if random parts of the system are broken / things aren't sent correctly - what happens.
#    Expect to add more edge cases to this over time.

import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory by joining the current directory with '..'
parent_dir = os.path.join(current_dir, "../..")

# Add the parent directory to the system path
sys.path.append(parent_dir)


import litellm
from litellm import embedding, completion


litellm.set_verbose = True
litellm.success_callback = ["posthog"]
litellm.failure_callback = ["slack", "sentry", "posthog"]


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]
model_val = None


def test_completion_with_empty_model():
    # test on empty
    try:
        response = completion(model=model_val, messages=messages)
    except Exception as e:
        print(f"error occurred: {e}")
        pass


# bad key
temp_key = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "bad-key"
# test on openai completion call
try:
    response = completion(model="gpt-3.5-turbo", messages=messages)
    print(f"response: {response}")
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass
os.environ["OPENAI_API_KEY"] = str(temp_key)  # this passes linting#5
