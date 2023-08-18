#### What this tests ####
#    This tests error logging (with custom user functions) for the `completion` + `embedding` endpoints without callbacks (i.e. slack, posthog, etc. not set)
#    Requirements: Remove any env keys you have related to slack/posthog/etc. + anthropic api key (cause an exception)

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = True

model_fallback_list = ["claude-instant-1", "gpt-3.5-turbo", "chatgpt-test"]

user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

for model in model_fallback_list:
    try:
        response = embedding(model="text-embedding-ada-002", input=[user_message])
        response = completion(model=model, messages=messages)
    except Exception as e:
        print(f"error occurred: {traceback.format_exc()}")
