#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import batch_completion

messages = [[{"role": "user", "content": "Hey, how's it going"}] for _ in range(5)]
print(messages[0:5])
print(len(messages))
model = "gpt-3.5-turbo"

result = batch_completion(model=model, messages=messages)
print(result)
print(len(result))
