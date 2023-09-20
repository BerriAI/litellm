import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from litellm import get_max_tokens, model_cost, open_ai_chat_completion_models

print(get_max_tokens("gpt-3.5-turbo"))

print(model_cost)
print(open_ai_chat_completion_models)
