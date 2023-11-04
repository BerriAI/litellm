import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from litellm import get_max_tokens, model_cost, open_ai_chat_completion_models

print(get_max_tokens("gpt-3.5-turbo"))

def test_get_gpt3_tokens():
    max_tokens = get_max_tokens("gpt-3.5-turbo")
    results = max_tokens['max_tokens']
    print(results)
# test_get_gpt3_tokens()

def test_get_palm_tokens():
    # # 🦄🦄🦄🦄🦄🦄🦄🦄
    max_tokens = get_max_tokens("palm/chat-bison")
    results = max_tokens['max_tokens']
    print(results)
# test_get_palm_tokens()
