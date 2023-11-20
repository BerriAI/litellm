import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from litellm import get_max_tokens, model_cost, open_ai_chat_completion_models

def test_get_gpt3_tokens():
    max_tokens = get_max_tokens("gpt-3.5-turbo")
    print(max_tokens)
    assert max_tokens==4097
    # print(results)
test_get_gpt3_tokens()

def test_get_palm_tokens():
    # # 🦄🦄🦄🦄🦄🦄🦄🦄
    max_tokens = get_max_tokens("palm/chat-bison")
    assert max_tokens == 4096
    print(max_tokens)
test_get_palm_tokens()

def test_zephyr_hf_tokens():
    max_tokens = get_max_tokens("huggingface/HuggingFaceH4/zephyr-7b-beta")
    print(max_tokens)
    assert max_tokens == 32768

test_zephyr_hf_tokens()