#### What this tests ####
#    This tests litellm.token_counter() function

import sys, os
import traceback
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from litellm import token_counter


def test_tokenizers():
    try: 
        ### test the openai, claude, cohere and llama2 tokenizers. 
        ### The tokenizer value should be different for all
        sample_text = "Hellö World, this is my input string!"

        # openai tokenizer 
        openai_tokens = token_counter(model="gpt-3.5-turbo", text=sample_text)

        # claude tokenizer
        claude_tokens = token_counter(model="claude-instant-1", text=sample_text)

        # cohere tokenizer
        cohere_tokens = token_counter(model="command-nightly", text=sample_text)

        # llama2 tokenizer 
        llama2_tokens = token_counter(model="meta-llama/Llama-2-7b-chat", text=sample_text)

        print(f"openai tokens: {openai_tokens}; claude tokens: {claude_tokens}; cohere tokens: {cohere_tokens}; llama2 tokens: {llama2_tokens}")

        # assert that all token values are different
        assert openai_tokens != cohere_tokens != llama2_tokens, "Token values are not different."
        
        return "It worked!"
    except Exception as e: 
        pytest.fail(f'An exception occured: {e}')

test_tokenizers()