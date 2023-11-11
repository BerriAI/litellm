#### What this tests ####
#    This tests litellm.token_counter() function

import sys, os
import traceback
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from litellm import token_counter, encode, decode


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
        
        print("test tokenizer: It worked!")
    except Exception as e: 
        pytest.fail(f'An exception occured: {e}')

# test_tokenizers()

def test_encoding_and_decoding(): 
    try: 
        sample_text = "Hellö World, this is my input string!"
        # openai encoding + decoding
        openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
        openai_text = decode(model="gpt-3.5-turbo", tokens=openai_tokens)

        assert openai_text == sample_text
        
        # claude encoding + decoding 
        claude_tokens = encode(model="claude-instant-1", text=sample_text)
        claude_text = decode(model="claude-instant-1", tokens=claude_tokens.ids)

        assert claude_text == sample_text

        # cohere encoding + decoding 
        cohere_tokens = encode(model="command-nightly", text=sample_text)
        cohere_text = decode(model="command-nightly", tokens=cohere_tokens.ids)

        assert cohere_text == sample_text

        # llama2 encoding + decoding
        llama2_tokens = encode(model="meta-llama/Llama-2-7b-chat", text=sample_text)
        llama2_text = decode(model="meta-llama/Llama-2-7b-chat", tokens=llama2_tokens.ids)

        assert llama2_text == sample_text
    except Exception as e: 
        pytest.fail(f'An exception occured: {e}')

test_encoding_and_decoding() 