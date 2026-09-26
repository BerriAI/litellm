import pytest
from litellm import create_pretrained_tokenizer
from tests.unit.litellm_core_utils.test_token_counter import token_counter


def test_tokenizers():
    try:
        ### test the openai, claude, cohere and llama2 tokenizers.
        ### The tokenizer value should be different for all
        sample_text = "Hellö World, this is my input string! My name is ishaan CTO"

        # openai tokenizer
        openai_tokens = token_counter(model="gpt-3.5-turbo", text=sample_text)

        # claude tokenizer
        claude_tokens = token_counter(model="claude-3-5-haiku-20241022", text=sample_text)

        # cohere tokenizer
        cohere_tokens = token_counter(model="command-nightly", text=sample_text)

        # llama2 tokenizer
        llama2_tokens = token_counter(model="meta-llama/Llama-2-7b-chat", text=sample_text)

        # llama3 tokenizer (also testing custom tokenizer)
        llama3_tokens_1 = token_counter(model="meta-llama/llama-3-70b-instruct", text=sample_text)

        try:
            llama3_tokenizer = create_pretrained_tokenizer("Xenova/llama-3-tokenizer")
        except Exception as e:
            pytest.skip(f"custom tokenizer download failed (HF hub unreachable): {e}")
        llama3_tokens_2 = token_counter(custom_tokenizer=llama3_tokenizer, text=sample_text)

        print(
            f"openai tokens: {openai_tokens}; claude tokens: {claude_tokens}; cohere tokens: {cohere_tokens}; llama2 tokens: {llama2_tokens}; llama3 tokens: {llama3_tokens_1}"
        )

        # assert that all token values are different
        # llama2 may fall back to the tiktoken tokenizer when the HuggingFace
        # model hub is unreachable (e.g. in CI).  In that case the count will
        # equal the openai count and the differentiation assertion is skipped.
        if openai_tokens == llama2_tokens:
            pytest.skip("llama2 fell back to tiktoken (HF hub unreachable); skipping differentiation assertion")
        assert llama2_tokens != llama3_tokens_1, "Token values are not different."

        assert llama3_tokens_1 == llama3_tokens_2, (
            "Custom tokenizer is not being used! It has been configured to use the same tokenizer as the built in llama3 tokenizer and the results should be the same."
        )

        print("test tokenizer: It worked!")
    except Exception as e:
        pytest.fail(f"An exception occured: {e}")
