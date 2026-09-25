import pytest

from tests.unit.litellm_core_utils.test_tokenizer import (
    UNICODE_TEXTS,
    assert_openai_encoding_exposes_the_tiktoken_vocabulary_surface,
    assert_openai_encoding_matches_python,
)

NETWORK_ENCODINGS = ("r50k_base", "gpt2")


@pytest.mark.parametrize("name", NETWORK_ENCODINGS)
@pytest.mark.parametrize("text", UNICODE_TEXTS)
def test_openai_encoding_matches_python_unicode_and_batches(name: str, text: str) -> None:
    assert_openai_encoding_matches_python(name, text)


@pytest.mark.parametrize("name", ("gpt2",))
def test_openai_encoding_exposes_the_tiktoken_vocabulary_surface(name: str) -> None:
    assert_openai_encoding_exposes_the_tiktoken_vocabulary_surface(name)
