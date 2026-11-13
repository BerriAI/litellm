from typing import Final

import pytest
import tiktoken

from litellm.rust_bridge import _native

pytestmark = pytest.mark.requires_rust_extension


def test_tiktoken_codec_round_trips_and_counts() -> None:
    tokenizer: Final = _native.Tokenizer.from_tiktoken("cl100k_base")
    encoded: Final = tokenizer.encode("hello world")

    assert tokenizer.name == "cl100k_base"
    assert tokenizer.count("hello world") == len(encoded)
    assert tokenizer.decode(encoded) == "hello world"


def test_tiktoken_codec_keeps_the_requested_encoding_name() -> None:
    assert _native.Tokenizer.from_tiktoken("gpt2").name == "gpt2"
    assert _native.Tokenizer.from_tiktoken("r50k_base").name == "r50k_base"
    assert _native.Tokenizer.from_tiktoken("gpt2").encode("hi") == _native.Tokenizer.from_tiktoken("r50k_base").encode(
        "hi"
    )


def test_tiktoken_codec_exposes_its_vocabulary() -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    tokenizer: Final = _native.Tokenizer.from_tiktoken("cl100k_base")

    assert tokenizer.special_tokens() == reference._special_tokens
    assert tokenizer.max_token_value() == reference.max_token_value
    assert tokenizer.token_byte_values() == reference.token_byte_values()
    assert tokenizer.encode_single_token(b"hello") == reference.encode_single_token("hello")
    assert tokenizer.is_special_token(reference.eot_token) and not tokenizer.is_special_token(0)
    with pytest.raises(KeyError):
        tokenizer.encode_single_token(b"<|not-a-token|>")


@pytest.mark.parametrize("text", ("\ud800", "x\udc80", "\ud83d\ude42"))
def test_tiktoken_encode_with_unstable_rejects_lone_surrogates(text: str) -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    tokenizer: Final = _native.Tokenizer.from_tiktoken("cl100k_base")

    with pytest.raises(UnicodeEncodeError):
        reference.encode_with_unstable(text)
    with pytest.raises(UnicodeEncodeError):
        tokenizer.encode_with_unstable(text, [])


@pytest.mark.parametrize("token", (-(2**100), -1, 0, 2**32 - 1, 2**32, 2**100, True, False))
def test_tiktoken_is_special_token_handles_ints_outside_u32(token: int) -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    tokenizer: Final = _native.Tokenizer.from_tiktoken("cl100k_base")

    assert tokenizer.is_special_token(token) is reference.is_special_token(token)


def test_unknown_tiktoken_encoding_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported tokenizer"):
        _native.Tokenizer.from_tiktoken("unknown-encoding")


def test_tiktoken_codec_decodes_truncated_unicode_like_python() -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    tokenizer: Final = _native.Tokenizer.from_tiktoken(reference.name)
    encoded: Final = reference.encode("🙂漢字")

    assert tuple(tokenizer.decode(encoded[:end]) for end in range(1, len(encoded) + 1)) == tuple(
        reference.decode(encoded[:end]) for end in range(1, len(encoded) + 1)
    )
