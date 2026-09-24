import json
from typing import Final

import pytest
import tiktoken
from tokenizers import Tokenizer as ReferenceTokenizer

from litellm.rust_bridge import _native
from litellm.utils import claude_json_str
from tests.test_litellm.litellm_core_utils.test_decode_special_tokens import TOKENIZER_JSON

pytestmark = pytest.mark.requires_rust_extension


def test_tiktoken_codec_round_trips_and_counts() -> None:
    tokenizer: Final = _native.Tokenizer.from_tiktoken("cl100k_base")
    encoded: Final = tokenizer.encode("hello world")

    assert tokenizer.name == "cl100k_base"
    assert tokenizer.count("hello world") == len(encoded)
    assert tokenizer.decode(encoded) == "hello world"


def test_huggingface_codec_skips_special_tokens() -> None:
    tokenizer: Final = _native.Tokenizer.from_json(claude_json_str)
    encoded: Final = tokenizer.encode("<SOS>hello<EOT>")

    assert "<SOS>" in tokenizer.decode(encoded, skip_special_tokens=False)
    assert tokenizer.decode(encoded, skip_special_tokens=True) == "hello"


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


def test_huggingface_codec_rejects_tiktoken_only_calls() -> None:
    tokenizer: Final = _native.Tokenizer.from_json(claude_json_str)
    with pytest.raises(ValueError, match="requires a tiktoken encoding"):
        tokenizer.token_byte_values()
    with pytest.raises(ValueError, match="requires a Hugging Face tokenizer"):
        _native.Tokenizer.from_tiktoken("cl100k_base").get_vocab()


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


FAST_TEXTS: Final = (
    "",
    "hello world <|endoftext|>",
    "café 漢字 ع 🙂 line\r\n  indented 123456789",
    "<SOS>x<EOT> a\u0301 ﬁ",
)


def test_fast_counting_is_an_opt_in_over_the_same_loaded_tokenizer() -> None:
    for tokenizer in (
        _native.Tokenizer.from_tiktoken("cl100k_base"),
        _native.Tokenizer.from_tiktoken("o200k_base"),
        _native.Tokenizer.from_json(claude_json_str),
    ):
        assert [tokenizer.count(text, fast=True) for text in FAST_TEXTS] == [
            tokenizer.count(text) for text in FAST_TEXTS
        ]


@pytest.mark.parametrize(
    "name", ("cl100k_base", "o200k_base", "o200k_harmony", "p50k_base", "p50k_edit", "r50k_base", "gpt2")
)
@pytest.mark.asyncio
async def test_token_counter_counts_over_a_shared_tokenizer(name: str) -> None:
    messages: Final = [{"role": "user", "content": "hello wide world"}, {"role": "assistant", "content": "ok"}]
    body: Final = json.dumps({"model": "gpt-4", "messages": messages}).encode()
    tokenizer: Final = _native.Tokenizer.from_tiktoken(name)
    reference: Final = tiktoken.get_encoding(name)
    for text in FAST_TEXTS:
        assert tokenizer.count(text, fast=True) == tokenizer.count(text) == len(reference.encode_ordinary(text))

    exact: Final = await _native.TokenCounter.from_tokenizer(tokenizer).acount_request(body)
    fast: Final = await _native.TokenCounter.from_tokenizer(tokenizer, fast=True).acount_request(body)

    assert exact == fast
    assert exact["input_tokens"] == 3 + sum(
        3 + len(reference.encode_ordinary(message["role"])) + len(reference.encode_ordinary(message["content"]))
        for message in messages
    )


@pytest.mark.parametrize("configured", (False, True))
@pytest.mark.asyncio
async def test_fast_count_preserves_huggingface_configuration(configured: bool) -> None:
    reference: Final = ReferenceTokenizer.from_str(TOKENIZER_JSON)
    if configured:
        reference.enable_truncation(max_length=3)
        reference.enable_padding(pad_id=0, pad_token="[UNK]", length=5)
    tokenizer: Final = _native.Tokenizer.from_json(reference.to_str())
    counter: Final = _native.TokenCounter.from_tokenizer(tokenizer, fast=True)
    for text in ("", "Hello", "Hello World Hello World", "[BOS] Hello"):
        expected: Final = len(reference.encode(text))
        assert tokenizer.count(text, fast=True) == tokenizer.count(text) == expected
        result: Final = await counter.acount_request(json.dumps({"prompt": text}).encode())
        assert result["input_tokens"] == expected
