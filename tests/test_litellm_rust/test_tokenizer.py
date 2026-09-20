from typing import Final

import pytest
import tiktoken

from litellm.rust_bridge import _native
from litellm.utils import claude_json_str

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


def test_tiktoken_encoding_for_model() -> None:
    assert _native.tiktoken_encoding_for_model("gpt-4o") == "o200k_base"
    assert _native.tiktoken_encoding_for_model("unknown-model") is None


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
