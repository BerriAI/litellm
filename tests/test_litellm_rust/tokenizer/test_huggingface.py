from typing import Final

import pytest

from litellm.rust_bridge import _native
from litellm.utils import claude_json_str

pytestmark = pytest.mark.requires_rust_extension


def test_huggingface_codec_skips_special_tokens() -> None:
    tokenizer: Final = _native.Tokenizer.from_json(claude_json_str)
    encoded: Final = tokenizer.encode("<SOS>hello<EOT>")

    assert "<SOS>" in tokenizer.decode(encoded, skip_special_tokens=False)
    assert tokenizer.decode(encoded, skip_special_tokens=True) == "hello"


def test_huggingface_codec_rejects_tiktoken_only_calls() -> None:
    tokenizer: Final = _native.Tokenizer.from_json(claude_json_str)
    with pytest.raises(ValueError, match="requires a tiktoken encoding"):
        tokenizer.token_byte_values()
    with pytest.raises(ValueError, match="requires a Hugging Face tokenizer"):
        _native.Tokenizer.from_tiktoken("cl100k_base").get_vocab()
