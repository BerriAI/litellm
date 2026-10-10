import json
from typing import Final

import pytest
import tiktoken
from tokenizers import Tokenizer as ReferenceTokenizer

from litellm.rust_bridge import _native
from litellm.utils import claude_json_str
from tests.unit.litellm_core_utils.test_decode_special_tokens import TOKENIZER_JSON

pytestmark = pytest.mark.requires_rust_extension


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
