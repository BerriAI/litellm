import json
from typing import Final

import pytest
import tiktoken

from litellm.rust_bridge import _native  # noqa: F401  # pyright: ignore[reportUnusedImport]  # the conftest import gate needs the compiled extension

FAST_TEXTS: Final = (
    "",
    "hello world <|endoftext|>",
    "caf\u00e9 \u6f22\u5b57 \u0639 \U0001f642 line\r\n  indented 123456789",
    "<SOS>x<EOT> a\u0301 \ufb01",
)


@pytest.mark.requires_rust_extension
@pytest.mark.parametrize(
    "name", ("cl100k_base", "o200k_base", "o200k_harmony", "p50k_base", "p50k_edit", "r50k_base", "gpt2")
)
@pytest.mark.asyncio
async def test_token_counter_counts_over_a_shared_tokenizer(
    name: str,
) -> None:
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
