from collections.abc import Callable
from functools import cache
from importlib import resources
from typing import Final

from litellm.litellm_core_utils.default_encoding import encoding as cl100k_base
from tokenizers import Tokenizer

CHINESE_TEXT: Final = "深度求索的分词器会把这段中文切成更少的词元。" * 20
SHORT_TEXT: Final = "Find record 14."


@cache
def _deepseek_tokenizer() -> Tokenizer:
    bundled: Final = resources.files("litellm.litellm_core_utils.tokenizers").joinpath("deepseek_tokenizer.json")
    return Tokenizer.from_str(bundled.read_text(encoding="utf-8"))


def deepseek_tokens(text: str) -> int:
    return len(_deepseek_tokenizer().encode(text).ids)


def cl100k_tokens(text: str) -> int:
    return len(cl100k_base.encode(text))


def chinese_minus_short(tokens: Callable[[str], int]) -> int:
    return tokens(CHINESE_TEXT) - tokens(SHORT_TEXT)


def assert_counted_with_deepseek(counted: int, expected: Callable[[Callable[[str], int]], int]) -> None:
    cl100k_count: Final = expected(cl100k_tokens)
    assert counted != cl100k_count, f"counted {counted}, the cl100k_base count, not DeepSeek's tokenizer"
    deepseek_count: Final = expected(deepseek_tokens)
    assert counted == deepseek_count, f"counted {counted}, DeepSeek's tokenizer counts {deepseek_count}"
