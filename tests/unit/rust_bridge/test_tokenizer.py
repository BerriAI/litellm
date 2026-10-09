from types import ModuleType
from typing import Final
from unittest.mock import patch

import pytest
import tiktoken
from tokenizers import Tokenizer

from litellm.litellm_core_utils.tokenizer import HuggingFaceTokenizer, OpenAIEncoding
from litellm.rust_bridge import tokenizer
from litellm.utils import claude_json_str
from tests.unit.litellm_core_utils.test_decode_special_tokens import TOKENIZER_JSON

TEXTS: Final = ("hello <|endoftext|> world", "café 漢字 🙂", "  def f():\n    return 1\n", "<SOS>hello<EOT> again")


@pytest.mark.requires_rust_extension
@pytest.mark.parametrize("name", ("cl100k_base", "o200k_base"))
@pytest.mark.parametrize("text", TEXTS)
def test_native_encoding_matches_tiktoken(name: str, text: str) -> None:
    native: Final = tokenizer.native_encoding(name)
    if native is None:
        pytest.skip("native extension is not built")
    encoding: Final = OpenAIEncoding.wrap(native)
    reference: Final = tiktoken.get_encoding(name)

    ids: Final = encoding.encode(text, disallowed_special=())
    assert ids == reference.encode(text, disallowed_special=())
    assert encoding.decode(ids) == reference.decode(ids)


@pytest.mark.requires_rust_extension
@pytest.mark.parametrize("text", TEXTS)
def test_native_anthropic_tokenizer_matches_python(text: str) -> None:
    native: Final = tokenizer.native_anthropic()
    if native is None:
        pytest.skip("native extension is not built")
    reference: Final = Tokenizer.from_str(claude_json_str)

    ids: Final = HuggingFaceTokenizer(native).encode(text).ids
    assert ids == reference.encode(text).ids
    assert HuggingFaceTokenizer(native).decode(ids) == reference.decode(ids)


@pytest.mark.requires_rust_extension
def test_native_custom_tokenizer_matches_python() -> None:
    factory: Final = tokenizer.TOKENIZER.load()
    if factory is None:
        pytest.skip("native extension is not built")
    native: Final = HuggingFaceTokenizer(factory.from_json(TOKENIZER_JSON))
    reference: Final = Tokenizer.from_str(TOKENIZER_JSON)

    assert native.encode("Hello World").ids == reference.encode("Hello World").ids
    assert native.decode(reference.encode("Hello World").ids) == reference.decode(reference.encode("Hello World").ids)


FAST_TEXTS: Final = (
    "",
    "hello world <|endoftext|>",
    "caf\u00e9 \u6f22\u5b57 \u0639 \U0001f642 line\r\n  indented 123456789",
    "<SOS>x<EOT> a\u0301 \ufb01",
)


@pytest.mark.requires_rust_extension
def test_tiktoken_codec_round_trips_and_counts(native: ModuleType) -> None:
    tokenizer: Final = native.Tokenizer.from_tiktoken("cl100k_base")
    encoded: Final = tokenizer.encode("hello world")

    assert tokenizer.name == "cl100k_base"
    assert tokenizer.count("hello world") == len(encoded)
    assert tokenizer.decode(encoded) == "hello world"


@pytest.mark.requires_rust_extension
def test_tiktoken_codec_keeps_the_requested_encoding_name(native: ModuleType) -> None:
    assert native.Tokenizer.from_tiktoken("gpt2").name == "gpt2"
    assert native.Tokenizer.from_tiktoken("r50k_base").name == "r50k_base"
    assert native.Tokenizer.from_tiktoken("gpt2").encode("hi") == native.Tokenizer.from_tiktoken("r50k_base").encode(
        "hi"
    )


@pytest.mark.requires_rust_extension
def test_tiktoken_codec_exposes_its_vocabulary(native: ModuleType) -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    tokenizer: Final = native.Tokenizer.from_tiktoken("cl100k_base")

    assert tokenizer.special_tokens() == reference._special_tokens
    assert tokenizer.max_token_value() == reference.max_token_value
    assert tokenizer.token_byte_values() == reference.token_byte_values()
    assert tokenizer.encode_single_token(b"hello") == reference.encode_single_token("hello")
    assert tokenizer.is_special_token(reference.eot_token) and not tokenizer.is_special_token(0)
    with pytest.raises(KeyError):
        tokenizer.encode_single_token(b"<|not-a-token|>")


@pytest.mark.requires_rust_extension
def test_unknown_tiktoken_encoding_raises_value_error(native: ModuleType) -> None:
    with pytest.raises(ValueError, match="unsupported tokenizer"):
        native.Tokenizer.from_tiktoken("unknown-encoding")


@pytest.mark.requires_rust_extension
def test_tiktoken_codec_decodes_truncated_unicode_like_python(native: ModuleType) -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    tokenizer: Final = native.Tokenizer.from_tiktoken(reference.name)
    encoded: Final = reference.encode("🙂漢字")

    assert tuple(tokenizer.decode(encoded[:end]) for end in range(1, len(encoded) + 1)) == tuple(
        reference.decode(encoded[:end]) for end in range(1, len(encoded) + 1)
    )


@pytest.mark.requires_rust_extension
def test_huggingface_codec_skips_special_tokens(native: ModuleType) -> None:
    tokenizer: Final = native.Tokenizer.from_json(claude_json_str)
    encoded: Final = tokenizer.encode("<SOS>hello<EOT>")

    assert "<SOS>" in tokenizer.decode(encoded, skip_special_tokens=False)
    assert tokenizer.decode(encoded, skip_special_tokens=True) == "hello"


@pytest.mark.requires_rust_extension
def test_huggingface_codec_rejects_tiktoken_only_calls(native: ModuleType) -> None:
    tokenizer: Final = native.Tokenizer.from_json(claude_json_str)
    with pytest.raises(ValueError, match="requires a tiktoken encoding"):
        tokenizer.token_byte_values()
    with pytest.raises(ValueError, match="requires a Hugging Face tokenizer"):
        native.Tokenizer.from_tiktoken("cl100k_base").get_vocab()


@pytest.mark.requires_rust_extension
def test_fast_counting_is_an_opt_in_over_the_same_loaded_tokenizer(native: ModuleType) -> None:
    for tokenizer in (
        native.Tokenizer.from_tiktoken("cl100k_base"),
        native.Tokenizer.from_tiktoken("o200k_base"),
        native.Tokenizer.from_json(claude_json_str),
    ):
        assert [tokenizer.count(text, fast=True) for text in FAST_TEXTS] == [
            tokenizer.count(text) for text in FAST_TEXTS
        ]


def test_python_tokenizer_missing_dependency_is_actionable() -> None:
    with patch.dict("sys.modules", {"tokenizers": None}):
        with pytest.raises(ImportError, match="pip install tokenizers") as error:
            tokenizer._python_tokenizer()
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert error.value.__cause__.name == "tokenizers"


def test_python_tokenizer_factory_preserves_installed_interface() -> None:
    result: Final = tokenizer._python_tokenizer().from_str(TOKENIZER_JSON)
    assert result.encode("Hello World").ids == Tokenizer.from_str(TOKENIZER_JSON).encode("Hello World").ids


def test_python_tokenizer_preserves_unrelated_import_failure() -> None:
    failure: Final = ModuleNotFoundError("broken tokenizer installation", name="unrelated_dependency")
    with patch("builtins.__import__", side_effect=failure):
        with pytest.raises(ModuleNotFoundError) as error:
            tokenizer._python_tokenizer()
    assert error.value is failure
