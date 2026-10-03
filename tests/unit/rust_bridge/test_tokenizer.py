from collections.abc import Callable
import sys
from typing import Final

import pytest
import tiktoken
from tokenizers import Tokenizer

from litellm.litellm_core_utils.tokenizer import HuggingFaceTokenizer, OpenAIEncoding
from litellm.rust_bridge import tokenizer
from litellm.utils import claude_json_str
from tests.unit.litellm_core_utils.test_decode_special_tokens import TOKENIZER_JSON

TEXTS: Final = ("hello <|endoftext|> world", "café 漢字 🙂", "  def f():\n    return 1\n", "<SOS>hello<EOT> again")


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


@pytest.mark.parametrize("text", TEXTS)
def test_native_anthropic_tokenizer_matches_python(text: str) -> None:
    native: Final = tokenizer.native_anthropic()
    if native is None:
        pytest.skip("native extension is not built")
    reference: Final = Tokenizer.from_str(claude_json_str)

    ids: Final = HuggingFaceTokenizer(native).encode(text).ids
    assert ids == reference.encode(text).ids
    assert HuggingFaceTokenizer(native).decode(ids) == reference.decode(ids)


def test_native_custom_tokenizer_matches_python() -> None:
    factory: Final = tokenizer.TOKENIZER.load()
    if factory is None:
        pytest.skip("native extension is not built")
    native: Final = HuggingFaceTokenizer(factory.from_json(TOKENIZER_JSON))
    reference: Final = Tokenizer.from_str(TOKENIZER_JSON)

    assert native.encode("Hello World").ids == reference.encode("Hello World").ids
    assert native.decode(reference.encode("Hello World").ids) == reference.decode(reference.encode("Hello World").ids)


def test_python_tokenizer_missing_dependency_explains_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    with pytest.raises(ImportError, match=r"litellm\[tokenizers\]") as error:
        tokenizer._python_huggingface_tokenizer()

    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert error.value.__cause__.name == "tokenizers"


def test_python_tokenizer_preserves_custom_encoding() -> None:
    custom: Final = tokenizer._python_huggingface_tokenizer().from_str(TOKENIZER_JSON)

    assert custom.encode("Hello World").ids == [3, 1, 2]
    assert custom.decode([3, 1, 2]) == "Hello World"


def test_python_tokenizer_preserves_invalid_definition_error() -> None:
    with pytest.raises(Exception, match=r".+") as error:
        tokenizer._python_huggingface_tokenizer().from_str("not json")

    assert not isinstance(error.value, ImportError)


def test_python_tokenizer_preserves_unrelated_missing_module(
    fail_optional_import: Callable[[str, ModuleNotFoundError], None],
) -> None:
    failure: Final = ModuleNotFoundError("unrelated dependency is broken", name="unrelated_dependency")
    fail_optional_import("tokenizers", failure)
    with pytest.raises(ModuleNotFoundError) as error:
        tokenizer._python_huggingface_tokenizer()

    assert error.value is failure
