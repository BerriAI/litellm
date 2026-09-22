from collections.abc import Generator
from typing import Final

import pytest
import tiktoken
from tokenizers import Tokenizer

import litellm
from litellm.litellm_core_utils.tokenizer import HuggingFaceTokenizer, OpenAIEncoding
from litellm.rust_bridge import configuration, tokenizer
from litellm.utils import _select_tokenizer
from tests.test_litellm.litellm_core_utils.test_decode_special_tokens import TOKENIZER_JSON


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    tokenizer.TOKENIZER.reset()
    configuration.reset_rust_configuration()


@pytest.mark.parametrize("environment", (None, "0", "1"))
@pytest.mark.parametrize("process", (None, False, True))
def test_tokenizer_factories_follow_rollout(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, process: bool | None
) -> None:
    configuration.rust(process)
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)
    enabled: Final = environment == "1" if environment is not None else process is True
    encoding: Final = tokenizer.get_encoding("cl100k_base")
    custom: Final = litellm.create_tokenizer(TOKENIZER_JSON)
    reference: Final = Tokenizer.from_str(TOKENIZER_JSON)

    assert isinstance(encoding, OpenAIEncoding if enabled else tiktoken.Encoding)
    assert isinstance(custom["tokenizer"], HuggingFaceTokenizer if enabled else Tokenizer)
    assert encoding.encode("café 漢字 🙂") == tiktoken.get_encoding(encoding.name).encode("café 漢字 🙂")
    assert litellm.encode(text="Hello World", custom_tokenizer=custom) == reference.encode("Hello World").ids
    assert litellm.token_counter(text="Hello World", custom_tokenizer=custom) == len(reference.encode("Hello World"))


def test_missing_native_binding_keeps_python_tokenizer_api() -> None:
    configuration.rust(True)
    tokenizer.TOKENIZER.override(None)
    encoding: Final = tokenizer.get_encoding("cl100k_base")
    custom: Final = litellm.create_tokenizer(TOKENIZER_JSON)["tokenizer"]

    assert isinstance(encoding, tiktoken.Encoding)
    assert isinstance(custom, Tokenizer)
    custom.enable_padding(pad_id=0, pad_token="[UNK]")
    assert [item.ids for item in custom.encode_batch(["Hello", "Hello World"])] == [[3, 1, 0], [3, 1, 2]]


def test_cached_selection_follows_backend_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)
    configuration.rust(True)
    native: Final = _select_tokenizer("dispatch-fixture")["tokenizer"]
    configuration.rust(False)
    python: Final = _select_tokenizer("dispatch-fixture")["tokenizer"]

    assert isinstance(native, OpenAIEncoding)
    assert isinstance(python, tiktoken.Encoding)
    assert native.encode("hello") == python.encode("hello")


def test_declined_native_factory_falls_back_before_tokenizing() -> None:
    from litellm.rust_bridge._native import RustBridgeDeclined

    class UnavailableTokenizer:
        @staticmethod
        def from_json(json: str) -> None:
            raise RustBridgeDeclined("huggingface feature is disabled")

    configuration.rust(True)
    binding: Final = tokenizer._as_factory(UnavailableTokenizer)
    tokenizer.TOKENIZER.override(binding)
    custom: Final = litellm.create_tokenizer(TOKENIZER_JSON)

    assert isinstance(custom["tokenizer"], Tokenizer)
    assert (
        litellm.decode(tokens=litellm.encode(text="Hello World", custom_tokenizer=custom), custom_tokenizer=custom)
        == "Hello World"
    )
