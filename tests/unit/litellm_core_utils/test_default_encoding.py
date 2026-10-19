import importlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

import litellm.litellm_core_utils.default_encoding as default_encoding

BUNDLED_TOKENIZERS: Final = Path(default_encoding.filename)


@pytest.fixture
def reload_default_encoding(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("CUSTOM_TIKTOKEN_CACHE_DIR", raising=False)
    yield
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("CUSTOM_TIKTOKEN_CACHE_DIR", raising=False)
    importlib.reload(default_encoding)


@pytest.mark.usefixtures("reload_default_encoding")
def test_tiktoken_cache_dir_defaults_to_bundled_tokenizers_for_non_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_NON_ROOT", "true")
    importlib.reload(default_encoding)
    assert Path(os.environ["TIKTOKEN_CACHE_DIR"]) == BUNDLED_TOKENIZERS
    assert BUNDLED_TOKENIZERS.name == "tokenizers"
    assert default_encoding.encoding.name == "cl100k_base"
    assert default_encoding.encoding.decode(default_encoding.encoding.encode("hello world")) == "hello world"


@pytest.mark.usefixtures("reload_default_encoding")
def test_custom_tiktoken_cache_dir_overrides_and_is_created(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_dir: Final = tmp_path / "tiktoken_cache"
    monkeypatch.setenv("CUSTOM_TIKTOKEN_CACHE_DIR", str(custom_dir))
    importlib.reload(default_encoding)
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(custom_dir)
    assert custom_dir.is_dir()
