import importlib
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
import tiktoken.load
from tiktoken_ext import openai_public

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


@pytest.mark.parametrize(
    "constructor",
    (openai_public.cl100k_base, openai_public.p50k_base, openai_public.o200k_base),
    ids=("cl100k_base", "p50k_base", "o200k_base"),
)
def test_bundled_tiktoken_cache_loads_without_downloads_or_rewrites(
    constructor: Callable[[], Mapping[str, object]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir: Final = tmp_path / "tokenizers"
    shutil.copytree(BUNDLED_TOKENIZERS, cache_dir)
    original_files: Final = {path.name: path.read_bytes() for path in cache_dir.iterdir() if path.is_file()}
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_dir))

    def refuse_download(url: str) -> bytes:
        pytest.fail(f"Bundled tiktoken cache required a download from {url}")

    monkeypatch.setattr(tiktoken.load, "read_file", refuse_download)

    assert constructor()["mergeable_ranks"]
    assert {path.name: path.read_bytes() for path in cache_dir.iterdir() if path.is_file()} == original_files
