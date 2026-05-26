import importlib
import os
from unittest.mock import MagicMock, patch

import litellm.litellm_core_utils.default_encoding as default_encoding


def _reload_default_encoding(monkeypatch, **env_overrides):
    """
    Helper to reload default_encoding with a clean TIKTOKEN_CACHE_DIR and
    specific environment overrides.
    """
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("CUSTOM_TIKTOKEN_CACHE_DIR", raising=False)
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)
    importlib.reload(default_encoding)


def test_default_encoding_uses_bundled_tokenizers_by_default(monkeypatch):
    """
    TIKTOKEN_CACHE_DIR should point at the bundled tokenizers directory
    when no CUSTOM_TIKTOKEN_CACHE_DIR is set, even in non-root environments.
    """
    _reload_default_encoding(monkeypatch, LITELLM_NON_ROOT="true")

    assert "TIKTOKEN_CACHE_DIR" in os.environ
    cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    assert "tokenizers" in cache_dir


def test_custom_tiktoken_cache_dir_override(monkeypatch, tmp_path):
    """
    CUSTOM_TIKTOKEN_CACHE_DIR must override the default bundled directory
    and the directory should be created if it does not exist.
    Reload with an empty custom dir would otherwise trigger tiktoken to
    download the vocab; we patch get_encoding so the test is offline-safe
    and does not depend on tiktoken's in-memory cache state.
    """
    custom_dir = tmp_path / "tiktoken_cache"
    with patch(
        "litellm.litellm_core_utils.default_encoding.tiktoken.get_encoding",
        return_value=MagicMock(),
    ):
        _reload_default_encoding(monkeypatch, CUSTOM_TIKTOKEN_CACHE_DIR=str(custom_dir))

    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    assert cache_dir == str(custom_dir)
    assert os.path.isdir(cache_dir)

    # Restore module to a clean state so default_encoding.encoding is a real
    # tiktoken Encoding, not the MagicMock, for any test that runs after this.
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("CUSTOM_TIKTOKEN_CACHE_DIR", raising=False)
    importlib.reload(default_encoding)
