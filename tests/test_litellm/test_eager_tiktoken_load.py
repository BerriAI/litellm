"""
Test for LITELLM_DISABLE_LAZY_LOADING environment variable.

This test verifies that when LITELLM_DISABLE_LAZY_LOADING is set,
encoding is loaded at import time (pre-#18070 behavior) instead of lazy loading.

This addresses issue #18659: VCR cassette creation broken by lazy loading.
For now, this only affects encoding as it was the only reported issue.
"""
import os
import sys

import pytest


def _litellm_module_names():
    return [name for name in sys.modules if name == "litellm" or name.startswith("litellm.")]


def _clear_litellm_modules():
    for module_name in _litellm_module_names():
        del sys.modules[module_name]


@pytest.fixture(autouse=True)
def cleanup_env_and_modules():
    """Restore litellm imports and environment after each test."""
    original_modules = {
        name: module for name, module in sys.modules.items() if name == "litellm" or name.startswith("litellm.")
    }
    original_disable_lazy_loading = os.environ.get("LITELLM_DISABLE_LAZY_LOADING")
    original_tiktoken_cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")

    yield

    _clear_litellm_modules()
    sys.modules.update(original_modules)

    if original_disable_lazy_loading is None:
        os.environ.pop("LITELLM_DISABLE_LAZY_LOADING", None)
    else:
        os.environ["LITELLM_DISABLE_LAZY_LOADING"] = original_disable_lazy_loading

    if original_tiktoken_cache_dir is None:
        os.environ.pop("TIKTOKEN_CACHE_DIR", None)
    else:
        os.environ["TIKTOKEN_CACHE_DIR"] = original_tiktoken_cache_dir


def test_eager_loading_enabled():
    """Test that encoding is loaded at import time when env var is set"""
    os.environ["LITELLM_DISABLE_LAZY_LOADING"] = "1"

    _clear_litellm_modules()

    import litellm

    assert hasattr(litellm, "encoding"), "Encoding should be available when eager loading is enabled"

    encoding = litellm.encoding
    assert encoding is not None, "Encoding should not be None"

    tokens = encoding.encode("Hello, world!")
    assert len(tokens) > 0, "Encoding should work"


def test_eager_loading_env_var_values():
    """Test that various env var values enable eager loading"""
    values = ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]

    for value in values:
        os.environ["LITELLM_DISABLE_LAZY_LOADING"] = value
        _clear_litellm_modules()

        import litellm

        assert hasattr(litellm, "encoding"), f"Encoding should be available for value: {value}"
        encoding = litellm.encoding
        tokens = encoding.encode("test")
        assert len(tokens) > 0


def test_lazy_loading_default():
    """Test that encoding is lazy loaded by default (when env var is not set)"""
    os.environ.pop("LITELLM_DISABLE_LAZY_LOADING", None)

    _clear_litellm_modules()

    import litellm

    encoding = litellm.encoding

    tokens = encoding.encode("Hello, world!")
    assert len(tokens) > 0, "Encoding should work"


def test_tiktoken_cache_dir_set_on_lazy_load():
    """Test that TIKTOKEN_CACHE_DIR is set when encoding is lazy loaded.

    This ensures the local tiktoken cache is used instead of downloading
    from the internet. Regression test for issue #19768.
    """
    os.environ.pop("LITELLM_DISABLE_LAZY_LOADING", None)
    os.environ.pop("TIKTOKEN_CACHE_DIR", None)

    _clear_litellm_modules()

    import litellm

    _ = litellm.encoding

    assert "TIKTOKEN_CACHE_DIR" in os.environ, "TIKTOKEN_CACHE_DIR should be set after lazy loading encoding"
    cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    assert "tokenizers" in cache_dir, f"TIKTOKEN_CACHE_DIR should point to tokenizers directory, got: {cache_dir}"
