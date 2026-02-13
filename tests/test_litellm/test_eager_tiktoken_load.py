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


def test_eager_loading_enabled():
    """Test that encoding is loaded at import time when env var is set"""
    # Set environment variable
    os.environ["LITELLM_DISABLE_LAZY_LOADING"] = "1"
    
    # Clear any cached modules to ensure fresh import
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith("litellm")]
    for module in modules_to_clear:
        del sys.modules[module]
    
    # Import litellm - encoding should be loaded immediately
    import litellm
    
    # Check that encoding is available (not lazy loaded)
    assert hasattr(litellm, "encoding"), "Encoding should be available when eager loading is enabled"
    
    # Verify it's actually the encoding object
    encoding = litellm.encoding
    assert encoding is not None, "Encoding should not be None"
    
    # Test that it works
    tokens = encoding.encode("Hello, world!")
    assert len(tokens) > 0, "Encoding should work"


def test_eager_loading_env_var_values():
    """Test that various env var values enable eager loading"""
    values = ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]
    
    for value in values:
        os.environ["LITELLM_DISABLE_LAZY_LOADING"] = value
        
        # Clear modules
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith("litellm")]
        for module in modules_to_clear:
            del sys.modules[module]
        
        import litellm
        assert hasattr(litellm, "encoding"), f"Encoding should be available for value: {value}"
        encoding = litellm.encoding
        tokens = encoding.encode("test")
        assert len(tokens) > 0


def test_lazy_loading_default():
    """Test that encoding is lazy loaded by default (when env var is not set)"""
    # Remove environment variable if set
    if "LITELLM_DISABLE_LAZY_LOADING" in os.environ:
        del os.environ["LITELLM_DISABLE_LAZY_LOADING"]
    
    # Clear any cached modules
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith("litellm")]
    for module in modules_to_clear:
        del sys.modules[module]
    
    # Import litellm - encoding should NOT be loaded yet
    import litellm
    
    # Encoding should be accessible via __getattr__ (lazy loading)
    encoding = litellm.encoding  # This triggers lazy loading
    
    # Verify it works
    tokens = encoding.encode("Hello, world!")
    assert len(tokens) > 0, "Encoding should work"


def test_tiktoken_cache_dir_set_on_lazy_load():
    """Test that TIKTOKEN_CACHE_DIR is set when encoding is lazy loaded.

    This ensures the local tiktoken cache is used instead of downloading
    from the internet. Regression test for issue #19768.
    """
    # Remove environment variables to ensure clean state
    if "LITELLM_DISABLE_LAZY_LOADING" in os.environ:
        del os.environ["LITELLM_DISABLE_LAZY_LOADING"]
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        del os.environ["TIKTOKEN_CACHE_DIR"]

    # Clear any cached modules
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith("litellm")]
    for module in modules_to_clear:
        del sys.modules[module]

    # Import litellm fresh
    import litellm

    # Access encoding (triggers lazy load)
    _ = litellm.encoding

    # Verify TIKTOKEN_CACHE_DIR is now set and points to local tokenizers
    assert "TIKTOKEN_CACHE_DIR" in os.environ, "TIKTOKEN_CACHE_DIR should be set after lazy loading encoding"
    cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    assert "tokenizers" in cache_dir, f"TIKTOKEN_CACHE_DIR should point to tokenizers directory, got: {cache_dir}"


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variable after each test"""
    yield
    if "LITELLM_DISABLE_LAZY_LOADING" in os.environ:
        del os.environ["LITELLM_DISABLE_LAZY_LOADING"]

