import importlib
import os
import shutil
import sys
import tempfile
import types
from typing import Final

# Ensure _native stub exists if running in an uncompiled source environment
if "litellm.rust_bridge._native" not in sys.modules:
    mock_native = types.ModuleType("litellm.rust_bridge._native")
    sys.modules["litellm.rust_bridge._native"] = mock_native


def test_default_encoding_directory_created() -> None:
    """Verify that default_encoding ensures the cache directory exists and is usable."""
    test_cache_dir: Final = os.path.join(tempfile.gettempdir(), "test_litellm_encoding_cache")
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir, ignore_errors=True)

    original_env: Final = os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR")
    try:
        os.environ["CUSTOM_TIKTOKEN_CACHE_DIR"] = test_cache_dir
        import litellm.litellm_core_utils.default_encoding as default_encoding

        importlib.reload(default_encoding)

        assert os.path.exists(test_cache_dir), "Cache dir should be created automatically"
        assert os.environ.get("TIKTOKEN_CACHE_DIR") == test_cache_dir
        assert default_encoding.encoding is not None
    finally:
        if original_env is not None:
            os.environ["CUSTOM_TIKTOKEN_CACHE_DIR"] = original_env
        else:
            os.environ.pop("CUSTOM_TIKTOKEN_CACHE_DIR", None)
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir, ignore_errors=True)


def test_default_encoding_fallback_on_uncreatable_dir() -> None:
    """Verify fallback to secure temporary directory when configured cache directory cannot be created."""
    invalid_dir: Final = "Z:\\non_existent_drive_for_test_12345\\tokenizers"
    original_env: Final = os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR")
    try:
        os.environ["CUSTOM_TIKTOKEN_CACHE_DIR"] = invalid_dir
        import litellm.litellm_core_utils.default_encoding as default_encoding

        importlib.reload(default_encoding)

        active_cache: Final = os.environ.get("TIKTOKEN_CACHE_DIR")
        assert active_cache is not None
        assert active_cache != invalid_dir, "Should have fallen back to a valid temp cache"
        assert os.path.exists(active_cache), "Fallback cache directory must exist"
        assert default_encoding.encoding is not None
    finally:
        if original_env is not None:
            os.environ["CUSTOM_TIKTOKEN_CACHE_DIR"] = original_env
        else:
            os.environ.pop("CUSTOM_TIKTOKEN_CACHE_DIR", None)
