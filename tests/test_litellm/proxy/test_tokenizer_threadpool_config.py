import pytest

from litellm.proxy._types import ConfigGeneralSettings


def test_config_general_settings_tokenizer_threadpool_fields():
    """ConfigGeneralSettings accepts tokenizer threadpool config fields."""
    config = ConfigGeneralSettings(
        tokenizer_threadpool_max_threads=4,
        tokenizer_threadpool_min_input_size_bytes=512000,
        tokenizer_threadpool_timeout=5.0,
    )
    assert config.tokenizer_threadpool_max_threads == 4
    assert config.tokenizer_threadpool_min_input_size_bytes == 512000
    assert config.tokenizer_threadpool_timeout == 5.0


def test_config_general_settings_tokenizer_threadpool_defaults():
    """Tokenizer threadpool fields default to None or 0."""
    config = ConfigGeneralSettings()
    assert config.tokenizer_threadpool_max_threads is None
    assert config.tokenizer_threadpool_min_input_size_bytes is None
    assert config.tokenizer_threadpool_timeout == 0
