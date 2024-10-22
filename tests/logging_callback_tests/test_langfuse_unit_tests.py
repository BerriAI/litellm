import json
import os
import sys
from datetime import datetime

from pydantic.main import Model

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
import litellm
import asyncio
import logging
from litellm._logging import verbose_logger
from litellm.integrations.langfuse import (
    LangFuseLogger,
    get_langfuse_logger_for_request,
    get_dynamic_langfuse_logging_config,
    _dynamic_langfuse_credentials_are_passed,
)
from litellm.types.utils import StandardCallbackDynamicParams
from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache


@pytest.fixture
def dynamic_logging_cache():
    return DynamicLoggingCache()


def test_get_langfuse_logger_for_request_with_dynamic_params(dynamic_logging_cache):
    """
    If StandardCallbackDynamicParams contain langfuse credentials the returned Langfuse logger should use the dynamic params

    the new Langfuse logger should be cached
    """
    standard_params = StandardCallbackDynamicParams(
        langfuse_public_key="test_public_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )

    result = get_langfuse_logger_for_request(
        standard_callback_dynamic_params=standard_params,
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=None,
    )

    assert isinstance(result, LangFuseLogger)
    assert result.public_key == "test_public_key"
    assert result.secret_key == "test_secret"
    assert result.langfuse_host == "https://test.langfuse.com"

    # Check if the logger is cached
    cached_logger = dynamic_logging_cache.get_cache(
        credentials={
            "langfuse_public_key": "test_public_key",
            "langfuse_secret": "test_secret",
            "langfuse_host": "https://test.langfuse.com",
        },
        service_name="langfuse",
    )
    assert cached_logger is result
