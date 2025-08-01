import json
import os
import sys
from typing import Optional

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import asyncio
from unittest.mock import patch

import pytest

import litellm
from litellm.constants import MAX_LANGFUSE_INITIALIZED_CLIENTS
from litellm.integrations.langfuse import langfuse as langfuse_module
from litellm.integrations.langfuse.langfuse import LangFuseLogger


def test_max_langfuse_clients_limit():
    """
    Test that the max langfuse clients limit is respected when initializing multiple clients
    """
    # Set max clients to 2 for testing
    # Patch both the constants module and the langfuse module to ensure it works in all environments
    with patch("litellm.constants.MAX_LANGFUSE_INITIALIZED_CLIENTS", 2), \
         patch("litellm.integrations.langfuse.langfuse.MAX_LANGFUSE_INITIALIZED_CLIENTS", 2):
        # Reset the counter
        litellm.initialized_langfuse_clients = 0

        # Mock the Langfuse constructor to handle unsupported parameters
        with patch("langfuse.Langfuse.__init__") as mock_init:
            def side_effect(*args, **kwargs):
                # Remove unsupported parameters
                kwargs.pop('sdk_integration', None)
                # Just return None as __init__ should
                return None
            
            mock_init.side_effect = side_effect

            # First client should succeed
            logger1 = LangFuseLogger(
                langfuse_public_key="test_key_1",
                langfuse_secret="test_secret_1",
                langfuse_host="https://test1.langfuse.com",
            )
            assert litellm.initialized_langfuse_clients == 1

            # Second client should succeed
            logger2 = LangFuseLogger(
                langfuse_public_key="test_key_2",
                langfuse_secret="test_secret_2",
                langfuse_host="https://test2.langfuse.com",
            )
            assert litellm.initialized_langfuse_clients == 2

            # Third client should fail with exception
            with pytest.raises(Exception) as exc_info:
                logger3 = LangFuseLogger(
                    langfuse_public_key="test_key_3",
                    langfuse_secret="test_secret_3",
                    langfuse_host="https://test3.langfuse.com",
                )

            # Verify the error message contains the expected text
            assert "Max langfuse clients reached" in str(exc_info.value)
            # Counter should still be 2 (third client failed to initialize)
            assert litellm.initialized_langfuse_clients == 2
