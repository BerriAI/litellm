"""
Test that add_deployment works without master_key set.

This test verifies the fix for the bug where saving LLM spend logs
failed when master_key was None. [https://github.com/BerriAI/litellm/issues/16428]
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest.mark.asyncio
async def test_add_deployment_without_master_key():
    """
    Test that add_deployment() works when master_key is None.

    This should not raise an exception anymore after the fix.
    Previously, it would raise: "Master key is not initialized or formatted"
    """
    # Set master_key to None
    with patch("litellm.proxy.proxy_server.master_key", None):
        # Mock the required dependencies
        mock_prisma_client = MagicMock(spec=PrismaClient)
        mock_prisma_client.db = MagicMock()
        mock_prisma_client.db.litellm_config = MagicMock()
        mock_prisma_client.db.litellm_config.find_first = AsyncMock(return_value=None)

        mock_proxy_logging = MagicMock(spec=ProxyLogging)

        # Create ProxyConfig instance
        proxy_config = ProxyConfig()

        # Mock the internal methods to avoid actual DB calls
        proxy_config._should_load_db_object = MagicMock(return_value=False)
        proxy_config._init_non_llm_objects_in_db = AsyncMock()

        # This should NOT raise an exception
        try:
            await proxy_config.add_deployment(
                prisma_client=mock_prisma_client,
                proxy_logging_obj=mock_proxy_logging,
            )
            # If we get here, the test passed
            assert True
        except ValueError as e:
            if "Master key is not initialized" in str(e):
                pytest.fail(f"add_deployment raised ValueError about master_key: {e}")
            raise
        except Exception as e:
            if "Master key is not initialized" in str(e):
                pytest.fail(f"add_deployment raised exception about master_key: {e}")
            raise


@pytest.mark.asyncio
async def test_add_deployment_without_salt_key_or_master_key():
    """
    Test that add_deployment() works when both master_key and LITELLM_SALT_KEY are None.

    This tests the scenario where the user runs proxy without any encryption keys,
    such as in a local/dev environment or when just saving spend logs.
    """
    # Remove LITELLM_SALT_KEY from environment
    old_salt_key = os.environ.pop("LITELLM_SALT_KEY", None)

    try:
        # Set master_key to None
        with patch("litellm.proxy.proxy_server.master_key", None):
            # Mock the required dependencies
            mock_prisma_client = MagicMock(spec=PrismaClient)
            mock_prisma_client.db = MagicMock()
            mock_prisma_client.db.litellm_config = MagicMock()
            mock_prisma_client.db.litellm_config.find_first = AsyncMock(return_value=None)

            mock_proxy_logging = MagicMock(spec=ProxyLogging)

            # Create ProxyConfig instance
            proxy_config = ProxyConfig()

            # Mock the internal methods
            proxy_config._should_load_db_object = MagicMock(return_value=False)
            proxy_config._init_non_llm_objects_in_db = AsyncMock()

            # This should NOT raise an exception
            try:
                await proxy_config.add_deployment(
                    prisma_client=mock_prisma_client,
                    proxy_logging_obj=mock_proxy_logging,
                )
                assert True
            except ValueError as e:
                if "Master key is not initialized" in str(e) or "Encryption key is not initialized" in str(e):
                    pytest.fail(f"add_deployment raised ValueError about encryption key: {e}")
                raise
            except Exception as e:
                if "Master key is not initialized" in str(e) or "Encryption key is not initialized" in str(e):
                    pytest.fail(f"add_deployment raised exception about encryption key: {e}")
                raise
    finally:
        # Restore LITELLM_SALT_KEY if it was set
        if old_salt_key:
            os.environ["LITELLM_SALT_KEY"] = old_salt_key


def test_add_deployment_sync_without_master_key():
    """
    Test that _add_deployment() (sync version) works when master_key is None.

    This tests the internal method used by add_deployment().
    """
    # Set master_key to None
    with patch("litellm.proxy.proxy_server.master_key", None):
        with patch("litellm.proxy.proxy_server.llm_router", None):
            # Create ProxyConfig instance
            proxy_config = ProxyConfig()

            # Call _add_deployment with empty model list
            # This should NOT raise an exception
            try:
                result = proxy_config._add_deployment(db_models=[])
                # Should return 0 because llm_router is None
                assert result == 0
            except Exception as e:
                if "Master key is not initialized" in str(e):
                    pytest.fail(f"_add_deployment raised exception about master_key: {e}")
                raise
