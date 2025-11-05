"""
Integration test for CyberArk Conjur Secret Manager.
"""
import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath("../.."))
from unittest.mock import patch
from litellm._uuid import uuid

# Set up environment variables for testing
os.environ["CYBERARK_API_KEY"] = "2syke5r262b6je2f4et1x3jptmry3frfx83t65e6417zad632e5qq8a"
os.environ["CYBERARK_API_BASE"] = "http://0.0.0.0:8080"
os.environ["CYBERARK_ACCOUNT"] = "default"
os.environ["CYBERARK_USERNAME"] = "admin"

from litellm.secret_managers.cyberark_secret_manager import CyberArkSecretManager


@pytest.mark.asyncio
async def test_cyberark_write_and_read_secret():
    """
    Integration test: Write a secret to CyberArk Conjur and read it back to validate.
    """
    with patch("litellm.proxy.proxy_server.premium_user", True):
        # Create CyberArk secret manager instance
        cyberark_manager = CyberArkSecretManager()

        # Generate unique secret name and value
        secret_name = f"test-secret-{uuid.uuid4()}"
        secret_value = f"test-value-{uuid.uuid4()}"

        # Write the secret
        write_response = await cyberark_manager.async_write_secret(
            secret_name=secret_name,
            secret_value=secret_value,
        )
        # Avoid logging write_response to prevent leaking secret names

        # Validate write was successful
        assert write_response["status"] == "success"

        # Read the secret back
        read_value = cyberark_manager.sync_read_secret(secret_name=secret_name)
        # Don't log secret value in clear text

        # Validate the secret exists and has the correct value
        assert read_value is not None
        assert read_value == secret_value

