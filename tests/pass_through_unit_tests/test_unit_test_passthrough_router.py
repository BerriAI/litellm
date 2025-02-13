import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath("../.."))  #

import unittest
from unittest.mock import patch
from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
    PassthroughEndpointRouter,
)

passthrough_endpoint_router = PassthroughEndpointRouter()

"""
1. Basic Usage
    - Set OpenAI, AssemblyAI, Anthropic, Cohere credentials
    - GET credentials from passthrough_endpoint_router

2. Basic Usage - when not using DB 
- No credentials set
- call GET credentials with provider name, assert that it reads the secret from the environment variable


3. Unit test for _get_default_env_variable_name_passthrough_endpoint
"""


class TestPassthroughEndpointRouter(unittest.TestCase):
    def setUp(self):
        self.router = PassthroughEndpointRouter()

    def test_set_and_get_credentials(self):
        """
        1. Basic Usage:
            - Set credentials for OpenAI, AssemblyAI, Anthropic, Cohere
            - GET credentials from passthrough_endpoint_router (from the memory store when available)
        """

        # OpenAI: standard (no region-specific logic)
        self.router.set_pass_through_credentials("openai", None, "openai_key")
        self.assertEqual(self.router.get_credentials("openai", None), "openai_key")

        # AssemblyAI: using an API base that contains 'eu' should trigger regional logic.
        api_base_eu = "https://api.eu.assemblyai.com"
        self.router.set_pass_through_credentials(
            "assemblyai", api_base_eu, "assemblyai_key"
        )
        # When calling get_credentials, pass the region "eu" (extracted from the API base)
        self.assertEqual(
            self.router.get_credentials("assemblyai", "eu"), "assemblyai_key"
        )

        # Anthropic: no region set
        self.router.set_pass_through_credentials("anthropic", None, "anthropic_key")
        self.assertEqual(
            self.router.get_credentials("anthropic", None), "anthropic_key"
        )

        # Cohere: no region set
        self.router.set_pass_through_credentials("cohere", None, "cohere_key")
        self.assertEqual(self.router.get_credentials("cohere", None), "cohere_key")

    def test_get_credentials_from_env(self):
        """
        2. Basic Usage - when not using the database:
            - No credentials set in memory
            - Call get_credentials with provider name and expect it to read from the environment variable (via get_secret_str)
        """
        # Patch the get_secret_str function within the router's module.
        with patch(
            "litellm.proxy.pass_through_endpoints.passthrough_endpoint_router.get_secret_str"
        ) as mock_get_secret:
            mock_get_secret.return_value = "env_openai_key"
            # For "openai", if credentials are not set, it should fallback to the env variable.
            result = self.router.get_credentials("openai", None)
            self.assertEqual(result, "env_openai_key")
            mock_get_secret.assert_called_once_with("OPENAI_API_KEY")

        with patch(
            "litellm.proxy.pass_through_endpoints.passthrough_endpoint_router.get_secret_str"
        ) as mock_get_secret:
            mock_get_secret.return_value = "env_cohere_key"
            result = self.router.get_credentials("cohere", None)
            self.assertEqual(result, "env_cohere_key")
            mock_get_secret.assert_called_once_with("COHERE_API_KEY")

        with patch(
            "litellm.proxy.pass_through_endpoints.passthrough_endpoint_router.get_secret_str"
        ) as mock_get_secret:
            mock_get_secret.return_value = "env_anthropic_key"
            result = self.router.get_credentials("anthropic", None)
            self.assertEqual(result, "env_anthropic_key")
            mock_get_secret.assert_called_once_with("ANTHROPIC_API_KEY")

        with patch(
            "litellm.proxy.pass_through_endpoints.passthrough_endpoint_router.get_secret_str"
        ) as mock_get_secret:
            mock_get_secret.return_value = "env_azure_key"
            result = self.router.get_credentials("azure", None)
            self.assertEqual(result, "env_azure_key")
            mock_get_secret.assert_called_once_with("AZURE_API_KEY")

    def test_default_env_variable_method(self):
        """
        3. Unit test for _get_default_env_variable_name_passthrough_endpoint:
            - Should return the provider in uppercase followed by _API_KEY.
        """
        self.assertEqual(
            PassthroughEndpointRouter._get_default_env_variable_name_passthrough_endpoint(
                "openai"
            ),
            "OPENAI_API_KEY",
        )
        self.assertEqual(
            PassthroughEndpointRouter._get_default_env_variable_name_passthrough_endpoint(
                "assemblyai"
            ),
            "ASSEMBLYAI_API_KEY",
        )
        self.assertEqual(
            PassthroughEndpointRouter._get_default_env_variable_name_passthrough_endpoint(
                "anthropic"
            ),
            "ANTHROPIC_API_KEY",
        )
        self.assertEqual(
            PassthroughEndpointRouter._get_default_env_variable_name_passthrough_endpoint(
                "cohere"
            ),
            "COHERE_API_KEY",
        )
