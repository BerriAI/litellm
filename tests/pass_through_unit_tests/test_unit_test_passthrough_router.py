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
from litellm.types.passthrough_endpoints.vertex_ai import VertexPassThroughCredentials

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

    def test_get_deployment_key(self):
        """Test _get_deployment_key with various inputs"""
        router = PassthroughEndpointRouter()

        # Test with valid inputs
        key = router._get_deployment_key("test-project", "us-central1")
        assert key == "test-project-us-central1"

        # Test with None values
        key = router._get_deployment_key(None, "us-central1")
        assert key is None

        key = router._get_deployment_key("test-project", None)
        assert key is None

        key = router._get_deployment_key(None, None)
        assert key is None

    def test_add_vertex_credentials(self):
        """Test add_vertex_credentials functionality"""
        router = PassthroughEndpointRouter()

        # Test adding valid credentials
        router.add_vertex_credentials(
            project_id="test-project",
            location="us-central1",
            vertex_credentials='{"credentials": "test-creds"}',
        )

        assert "test-project-us-central1" in router.deployment_key_to_vertex_credentials
        creds = router.deployment_key_to_vertex_credentials["test-project-us-central1"]
        assert creds.vertex_project == "test-project"
        assert creds.vertex_location == "us-central1"
        assert creds.vertex_credentials == '{"credentials": "test-creds"}'

        # Test adding with None values
        router.add_vertex_credentials(
            project_id=None,
            location=None,
            vertex_credentials='{"credentials": "test-creds"}',
        )
        # Should not add None values
        assert len(router.deployment_key_to_vertex_credentials) == 1

    def test_default_credentials(self):
        """
        Test get_vertex_credentials with stored credentials.

        Tests if default credentials are used if set.

        Tests if no default credentials are used, if no default set
        """
        router = PassthroughEndpointRouter()
        router.add_vertex_credentials(
            project_id="test-project",
            location="us-central1",
            vertex_credentials='{"credentials": "test-creds"}',
        )

        creds = router.get_vertex_credentials(
            project_id="test-project", location="us-central2"
        )

        assert creds is None

    def test_get_vertex_env_vars(self):
        """Test that _get_vertex_env_vars correctly reads environment variables"""
        # Set environment variables for the test
        os.environ["DEFAULT_VERTEXAI_PROJECT"] = "test-project-123"
        os.environ["DEFAULT_VERTEXAI_LOCATION"] = "us-central1"
        os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/creds"

        try:
            result = self.router._get_vertex_env_vars()
            print(result)

            # Verify the result
            assert isinstance(result, VertexPassThroughCredentials)
            assert result.vertex_project == "test-project-123"
            assert result.vertex_location == "us-central1"
            assert result.vertex_credentials == "/path/to/creds"

        finally:
            # Clean up environment variables
            del os.environ["DEFAULT_VERTEXAI_PROJECT"]
            del os.environ["DEFAULT_VERTEXAI_LOCATION"]
            del os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"]

    def test_set_default_vertex_config(self):
        """Test set_default_vertex_config with various inputs"""
        # Test with None config - set environment variables first
        os.environ["DEFAULT_VERTEXAI_PROJECT"] = "env-project"
        os.environ["DEFAULT_VERTEXAI_LOCATION"] = "env-location"
        os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"] = "env-creds"
        os.environ["GOOGLE_CREDS"] = "secret-creds"

        try:
            # Test with None config
            self.router.set_default_vertex_config()

            assert self.router.default_vertex_config.vertex_project == "env-project"
            assert self.router.default_vertex_config.vertex_location == "env-location"
            assert self.router.default_vertex_config.vertex_credentials == "env-creds"

            # Test with valid config.yaml settings on vertex_config
            test_config = {
                "vertex_project": "my-project-123",
                "vertex_location": "us-central1",
                "vertex_credentials": "path/to/creds",
            }
            self.router.set_default_vertex_config(test_config)

            assert self.router.default_vertex_config.vertex_project == "my-project-123"
            assert self.router.default_vertex_config.vertex_location == "us-central1"
            assert (
                self.router.default_vertex_config.vertex_credentials == "path/to/creds"
            )

            # Test with environment variable reference
            test_config = {
                "vertex_project": "my-project-123",
                "vertex_location": "us-central1",
                "vertex_credentials": "os.environ/GOOGLE_CREDS",
            }
            self.router.set_default_vertex_config(test_config)

            assert (
                self.router.default_vertex_config.vertex_credentials == "secret-creds"
            )

        finally:
            # Clean up environment variables
            del os.environ["DEFAULT_VERTEXAI_PROJECT"]
            del os.environ["DEFAULT_VERTEXAI_LOCATION"]
            del os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"]
            del os.environ["GOOGLE_CREDS"]

    def test_vertex_passthrough_router_init(self):
        """Test VertexPassThroughRouter initialization"""
        router = PassthroughEndpointRouter()
        assert isinstance(router.deployment_key_to_vertex_credentials, dict)
        assert len(router.deployment_key_to_vertex_credentials) == 0

    def test_get_vertex_credentials_none(self):
        """Test get_vertex_credentials with various inputs"""
        router = PassthroughEndpointRouter()

        router.set_default_vertex_config(
            config={
                "vertex_project": None,
                "vertex_location": None,
                "vertex_credentials": None,
            }
        )

        # Test with None project_id and location - should return default config
        creds = router.get_vertex_credentials(None, None)
        assert isinstance(creds, VertexPassThroughCredentials)

        # Test with valid project_id and location but no stored credentials
        creds = router.get_vertex_credentials("test-project", "us-central1")
        assert isinstance(creds, VertexPassThroughCredentials)
        assert creds.vertex_project is None
        assert creds.vertex_location is None
        assert creds.vertex_credentials is None

    def test_get_vertex_credentials_stored(self):
        """Test get_vertex_credentials with stored credentials"""
        router = PassthroughEndpointRouter()
        router.add_vertex_credentials(
            project_id="test-project",
            location="us-central1",
            vertex_credentials='{"credentials": "test-creds"}',
        )

        creds = router.get_vertex_credentials(
            project_id="test-project", location="us-central1"
        )
        assert creds.vertex_project == "test-project"
        assert creds.vertex_location == "us-central1"
        assert creds.vertex_credentials == '{"credentials": "test-creds"}'
