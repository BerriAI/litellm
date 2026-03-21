"""
Test SSL verification for AWS Bedrock boto3 clients.

This test ensures that custom CA certificates are properly passed to all boto3 clients
(STS and Bedrock services) to support internal certificate authorities.

Issue: https://github.com/BerriAI/litellm/issues/XXXX
User reported that SSL_CERT_FILE environment variable and ssl_verify config were not
being applied to boto3 clients, causing "certificate verify failed" errors.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import init_bedrock_client


class TestBedrockSSLVerify:
    """Test suite for SSL verification in Bedrock boto3 clients."""

    def test_base_aws_llm_get_ssl_verify_default(self):
        """Test that _get_ssl_verify returns default value when no custom config is set."""
        base_aws = BaseAWSLLM()
        
        # Clear any environment variables
        os.environ.pop("SSL_VERIFY", None)
        os.environ.pop("SSL_CERT_FILE", None)
        
        # Reset litellm.ssl_verify to default
        litellm.ssl_verify = True
        
        ssl_verify = base_aws._get_ssl_verify()
        assert ssl_verify is True

    def test_base_aws_llm_get_ssl_verify_false(self):
        """Test that _get_ssl_verify returns False when SSL verification is disabled."""
        base_aws = BaseAWSLLM()
        
        # Set SSL_VERIFY to False via environment
        os.environ["SSL_VERIFY"] = "False"
        
        ssl_verify = base_aws._get_ssl_verify()
        assert ssl_verify is False
        
        # Clean up
        os.environ.pop("SSL_VERIFY", None)

    def test_base_aws_llm_get_ssl_verify_custom_ca_bundle(self):
        """Test that _get_ssl_verify returns custom CA bundle path when SSL_CERT_FILE is set."""
        base_aws = BaseAWSLLM()
        
        # Create a temporary CA bundle file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("FAKE CERTIFICATE FOR TESTING\n")
            f.write("-----END CERTIFICATE-----\n")
            ca_bundle_path = f.name
        
        try:
            # Set SSL_CERT_FILE environment variable
            os.environ["SSL_CERT_FILE"] = ca_bundle_path
            os.environ.pop("SSL_VERIFY", None)
            litellm.ssl_verify = True
            
            ssl_verify = base_aws._get_ssl_verify()
            assert ssl_verify == ca_bundle_path
        finally:
            # Clean up
            os.environ.pop("SSL_CERT_FILE", None)
            os.unlink(ca_bundle_path)

    def test_base_aws_llm_get_ssl_verify_litellm_config(self):
        """Test that _get_ssl_verify uses litellm.ssl_verify when set."""
        base_aws = BaseAWSLLM()
        
        # Clear environment variables
        os.environ.pop("SSL_VERIFY", None)
        os.environ.pop("SSL_CERT_FILE", None)
        
        # Create a temporary CA bundle file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("FAKE CERTIFICATE FOR TESTING\n")
            f.write("-----END CERTIFICATE-----\n")
            ca_bundle_path = f.name
        
        try:
            # Set litellm.ssl_verify to custom CA bundle
            litellm.ssl_verify = ca_bundle_path
            
            ssl_verify = base_aws._get_ssl_verify()
            # When ssl_verify is a path, it should be returned directly
            assert ssl_verify == ca_bundle_path
        finally:
            # Clean up
            litellm.ssl_verify = True
            os.unlink(ca_bundle_path)

    @patch("boto3.client")
    def test_init_bedrock_client_passes_ssl_verify_to_sts(self, mock_boto3_client):
        """Test that init_bedrock_client passes ssl_verify to STS client."""
        # Create a temporary CA bundle file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("FAKE CERTIFICATE FOR TESTING\n")
            f.write("-----END CERTIFICATE-----\n")
            ca_bundle_path = f.name
        
        try:
            # Set SSL_CERT_FILE environment variable
            os.environ["SSL_CERT_FILE"] = ca_bundle_path
            litellm.ssl_verify = True
            
            # Mock the STS client and Bedrock client
            mock_sts_client = MagicMock()
            mock_sts_response = {
                "Credentials": {
                    "AccessKeyId": "test_access_key",
                    "SecretAccessKey": "test_secret_key",
                    "SessionToken": "test_session_token",
                }
            }
            mock_sts_client.assume_role.return_value = mock_sts_response
            
            mock_bedrock_client = MagicMock()
            
            # Configure mock to return different clients based on service name
            def side_effect(service_name=None, **kwargs):
                if service_name == "sts":
                    return mock_sts_client
                elif service_name == "bedrock-runtime":
                    return mock_bedrock_client
                return MagicMock()
            
            mock_boto3_client.side_effect = side_effect
            
            # Call init_bedrock_client with role assumption
            client = init_bedrock_client(
                aws_region_name="us-west-2",
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                aws_role_name="arn:aws:iam::123456789012:role/test-role",
                aws_session_name="test-session",
            )
            
            # Verify that boto3.client was called with verify parameter for STS
            sts_calls = [
                call for call in mock_boto3_client.call_args_list
                if (len(call[0]) > 0 and call[0][0] == "sts") or 
                   ("service_name" not in call[1])  # STS calls don't use service_name kwarg
            ]
            
            assert len(sts_calls) > 0, "STS client should have been created"
            
            # Check that verify parameter was passed to STS client
            sts_call = sts_calls[0]
            assert "verify" in sts_call[1], "verify parameter should be passed to STS client"
            assert sts_call[1]["verify"] == ca_bundle_path, f"verify should be set to CA bundle path, got {sts_call[1]['verify']}"
            
            # Verify that boto3.client was called with verify parameter for Bedrock
            bedrock_calls = [
                call for call in mock_boto3_client.call_args_list
                if "service_name" in call[1] and call[1]["service_name"] == "bedrock-runtime"
            ]
            
            assert len(bedrock_calls) > 0, "Bedrock client should have been created"
            
            bedrock_call = bedrock_calls[0]
            assert "verify" in bedrock_call[1], "verify parameter should be passed to Bedrock client"
            assert bedrock_call[1]["verify"] == ca_bundle_path, f"verify should be set to CA bundle path, got {bedrock_call[1]['verify']}"
            
        finally:
            # Clean up
            os.environ.pop("SSL_CERT_FILE", None)
            os.unlink(ca_bundle_path)

    @patch("boto3.client")
    def test_base_aws_llm_auth_with_role_passes_ssl_verify(self, mock_boto3_client):
        """Test that _auth_with_aws_role passes ssl_verify to STS client."""
        base_aws = BaseAWSLLM()
        
        # Create a temporary CA bundle file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("FAKE CERTIFICATE FOR TESTING\n")
            f.write("-----END CERTIFICATE-----\n")
            ca_bundle_path = f.name
        
        try:
            # Set SSL_CERT_FILE environment variable
            os.environ["SSL_CERT_FILE"] = ca_bundle_path
            litellm.ssl_verify = True
            
            # Mock the STS client
            mock_sts_client = MagicMock()
            mock_sts_response = {
                "Credentials": {
                    "AccessKeyId": "test_access_key",
                    "SecretAccessKey": "test_secret_key",
                    "SessionToken": "test_session_token",
                    "Expiration": "2025-01-10T00:00:00Z",
                }
            }
            
            # Convert Expiration to datetime
            from datetime import datetime, timezone
            mock_sts_response["Credentials"]["Expiration"] = datetime.now(timezone.utc)
            
            mock_sts_client.assume_role.return_value = mock_sts_response
            mock_boto3_client.return_value = mock_sts_client
            
            # Call _auth_with_aws_role
            credentials, ttl = base_aws._auth_with_aws_role(
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                aws_session_token=None,
                aws_role_name="arn:aws:iam::123456789012:role/test-role",
                aws_session_name="test-session",
            )
            
            # Verify that boto3.client was called with verify parameter
            assert mock_boto3_client.called, "boto3.client should have been called"
            
            call_kwargs = mock_boto3_client.call_args[1]
            assert "verify" in call_kwargs, "verify parameter should be passed to STS client"
            assert call_kwargs["verify"] == ca_bundle_path, f"verify should be set to CA bundle path, got {call_kwargs['verify']}"
            
        finally:
            # Clean up
            os.environ.pop("SSL_CERT_FILE", None)
            os.unlink(ca_bundle_path)

    @patch("litellm.llms.bedrock.base_aws_llm.get_secret")
    @patch("boto3.client")
    def test_base_aws_llm_auth_with_web_identity_passes_ssl_verify(self, mock_boto3_client, mock_get_secret):
        """Test that _auth_with_web_identity_token passes ssl_verify to STS client."""
        base_aws = BaseAWSLLM()
        
        # Create a temporary CA bundle file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("FAKE CERTIFICATE FOR TESTING\n")
            f.write("-----END CERTIFICATE-----\n")
            ca_bundle_path = f.name
        
        try:
            # Set SSL_CERT_FILE environment variable
            os.environ["SSL_CERT_FILE"] = ca_bundle_path
            litellm.ssl_verify = True
            
            # Mock get_secret to return the token
            mock_get_secret.return_value = "mocked_oidc_token"
            
            # Mock the STS client
            mock_sts_client = MagicMock()
            mock_sts_response = {
                "Credentials": {
                    "AccessKeyId": "test_access_key",
                    "SecretAccessKey": "test_secret_key",
                    "SessionToken": "test_session_token",
                },
                "PackedPolicySize": 100,
            }
            
            mock_sts_client.assume_role_with_web_identity.return_value = mock_sts_response
            
            # Mock boto3.Session
            mock_session = MagicMock()
            mock_credentials = MagicMock()
            mock_session.get_credentials.return_value = mock_credentials
            
            mock_boto3_client.return_value = mock_sts_client
            
            with patch("boto3.Session", return_value=mock_session):
                # Call _auth_with_web_identity_token
                credentials, ttl = base_aws._auth_with_web_identity_token(
                    aws_web_identity_token="test_token",
                    aws_role_name="arn:aws:iam::123456789012:role/test-role",
                    aws_session_name="test-session",
                    aws_region_name="us-west-2",
                    aws_sts_endpoint=None,
                )
            
            # Verify that boto3.client was called with verify parameter
            assert mock_boto3_client.called, "boto3.client should have been called"
            
            call_kwargs = mock_boto3_client.call_args[1]
            assert "verify" in call_kwargs, "verify parameter should be passed to STS client"
            assert call_kwargs["verify"] == ca_bundle_path, f"verify should be set to CA bundle path, got {call_kwargs['verify']}"
            
        finally:
            # Clean up
            os.environ.pop("SSL_CERT_FILE", None)
            os.unlink(ca_bundle_path)

    def test_ssl_verify_priority_env_over_litellm_config(self):
        """Test that SSL_VERIFY environment variable takes priority over litellm.ssl_verify."""
        base_aws = BaseAWSLLM()
        
        # Set litellm.ssl_verify to True
        litellm.ssl_verify = True
        
        # Set SSL_VERIFY environment variable to False
        os.environ["SSL_VERIFY"] = "False"
        
        try:
            ssl_verify = base_aws._get_ssl_verify()
            assert ssl_verify is False, "Environment variable should take priority"
        finally:
            # Clean up
            os.environ.pop("SSL_VERIFY", None)
            litellm.ssl_verify = True

    def test_ssl_cert_file_priority_over_default(self):
        """Test that SSL_CERT_FILE takes priority when ssl_verify is True."""
        base_aws = BaseAWSLLM()
        
        # Create a temporary CA bundle file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("FAKE CERTIFICATE FOR TESTING\n")
            f.write("-----END CERTIFICATE-----\n")
            ca_bundle_path = f.name
        
        try:
            # Set SSL_CERT_FILE environment variable
            os.environ["SSL_CERT_FILE"] = ca_bundle_path
            os.environ.pop("SSL_VERIFY", None)
            litellm.ssl_verify = True
            
            ssl_verify = base_aws._get_ssl_verify()
            assert ssl_verify == ca_bundle_path, "SSL_CERT_FILE should be used when ssl_verify is True"
        finally:
            # Clean up
            os.environ.pop("SSL_CERT_FILE", None)
            os.unlink(ca_bundle_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
