"""
Tests for AWS credential validation in credential_endpoints.

This module tests the validate_aws_credential function and endpoint.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)

from fastapi import HTTPException
from litellm.proxy.credential_endpoints.endpoints import validate_aws_credential


class TestValidateAwsCredential:
    """Tests for validate_aws_credential function"""

    def test_credential_not_found_raises_404(self):
        """Test that missing credential raises 404"""
        with patch(
            "litellm.proxy.credential_endpoints.endpoints.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = {}

            with pytest.raises(HTTPException) as exc_info:
                validate_aws_credential("nonexistent-cred")

            assert exc_info.value.status_code == 404
            assert "not found" in exc_info.value.detail

    def test_credential_without_aws_params_raises_400(self):
        """Test that credential without AWS params raises 400"""
        with patch(
            "litellm.proxy.credential_endpoints.endpoints.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = {
                "api_key": "some-key",
                "endpoint": "some-endpoint",
            }

            with pytest.raises(HTTPException) as exc_info:
                validate_aws_credential("non-aws-cred")

            assert exc_info.value.status_code == 400
            assert "does not contain any AWS" in exc_info.value.detail

    def test_valid_credential_returns_success(self):
        """Test that valid AWS credential returns success"""
        with patch(
            "litellm.proxy.credential_endpoints.endpoints.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = {
                "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
                "aws_region_name": "us-west-2",
            }

            result = validate_aws_credential("my-aws-cred")

            assert result["valid"] is True
            assert "aws_role_name" in result["credential_params"]
            assert "aws_region_name" in result["credential_params"]

    def test_role_assumption_test_success(self):
        """Test role assumption test succeeds"""
        with patch(
            "litellm.proxy.credential_endpoints.endpoints.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = {
                "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
                "aws_region_name": "us-west-2",
            }

            with patch(
                "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM"
            ) as mock_llm:
                mock_instance = MagicMock()
                mock_llm.return_value = mock_instance
                # Successful credential retrieval
                mock_instance.get_credentials.return_value = MagicMock()

                result = validate_aws_credential(
                    "my-aws-cred", test_role_assumption=True
                )

                assert result["valid"] is True
                assert result["role_assumption_tested"] is True
                mock_instance.get_credentials.assert_called_once()

    def test_role_assumption_test_failure(self):
        """Test role assumption test failure raises 400"""
        with patch(
            "litellm.proxy.credential_endpoints.endpoints.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = {
                "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
                "aws_region_name": "us-west-2",
            }

            with patch(
                "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM"
            ) as mock_llm:
                mock_instance = MagicMock()
                mock_llm.return_value = mock_instance
                # Simulate role assumption failure
                mock_instance.get_credentials.side_effect = Exception("AccessDenied")

                with pytest.raises(HTTPException) as exc_info:
                    validate_aws_credential("my-aws-cred", test_role_assumption=True)

                assert exc_info.value.status_code == 400
                assert "Failed to assume role" in exc_info.value.detail

    def test_all_aws_params_detected(self):
        """Test that all AWS params are detected"""
        all_aws_params = {
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
            "aws_region_name": "us-west-2",
            "aws_session_name": "session",
            "aws_profile_name": "profile",
            "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
            "aws_web_identity_token": "token",
            "aws_sts_endpoint": "endpoint",
            "aws_bedrock_runtime_endpoint": "endpoint",
            "aws_external_id": "external-id",
        }

        with patch(
            "litellm.proxy.credential_endpoints.endpoints.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = all_aws_params

            result = validate_aws_credential("full-aws-cred")

            assert result["valid"] is True
            # All params should be detected
            for param in all_aws_params.keys():
                assert param in result["credential_params"]
