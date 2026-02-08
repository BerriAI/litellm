"""
Tests for AWS credentials support in litellm_pre_call_utils.py

This module tests the per-request, per-key, and per-team AWS credential
merging functionality that enables RBAC for AWS Bedrock models.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    AWS_CREDENTIAL_PARAMS,
    _extract_aws_credentials_from_metadata,
    _get_aws_credentials_from_credential_name,
    add_aws_credentials_to_request,
)


class TestExtractAwsCredentialsFromMetadata:
    """Tests for _extract_aws_credentials_from_metadata function"""

    def test_extract_aws_credentials_from_empty_metadata(self):
        """Test that empty metadata returns empty dict"""
        result = _extract_aws_credentials_from_metadata(None)
        assert result == {}

        result = _extract_aws_credentials_from_metadata({})
        assert result == {}

    def test_extract_aws_role_name_from_metadata(self):
        """Test extraction of aws_role_name from metadata"""
        metadata = {
            "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
            "other_field": "value",
        }
        result = _extract_aws_credentials_from_metadata(metadata)
        assert result == {"aws_role_name": "arn:aws:iam::123456789:role/MyRole"}

    def test_extract_multiple_aws_params_from_metadata(self):
        """Test extraction of multiple AWS params from metadata"""
        metadata = {
            "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
            "aws_region_name": "us-west-2",
            "aws_session_name": "my-session",
            "aws_external_id": "external-123",
            "other_field": "value",
        }
        result = _extract_aws_credentials_from_metadata(metadata)
        assert result == {
            "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
            "aws_region_name": "us-west-2",
            "aws_session_name": "my-session",
            "aws_external_id": "external-123",
        }

    def test_extract_all_aws_params_from_metadata(self):
        """Test that all AWS params are extracted correctly"""
        metadata = {param: f"value_{param}" for param in AWS_CREDENTIAL_PARAMS}
        metadata["non_aws_param"] = "should_not_be_extracted"

        result = _extract_aws_credentials_from_metadata(metadata)

        assert len(result) == len(AWS_CREDENTIAL_PARAMS)
        assert "non_aws_param" not in result
        for param in AWS_CREDENTIAL_PARAMS:
            assert result[param] == f"value_{param}"


class TestGetAwsCredentialsFromCredentialName:
    """Tests for _get_aws_credentials_from_credential_name function"""

    def test_credential_not_found_returns_empty(self):
        """Test that missing credential returns empty dict"""
        with patch(
            "litellm.litellm_core_utils.credential_accessor.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = {}
            result = _get_aws_credentials_from_credential_name("nonexistent")
            assert result == {}

    def test_credential_with_aws_params(self):
        """Test that credentials with AWS params are returned correctly"""
        with patch(
            "litellm.litellm_core_utils.credential_accessor.CredentialAccessor.get_credential_values"
        ) as mock_get:
            mock_get.return_value = {
                "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
                "aws_region_name": "us-west-2",
                "api_key": "should_not_be_included",
            }
            result = _get_aws_credentials_from_credential_name("my-credential")
            assert result == {
                "aws_role_name": "arn:aws:iam::123456789:role/MyRole",
                "aws_region_name": "us-west-2",
            }
            assert "api_key" not in result


class TestAddAwsCredentialsToRequest:
    """Tests for add_aws_credentials_to_request function"""

    def test_no_credentials_returns_unchanged_data(self):
        """Test that data without credentials is unchanged"""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            metadata={},
            team_metadata={},
        )
        data = {"model": "bedrock/anthropic.claude-3-5-sonnet", "metadata": {}}

        result = add_aws_credentials_to_request(data, user_api_key_dict, "metadata")

        assert "aws_role_name" not in result

    def test_team_level_aws_credential_name(self):
        """Test that team-level aws_credential_name is used"""
        with patch(
            "litellm.proxy.litellm_pre_call_utils._get_aws_credentials_from_credential_name"
        ) as mock_get:
            mock_get.return_value = {
                "aws_role_name": "arn:aws:iam::123456789:role/TeamRole",
                "aws_region_name": "us-east-1",
            }

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                metadata={},
                team_metadata={"aws_credential_name": "team-bedrock-creds"},
            )
            data = {"model": "bedrock/anthropic.claude-3-5-sonnet", "metadata": {}}

            result = add_aws_credentials_to_request(data, user_api_key_dict, "metadata")

            mock_get.assert_called_with("team-bedrock-creds")
            assert result["aws_role_name"] == "arn:aws:iam::123456789:role/TeamRole"
            assert result["aws_region_name"] == "us-east-1"

    def test_key_level_aws_credential_name_overrides_team(self):
        """Test that key-level credentials override team-level"""
        with patch(
            "litellm.proxy.litellm_pre_call_utils._get_aws_credentials_from_credential_name"
        ) as mock_get:
            # Return different credentials based on the credential name
            def side_effect(name):
                if name == "key-bedrock-creds":
                    return {
                        "aws_role_name": "arn:aws:iam::123456789:role/KeyRole",
                        "aws_region_name": "us-west-2",
                    }
                elif name == "team-bedrock-creds":
                    return {
                        "aws_role_name": "arn:aws:iam::123456789:role/TeamRole",
                        "aws_region_name": "us-east-1",
                    }
                return {}

            mock_get.side_effect = side_effect

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                metadata={"aws_credential_name": "key-bedrock-creds"},
                team_metadata={"aws_credential_name": "team-bedrock-creds"},
            )
            data = {"model": "bedrock/anthropic.claude-3-5-sonnet", "metadata": {}}

            result = add_aws_credentials_to_request(data, user_api_key_dict, "metadata")

            # Key-level should override team-level
            assert result["aws_role_name"] == "arn:aws:iam::123456789:role/KeyRole"
            assert result["aws_region_name"] == "us-west-2"

    def test_request_metadata_aws_credential_name_overrides_key(self):
        """Test that request-level aws_credential_name overrides key-level"""
        with patch(
            "litellm.proxy.litellm_pre_call_utils._get_aws_credentials_from_credential_name"
        ) as mock_get:
            def side_effect(name):
                if name == "request-bedrock-creds":
                    return {
                        "aws_role_name": "arn:aws:iam::123456789:role/RequestRole",
                        "aws_region_name": "eu-west-1",
                    }
                elif name == "key-bedrock-creds":
                    return {
                        "aws_role_name": "arn:aws:iam::123456789:role/KeyRole",
                        "aws_region_name": "us-west-2",
                    }
                return {}

            mock_get.side_effect = side_effect

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                metadata={"aws_credential_name": "key-bedrock-creds"},
                team_metadata={},
            )
            data = {
                "model": "bedrock/anthropic.claude-3-5-sonnet",
                "metadata": {"aws_credential_name": "request-bedrock-creds"},
            }

            result = add_aws_credentials_to_request(data, user_api_key_dict, "metadata")

            # Request-level should override key-level
            assert result["aws_role_name"] == "arn:aws:iam::123456789:role/RequestRole"
            assert result["aws_region_name"] == "eu-west-1"

    def test_direct_aws_params_in_metadata_highest_priority(self):
        """Test that direct AWS params in metadata have highest priority"""
        with patch(
            "litellm.proxy.litellm_pre_call_utils._get_aws_credentials_from_credential_name"
        ) as mock_get:
            mock_get.return_value = {
                "aws_role_name": "arn:aws:iam::123456789:role/CredentialRole",
                "aws_region_name": "us-west-2",
            }

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                metadata={"aws_credential_name": "key-bedrock-creds"},
                team_metadata={},
            )
            # Direct AWS params in metadata override credential lookup
            data = {
                "model": "bedrock/anthropic.claude-3-5-sonnet",
                "metadata": {
                    "aws_credential_name": "request-bedrock-creds",
                    "aws_role_name": "arn:aws:iam::123456789:role/DirectRole",
                },
            }

            result = add_aws_credentials_to_request(data, user_api_key_dict, "metadata")

            # Direct params should win
            assert result["aws_role_name"] == "arn:aws:iam::123456789:role/DirectRole"

    def test_existing_data_params_not_overwritten(self):
        """Test that existing params in data are not overwritten"""
        with patch(
            "litellm.proxy.litellm_pre_call_utils._get_aws_credentials_from_credential_name"
        ) as mock_get:
            mock_get.return_value = {
                "aws_role_name": "arn:aws:iam::123456789:role/CredentialRole",
                "aws_region_name": "us-west-2",
            }

            user_api_key_dict = UserAPIKeyAuth(
                api_key="test-key",
                metadata={"aws_credential_name": "key-bedrock-creds"},
                team_metadata={},
            )
            # Pre-existing param in data
            data = {
                "model": "bedrock/anthropic.claude-3-5-sonnet",
                "metadata": {},
                "aws_role_name": "arn:aws:iam::123456789:role/ExistingRole",
            }

            result = add_aws_credentials_to_request(data, user_api_key_dict, "metadata")

            # Existing param should not be overwritten
            assert result["aws_role_name"] == "arn:aws:iam::123456789:role/ExistingRole"
            # But other params should be added
            assert result["aws_region_name"] == "us-west-2"


class TestAwsCredentialParams:
    """Tests for the AWS_CREDENTIAL_PARAMS constant"""

    def test_aws_credential_params_contains_expected_params(self):
        """Test that AWS_CREDENTIAL_PARAMS contains all expected params"""
        expected_params = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_session_name",
            "aws_profile_name",
            "aws_role_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
            "aws_bedrock_runtime_endpoint",
            "aws_external_id",
        ]
        for param in expected_params:
            assert param in AWS_CREDENTIAL_PARAMS
