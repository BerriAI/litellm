"""
Test cases for SageMaker embedding role assumption support

This module tests that the SageMaker embedding handler properly supports
AWS IAM role assumption via aws_role_name and aws_session_name parameters,
matching the behavior of the completion handler.
"""

import json
import os
import sys
from datetime import timezone
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

from botocore.credentials import Credentials

from litellm.llms.sagemaker.completion.handler import SagemakerLLM
from litellm.types.utils import EmbeddingResponse


class TestSagemakerEmbeddingRoleAssumption:
    """Test that SageMaker embedding supports role assumption like completion does"""

    def setup_method(self):
        self.sagemaker_llm = SagemakerLLM()

    def test_embedding_uses_load_credentials(self):
        """
        Test that embedding() calls _load_credentials() to support role assumption.
        This ensures aws_role_name and aws_session_name parameters are properly handled.
        """
        # Mock credentials that would be returned after role assumption
        mock_credentials = Credentials(
            access_key="assumed-access-key",
            secret_key="assumed-secret-key",
            token="assumed-session-token",
        )

        # Mock the SageMaker client response
        mock_sagemaker_client = MagicMock()
        mock_sagemaker_client.invoke_endpoint.return_value = {
            "Body": MagicMock(
                read=MagicMock(return_value=json.dumps({"embedding": [[0.1, 0.2, 0.3]]}).encode())
            )
        }

        # Mock boto3.Session to return our mock client
        mock_session = MagicMock()
        mock_session.client.return_value = mock_sagemaker_client

        with patch.object(
            self.sagemaker_llm, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ) as mock_load_creds, patch("boto3.Session", return_value=mock_session):

            # Create mock logging object
            mock_logging = MagicMock()

            optional_params = {
                "aws_role_name": "arn:aws:iam::123456789012:role/TestRole",
                "aws_session_name": "test-session",
            }

            self.sagemaker_llm.embedding(
                model="test-endpoint",
                input=["hello world"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=mock_logging,
                optional_params=optional_params,
            )

            # Verify _load_credentials was called with the optional_params
            mock_load_creds.assert_called_once()

            # Verify boto3.Session was created with the assumed credentials
            mock_session_calls = mock_session.client.call_args_list
            assert len(mock_session_calls) == 1
            assert mock_session_calls[0] == call(service_name="sagemaker-runtime")

    def test_embedding_role_assumption_with_sts(self):
        """
        Test the full role assumption flow for embeddings, similar to completion.
        Verifies that STS assume_role is called when aws_role_name is provided.
        """
        # Mock the STS client for role assumption
        mock_sts_client = MagicMock()

        # Mock the STS response with proper expiration handling
        mock_expiry = MagicMock()
        mock_expiry.tzinfo = timezone.utc
        time_diff = MagicMock()
        time_diff.total_seconds.return_value = 3600
        mock_expiry.__sub__ = MagicMock(return_value=time_diff)

        mock_sts_response = {
            "Credentials": {
                "AccessKeyId": "assumed-access-key",
                "SecretAccessKey": "assumed-secret-key",
                "SessionToken": "assumed-session-token",
                "Expiration": mock_expiry,
            }
        }
        mock_sts_client.assume_role.return_value = mock_sts_response

        # Mock the SageMaker client response
        mock_sagemaker_client = MagicMock()
        mock_sagemaker_client.invoke_endpoint.return_value = {
            "Body": MagicMock(
                read=MagicMock(return_value=json.dumps({"embedding": [[0.1, 0.2, 0.3]]}).encode())
            )
        }

        # Mock boto3.Session for SageMaker client creation
        mock_session = MagicMock()
        mock_session.client.return_value = mock_sagemaker_client

        def mock_boto3_client(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            return mock_sagemaker_client

        with patch("boto3.client", side_effect=mock_boto3_client), \
             patch("boto3.Session", return_value=mock_session):

            mock_logging = MagicMock()

            optional_params = {
                "aws_role_name": "arn:aws:iam::123456789012:role/CrossAccountRole",
                "aws_session_name": "litellm-embedding-session",
                "aws_region_name": "us-east-1",
            }

            self.sagemaker_llm.embedding(
                model="test-endpoint",
                input=["hello world"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=mock_logging,
                optional_params=optional_params,
            )

            # Verify STS assume_role was called with correct parameters
            mock_sts_client.assume_role.assert_called_once()
            call_args = mock_sts_client.assume_role.call_args
            assert call_args[1]["RoleArn"] == "arn:aws:iam::123456789012:role/CrossAccountRole"
            assert call_args[1]["RoleSessionName"] == "litellm-embedding-session"

    def test_embedding_without_role_assumption(self):
        """
        Test that embedding works without role assumption when aws_role_name is not provided.
        Should use default credentials from environment/instance profile.
        """
        # Mock the SageMaker client response
        mock_sagemaker_client = MagicMock()
        mock_sagemaker_client.invoke_endpoint.return_value = {
            "Body": MagicMock(
                read=MagicMock(return_value=json.dumps({"embedding": [[0.1, 0.2, 0.3]]}).encode())
            )
        }

        mock_session = MagicMock()
        mock_session.client.return_value = mock_sagemaker_client

        # Mock credentials returned from environment
        mock_credentials = Credentials(
            access_key="env-access-key",
            secret_key="env-secret-key",
            token=None,
        )

        with patch.object(
            self.sagemaker_llm, "_load_credentials", return_value=(mock_credentials, "us-west-2")
        ), patch("boto3.Session", return_value=mock_session):

            mock_logging = MagicMock()

            # No aws_role_name provided
            optional_params = {
                "aws_region_name": "us-west-2",
            }

            result = self.sagemaker_llm.embedding(
                model="test-endpoint",
                input=["hello world"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=mock_logging,
                optional_params=optional_params,
            )

            # Should still work and return embeddings
            assert result is not None

    def test_embedding_session_created_with_assumed_credentials(self):
        """
        Test that boto3.Session is created with the credentials from role assumption.
        This verifies the credentials flow from _load_credentials to the SageMaker client.
        """
        mock_credentials = Credentials(
            access_key="assumed-key",
            secret_key="assumed-secret",
            token="assumed-token",
        )

        mock_sagemaker_client = MagicMock()
        mock_sagemaker_client.invoke_endpoint.return_value = {
            "Body": MagicMock(
                read=MagicMock(return_value=json.dumps({"embedding": [[0.1, 0.2, 0.3]]}).encode())
            )
        }

        with patch.object(
            self.sagemaker_llm, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ), patch("boto3.Session") as mock_session_class:

            mock_session = MagicMock()
            mock_session.client.return_value = mock_sagemaker_client
            mock_session_class.return_value = mock_session

            mock_logging = MagicMock()

            self.sagemaker_llm.embedding(
                model="test-endpoint",
                input=["hello world"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=mock_logging,
                optional_params={},
            )

            # Verify Session was created with the assumed credentials
            mock_session_class.assert_called_once_with(
                aws_access_key_id="assumed-key",
                aws_secret_access_key="assumed-secret",
                aws_session_token="assumed-token",
                region_name="us-east-1",
            )
