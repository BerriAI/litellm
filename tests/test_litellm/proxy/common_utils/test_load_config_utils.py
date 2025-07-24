from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from litellm.proxy.common_utils.load_config_utils import get_file_contents_from_s3


class TestGetFileContentsFromS3:
    """Test suite for S3 config loading functionality."""

    @patch('boto3.client')
    @patch('litellm.main.bedrock_converse_chat_completion')
    @patch('yaml.safe_load')
    def test_get_file_contents_from_s3_no_temp_file_creation(
        self, mock_yaml_load, mock_bedrock, mock_boto3_client
    ):
        """
        Test that get_file_contents_from_s3 doesn't create temporary files
        and uses yaml.safe_load directly on the S3 response content.

        Note: It's critical that yaml.safe_load is used

        Relevant issue/PR: https://github.com/BerriAI/litellm/pull/12078
        """
        # Mock credentials
        mock_credentials = MagicMock()
        mock_credentials.access_key = "test_access_key"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_token"
        mock_bedrock.get_credentials.return_value = mock_credentials

        # Mock S3 client and response
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client
        
        # Mock S3 response with YAML content
        yaml_content = """
        model_list:
          - model_name: gpt-3.5-turbo
            litellm_params:
              model: gpt-3.5-turbo
        """
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = yaml_content.encode('utf-8')
        mock_s3_response = {
            'Body': mock_response_body
        }
        mock_s3_client.get_object.return_value = mock_s3_response

        # Mock yaml.safe_load to return parsed config
        expected_config = {
            'model_list': [{
                'model_name': 'gpt-3.5-turbo',
                'litellm_params': {
                    'model': 'gpt-3.5-turbo'
                }
            }]
        }
        mock_yaml_load.return_value = expected_config

        # Call the function
        bucket_name = "test-bucket"
        object_key = "config.yaml"
        result = get_file_contents_from_s3(bucket_name, object_key)

        # Assertions
        assert result == expected_config
        
        # Verify S3 client was created with correct credentials
        mock_boto3_client.assert_called_once_with(
            "s3",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            aws_session_token="test_token"
        )
        
        # Verify S3 get_object was called with correct parameters
        mock_s3_client.get_object.assert_called_once_with(
            Bucket=bucket_name,
            Key=object_key
        )
        
        # Verify the response body was read and decoded
        mock_response_body.read.assert_called_once()
        
        # Verify yaml.safe_load was called with the decoded content
        mock_yaml_load.assert_called_once_with(yaml_content)


