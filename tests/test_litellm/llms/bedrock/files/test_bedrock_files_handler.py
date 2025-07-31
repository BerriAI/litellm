"""
Test cases for BedrockFilesHandler

Tests the S3 file upload functionality for Bedrock batch processing.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.llms.bedrock.files.handler import BedrockFilesHandler
from litellm.types.llms.openai import CreateFileRequest


class TestBedrockFilesHandler:
    """Test cases for BedrockFilesHandler"""

    def setup_method(self):
        """Set up test fixtures"""
        self.handler = BedrockFilesHandler()

    def test_get_s3_config_success(self):
        """Test successful S3 configuration retrieval"""
        with patch('litellm.llms.bedrock.files.handler.get_secret_str') as mock_get_secret:
            mock_get_secret.side_effect = lambda key: {
                'LITELLM_BEDROCK_BATCH_BUCKET': 'test-bucket',
                'AWS_REGION': 'us-west-2'
            }.get(key)
            
            config = self.handler._get_s3_config()
            
            assert config['bucket_name'] == 'test-bucket'
            assert config['region_name'] == 'us-west-2'

    def test_get_s3_config_missing_bucket(self):
        """Test S3 configuration with missing bucket"""
        with patch('litellm.llms.bedrock.files.handler.get_secret_str') as mock_get_secret:
            mock_get_secret.return_value = None
            
            with pytest.raises(ValueError, match="LITELLM_BEDROCK_BATCH_BUCKET"):
                self.handler._get_s3_config()

    def test_generate_s3_key(self):
        """Test S3 key generation"""
        key = self.handler._generate_s3_key("test.jsonl", "batch")
        
        assert key.startswith("batch/")
        assert key.endswith(".jsonl")
        assert len(key.split("/")[1].split(".")[0]) == 36  # UUID length

    # Note: Integration test removed due to cross-branch dependency issues
    # Will be added back when all bedrock components are merged