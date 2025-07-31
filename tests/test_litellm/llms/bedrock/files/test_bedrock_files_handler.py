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

    @pytest.mark.asyncio
    async def test_create_file_api_integration(self):
        """Test that create_file is properly wired into the API"""
        from litellm.types.llms.openai import OpenAIFileObject
        
        # Mock the dependencies
        with patch.object(self.handler, '_get_s3_config', return_value={
            'bucket_name': 'test-bucket',
            'region_name': 'us-east-1'
        }), \
        patch.object(self.handler, '_upload_to_s3', new_callable=AsyncMock) as mock_upload, \
        patch.object(self.handler, 'get_credentials', return_value=MagicMock()):
            
            mock_upload.return_value = {
                's3_uri': 's3://test-bucket/batch/test.jsonl',
                's3_key': 'batch/test.jsonl',
                'bucket_name': 'test-bucket',
                'region_name': 'us-east-1'
            }
            
            request = CreateFileRequest(
                file=b'{"test": "data"}',
                purpose="batch",
                filename="test.jsonl"
            )
            
            result = await self.handler.async_create_file(
                _is_async=True,
                create_file_data=request,
                timeout=30.0,
                max_retries=3
            )
            
            assert isinstance(result, OpenAIFileObject)
            assert result.filename == "test.jsonl"
            assert result.purpose == "batch"
            assert result.status == "processed"
            assert hasattr(result, '_hidden_params')
            assert 's3_uri' in result._hidden_params