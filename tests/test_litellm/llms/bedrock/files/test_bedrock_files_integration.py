"""
Test Bedrock files integration with main files API
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import SpecialEnums


class TestBedrockFilesIntegration:
    """Test integration of Bedrock files with main litellm API"""

    @pytest.mark.asyncio
    async def test_litellm_afile_content_bedrock_provider_with_s3_uri(self):
        """Test litellm.afile_content with bedrock provider using direct S3 URI"""
        file_id = "s3://test-bucket/test-file.jsonl"
        expected_content = b'{"recordId": "request-1", "modelInput": {}, "modelOutput": {}}'

        # Mock the bedrock_files_instance.file_content method
        with patch(
            "litellm.files.main.bedrock_files_instance.file_content",
            new_callable=AsyncMock,
        ) as mock_file_content:
            # Create a mock HttpxBinaryResponseContent response
            import httpx

            mock_response = httpx.Response(
                status_code=200,
                content=expected_content,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request(
                    method="GET", url="s3://test-bucket/test-file.jsonl"
                ),
            )
            mock_file_content.return_value = HttpxBinaryResponseContent(
                response=mock_response
            )

            # Call litellm.afile_content
            result = await litellm.afile_content(
                file_id=file_id,
                custom_llm_provider="bedrock",
                aws_region_name="us-west-2",
            )

            # Verify the result
            assert isinstance(result, HttpxBinaryResponseContent)
            assert result.response.content == expected_content
            assert result.response.status_code == 200

            # Verify the mock was called with correct parameters
            mock_file_content.assert_called_once()
            call_kwargs = mock_file_content.call_args.kwargs
            assert call_kwargs["_is_async"] is True
            assert call_kwargs["file_content_request"]["file_id"] == file_id

    @pytest.mark.asyncio
    async def test_litellm_afile_content_bedrock_provider_with_unified_file_id(self):
        """Test litellm.afile_content with bedrock provider using unified file ID"""
        # Create a unified file ID
        s3_uri = "s3://test-bucket/batch-outputs/output.jsonl"
        unified_id = "test-unified-id-123"
        model_id = "test-model-id-456"
        
        unified_file_id_str = f"litellm_proxy:application/json;unified_id,{unified_id};target_model_names,;llm_output_file_id,{s3_uri};llm_output_file_model_id,{model_id}"
        encoded_file_id = base64.urlsafe_b64encode(unified_file_id_str.encode()).decode().rstrip("=")
        
        expected_content = b'{"recordId": "request-1", "modelInput": {}, "modelOutput": {}}'

        # Mock the bedrock_files_instance.file_content method
        with patch(
            "litellm.files.main.bedrock_files_instance.file_content",
            new_callable=AsyncMock,
        ) as mock_file_content:
            # Create a mock HttpxBinaryResponseContent response
            import httpx

            mock_response = httpx.Response(
                status_code=200,
                content=expected_content,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request(method="GET", url=s3_uri),
            )
            mock_file_content.return_value = HttpxBinaryResponseContent(
                response=mock_response
            )

            # Call litellm.afile_content with unified file ID
            result = await litellm.afile_content(
                file_id=encoded_file_id,
                custom_llm_provider="bedrock",
                aws_region_name="us-west-2",
            )

            # Verify the result
            assert isinstance(result, HttpxBinaryResponseContent)
            assert result.response.content == expected_content
            assert result.response.status_code == 200

            # Verify the mock was called - the handler should extract S3 URI from unified file ID
            mock_file_content.assert_called_once()
            call_kwargs = mock_file_content.call_args.kwargs
            assert call_kwargs["_is_async"] is True
            # The handler extracts S3 URI from the unified file ID
            assert call_kwargs["file_content_request"]["file_id"] == encoded_file_id
