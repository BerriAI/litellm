"""
Test Vertex AI binary file upload functionality

This test ensures that binary files (like PDFs, images) are correctly handled
during upload without attempting UTF-8 decoding, which would cause errors.

Regression test for: UTF-8 codec error when uploading binary files
"""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from litellm.llms.custom_httpx.llm_http_handler import AsyncHTTPHandler
from litellm.llms.vertex_ai.files.transformation import VertexAIFilesConfig
from litellm.types.llms.openai import CreateFileRequest


class TestVertexAIBinaryFileUpload:
    """Test binary file upload handling for Vertex AI"""

    def setup_method(self):
        """Setup test method"""
        self.http_handler = AsyncHTTPHandler()
        self.vertex_config = VertexAIFilesConfig()

    @pytest.mark.asyncio
    async def test_pdf_file_upload_bytes_handling(self):
        """
        Test that PDF binary data is correctly handled without UTF-8 decoding.
        
        This is a regression test for the error:
        'utf-8' codec can't decode byte 0xc4 in position 10: invalid continuation byte
        """
        # Create mock PDF binary data (with non-UTF-8 bytes)
        # PDF files start with %PDF- and contain binary data
        mock_pdf_content = b"%PDF-1.4\n%\xc4\xe5\xf2\xe5\xeb\xa7\xf3\xa0\xd0\xc4\xc6\n"
        mock_pdf_content += b"\x00\x01\x02\x03\xff\xfe\xfd" * 100  # Add more binary data
        
        # Create file object
        file_obj = io.BytesIO(mock_pdf_content)
        file_obj.name = "test_document.pdf"
        
        # Create file request
        create_file_data: CreateFileRequest = {
            "file": file_obj,
            "purpose": "user_data",
        }
        
        # Transform the request
        transformed_request = self.vertex_config.transform_create_file_request(
            model="vertex_ai/gemini-flash",
            create_file_data=create_file_data,
            optional_params={},
            litellm_params={},
        )
        
        # Verify the transformation returns bytes (not string)
        assert isinstance(transformed_request, bytes), (
            f"Expected bytes for binary file, got {type(transformed_request)}"
        )
        
        # Verify the bytes match the original content
        assert transformed_request == mock_pdf_content, (
            "Transformed request should preserve binary content exactly"
        )
        
        # Verify that the bytes contain non-UTF-8 characters
        # This should raise UnicodeDecodeError if we try to decode
        with pytest.raises(UnicodeDecodeError):
            transformed_request.decode("utf-8")

    @pytest.mark.asyncio
    async def test_image_file_upload_bytes_handling(self):
        """Test that image binary data (PNG) is correctly handled"""
        # Create mock PNG binary data (PNG signature + some binary data)
        mock_png_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        mock_png_content += b"\x00\x01\x02\x03\xff\xfe\xfd" * 50
        
        file_obj = io.BytesIO(mock_png_content)
        file_obj.name = "test_image.png"
        
        create_file_data: CreateFileRequest = {
            "file": file_obj,
            "purpose": "user_data",
        }
        
        transformed_request = self.vertex_config.transform_create_file_request(
            model="vertex_ai/gemini-flash",
            create_file_data=create_file_data,
            optional_params={},
            litellm_params={},
        )
        
        # Verify bytes are preserved
        assert isinstance(transformed_request, bytes)
        assert transformed_request == mock_png_content

    @pytest.mark.asyncio
    async def test_http_handler_accepts_bytes_without_decoding(self):
        """
        Test that httpx correctly accepts binary data without decoding.
        
        This test verifies that bytes can be passed to httpx's post/put methods
        without needing UTF-8 decoding, which is the core of our fix.
        """
        # Create mock binary data with non-UTF-8 bytes
        mock_binary_data = b"\x00\x01\x02\x03\xff\xfe\xfd\xc4\xe5\xf2"
        
        # Test that httpx accepts bytes in the data parameter
        # We're testing the behavior, not making an actual request
        
        # Verify that attempting to decode would fail (proving it's binary)
        with pytest.raises(UnicodeDecodeError):
            mock_binary_data.decode("utf-8")
        
        # Verify that httpx Request accepts bytes
        try:
            request = httpx.Request(
                method="POST",
                url="https://example.com/upload",
                data=mock_binary_data,
                headers={"Content-Type": "application/octet-stream"},
            )
            # If we get here, httpx accepts bytes - which is what we need
            assert request.content == mock_binary_data
        except Exception as e:
            pytest.fail(f"httpx should accept bytes in data parameter: {e}")
        
        # Document the expected behavior
        assert isinstance(mock_binary_data, bytes), (
            "Binary file data should remain as bytes"
        )

    @pytest.mark.asyncio
    async def test_jsonl_file_upload_returns_string(self):
        """
        Test that JSONL files (text) are correctly transformed to strings.
        
        This ensures we handle both binary and text files correctly.
        """
        # Create mock JSONL content
        mock_jsonl_content = (
            '{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", '
            '"body": {"model": "gemini-flash", "messages": [{"role": "user", "content": "Hello"}]}}\n'
        )
        
        file_obj = io.BytesIO(mock_jsonl_content.encode("utf-8"))
        file_obj.name = "batch_requests.jsonl"
        
        create_file_data: CreateFileRequest = {
            "file": file_obj,
            "purpose": "batch",
        }
        
        transformed_request = self.vertex_config.transform_create_file_request(
            model="vertex_ai/gemini-flash",
            create_file_data=create_file_data,
            optional_params={},
            litellm_params={},
        )
        
        # JSONL files should be transformed to string
        assert isinstance(transformed_request, str), (
            f"Expected string for JSONL file, got {type(transformed_request)}"
        )

    @pytest.mark.asyncio
    async def test_mixed_file_types_in_sequence(self):
        """
        Test uploading different file types in sequence to ensure no state pollution.
        """
        # Test 1: Upload binary file
        binary_content = b"\x00\x01\x02\x03\xff\xfe\xfd"
        binary_file = io.BytesIO(binary_content)
        binary_file.name = "binary.dat"
        
        binary_request: CreateFileRequest = {
            "file": binary_file,
            "purpose": "user_data",
        }
        
        result1 = self.vertex_config.transform_create_file_request(
            model="vertex_ai/gemini-flash",
            create_file_data=binary_request,
            optional_params={},
            litellm_params={},
        )
        assert isinstance(result1, bytes)
        
        # Test 2: Upload JSONL file
        jsonl_content = '{"test": "data"}\n'
        jsonl_file = io.BytesIO(jsonl_content.encode("utf-8"))
        jsonl_file.name = "batch.jsonl"
        
        jsonl_request: CreateFileRequest = {
            "file": jsonl_file,
            "purpose": "batch",
        }
        
        result2 = self.vertex_config.transform_create_file_request(
            model="vertex_ai/gemini-flash",
            create_file_data=jsonl_request,
            optional_params={},
            litellm_params={},
        )
        assert isinstance(result2, str)
        
        # Test 3: Upload another binary file
        binary_content2 = b"\xc4\xe5\xf2\xe5\xeb"
        binary_file2 = io.BytesIO(binary_content2)
        binary_file2.name = "binary2.dat"
        
        binary_request2: CreateFileRequest = {
            "file": binary_file2,
            "purpose": "user_data",
        }
        
        result3 = self.vertex_config.transform_create_file_request(
            model="vertex_ai/gemini-flash",
            create_file_data=binary_request2,
            optional_params={},
            litellm_params={},
        )
        assert isinstance(result3, bytes)

    def test_bytes_type_preservation_documentation(self):
        """
        Documentation test: Verify that bytes are the correct type for binary uploads.
        
        This test documents the expected behavior:
        - Binary files (PDF, images, etc.) should remain as bytes
        - Text files (JSONL) should be strings
        - httpx accepts both bytes and strings in the 'data' parameter
        - bytes should NEVER be decoded to UTF-8 for binary files
        """
        # This is a documentation test - it always passes
        # but serves as a reference for the expected behavior
        
        expected_behavior = {
            "binary_files": {
                "input_type": "bytes",
                "output_type": "bytes",
                "examples": ["PDF", "PNG", "JPEG", "binary data"],
                "http_method": "POST or PUT",
                "encoding": "none - preserve raw bytes",
            },
            "text_files": {
                "input_type": "str or bytes",
                "output_type": "str",
                "examples": ["JSONL", "CSV", "TXT"],
                "http_method": "POST",
                "encoding": "UTF-8",
            },
        }
        
        assert expected_behavior["binary_files"]["encoding"] == "none - preserve raw bytes"
        assert expected_behavior["text_files"]["encoding"] == "UTF-8"
