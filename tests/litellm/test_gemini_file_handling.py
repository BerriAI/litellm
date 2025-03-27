import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import base64

# Parent directory to sys.path to import litellm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from litellm.types.files import (
    FileType, 
    requires_base64_encoding,
    is_gemini_1_5_accepted_file_type,
    get_file_type_from_extension
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _get_mime_type_from_url, 
    _process_gemini_file
)


class TestGeminiFileHandling(unittest.TestCase):
    
    def test_requires_base64_encoding(self):
        """Test that the requires_base64_encoding function correctly identifies text vs binary formats"""
        # Text formats shouldn't require base64
        self.assertFalse(requires_base64_encoding("text/markdown"))
        self.assertFalse(requires_base64_encoding("text/plain"))
        self.assertFalse(requires_base64_encoding("text/csv"))
        self.assertFalse(requires_base64_encoding("text/html"))
        self.assertFalse(requires_base64_encoding("application/json"))
        self.assertFalse(requires_base64_encoding("application/xml"))
        
        # Binary formats should require base64
        self.assertTrue(requires_base64_encoding("image/png"))
        self.assertTrue(requires_base64_encoding("image/jpeg"))
        self.assertTrue(requires_base64_encoding("application/pdf"))
        self.assertTrue(requires_base64_encoding("video/mp4"))
        self.assertTrue(requires_base64_encoding("audio/mp3"))
    
    def test_get_mime_type_from_url(self):
        """Test that the _get_mime_type_from_url function correctly identifies MIME types from URLs"""
        # Test image formats
        self.assertEqual(_get_mime_type_from_url("https://example.com/image.jpg"), "image/jpeg")
        self.assertEqual(_get_mime_type_from_url("https://example.com/pic.jpeg"), "image/jpeg")
        self.assertEqual(_get_mime_type_from_url("https://example.com/icon.png"), "image/png")
        self.assertEqual(_get_mime_type_from_url("https://example.com/banner.webp"), "image/webp")
        
        # Test text formats
        self.assertEqual(_get_mime_type_from_url("https://example.com/doc.md"), "text/markdown")
        self.assertEqual(_get_mime_type_from_url("https://example.com/doc.markdown"), "text/markdown")
        self.assertEqual(_get_mime_type_from_url("https://example.com/notes.txt"), "text/plain")
        self.assertEqual(_get_mime_type_from_url("https://example.com/data.json"), "application/json")
        self.assertEqual(_get_mime_type_from_url("https://example.com/data.xml"), "application/xml")
        self.assertEqual(_get_mime_type_from_url("https://example.com/data.csv"), "text/csv")
        
        # Test other formats
        self.assertEqual(_get_mime_type_from_url("https://example.com/doc.pdf"), "application/pdf")
        self.assertEqual(_get_mime_type_from_url("https://example.com/video.mp4"), "video/mp4")
        
        # Test unknown format
        self.assertIsNone(_get_mime_type_from_url("https://example.com/unknown.xyz"))
    
    def test_gemini_accepted_file_types(self):
        """Test that the GEMINI_1_5_ACCEPTED_FILE_TYPES includes all supported file types"""
        # Check common file types
        self.assertTrue(is_gemini_1_5_accepted_file_type(FileType.PNG))
        self.assertTrue(is_gemini_1_5_accepted_file_type(FileType.JPEG))
        self.assertTrue(is_gemini_1_5_accepted_file_type(FileType.PDF))
        self.assertTrue(is_gemini_1_5_accepted_file_type(FileType.MP4))
        self.assertTrue(is_gemini_1_5_accepted_file_type(FileType.TXT))
        
        # Check newly added types
        for extension in [".md", ".json", ".xml", ".csv"]:
            file_type = get_file_type_from_extension(extension[1:])
            self.assertTrue(
                is_gemini_1_5_accepted_file_type(file_type),
                f"File type for {extension} not in GEMINI_1_5_ACCEPTED_FILE_TYPES"
            )
    
    @patch('litellm.llms.vertex_ai.gemini.transformation.convert_to_anthropic_image_obj')
    @patch('httpx.get')
    def test_process_gemini_file(self, mock_httpx_get, mock_convert):
        """Test that _process_gemini_file handles different file types correctly"""
        # Setup mock for FileDataType and PartType
        with patch('litellm.llms.vertex_ai.gemini.transformation.FileDataType') as mock_file_data:
            with patch('litellm.llms.vertex_ai.gemini.transformation.PartType') as mock_part_type:
                # Configure mocks
                mock_file_data.return_value = "file_data_mock"
                mock_part_type.return_value = "part_type_mock"
                
                # Test text file (markdown) - should use file_uri directly for recognized MIME types
                result = _process_gemini_file("https://example.com/doc.md")
                
                # Verify FileDataType was called with correct params for text file
                mock_file_data.assert_called_with(
                    file_uri="https://example.com/doc.md", 
                    mime_type="text/markdown"
                )
                
                # Verify httpx.get was NOT called (as it uses file_uri directly)
                mock_httpx_get.assert_not_called()
                
                # Reset mocks
                mock_file_data.reset_mock()
                mock_part_type.reset_mock()
                
                # Test binary file handling path for unrecognized URLs
                mock_convert.return_value = {
                    "data": base64.b64encode(b"binary content").decode(),
                    "media_type": "image/png"
                }
                
                # Test with a URL that doesn't have a known MIME type
                mock_httpx_get.return_value = MagicMock()
                result = _process_gemini_file("https://example.com/unknown-format")
                
                # For unknown formats, it should call convert_to_anthropic_image_obj
                mock_convert.assert_called_once()


if __name__ == "__main__":
    unittest.main()