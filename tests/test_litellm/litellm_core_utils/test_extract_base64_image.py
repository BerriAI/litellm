"""
Unit tests for _extract_base64_data and extract_images_from_message functions.

These tests verify that base64 image data is correctly extracted from data URLs,
which fixes the Ollama error "illegal base64 data at input byte 4".

Related issue: https://github.com/BerriAI/litellm/issues/18338
"""
import pytest

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _extract_base64_data,
    extract_images_from_message,
)


class TestExtractBase64Data:
    """Tests for _extract_base64_data function"""

    def test_extract_base64_from_png_data_url(self):
        """Test extracting base64 data from a PNG data URL"""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        expected = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        assert _extract_base64_data(data_url) == expected

    def test_extract_base64_from_jpeg_data_url(self):
        """Test extracting base64 data from a JPEG data URL"""
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
        expected = "/9j/4AAQSkZJRgABAQAAAQABAAD"
        assert _extract_base64_data(data_url) == expected

    def test_extract_base64_from_gif_data_url(self):
        """Test extracting base64 data from a GIF data URL"""
        data_url = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP"
        expected = "R0lGODlhAQABAIAAAAAAAP"
        assert _extract_base64_data(data_url) == expected

    def test_regular_url_unchanged(self):
        """Test that regular HTTP URLs are returned unchanged"""
        url = "https://example.com/image.png"
        assert _extract_base64_data(url) == url

    def test_file_path_unchanged(self):
        """Test that file paths are returned unchanged"""
        path = "/path/to/image.png"
        assert _extract_base64_data(path) == path

    def test_data_url_without_base64_unchanged(self):
        """Test that data URLs without base64 encoding are returned unchanged"""
        # This is a data URL with URL encoding, not base64
        url = "data:text/plain,Hello%20World"
        assert _extract_base64_data(url) == url

    def test_base64_data_with_special_chars(self):
        """Test extracting base64 data that contains valid special characters"""
        # Base64 can contain +, /, and = characters
        data_url = "data:image/png;base64,abc+def/ghi==="
        expected = "abc+def/ghi==="
        assert _extract_base64_data(data_url) == expected


class TestExtractImagesFromMessage:
    """Tests for extract_images_from_message function"""

    def test_extract_from_message_with_data_url_string(self):
        """Test extracting images when image_url is a string data URL"""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64,iVBORw0KGgo",
                }
            ],
        }
        result = extract_images_from_message(message)
        assert result == ["iVBORw0KGgo"]

    def test_extract_from_message_with_data_url_dict(self):
        """Test extracting images when image_url is a dict with url key"""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"},
                }
            ],
        }
        result = extract_images_from_message(message)
        assert result == ["iVBORw0KGgo"]

    def test_extract_from_message_with_regular_url(self):
        """Test that regular URLs are preserved"""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                }
            ],
        }
        result = extract_images_from_message(message)
        assert result == ["https://example.com/image.png"]

    def test_extract_multiple_images(self):
        """Test extracting multiple images from a single message"""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64,image1base64",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,image2base64"},
                },
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image3.png",
                },
            ],
        }
        result = extract_images_from_message(message)
        assert result == [
            "image1base64",
            "image2base64",
            "https://example.com/image3.png",
        ]

    def test_empty_content(self):
        """Test message with empty content"""
        message = {"role": "user", "content": []}
        result = extract_images_from_message(message)
        assert result == []

    def test_no_images_in_content(self):
        """Test message with content but no images"""
        message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello world"}],
        }
        result = extract_images_from_message(message)
        assert result == []

    def test_string_content(self):
        """Test message with string content (no images possible)"""
        message = {"role": "user", "content": "Hello world"}
        result = extract_images_from_message(message)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
