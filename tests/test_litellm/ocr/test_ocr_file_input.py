"""
Tests for OCR file input support.

Tests that:
1. The SDK document parameter with type="file" correctly converts file paths,
   file objects, and raw bytes to base64 data URIs before sending to providers.
2. The proxy _build_document_from_upload helper correctly handles uploaded file bytes.
3. The proxy rejects type="file" documents received via JSON (security guard).
4. The proxy returns user-friendly errors for invalid JSON bodies.
"""
import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from litellm.ocr.main import convert_file_document_to_url_document, get_mime_type


class TestGetMimeType:
    def test_should_detect_pdf_mime_type(self):
        assert get_mime_type("document.pdf") == "application/pdf"

    def test_should_detect_png_mime_type(self):
        assert get_mime_type("image.png") == "image/png"

    def test_should_detect_jpg_mime_type(self):
        assert get_mime_type("photo.jpg") == "image/jpeg"

    def test_should_detect_jpeg_mime_type(self):
        assert get_mime_type("photo.jpeg") == "image/jpeg"

    def test_should_detect_gif_mime_type(self):
        assert get_mime_type("animation.gif") == "image/gif"

    def test_should_detect_webp_mime_type(self):
        assert get_mime_type("image.webp") == "image/webp"

    def test_should_detect_tiff_mime_type(self):
        assert get_mime_type("scan.tiff") == "image/tiff"

    def test_should_detect_tif_mime_type(self):
        assert get_mime_type("scan.tif") == "image/tiff"

    def test_should_detect_bmp_mime_type(self):
        assert get_mime_type("bitmap.bmp") == "image/bmp"

    def test_should_be_case_insensitive(self):
        assert get_mime_type("DOCUMENT.PDF") == "application/pdf"
        assert get_mime_type("IMAGE.PNG") == "image/png"

    def test_should_fallback_for_unknown_extension(self):
        result = get_mime_type("file.xyz123")
        assert isinstance(result, str)


class TestConvertFileDocumentToUrlDocument:
    def test_should_convert_pdf_file_path_to_document_url(self):
        """File path to a PDF should produce type=document_url with base64 data URI."""
        pdf_content = b"%PDF-1.4 test content"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_content)
            f.flush()
            tmp_path = f.name

        try:
            result = convert_file_document_to_url_document(
                {"type": "file", "file": tmp_path}
            )

            assert result["type"] == "document_url"
            assert result["document_url"].startswith("data:application/pdf;base64,")

            b64_data = result["document_url"].split(";base64,")[1]
            assert base64.b64decode(b64_data) == pdf_content
        finally:
            os.unlink(tmp_path)

    def test_should_convert_image_file_path_to_image_url(self):
        """File path to a PNG image should produce type=image_url with base64 data URI."""
        png_content = b"\x89PNG\r\n\x1a\n fake png content"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_content)
            f.flush()
            tmp_path = f.name

        try:
            result = convert_file_document_to_url_document(
                {"type": "file", "file": tmp_path}
            )

            assert result["type"] == "image_url"
            assert result["image_url"].startswith("data:image/png;base64,")

            b64_data = result["image_url"].split(";base64,")[1]
            assert base64.b64decode(b64_data) == png_content
        finally:
            os.unlink(tmp_path)

    def test_should_convert_pathlib_path(self):
        """pathlib.Path objects should work the same as string paths."""
        content = b"test pdf content"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = Path(f.name)

        try:
            result = convert_file_document_to_url_document(
                {"type": "file", "file": tmp_path}
            )

            assert result["type"] == "document_url"
            assert result["document_url"].startswith("data:application/pdf;base64,")
        finally:
            os.unlink(str(tmp_path))

    def test_should_convert_raw_bytes(self):
        """Raw bytes should be converted using a fallback MIME type."""
        content = b"raw bytes content"

        result = convert_file_document_to_url_document(
            {"type": "file", "file": content}
        )

        assert result["type"] == "document_url"
        assert "base64," in result["document_url"]

        b64_data = result["document_url"].split(";base64,")[1]
        assert base64.b64decode(b64_data) == content

    def test_should_convert_raw_bytes_with_explicit_mime_type(self):
        """Raw bytes with explicit mime_type should use the specified MIME type."""
        content = b"raw pdf content"

        result = convert_file_document_to_url_document(
            {"type": "file", "file": content, "mime_type": "application/pdf"}
        )

        assert result["type"] == "document_url"
        assert result["document_url"].startswith("data:application/pdf;base64,")

    def test_should_convert_raw_bytes_with_image_mime_type(self):
        """Raw bytes with an image MIME type should produce type=image_url."""
        content = b"raw image content"

        result = convert_file_document_to_url_document(
            {"type": "file", "file": content, "mime_type": "image/jpeg"}
        )

        assert result["type"] == "image_url"
        assert result["image_url"].startswith("data:image/jpeg;base64,")

    def test_should_convert_file_like_object(self):
        """BytesIO and other file-like objects should be supported."""
        content = b"file-like content"
        file_obj = BytesIO(content)

        result = convert_file_document_to_url_document(
            {"type": "file", "file": file_obj}
        )

        assert result["type"] == "document_url"
        assert "base64," in result["document_url"]

    def test_should_convert_file_like_object_with_name(self):
        """File-like objects with a .name attribute should detect MIME from the name."""
        content = b"file-like png content"
        file_obj = BytesIO(content)
        file_obj.name = "test_image.png"

        result = convert_file_document_to_url_document(
            {"type": "file", "file": file_obj}
        )

        assert result["type"] == "image_url"
        assert result["image_url"].startswith("data:image/png;base64,")

    def test_should_raise_error_for_missing_file_field(self):
        """Missing 'file' field should raise ValueError."""
        with pytest.raises(ValueError, match="must include a 'file' field"):
            convert_file_document_to_url_document({"type": "file"})

    def test_should_raise_error_for_nonexistent_file_path(self):
        """Non-existent file path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            convert_file_document_to_url_document(
                {"type": "file", "file": "/nonexistent/path/to/file.pdf"}
            )

    def test_should_raise_error_for_empty_file(self):
        """Empty file should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="File is empty"):
                convert_file_document_to_url_document(
                    {"type": "file", "file": tmp_path}
                )
        finally:
            os.unlink(tmp_path)

    def test_should_raise_error_for_unsupported_type(self):
        """Unsupported file input types should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file input type"):
            convert_file_document_to_url_document({"type": "file", "file": 12345})

    def test_should_raise_error_for_invalid_mime_type(self):
        """MIME types with special characters should be rejected."""
        content = b"some content"
        with pytest.raises(ValueError, match="Invalid MIME type"):
            convert_file_document_to_url_document(
                {"type": "file", "file": content, "mime_type": "text/html; charset=utf-8\nX-Injected: true"}
            )

    def test_should_override_mime_type_for_file_path(self):
        """Explicit mime_type should override auto-detection from extension."""
        content = b"some content"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            result = convert_file_document_to_url_document(
                {"type": "file", "file": tmp_path, "mime_type": "image/png"}
            )

            assert result["type"] == "image_url"
            assert result["image_url"].startswith("data:image/png;base64,")
        finally:
            os.unlink(tmp_path)


class TestBuildDocumentFromUpload:
    """Test the proxy endpoint's file upload to document conversion helper."""

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        """Import the proxy helper, skip if proxy deps aren't installed."""
        try:
            from litellm.proxy.ocr_endpoints.endpoints import (
                _build_document_from_upload,
            )

            self._build = _build_document_from_upload
        except ImportError:
            pytest.skip("Proxy dependencies (fastapi/orjson) not installed")

    def test_should_build_document_url_for_pdf(self):
        content = b"%PDF-1.4 test content"

        result = self._build(
            file_content=content,
            filename="document.pdf",
            content_type="application/pdf",
        )

        assert result["type"] == "document_url"
        assert result["document_url"].startswith("data:application/pdf;base64,")

        b64_data = result["document_url"].split(";base64,")[1]
        assert base64.b64decode(b64_data) == content

    def test_should_build_image_url_for_png(self):
        content = b"\x89PNG fake png"

        result = self._build(
            file_content=content,
            filename="screenshot.png",
            content_type="image/png",
        )

        assert result["type"] == "image_url"
        assert result["image_url"].startswith("data:image/png;base64,")

    def test_should_build_image_url_for_jpeg(self):
        content = b"\xff\xd8\xff fake jpeg"

        result = self._build(
            file_content=content,
            filename="photo.jpg",
            content_type="image/jpeg",
        )

        assert result["type"] == "image_url"
        assert result["image_url"].startswith("data:image/jpeg;base64,")

    def test_should_detect_mime_from_filename_when_content_type_is_octet_stream(self):
        content = b"pdf content"

        result = self._build(
            file_content=content,
            filename="report.pdf",
            content_type="application/octet-stream",
        )

        assert result["type"] == "document_url"
        assert result["document_url"].startswith("data:application/pdf;base64,")

    def test_should_detect_mime_from_filename_when_content_type_is_none(self):
        content = b"png content"

        result = self._build(
            file_content=content,
            filename="image.png",
            content_type=None,
        )

        assert result["type"] == "image_url"
        assert result["image_url"].startswith("data:image/png;base64,")

    def test_should_fallback_to_octet_stream_for_unknown(self):
        content = b"unknown content"

        result = self._build(
            file_content=content,
            filename=None,
            content_type=None,
        )

        assert result["type"] == "document_url"
        assert "application/octet-stream" in result["document_url"]

    def test_should_preserve_base64_content_correctly(self):
        content = b"Hello, World! \x00\x01\x02\xff"

        result = self._build(
            file_content=content,
            filename="test.pdf",
            content_type="application/pdf",
        )

        b64_data = result["document_url"].split(";base64,")[1]
        assert base64.b64decode(b64_data) == content

    def test_should_strip_mime_parameters_from_content_type(self):
        """Content-Type with parameters (e.g. charset) should be stripped to the base MIME type."""
        content = b"%PDF-1.4 test"

        result = self._build(
            file_content=content,
            filename="doc.pdf",
            content_type="application/pdf; charset=utf-8",
        )

        assert result["type"] == "document_url"
        assert result["document_url"].startswith("data:application/pdf;base64,")

    def test_should_strip_mime_parameters_with_multiple_params(self):
        """Content-Type with multiple parameters should still be stripped correctly."""
        content = b"image data"

        result = self._build(
            file_content=content,
            filename="img.png",
            content_type="image/png; charset=utf-8; boundary=something",
        )

        assert result["type"] == "image_url"
        assert result["image_url"].startswith("data:image/png;base64,")


class TestProxySecurityGuard:
    """Test that the proxy rejects type='file' documents in JSON requests
    and that multipart form fields cannot override the constructed document."""

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        """Import the proxy helpers, skip if proxy deps aren't installed."""
        try:
            from litellm.proxy.ocr_endpoints.endpoints import (
                _parse_multipart_form,
                _parse_ocr_request,
            )

            self._parse = _parse_ocr_request
            self._parse_multipart = _parse_multipart_form
        except ImportError:
            pytest.skip("Proxy dependencies (fastapi/orjson) not installed")

    @pytest.mark.asyncio
    async def test_should_reject_file_type_document_in_json_body(self):
        """type='file' in a JSON body must be rejected to prevent server-side file reads."""
        body = orjson.dumps(
            {
                "model": "mistral/mistral-ocr-latest",
                "document": {"type": "file", "file": "/etc/passwd"},
            }
        )

        mock_request = MagicMock()
        mock_request.headers = {"content-type": "application/json"}
        mock_request.body = AsyncMock(return_value=body)
        mock_request._form = None

        with pytest.raises(ValueError, match="not supported through the JSON API"):
            await self._parse(mock_request)

    @pytest.mark.asyncio
    async def test_should_accept_document_url_type_in_json_body(self):
        """type='document_url' in a JSON body should pass through normally."""
        expected = {
            "model": "mistral/mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf",
            },
        }
        body = orjson.dumps(expected)

        mock_request = MagicMock()
        mock_request.headers = {"content-type": "application/json"}
        mock_request.body = AsyncMock(return_value=body)
        mock_request._form = None

        result = await self._parse(mock_request)
        assert result["document"]["type"] == "document_url"

    @pytest.mark.asyncio
    async def test_should_raise_on_invalid_json_body(self):
        """Invalid JSON should produce a user-friendly ValueError."""
        mock_request = MagicMock()
        mock_request.headers = {"content-type": "application/json"}
        mock_request.body = AsyncMock(return_value=b"not valid json{{{")
        mock_request._form = None

        with pytest.raises(ValueError, match="Invalid JSON in request body"):
            await self._parse(mock_request)

    @pytest.mark.asyncio
    async def test_should_ignore_document_form_field_injection(self):
        """A 'document' form field must not override the document built from the uploaded file."""
        from starlette.datastructures import UploadFile

        file_content = b"%PDF-1.4 legit content"
        upload = UploadFile(filename="legit.pdf", file=BytesIO(file_content))

        injected = '{"type": "file", "file": "/etc/passwd"}'

        mock_form = {
            "file": upload,
            "model": "mistral/mistral-ocr-latest",
            "document": injected,
        }

        mock_request = MagicMock()
        mock_request.headers = {"content-type": "multipart/form-data; boundary=---"}
        mock_request.form = AsyncMock(return_value=mock_form)

        result = await self._parse_multipart(mock_request)

        assert result["document"]["type"] == "document_url"
        assert result["document"]["document_url"].startswith("data:application/pdf;base64,")
        assert result["model"] == "mistral/mistral-ocr-latest"
