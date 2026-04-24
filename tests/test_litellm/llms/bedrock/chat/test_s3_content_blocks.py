"""
Tests for Bedrock S3 URL pass-through support (PR1).

When users provide s3:// URLs in OpenAI-format messages, LiteLLM should
map them to Bedrock's native s3Location blocks without downloading or
base64-encoding.
"""

import hashlib
import os
from unittest.mock import patch

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import BedrockImageProcessor


def _mock_supports_s3(model: str, custom_llm_provider=None) -> bool:
    """Mock supports_s3_input that returns True for Nova models."""
    return "nova" in model.lower() and "micro" not in model.lower()


class TestS3ImageBlocks:
    """Test S3 URL → ImageBlock creation."""

    def test_should_create_s3_image_block_from_jpg_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/image.jpg")
        assert result["image"]["format"] == "jpeg"
        assert result["image"]["source"]["s3Location"]["uri"] == "s3://bucket/image.jpg"
        assert "bytes" not in result["image"]["source"]

    def test_should_create_s3_image_block_from_png_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/image.png")
        assert result["image"]["format"] == "png"
        assert result["image"]["source"]["s3Location"]["uri"] == "s3://bucket/image.png"

    def test_should_create_s3_image_block_from_gif_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/image.gif")
        assert result["image"]["format"] == "gif"

    def test_should_create_s3_image_block_from_webp_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/image.webp")
        assert result["image"]["format"] == "webp"

    def test_should_create_s3_image_block_from_jpeg_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/photo.jpeg")
        assert result["image"]["format"] == "jpeg"


class TestS3DocumentBlocks:
    """Test S3 URL → DocumentBlock creation."""

    def test_should_create_s3_document_block_from_pdf_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/doc.pdf")
        assert result["document"]["format"] == "pdf"
        assert (
            result["document"]["source"]["s3Location"]["uri"] == "s3://bucket/doc.pdf"
        )
        assert "name" in result["document"]
        assert "bytes" not in result["document"]["source"]

    def test_should_create_s3_document_block_from_docx_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/doc.docx")
        assert result["document"]["format"] == "docx"

    def test_should_create_s3_document_block_from_xlsx_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/data.xlsx")
        assert result["document"]["format"] == "xlsx"

    def test_should_create_s3_document_block_from_md_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/readme.md")
        assert result["document"]["format"] == "md"

    def test_should_create_s3_document_block_from_txt_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/notes.txt")
        assert result["document"]["format"] == "txt"

    def test_should_create_s3_document_block_from_csv_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/data.csv")
        assert result["document"]["format"] == "csv"

    def test_should_create_s3_document_block_from_html_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/page.html")
        assert result["document"]["format"] == "html"

    def test_should_generate_deterministic_document_names(self):
        r1 = BedrockImageProcessor.process_image_sync("s3://bucket/doc.pdf")
        r2 = BedrockImageProcessor.process_image_sync("s3://bucket/doc.pdf")
        assert r1["document"]["name"] == r2["document"]["name"]

        r3 = BedrockImageProcessor.process_image_sync("s3://bucket/other.pdf")
        assert r1["document"]["name"] != r3["document"]["name"]


class TestS3VideoBlocks:
    """Test S3 URL → VideoBlock creation."""

    def test_should_create_s3_video_block_from_mp4_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/video.mp4")
        assert result["video"]["format"] == "mp4"
        assert result["video"]["source"]["s3Location"]["uri"] == "s3://bucket/video.mp4"
        assert "bytes" not in result["video"]["source"]

    def test_should_create_s3_video_block_from_mov_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/video.mov")
        assert result["video"]["format"] == "mov"

    def test_should_create_s3_video_block_from_mkv_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/video.mkv")
        assert result["video"]["format"] == "mkv"

    def test_should_create_s3_video_block_from_3gp_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/video.3gp")
        assert result["video"]["format"] == "3gp"

    def test_should_create_s3_video_block_from_3gpp_url(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/video.3gpp")
        assert result["video"]["format"] == "3gp"


class TestS3ErrorCases:
    """Test error handling for S3 URLs."""

    def test_should_raise_error_for_s3_url_without_extension(self):
        with pytest.raises(ValueError, match="no extension"):
            BedrockImageProcessor.process_image_sync("s3://bucket/noext")

    def test_should_raise_error_for_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            BedrockImageProcessor.process_image_sync("s3://bucket/file.xyz")

    def test_should_not_raise_for_s3_url_without_extension_when_format_provided(self):
        """format override bypasses extension requirement."""
        result = BedrockImageProcessor.process_image_sync(
            "s3://bucket/blob", format="image/png"
        )
        assert result["image"]["format"] == "png"


class TestS3EdgeCases:
    """Test edge cases: uppercase extensions, dots in path, etc."""

    def test_should_handle_uppercase_extension(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/IMAGE.JPG")
        assert result["image"]["format"] == "jpeg"

    def test_should_handle_mixed_case_extension(self):
        result = BedrockImageProcessor.process_image_sync("s3://bucket/doc.Pdf")
        assert result["document"]["format"] == "pdf"

    def test_should_handle_dots_in_path(self):
        result = BedrockImageProcessor.process_image_sync(
            "s3://bucket/path/to/file.with.dots.jpg"
        )
        assert result["image"]["format"] == "jpeg"
        assert (
            result["image"]["source"]["s3Location"]["uri"]
            == "s3://bucket/path/to/file.with.dots.jpg"
        )

    def test_should_handle_deeply_nested_s3_path(self):
        url = "s3://my-bucket/a/b/c/d/report.pdf"
        result = BedrockImageProcessor.process_image_sync(url)
        assert result["document"]["format"] == "pdf"
        assert result["document"]["source"]["s3Location"]["uri"] == url


class TestS3FormatOverride:
    """Test explicit format override for S3 URLs."""

    def test_should_handle_s3_url_with_explicit_format_override(self):
        # Even though extension is .dat, explicit format says pdf
        result = BedrockImageProcessor.process_image_sync(
            "s3://bucket/file.dat", format="application/pdf"
        )
        assert result["document"]["format"] == "pdf"

    def test_should_handle_s3_url_with_simple_format_override(self):
        result = BedrockImageProcessor.process_image_sync(
            "s3://bucket/file.dat", format="png"
        )
        assert result["image"]["format"] == "png"


class TestS3AsyncProcessing:
    """Test async S3 URL processing."""

    @pytest.mark.asyncio
    async def test_should_create_s3_image_block_async(self):
        result = await BedrockImageProcessor.process_image_async(
            "s3://bucket/image.jpg", format=None
        )
        assert result["image"]["format"] == "jpeg"
        assert result["image"]["source"]["s3Location"]["uri"] == "s3://bucket/image.jpg"

    @pytest.mark.asyncio
    async def test_should_create_s3_document_block_async(self):
        result = await BedrockImageProcessor.process_image_async(
            "s3://bucket/doc.pdf", format=None
        )
        assert result["document"]["format"] == "pdf"

    @pytest.mark.asyncio
    async def test_should_create_s3_video_block_async(self):
        result = await BedrockImageProcessor.process_image_async(
            "s3://bucket/video.mp4", format=None
        )
        assert result["video"]["format"] == "mp4"


class TestExistingPathsUnbroken:
    """Ensure existing base64 and HTTPS paths still work."""

    def test_should_still_handle_base64_urls(self):
        # Minimal valid base64 PNG
        b64_url = "data:image/png;base64,iVBORw0KGgo="
        result = BedrockImageProcessor.process_image_sync(b64_url)
        assert result["image"]["format"] == "png"
        assert "bytes" in result["image"]["source"]
        assert "s3Location" not in result["image"]["source"]


@patch(
    "litellm.litellm_core_utils.prompt_templates.factory.supports_s3_input",
    _mock_supports_s3,
)
class TestFullMessageTransformation:
    """Test S3 URLs through full Bedrock message transformation."""

    def test_should_pass_s3_image_url_through_full_message_transformation(self):
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _bedrock_converse_messages_pt,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "s3://my-bucket/photos/cat.jpg"},
                    },
                ],
            }
        ]
        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="us.amazon.nova-lite-v1:0",
            llm_provider="bedrock",
        )
        # Find the image block in the result
        content_blocks = result[0]["content"]
        image_blocks = [b for b in content_blocks if "image" in b]
        assert len(image_blocks) == 1
        assert (
            image_blocks[0]["image"]["source"]["s3Location"]["uri"]
            == "s3://my-bucket/photos/cat.jpg"
        )
        assert image_blocks[0]["image"]["format"] == "jpeg"

    def test_should_pass_s3_video_url_through_full_message_transformation(self):
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _bedrock_converse_messages_pt,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "s3://my-bucket/videos/demo.mp4"},
                    },
                ],
            }
        ]
        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="us.amazon.nova-pro-v1:0",
            llm_provider="bedrock",
        )
        content_blocks = result[0]["content"]
        video_blocks = [b for b in content_blocks if "video" in b]
        assert len(video_blocks) == 1
        assert (
            video_blocks[0]["video"]["source"]["s3Location"]["uri"]
            == "s3://my-bucket/videos/demo.mp4"
        )
        assert video_blocks[0]["video"]["format"] == "mp4"

    def test_should_pass_s3_file_url_through_full_message_transformation(self):
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _bedrock_converse_messages_pt,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this document"},
                    {
                        "type": "file",
                        "file": {
                            "file_data": "s3://my-bucket/docs/report.pdf",
                        },
                    },
                ],
            }
        ]
        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="us.amazon.nova-lite-v1:0",
            llm_provider="bedrock",
        )
        content_blocks = result[0]["content"]
        doc_blocks = [b for b in content_blocks if "document" in b]
        assert len(doc_blocks) == 1
        assert (
            doc_blocks[0]["document"]["source"]["s3Location"]["uri"]
            == "s3://my-bucket/docs/report.pdf"
        )
        assert doc_blocks[0]["document"]["format"] == "pdf"

    @pytest.mark.asyncio
    async def test_should_pass_s3_video_url_through_async_message_transformation(self):
        from litellm.litellm_core_utils.prompt_templates.factory import (
            BedrockConverseMessagesProcessor,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "s3://my-bucket/videos/demo.mp4"},
                    },
                ],
            }
        ]
        result = (
            await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
                messages=messages,
                model="us.amazon.nova-pro-v1:0",
                llm_provider="bedrock",
            )
        )
        content_blocks = result[0]["content"]
        video_blocks = [b for b in content_blocks if "video" in b]
        assert len(video_blocks) == 1
        assert (
            video_blocks[0]["video"]["source"]["s3Location"]["uri"]
            == "s3://my-bucket/videos/demo.mp4"
        )
        assert video_blocks[0]["video"]["format"] == "mp4"

    def test_should_handle_video_url_as_plain_string(self):
        """video_url can be a plain string instead of a dict."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _bedrock_converse_messages_pt,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "video_url", "video_url": "s3://bucket/video.mp4"},
                ],
            }
        ]
        result = _bedrock_converse_messages_pt(
            messages=messages,
            model="us.amazon.nova-pro-v1:0",
            llm_provider="bedrock",
        )
        content_blocks = result[0]["content"]
        video_blocks = [b for b in content_blocks if "video" in b]
        assert len(video_blocks) == 1
        assert video_blocks[0]["video"]["format"] == "mp4"

    def test_should_still_handle_https_image_url(self):
        """Regression: HTTPS URLs should still be downloaded and base64-encoded, not treated as S3."""
        from unittest.mock import patch

        from litellm.litellm_core_utils.prompt_templates.factory import (
            _bedrock_converse_messages_pt,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/photo.jpg"},
                    },
                ],
            }
        ]

        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with patch.object(
            BedrockImageProcessor,
            "get_image_details",
            return_value=(fake_bytes, "image/jpeg"),
        ):
            result = _bedrock_converse_messages_pt(
                messages=messages,
                model="us.amazon.nova-lite-v1:0",
                llm_provider="bedrock",
            )
        content_blocks = result[0]["content"]
        img_blocks = [b for b in content_blocks if "image" in b]
        assert len(img_blocks) == 1
        # Should have inline bytes, NOT s3Location
        assert "bytes" in img_blocks[0]["image"]["source"]
        assert "s3Location" not in img_blocks[0]["image"]["source"]

    def test_should_accept_s3_video_url_in_extract(self):
        """s3:// video URLs should pass through extraction."""
        url = BedrockImageProcessor._extract_video_url(
            {"type": "video_url", "video_url": {"url": "s3://bucket/video.mp4"}}
        )
        assert url == "s3://bucket/video.mp4"


@patch(
    "litellm.litellm_core_utils.prompt_templates.factory.supports_s3_input",
    _mock_supports_s3,
)
class TestS3ModelGuard:
    """Test that S3 URLs are rejected for models that don't support s3Location."""

    def test_should_reject_s3_url_for_unsupported_model(self):
        with pytest.raises(ValueError, match="does not support s3://"):
            BedrockImageProcessor.process_image_sync(
                "s3://bucket/image.jpg",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                custom_llm_provider="bedrock",
            )

    def test_should_allow_s3_url_for_nova_model(self):
        result = BedrockImageProcessor.process_image_sync(
            "s3://bucket/image.jpg",
            model="us.amazon.nova-lite-v1:0",
            custom_llm_provider="bedrock",
        )
        assert result["image"]["source"]["s3Location"]["uri"] == "s3://bucket/image.jpg"

    def test_should_allow_s3_url_when_model_not_specified(self):
        """When no model is provided, allow S3 pass-through (caller's responsibility)."""
        result = BedrockImageProcessor.process_image_sync("s3://bucket/image.jpg")
        assert result["image"]["source"]["s3Location"]["uri"] == "s3://bucket/image.jpg"

    @pytest.mark.asyncio
    async def test_should_reject_s3_url_for_unsupported_model_async(self):
        with pytest.raises(ValueError, match="does not support s3://"):
            await BedrockImageProcessor.process_image_async(
                "s3://bucket/image.jpg",
                format=None,
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                custom_llm_provider="bedrock",
            )
