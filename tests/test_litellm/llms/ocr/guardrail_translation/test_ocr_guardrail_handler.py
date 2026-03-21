"""
Unit tests for OCR Guardrail Translation Handler
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo
from litellm.llms.mistral.ocr.guardrail_translation.handler import OCRHandler
from litellm.types.utils import CallTypes


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing"""

    async def apply_guardrail(
        self, inputs: dict, request_data: dict, input_type: str, **kwargs
    ) -> dict:
        texts = inputs.get("texts", [])
        return {"texts": [f"{text} [GUARDRAILED]" for text in texts]}


class BlockingGuardrail(CustomGuardrail):
    """Mock guardrail that raises on forbidden content"""

    async def apply_guardrail(
        self, inputs: dict, request_data: dict, input_type: str, **kwargs
    ) -> dict:
        texts = inputs.get("texts", [])
        for text in texts:
            if "FORBIDDEN" in text:
                raise ValueError("Content blocked by guardrail")
        return {"texts": texts}


class TestHandlerDiscovery:
    """Test that the handler is properly discovered"""

    def test_handler_discovered_for_ocr(self):
        """Test that ocr CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.ocr)
        assert handler_class == OCRHandler

    def test_handler_discovered_for_aocr(self):
        """Test that aocr CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.aocr)
        assert handler_class == OCRHandler


class TestInputProcessing:
    """Test input processing functionality"""

    @pytest.mark.asyncio
    async def test_process_document_url(self):
        """Test processing a document_url input"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "mistral/mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234",
            },
        }

        result = await handler.process_input_messages(data, guardrail)

        # Document should be unchanged (guardrail can reject but not modify URL)
        assert result["document"]["document_url"] == "https://arxiv.org/pdf/2201.04234"
        assert result["model"] == "mistral/mistral-ocr-latest"

    @pytest.mark.asyncio
    async def test_process_image_url(self):
        """Test processing an image_url input"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "mistral/mistral-ocr-latest",
            "document": {
                "type": "image_url",
                "image_url": "https://example.com/image.png",
            },
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["document"]["image_url"] == "https://example.com/image.png"

    @pytest.mark.asyncio
    async def test_process_no_document(self):
        """Test processing when no document is provided"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "mistral/mistral-ocr-latest"}

        result = await handler.process_input_messages(data, guardrail)

        assert result == data
        assert "document" not in result

    @pytest.mark.asyncio
    async def test_process_invalid_document(self):
        """Test processing when document is not a dict"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "mistral/mistral-ocr-latest", "document": "not_a_dict"}

        result = await handler.process_input_messages(data, guardrail)

        assert result == data

    @pytest.mark.asyncio
    async def test_input_blocking_guardrail(self):
        """Test that a blocking guardrail can reject OCR input"""
        handler = OCRHandler()
        guardrail = BlockingGuardrail(guardrail_name="blocker")

        data = {
            "model": "mistral/mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "https://example.com/FORBIDDEN_document.pdf",
            },
        }

        with pytest.raises(ValueError, match="Content blocked by guardrail"):
            await handler.process_input_messages(data, guardrail)


class TestOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_single_page(self):
        """Test processing OCR response with a single page"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = OCRResponse(
            pages=[OCRPage(index=0, markdown="Hello world from OCR")],
            model="mistral/mistral-ocr-latest",
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.pages[0].markdown == "Hello world from OCR [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_multiple_pages(self):
        """Test processing OCR response with multiple pages"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = OCRResponse(
            pages=[
                OCRPage(index=0, markdown="Page one content"),
                OCRPage(index=1, markdown="Page two content"),
                OCRPage(index=2, markdown="Page three content"),
            ],
            model="mistral/mistral-ocr-latest",
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.pages[0].markdown == "Page one content [GUARDRAILED]"
        assert result.pages[1].markdown == "Page two content [GUARDRAILED]"
        assert result.pages[2].markdown == "Page three content [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_empty_pages(self):
        """Test processing OCR response with no pages"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = OCRResponse(
            pages=[],
            model="mistral/mistral-ocr-latest",
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.pages == []

    @pytest.mark.asyncio
    async def test_process_page_with_empty_markdown(self):
        """Test processing page where markdown is empty"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = OCRResponse(
            pages=[
                OCRPage(index=0, markdown=""),
                OCRPage(index=1, markdown="Non-empty content"),
            ],
            model="mistral/mistral-ocr-latest",
        )

        result = await handler.process_output_response(response, guardrail)

        # Empty markdown page should be skipped
        assert result.pages[0].markdown == ""
        # Non-empty page should be guardrailed
        assert result.pages[1].markdown == "Non-empty content [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_preserves_page_metadata(self):
        """Test that guardrail processing preserves page metadata"""
        handler = OCRHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = OCRResponse(
            pages=[
                OCRPage(index=0, markdown="Page content"),
            ],
            model="mistral/mistral-ocr-latest",
            usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=1024),
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.pages[0].index == 0
        assert result.pages[0].markdown == "Page content [GUARDRAILED]"
        assert result.model == "mistral/mistral-ocr-latest"
        assert result.usage_info.pages_processed == 1

    @pytest.mark.asyncio
    async def test_output_blocking_guardrail(self):
        """Test that a blocking guardrail can reject OCR output"""
        handler = OCRHandler()
        guardrail = BlockingGuardrail(guardrail_name="blocker")

        response = OCRResponse(
            pages=[OCRPage(index=0, markdown="This contains FORBIDDEN text")],
            model="mistral/mistral-ocr-latest",
        )

        with pytest.raises(ValueError, match="Content blocked by guardrail"):
            await handler.process_output_response(response, guardrail)


class TestPIIMaskingScenario:
    """Test real-world scenario: PII masking in OCR output"""

    @pytest.mark.asyncio
    async def test_pii_masking_in_ocr_pages(self):
        """Test that PII can be masked from OCR extracted text"""

        class PIIMaskingGuardrail(CustomGuardrail):
            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                texts = inputs.get("texts", [])
                masked_texts = []
                for text in texts:
                    masked = re.sub(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                        "[EMAIL_REDACTED]",
                        text,
                    )
                    masked = re.sub(
                        r"\b\d{3}-\d{2}-\d{4}\b",
                        "[SSN_REDACTED]",
                        masked,
                    )
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = OCRHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        response = OCRResponse(
            pages=[
                OCRPage(
                    index=0,
                    markdown="Name: John Doe\nEmail: john@example.com\nSSN: 123-45-6789",
                ),
                OCRPage(
                    index=1,
                    markdown="Contact: jane@corp.com for details",
                ),
            ],
            model="mistral/mistral-ocr-latest",
        )

        result = await handler.process_output_response(response, guardrail)

        assert "john@example.com" not in result.pages[0].markdown
        assert "123-45-6789" not in result.pages[0].markdown
        assert "[EMAIL_REDACTED]" in result.pages[0].markdown
        assert "[SSN_REDACTED]" in result.pages[0].markdown
        assert "jane@corp.com" not in result.pages[1].markdown
        assert "[EMAIL_REDACTED]" in result.pages[1].markdown
