"""
Unit tests for Cohere Rerank Guardrail Translation Handler
"""

import asyncio
import os
import sys
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.cohere.rerank.guardrail_translation.handler import CohereRerankHandler
from litellm.types.rerank import RerankResponse
from litellm.types.utils import CallTypes


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing"""

    async def apply_guardrail(
        self, inputs: dict, request_data: dict, input_type: str, **kwargs
    ) -> dict:
        texts = inputs.get("texts", [])
        return {"texts": [f"{text} [GUARDRAILED]" for text in texts]}


class TestHandlerDiscovery:
    """Test that the handler is properly discovered"""

    def test_handler_discovered_for_rerank(self):
        """Test that rerank CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.rerank)
        assert handler_class == CohereRerankHandler

    def test_handler_discovered_for_arerank(self):
        """Test that arerank CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.arerank)
        assert handler_class == CohereRerankHandler


class TestInputProcessing:
    """Test input processing functionality"""

    @pytest.mark.asyncio
    async def test_process_query_only(self):
        """Test processing just the query"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "rerank-english-v3.0",
            "query": "What is machine learning?",
            "documents": ["Doc 1", "Doc 2"],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Query should be guardrailed
        assert result["query"] == "What is machine learning? [GUARDRAILED]"
        # Documents should be unchanged
        assert result["documents"] == ["Doc 1", "Doc 2"]

    @pytest.mark.asyncio
    async def test_process_query_with_dict_documents(self):
        """Test that query is guardrailed but dict documents are unchanged"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "rerank-english-v3.0",
            "query": "Python programming",
            "documents": [
                {"text": "Python is a programming language.", "id": "doc1"},
                {"text": "JavaScript is used for web development.", "id": "doc2"},
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Query should be guardrailed
        assert result["query"] == "Python programming [GUARDRAILED]"

        # Documents should be completely unchanged
        assert result["documents"][0] == {
            "text": "Python is a programming language.",
            "id": "doc1",
        }
        assert result["documents"][1] == {
            "text": "JavaScript is used for web development.",
            "id": "doc2",
        }

    @pytest.mark.asyncio
    async def test_process_no_query(self):
        """Test processing when query is missing"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "rerank-english-v3.0",
            "documents": ["Document 1", "Document 2"],
        }

        result = await handler.process_input_messages(data, guardrail)

        # No query to process, documents unchanged
        assert "query" not in result
        assert result["documents"] == ["Document 1", "Document 2"]

    @pytest.mark.asyncio
    async def test_process_empty_documents(self):
        """Test processing with empty documents list"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "rerank-english-v3.0",
            "query": "Test query",
            "documents": [],
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["query"] == "Test query [GUARDRAILED]"
        assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_process_long_query(self):
        """Test processing a long query"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        long_query = "This is a very long search query. " * 20

        data = {
            "model": "rerank-english-v3.0",
            "query": long_query,
            "documents": ["Doc 1"],
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["query"] == f"{long_query} [GUARDRAILED]"
        assert long_query in result["query"]
        # Document unchanged
        assert result["documents"] == ["Doc 1"]


class TestOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_output_processing_is_noop(self):
        """Test that output processing returns response unchanged"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create a mock RerankResponse
        response = RerankResponse(
            id="rerank-123",
            results=[
                {"index": 0, "relevance_score": 0.98},
                {"index": 2, "relevance_score": 0.75},
            ],
        )

        result = await handler.process_output_response(response, guardrail)

        # Should return unchanged
        assert result == response
        assert result.id == "rerank-123"
        assert len(result.results) == 2


class TestPIIMaskingScenario:
    """Test real-world scenario: PII masking in rerank query"""

    @pytest.mark.asyncio
    async def test_pii_masking_in_query(self):
        """Test that PII can be masked from query only"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                import re

                texts = inputs.get("texts", [])
                masked_texts = []
                for text in texts:
                    masked = re.sub(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                        "[EMAIL_REDACTED]",
                        text,
                    )
                    masked = masked.replace("John Doe", "[NAME_REDACTED]")
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = CohereRerankHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        data = {
            "model": "rerank-english-v3.0",
            "query": "Find documents about John Doe at john@example.com",
            "documents": [
                "Document 1 content",
                "Document 2 content",
                "Document 3 content",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify PII was masked in query
        assert "john@example.com" not in result["query"]
        assert "John Doe" not in result["query"]
        assert "[EMAIL_REDACTED]" in result["query"]
        assert "[NAME_REDACTED]" in result["query"]

        # Verify documents were NOT modified
        assert result["documents"] == [
            "Document 1 content",
            "Document 2 content",
            "Document 3 content",
        ]

    @pytest.mark.asyncio
    async def test_multiple_pii_types_in_query(self):
        """Test masking multiple PII types in query"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                import re

                texts = inputs.get("texts", [])
                masked_texts = []
                for text in texts:
                    # Mask emails
                    masked = re.sub(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                        "[EMAIL_REDACTED]",
                        text,
                    )
                    # Mask phone numbers
                    masked = re.sub(r"\d{3}-\d{3}-\d{4}", "[PHONE_REDACTED]", masked)
                    # Mask names
                    masked = masked.replace("Alice Smith", "[NAME_REDACTED]")
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = CohereRerankHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        data = {
            "model": "rerank-english-v3.0",
            "query": "Search for Alice Smith at alice@company.com or call 555-123-4567",
            "documents": ["Doc 1"],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify all PII types were masked in query
        assert "alice@company.com" not in result["query"]
        assert "Alice Smith" not in result["query"]
        assert "555-123-4567" not in result["query"]
        assert "[EMAIL_REDACTED]" in result["query"]
        assert "[NAME_REDACTED]" in result["query"]
        assert "[PHONE_REDACTED]" in result["query"]

        # Documents unchanged
        assert result["documents"] == ["Doc 1"]


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test with empty query"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "rerank-english-v3.0",
            "query": "",
            "documents": ["Doc 1", "Doc 2"],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Empty string should still be processed
        assert result["query"] == " [GUARDRAILED]"
        # Documents unchanged
        assert result["documents"] == ["Doc 1", "Doc 2"]

    @pytest.mark.asyncio
    async def test_none_query(self):
        """Test with None query"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "rerank-english-v3.0",
            "query": None,
            "documents": ["Doc 1", "Doc 2"],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Query None should be unchanged
        assert result["query"] is None
        # Documents unchanged
        assert result["documents"] == ["Doc 1", "Doc 2"]

    @pytest.mark.asyncio
    async def test_query_with_special_characters(self):
        """Test query with special characters"""
        handler = CohereRerankHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "rerank-english-v3.0",
            "query": "Search for: @user #hashtag & more!",
            "documents": ["Doc 1"],
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["query"] == "Search for: @user #hashtag & more! [GUARDRAILED]"
        assert result["documents"] == ["Doc 1"]


class TestContentFilteringScenario:
    """Test real-world scenario: Content filtering in rerank query"""

    @pytest.mark.asyncio
    async def test_content_filtering_query(self):
        """Test filtering inappropriate content from query only"""

        class ContentFilterGuardrail(CustomGuardrail):
            """Mock content filter guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                bad_words = ["inappropriate", "offensive"]
                texts = inputs.get("texts", [])
                filtered_texts = []
                for text in texts:
                    filtered = text
                    for word in bad_words:
                        filtered = filtered.replace(word, "[FILTERED]")
                    filtered_texts.append(filtered)
                return {"texts": filtered_texts}

        handler = CohereRerankHandler()
        guardrail = ContentFilterGuardrail(guardrail_name="content_filter")

        data = {
            "model": "rerank-english-v3.0",
            "query": "Find inappropriate and offensive content",
            "documents": [
                "Document 1 content",
                "Document 2 content",
                "Document 3 content",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Query should be filtered
        assert "inappropriate" not in result["query"]
        assert "offensive" not in result["query"]
        assert "[FILTERED]" in result["query"]

        # Documents should be unchanged
        assert result["documents"] == [
            "Document 1 content",
            "Document 2 content",
            "Document 3 content",
        ]
