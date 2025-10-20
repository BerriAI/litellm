"""
Integration tests for Vertex AI rerank functionality.
These tests demonstrate end-to-end usage of the Vertex AI rerank feature.
"""
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.vertex_ai.rerank.transformation import VertexAIRerankConfig


class TestVertexAIRerankIntegration:
    def setup_method(self):
        self.config = VertexAIRerankConfig()
        self.model = "semantic-ranker-default@latest"

    @patch('litellm.llms.vertex_ai.rerank.transformation.VertexAIRerankConfig._ensure_access_token')
    def test_end_to_end_rerank_flow(self, mock_ensure_access_token):
        """Test complete rerank flow from request to response."""
        # Mock authentication
        mock_ensure_access_token.return_value = ("test-access-token", "test-project-123")
        
        # Test documents
        documents = [
            "Gemini is a cutting edge large language model created by Google.",
            "The Gemini zodiac symbol often depicts two figures standing side-by-side.",
            "Gemini is a constellation that can be seen in the night sky.",
            "Google's Gemini AI model represents a significant advancement in artificial intelligence technology."
        ]
        query = "What is Google Gemini?"
        
        # Step 1: Test request transformation
        with patch.object(self.config, 'get_vertex_ai_credentials', return_value=None), \
             patch.object(self.config, 'get_vertex_ai_project', return_value="test-project-123"):
            
            # Validate environment
            headers = self.config.validate_environment(
                headers={},
                model=self.model,
                api_key=None
            )
            
            # Transform request
            request_data = self.config.transform_rerank_request(
                model=self.model,
                optional_rerank_params={
                    "query": query,
                    "documents": documents,
                    "top_n": 2,
                    "return_documents": True
                },
                headers=headers
            )
            
            # Verify request structure
            assert request_data["model"] == self.model
            assert request_data["query"] == query
            assert request_data["topN"] == 2
            assert request_data["ignoreRecordDetailsInResponse"] == False
            assert len(request_data["records"]) == 4
            
            # Verify record structure
            for i, record in enumerate(request_data["records"]):
                assert record["id"] == str(i)  # 0-based indexing
                assert "title" in record
                assert "content" in record
                assert record["content"] == documents[i]
        
        # Step 2: Test response transformation
        # Mock Vertex AI Discovery Engine response
        mock_response_data = {
            "records": [
                {
                    "id": "3",
                    "score": 0.95,
                    "title": "Google's Gemini AI model",
                    "content": "Google's Gemini AI model represents a significant advancement in artificial intelligence technology."
                },
                {
                    "id": "0",
                    "score": 0.92,
                    "title": "Gemini is a",
                    "content": "Gemini is a cutting edge large language model created by Google."
                }
            ]
        }
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.text = '{"records": [{"id": "3", "score": 0.95, "title": "Google\'s Gemini AI model", "content": "Google\'s Gemini AI model represents a significant advancement in artificial intelligence technology."}, {"id": "0", "score": 0.92, "title": "Gemini is a", "content": "Gemini is a cutting edge large language model created by Google."}]}'
        
        mock_logging = MagicMock()
        
        # Transform response
        from litellm.types.rerank import RerankResponse
        model_response = RerankResponse()
        
        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )
        
        # Verify response structure
        assert result.id == f"vertex_ai_rerank_{self.model}"
        assert len(result.results) == 2
        
        # Results should be sorted by relevance score (descending)
        assert result.results[0]["index"] == 3  # Highest score
        assert result.results[0]["relevance_score"] == 0.95
        assert result.results[1]["index"] == 0  # Second highest score
        assert result.results[1]["relevance_score"] == 0.92
        
        # Verify metadata
        assert result.meta["billed_units"]["search_units"] == 2

    def test_return_documents_false_flow(self):
        """Test rerank flow when return_documents=False (ID-only response)."""
        documents = ["doc1", "doc2", "doc3"]
        query = "test query"
        
        # Transform request with return_documents=False
        request_data = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params={
                "query": query,
                "documents": documents,
                "return_documents": False
            },
            headers={}
        )
        
        # Verify ignoreRecordDetailsInResponse is True
        assert request_data["ignoreRecordDetailsInResponse"] == True
        
        # Mock response with only IDs
        mock_response_data = {
            "records": [
                {"id": "1"},
                {"id": "0"},
                {"id": "2"}
            ]
        }
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_response.text = '{"records": [{"id": "1"}, {"id": "0"}, {"id": "2"}]}'
        
        mock_logging = MagicMock()
        
        # Transform response
        from litellm.types.rerank import RerankResponse
        model_response = RerankResponse()
        
        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )
        
        # Verify response structure with default scores
        assert len(result.results) == 3
        for result_item in result.results:
            assert result_item["relevance_score"] == 1.0  # Default score when details are ignored
            assert "index" in result_item

    def test_document_title_generation(self):
        """Test that document titles are generated correctly from content."""
        documents = [
            "This is a very long document with many words that should be truncated to only the first three words for the title",
            "Short doc",
            "Another document with multiple words here and more content"
        ]
        
        request_data = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params={
                "query": "test query",
                "documents": documents
            },
            headers={}
        )
        
        # Verify title generation
        assert request_data["records"][0]["title"] == "This is a"  # First 3 words
        assert request_data["records"][1]["title"] == "Short doc"  # Less than 3 words
        assert request_data["records"][2]["title"] == "Another document with"  # First 3 words

    def test_dictionary_document_handling(self):
        """Test handling of dictionary-format documents."""
        documents = [
            {"text": "Gemini is a cutting edge large language model created by Google.", "title": "Custom Title 1"},
            {"text": "The Gemini zodiac symbol often depicts two figures standing side-by-side."},
            {"text": "Gemini is a constellation that can be seen in the night sky.", "title": "Custom Title 3"}
        ]
        
        request_data = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params={
                "query": "test query",
                "documents": documents
            },
            headers={}
        )
        
        # Verify custom titles are used when provided
        assert request_data["records"][0]["title"] == "Custom Title 1"
        assert request_data["records"][1]["title"] == "The Gemini zodiac"  # Generated from first 3 words
        assert request_data["records"][2]["title"] == "Custom Title 3"
        
        # Verify content is extracted correctly
        assert request_data["records"][0]["content"] == "Gemini is a cutting edge large language model created by Google."
        assert request_data["records"][1]["content"] == "The Gemini zodiac symbol often depicts two figures standing side-by-side."
        assert request_data["records"][2]["content"] == "Gemini is a constellation that can be seen in the night sky."
