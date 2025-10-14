"""
Tests for Vertex AI rerank transformation functionality.
Based on the test patterns from other rerank providers and the current Vertex AI implementation.
"""
import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.vertex_ai.rerank.transformation import VertexAIRerankConfig
from litellm.types.rerank import RerankResponse


class TestVertexAIRerankTransform:
    def setup_method(self):
        self.config = VertexAIRerankConfig()
        self.model = "semantic-ranker-default@latest"

    def test_get_complete_url(self):
        """Test URL generation for Vertex AI Discovery Engine rerank API."""
        # Test with project ID from environment
        with patch.dict(os.environ, {"VERTEXAI_PROJECT": "test-project-123"}):
            url = self.config.get_complete_url(api_base=None, model=self.model)
            expected_url = "https://discoveryengine.googleapis.com/v1/projects/test-project-123/locations/global/rankingConfigs/default_ranking_config:rank"
            assert url == expected_url

        # Test with litellm.vertex_project
        with patch.dict(os.environ, {}, clear=True):
            import litellm
            # Set vertex_project attribute if it doesn't exist
            if not hasattr(litellm, 'vertex_project'):
                litellm.vertex_project = None
            original_project = litellm.vertex_project
            litellm.vertex_project = "litellm-project-456"
            try:
                url = self.config.get_complete_url(api_base=None, model=self.model)
                expected_url = "https://discoveryengine.googleapis.com/v1/projects/litellm-project-456/locations/global/rankingConfigs/default_ranking_config:rank"
                assert url == expected_url
            finally:
                litellm.vertex_project = original_project

        # Test error when no project ID is available
        with patch.dict(os.environ, {}, clear=True):
            import litellm
            # Set vertex_project to None to ensure no project ID is available
            if not hasattr(litellm, 'vertex_project'):
                litellm.vertex_project = None
            original_project = litellm.vertex_project
            litellm.vertex_project = None
            try:
                with pytest.raises(ValueError, match="Vertex AI project ID is required"):
                    self.config.get_complete_url(api_base=None, model=self.model)
            finally:
                litellm.vertex_project = original_project

    @patch('litellm.llms.vertex_ai.rerank.transformation.VertexAIRerankConfig._ensure_access_token')
    def test_validate_environment(self, mock_ensure_access_token):
        """Test environment validation and header setup."""
        # Mock the authentication
        mock_ensure_access_token.return_value = ("test-access-token", "test-project-123")
        
        # Mock the credential and project methods
        with patch.object(self.config, 'get_vertex_ai_credentials', return_value=None), \
             patch.object(self.config, 'get_vertex_ai_project', return_value="test-project-123"):
            
            headers = self.config.validate_environment(
                headers={},
                model=self.model,
                api_key=None
            )
            
            expected_headers = {
                "Authorization": "Bearer test-access-token",
                "Content-Type": "application/json",
                "X-Goog-User-Project": "test-project-123"
            }
            assert headers == expected_headers

    def test_transform_rerank_request_basic(self):
        """Test basic request transformation for Vertex AI Discovery Engine format."""
        optional_params = {
            "query": "What is Google Gemini?",
            "documents": [
                "Gemini is a cutting edge large language model created by Google.",
                "The Gemini zodiac symbol often depicts two figures standing side-by-side."
            ],
            "top_n": 2
        }
        
        request_data = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params=optional_params,
            headers={}
        )
        
        # Verify basic structure
        assert request_data["model"] == self.model
        assert request_data["query"] == "What is Google Gemini?"
        assert request_data["topN"] == 2
        assert "records" in request_data
        assert len(request_data["records"]) == 2
        
        # Verify record structure
        for i, record in enumerate(request_data["records"]):
            assert "id" in record
            assert "title" in record
            assert "content" in record
            assert record["id"] == str(i)  # 0-based indexing
            assert len(record["title"].split()) <= 3  # First 3 words as title

    def test_transform_rerank_request_with_dict_documents(self):
        """Test request transformation with dictionary documents."""
        optional_params = {
            "query": "What is Google Gemini?",
            "documents": [
                {"text": "Gemini is a cutting edge large language model created by Google.", "title": "Custom Title 1"},
                {"text": "The Gemini zodiac symbol often depicts two figures standing side-by-side."}
            ]
        }
        
        request_data = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params=optional_params,
            headers={}
        )
        
        # Verify record structure with custom titles
        assert request_data["records"][0]["title"] == "Custom Title 1"
        assert request_data["records"][1]["title"] == "The Gemini zodiac"  # First 3 words

    def test_transform_rerank_request_return_documents_mapping(self):
        """Test return_documents to ignoreRecordDetailsInResponse mapping."""
        # Test return_documents=True (default)
        optional_params_true = {
            "query": "test query",
            "documents": ["doc1", "doc2"],
            "return_documents": True
        }
        
        request_data_true = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params=optional_params_true,
            headers={}
        )
        assert request_data_true["ignoreRecordDetailsInResponse"] == False
        
        # Test return_documents=False
        optional_params_false = {
            "query": "test query",
            "documents": ["doc1", "doc2"],
            "return_documents": False
        }
        
        request_data_false = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params=optional_params_false,
            headers={}
        )
        assert request_data_false["ignoreRecordDetailsInResponse"] == True
        
        # Test return_documents not specified (should default to True)
        optional_params_default = {
            "query": "test query",
            "documents": ["doc1", "doc2"]
        }
        
        request_data_default = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params=optional_params_default,
            headers={}
        )
        assert request_data_default["ignoreRecordDetailsInResponse"] == False

    def test_transform_rerank_request_missing_required_params(self):
        """Test that transform_rerank_request handles missing required parameters."""
        # Test missing query
        with pytest.raises(ValueError, match="query is required for Vertex AI rerank"):
            self.config.transform_rerank_request(
                model=self.model,
                optional_rerank_params={"documents": ["doc1"]},
                headers={}
            )
        
        # Test missing documents
        with pytest.raises(ValueError, match="documents is required for Vertex AI rerank"):
            self.config.transform_rerank_request(
                model=self.model,
                optional_rerank_params={"query": "test query"},
                headers={}
            )

    def test_transform_rerank_response_success(self):
        """Test successful response transformation."""
        # Mock Vertex AI Discovery Engine response format
        response_data = {
            "records": [
                {
                    "id": "1",
                    "score": 0.98,
                    "title": "The Science of a Blue Sky",
                    "content": "The sky appears blue due to a phenomenon called Rayleigh scattering."
                },
                {
                    "id": "0",
                    "score": 0.64,
                    "title": "The Color of the Sky: A Poem",
                    "content": "A canvas stretched across the day, Where sunlight learns to dance and play."
                }
            ]
        }
        
        # Create mock httpx response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)
        
        # Create mock logging object
        mock_logging = MagicMock()
        
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
        assert result.results[0]["index"] == 1  # Converted back to 0-based index
        assert result.results[0]["relevance_score"] == 0.98
        assert result.results[1]["index"] == 0
        assert result.results[1]["relevance_score"] == 0.64
        
        # Verify metadata
        assert result.meta["billed_units"]["search_units"] == 2

    def test_transform_rerank_response_with_ignore_record_details(self):
        """Test response transformation when ignoreRecordDetailsInResponse=true."""
        # Mock response with only IDs (when ignoreRecordDetailsInResponse=true)
        response_data = {
            "records": [
                {"id": "1"},
                {"id": "0"}
            ]
        }
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)
        
        mock_logging = MagicMock()
        model_response = RerankResponse()
        
        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )
        
        # Verify response structure with default scores
        assert len(result.results) == 2
        assert result.results[0]["index"] == 1  # 0-based index
        assert result.results[0]["relevance_score"] == 1.0  # Default score
        assert result.results[1]["index"] == 0
        assert result.results[1]["relevance_score"] == 1.0

    def test_transform_rerank_response_json_error(self):
        """Test response transformation with JSON parsing error."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.text = "Invalid JSON response"
        
        mock_logging = MagicMock()
        model_response = RerankResponse()
        
        with pytest.raises(ValueError, match="Failed to parse response"):
            self.config.transform_rerank_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
            )

    def test_get_supported_cohere_rerank_params(self):
        """Test getting supported parameters for Vertex AI rerank."""
        supported_params = self.config.get_supported_cohere_rerank_params(self.model)
        expected_params = ["query", "documents", "top_n", "return_documents"]
        assert supported_params == expected_params

    def test_map_cohere_rerank_params(self):
        """Test parameter mapping for Vertex AI rerank."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"documents": ["doc1", "doc2"]},
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
            top_n=2,
            return_documents=True
        )
        
        expected_params = {
            "query": "test query",
            "documents": ["doc1", "doc2"],
            "top_n": 2,
            "return_documents": True
        }
        assert params == expected_params

    def test_title_generation_from_content(self):
        """Test that titles are generated correctly from document content."""
        optional_params = {
            "query": "test query",
            "documents": [
                "This is a very long document with many words that should be truncated to only the first three words for the title",
                "Short doc",
                "Another document with multiple words here"
            ]
        }
        
        request_data = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params=optional_params,
            headers={}
        )
        
        # Verify title generation
        assert request_data["records"][0]["title"] == "This is a"  # First 3 words
        assert request_data["records"][1]["title"] == "Short doc"  # Less than 3 words
        assert request_data["records"][2]["title"] == "Another document with"  # First 3 words

    def test_record_id_generation(self):
        """Test that record IDs are generated correctly with 0-based indexing."""
        optional_params = {
            "query": "test query",
            "documents": ["doc1", "doc2", "doc3", "doc4"]
        }
        
        request_data = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params=optional_params,
            headers={}
        )
        
        # Verify 0-based indexing
        for i, record in enumerate(request_data["records"]):
            assert record["id"] == str(i)
