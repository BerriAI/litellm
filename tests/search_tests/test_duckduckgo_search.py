"""
Tests for DuckDuckGo Search API integration.
"""
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)

import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestDuckDuckGoSearch(BaseSearchTest):
    """
    Tests for DuckDuckGo Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for DuckDuckGo Search.
        """
        return "duckduckgo"
    
    @pytest.mark.asyncio
    async def test_basic_search(self):
        """
        Test basic search functionality with a simple query.
        """
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm._turn_on_debug()
        search_provider = self.get_search_provider()
        print("Search Provider=", search_provider)

        try:
            response = await litellm.asearch(
                query="india",
                search_provider=search_provider,
            )
            print("Search response=", response.model_dump_json(indent=4))

            print(f"\n{'='*80}")
            print(f"Response type: {type(response)}")
            print(f"Response object: {response.object if hasattr(response, 'object') else 'N/A'}")
            
            # Check if response has expected Search format
            assert hasattr(response, "results"), "Response should have 'results' attribute"
            assert hasattr(response, "object"), "Response should have 'object' attribute"
            assert response.object == "search", f"Expected object='search', got '{response.object}'"
            
            # Validate results structure
            assert isinstance(response.results, list), "results should be a list"
            assert len(response.results) > 0, "Should have at least one result"
            
            # Check first result structure
            first_result = response.results[0]
            assert hasattr(first_result, "title"), "Result should have 'title' attribute"
            assert hasattr(first_result, "url"), "Result should have 'url' attribute"
            assert hasattr(first_result, "snippet"), "Result should have 'snippet' attribute"
            
            print(f"Total results: {len(response.results)}")
            print(f"First result title: {first_result.title}")
            print(f"First result URL: {first_result.url}")
            print(f"First result snippet: {first_result.snippet[:100]}...")
            print(f"{'='*80}\n")
            
            assert len(first_result.title) > 0, "Title should not be empty"
            assert len(first_result.url) > 0, "URL should not be empty"
            assert len(first_result.snippet) > 0, "Snippet should not be empty"
            
            # Validate cost tracking in _hidden_params
            assert hasattr(response, "_hidden_params"), "Response should have '_hidden_params' attribute"
            hidden_params = response._hidden_params
            assert "response_cost" in hidden_params, "_hidden_params should contain 'response_cost'"
            
            response_cost = hidden_params["response_cost"]
            assert response_cost is not None, "response_cost should not be None"
            assert isinstance(response_cost, (int, float)), "response_cost should be a number"
            assert response_cost == 0, "response_cost should be 0"
            
            print(f"Cost tracking: ${response_cost:.6f}")
            
        except Exception as e:
            pytest.fail(f"Search call failed: {str(e)}")


    def test_search_response_structure(self):
        """
        Test that the Search response has the correct structure.
        """
        litellm.set_verbose = True
        search_provider = self.get_search_provider()

        response = litellm.search(
            query="india",
            search_provider=search_provider,
        )

        # Validate response structure
        assert hasattr(response, "results"), "Response should have 'results' attribute"
        assert hasattr(response, "object"), "Response should have 'object' attribute"
        
        assert isinstance(response.results, list), "results should be a list"
        assert len(response.results) > 0, "Should have at least one result"
        assert response.object == "search", "object should be 'search'"
        
        # Validate first result structure
        first_result = response.results[0]
        assert hasattr(first_result, "title"), "Result should have 'title' attribute"
        assert hasattr(first_result, "url"), "Result should have 'url' attribute"
        assert hasattr(first_result, "snippet"), "Result should have 'snippet' attribute"
        assert isinstance(first_result.title, str), "title should be a string"
        assert isinstance(first_result.url, str), "url should be a string"
        assert isinstance(first_result.snippet, str), "snippet should be a string"
        
        print(f"\nResponse structure validated:")
        print(f"  - object: {response.object}")
        print(f"  - results: {len(response.results)}")
        print(f"  - first result has all required fields")

class TestDuckDuckGoSearchMocked:
    """
    Tests for DuckDuckGo Search functionality with mocked network responses.
    """
    
    @pytest.mark.asyncio
    async def test_duckduckgo_search_request_payload(self):
        """
        Test that validates the DuckDuckGo search request payload structure without making real API calls.
        """
        # Create a mock response matching DuckDuckGo API format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Abstract": "",
            "AbstractSource": "Wikipedia",
            "AbstractText": "Python is a high-level programming language.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "Answer": "",
            "AnswerType": "",
            "Definition": "",
            "DefinitionSource": "",
            "DefinitionURL": "",
            "Entity": "",
            "Heading": "Python (programming language)",
            "Image": "",
            "ImageHeight": 0,
            "ImageIsLogo": 0,
            "ImageWidth": 0,
            "Infobox": "",
            "Redirect": "",
            "RelatedTopics": [
                {
                    "FirstURL": "https://duckduckgo.com/Python_programming",
                    "Icon": {
                        "Height": "",
                        "URL": "/i/python.png",
                        "Width": ""
                    },
                    "Result": "<a href=\"https://duckduckgo.com/Python_programming\">Python Programming</a> A general-purpose programming language.",
                    "Text": "Python Programming - A general-purpose programming language."
                },
                {
                    "FirstURL": "https://duckduckgo.com/Python_packages",
                    "Icon": {
                        "Height": "",
                        "URL": "",
                        "Width": ""
                    },
                    "Result": "<a href=\"https://duckduckgo.com/Python_packages\">Python Packages</a> Package management in Python.",
                    "Text": "Python Packages - Package management in Python."
                }
            ],
            "Results": [],
            "Type": "A",
            "meta": {
                "attribution": None,
                "blockgroup": None,
                "created_date": None,
                "description": "Wikipedia",
                "designer": None,
                "dev_date": None,
                "dev_milestone": "live",
                "developer": [
                    {
                        "name": "DDG Team",
                        "type": "ddg",
                        "url": "http://www.duckduckhack.com"
                    }
                ],
                "example_query": "python programming",
                "id": "wikipedia_fathead",
                "is_stackexchange": None,
                "js_callback_name": "wikipedia",
                "live_date": None,
                "maintainer": {
                    "github": "duckduckgo"
                },
                "name": "Wikipedia",
                "perl_module": "DDG::Fathead::Wikipedia",
                "producer": None,
                "production_state": "online",
                "repo": "fathead",
                "signal_from": "wikipedia_fathead",
                "src_domain": "en.wikipedia.org",
                "src_id": 1,
                "src_name": "Wikipedia",
                "src_options": {
                    "directory": "",
                    "is_fanon": 0,
                    "is_mediawiki": 1,
                    "is_wikipedia": 1,
                    "language": "en",
                    "min_abstract_length": "20",
                    "skip_abstract": 0,
                    "skip_abstract_paren": 0,
                    "skip_end": "0",
                    "skip_icon": 0,
                    "skip_image_name": 0,
                    "skip_qr": "",
                    "source_skip": "",
                    "src_info": ""
                },
                "src_url": None,
                "status": "live",
                "tab": "About",
                "topic": [
                    "productivity"
                ],
                "unsafe": 0
            }
        }
        
        # Mock the httpx AsyncClient get method (DuckDuckGo uses GET)
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            # Make the search call
            response = await litellm.asearch(
                query="python programming",
                search_provider="duckduckgo",
                max_results=5
            )
            
            # Verify the get method was called once
            assert mock_get.call_count == 1
            
            # Get the actual call arguments
            call_args = mock_get.call_args
            
            # Verify URL contains the query with proper URL encoding
            url = call_args.kwargs["url"]
            assert "api.duckduckgo.com" in url
            # URL should be properly encoded with %20 for spaces
            assert ("q=python+programming" in url or "q=python%20programming" in url)
            assert "format=json" in url
            
            # Verify response structure
            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) > 0
            
            # Verify first result (Abstract)
            first_result = response.results[0]
            assert first_result.title == "Python (programming language)"
            assert first_result.url == "https://en.wikipedia.org/wiki/Python_(programming_language)"
            assert "Python is a high-level programming language" in first_result.snippet
            
            # Verify related topics are included
            assert len(response.results) >= 2  # Abstract + at least one related topic
            
    @pytest.mark.asyncio
    async def test_duckduckgo_search_disambiguation(self):
        """
        Test handling of disambiguation results from DuckDuckGo.
        """
        # Create a mock response with disambiguation type
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Abstract": "",
            "AbstractSource": "Wikipedia",
            "AbstractText": "",
            "AbstractURL": "https://en.wikipedia.org/wiki/India_(disambiguation)",
            "Answer": "",
            "AnswerType": "",
            "Definition": "",
            "DefinitionSource": "",
            "DefinitionURL": "",
            "Entity": "",
            "Heading": "India",
            "Image": "",
            "ImageHeight": 0,
            "ImageIsLogo": 0,
            "ImageWidth": 0,
            "Infobox": "",
            "Redirect": "",
            "RelatedTopics": [
                {
                    "FirstURL": "https://duckduckgo.com/India",
                    "Icon": {
                        "Height": "",
                        "URL": "/i/cef47a13.png",
                        "Width": ""
                    },
                    "Result": "<a href=\"https://duckduckgo.com/India\">India</a> A country in South Asia.",
                    "Text": "India - A country in South Asia."
                },
                {
                    "Name": "Related Topics",
                    "Topics": [
                        {
                            "FirstURL": "https://duckduckgo.com/d/Indus",
                            "Icon": {
                                "Height": "",
                                "URL": "",
                                "Width": ""
                            },
                            "Result": "<a href=\"https://duckduckgo.com/d/Indus\">Indus</a> See related meanings for the word 'Indus'.",
                            "Text": "Indus - See related meanings for the word 'Indus'."
                        }
                    ]
                }
            ],
            "Results": [],
            "Type": "D",
            "meta": {}
        }
        
        # Mock the httpx AsyncClient get method
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            # Make the search call
            response = await litellm.asearch(
                query="India",
                search_provider="duckduckgo"
            )
            
            # Verify response structure
            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            
            # Should have results from both direct topics and nested topics
            assert len(response.results) >= 2
            
            # Verify nested topics are processed
            urls = [result.url for result in response.results]
            assert any("India" in url for url in urls)
            assert any("Indus" in url for url in urls)
