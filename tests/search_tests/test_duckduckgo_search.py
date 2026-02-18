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
            
            # Verify URL contains the query
            url = call_args.kwargs["url"]
            assert "api.duckduckgo.com" in url
            assert "q=python" in url or "q=python%20programming" in url
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
