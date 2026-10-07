from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest


class TestDuckDuckGoSearchMocked:
    """
    Tests for DuckDuckGo Search functionality with mocked network responses.
    """

    @pytest.mark.asyncio
    async def test_duckduckgo_search_request_payload(self):
        """
        Test that validates the DuckDuckGo search request payload structure without making real API calls.
        """
        mock_response: Final = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
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
                            "Icon": {"Height": "", "URL": "/i/python.png", "Width": ""},
                            "Result": '<a href="https://duckduckgo.com/Python_programming">Python Programming</a> A general-purpose programming language.',
                            "Text": "Python Programming - A general-purpose programming language.",
                        },
                        {
                            "FirstURL": "https://duckduckgo.com/Python_packages",
                            "Icon": {"Height": "", "URL": "", "Width": ""},
                            "Result": '<a href="https://duckduckgo.com/Python_packages">Python Packages</a> Package management in Python.',
                            "Text": "Python Packages - Package management in Python.",
                        },
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
                                "url": "http://www.duckduckhack.com",
                            }
                        ],
                        "example_query": "python programming",
                        "id": "wikipedia_fathead",
                        "is_stackexchange": None,
                        "js_callback_name": "wikipedia",
                        "live_date": None,
                        "maintainer": {"github": "duckduckgo"},
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
                            "src_info": "",
                        },
                        "src_url": None,
                        "status": "live",
                        "tab": "About",
                        "topic": ["productivity"],
                        "unsafe": 0,
                    },
                }
            ),
        )

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_get:
            response: Final = await litellm.asearch(
                query="python programming", search_provider="duckduckgo", max_results=5
            )

            assert mock_get.call_count == 1

            call_args: Final = mock_get.call_args
            url: Final = call_args.kwargs["url"]
            assert "api.duckduckgo.com" in url
            assert "q=python+programming" in url or "q=python%20programming" in url
            assert "format=json" in url

            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) > 0

            first_result: Final = response.results[0]
            assert first_result.title == "Python (programming language)"
            assert first_result.url == "https://en.wikipedia.org/wiki/Python_(programming_language)"
            assert "Python is a high-level programming language" in first_result.snippet

            assert len(response.results) >= 2

    @pytest.mark.asyncio
    async def test_duckduckgo_search_disambiguation(self):
        """
        Test handling of disambiguation results from DuckDuckGo.
        """
        mock_response: Final = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
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
                            "Icon": {"Height": "", "URL": "/i/cef47a13.png", "Width": ""},
                            "Result": '<a href="https://duckduckgo.com/India">India</a> A country in South Asia.',
                            "Text": "India - A country in South Asia.",
                        },
                        {
                            "Name": "Related Topics",
                            "Topics": [
                                {
                                    "FirstURL": "https://duckduckgo.com/d/Indus",
                                    "Icon": {"Height": "", "URL": "", "Width": ""},
                                    "Result": "<a href=\"https://duckduckgo.com/d/Indus\">Indus</a> See related meanings for the word 'Indus'.",
                                    "Text": "Indus - See related meanings for the word 'Indus'.",
                                }
                            ],
                        },
                    ],
                    "Results": [],
                    "Type": "D",
                    "meta": {},
                }
            ),
        )

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response: Final = await litellm.asearch(query="India", search_provider="duckduckgo")

            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"

            assert len(response.results) >= 2

            urls: Final = tuple(result.url for result in response.results)
            assert any("India" in url for url in urls)
            assert any("Indus" in url for url in urls)
