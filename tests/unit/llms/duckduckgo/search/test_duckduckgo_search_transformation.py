from collections.abc import Iterator
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm


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

        # Mock the httpx AsyncClient get method (DuckDuckGo uses GET)
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            # Make the search call
            response = await litellm.asearch(
                query="python programming", search_provider="duckduckgo", max_results=5
            )

            # Verify the get method was called once
            assert mock_get.call_count == 1

            # Get the actual call arguments
            call_args = mock_get.call_args

            # Verify URL contains the query with proper URL encoding
            url = call_args.kwargs["url"]
            assert "api.duckduckgo.com" in url
            # URL should be properly encoded with %20 for spaces
            assert "q=python+programming" in url or "q=python%20programming" in url
            assert "format=json" in url

            # Verify response structure
            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) > 0

            # Verify first result (Abstract)
            first_result = response.results[0]
            assert first_result.title == "Python (programming language)"
            assert (
                first_result.url
                == "https://en.wikipedia.org/wiki/Python_(programming_language)"
            )
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

        # Mock the httpx AsyncClient get method
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            # Make the search call
            response = await litellm.asearch(
                query="India", search_provider="duckduckgo"
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


_DDG_INSTANT_ANSWER: Final = {
    "AbstractText": "India is a country in South Asia.",
    "AbstractURL": "https://en.wikipedia.org/wiki/India",
    "Heading": "India",
    "RelatedTopics": [
        {"FirstURL": f"https://example.com/{index}", "Text": f"Topic {index} - snippet text for topic {index}."}
        for index in range(10)
    ],
    "Results": [],
    "Type": "D",
}


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_duckduckgo_search_response_structure_and_max_results(
    respx_mock: respx.MockRouter, httpx_transport: None
) -> None:
    route: Final = respx_mock.get(url__startswith="https://api.duckduckgo.com/").mock(
        return_value=httpx.Response(200, json=_DDG_INSTANT_ANSWER)
    )

    response: Final = await litellm.asearch(query="india", search_provider="duckduckgo", max_results=5)

    assert route.call_count == 1
    sent_params: Final = route.calls[0].request.url.params
    assert sent_params["q"] == "india"
    assert sent_params["format"] == "json"
    assert sent_params["_max_results"] == "5"
    assert response.object == "search"
    assert [result.url for result in response.results] == [
        "https://en.wikipedia.org/wiki/India",
        "https://example.com/0",
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    first_result: Final = response.results[0]
    assert first_result.title == "India"
    assert first_result.snippet == "India is a country in South Asia."
    assert response.results[1].title == "Topic 0"
    assert response.results[1].snippet == "snippet text for topic 0."
    assert response._hidden_params["response_cost"] == litellm.model_cost["duckduckgo/search"]["input_cost_per_query"]  # pyright: ignore[reportPrivateUsage]  # cost is only surfaced on _hidden_params


def test_duckduckgo_sync_search_returns_typed_results_without_a_limit(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.get(url__startswith="https://api.duckduckgo.com/").mock(
        return_value=httpx.Response(200, json=_DDG_INSTANT_ANSWER)
    )

    response: Final = litellm.search(query="india", search_provider="duckduckgo")

    assert route.call_count == 1
    assert "_max_results" not in route.calls[0].request.url.params
    assert response.object == "search"
    assert len(response.results) == 11
    assert all(
        isinstance(result.title, str) and isinstance(result.url, str) and isinstance(result.snippet, str)
        for result in response.results
    )
    assert response.results[-1].url == "https://example.com/9"
