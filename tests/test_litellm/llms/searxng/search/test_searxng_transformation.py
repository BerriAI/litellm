"""
Regression tests for HTTP status handling in ``SearXNGSearchConfig``.

The adapter used to build a ``SearchResponse`` from ``raw_response.json()``
without inspecting the status code. A 429 or 503 has no ``results`` key, so the
caller received a successful response with zero results — indistinguishable
from a query that genuinely matched nothing, which meant fallback between
configured search tools never fired.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.searxng.search.transformation import SearXNGSearchConfig


@pytest.fixture
def config() -> SearXNGSearchConfig:
    return SearXNGSearchConfig()


@pytest.fixture
def logging_obj() -> MagicMock:
    return MagicMock()


def _response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("GET", "https://searxng.example.com/search"),
    )


@pytest.mark.parametrize("status_code", [429, 503])
def test_retryable_status_is_raised_not_parsed_as_empty(
    config: SearXNGSearchConfig, logging_obj: MagicMock, status_code: int
) -> None:
    """A rate-limited or unavailable instance must surface as an error."""
    with pytest.raises(BaseLLMException) as exc_info:
        config.transform_search_response(
            raw_response=_response(status_code, {"error": "upstream unavailable"}),
            logging_obj=logging_obj,
        )

    assert exc_info.value.status_code == status_code
    assert str(status_code) in str(exc_info.value)


def test_error_message_does_not_echo_upstream_body(config: SearXNGSearchConfig, logging_obj: MagicMock) -> None:
    """The upstream body can echo the original query; keep it out of the error."""
    with pytest.raises(BaseLLMException) as exc_info:
        config.transform_search_response(
            raw_response=_response(500, {"error": "failed searching for confidential-term"}),
            logging_obj=logging_obj,
        )

    assert "confidential-term" not in str(exc_info.value)


def test_success_status_still_parses_normally(config: SearXNGSearchConfig, logging_obj: MagicMock) -> None:
    """2xx behaviour is unchanged."""
    response = config.transform_search_response(
        raw_response=_response(200, {"results": [{"title": "Still works", "url": "https://example.com"}]}),
        logging_obj=logging_obj,
    )

    assert len(response.results) == 1
    assert response.results[0].title == "Still works"
