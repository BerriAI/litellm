"""
Webz.io News Search provider implementation for LiteLLM.
Endpoint: POST https://api.webz.io/api/news/context
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    SearchResponse,
    SearchResult,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.secret_managers.main import get_secret_str

WEBZ_API_URL: Final = "https://api.webz.io/api/news/context"


def _build_webz_payload(
    query: str | list[str],
    max_results: int | None = 5,
    country: str | None = None,
    language: str | None = None,
    domain: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Build Webz News Search JSON payload.
    Maps:
      - query -> query
      - max_results -> k (defaults to 5)
      - country -> country
      - language -> language
      - domain -> domain (or first item of search_domain_filter)
    """
    if isinstance(query, list):
        query_str = " ".join(query)
    else:
        query_str = str(query)

    k = kwargs.get("k")
    if k is None:
        k = max_results if max_results is not None else 5
    payload: dict[str, Any] = {
        "query": query_str,
        "k": k,
    }

    resolved_country = country if country is not None else kwargs.get("country")
    if resolved_country is not None:
        payload["country"] = resolved_country

    resolved_language = language if language is not None else kwargs.get("language")
    if resolved_language is not None:
        payload["language"] = resolved_language

    resolved_domain = domain if domain is not None else kwargs.get("domain")
    if resolved_domain is None and "search_domain_filter" in kwargs and kwargs["search_domain_filter"]:
        domain_filter = kwargs["search_domain_filter"]
        if isinstance(domain_filter, list) and len(domain_filter) > 0:
            resolved_domain = domain_filter[0]
        elif isinstance(domain_filter, str):
            resolved_domain = domain_filter

    if resolved_domain is not None:
        payload["domain"] = resolved_domain

    # Pass through other filter/custom parameters if provided
    excluded_keys = {
        "query",
        "max_results",
        "k",
        "country",
        "language",
        "domain",
        "search_domain_filter",
        "max_tokens_per_page",
        "api_key",
        "api_base",
        "timeout",
        "extra_headers",
        "client",
        "asearch",
        "search_provider",
    }
    for key, value in kwargs.items():
        if key not in excluded_keys and key not in payload and value is not None:
            payload[key] = value

    return payload


def _parse_webz_response(response_data: Mapping[str, Any]) -> SearchResponse:
    """
    Parse Webz.io JSON response into LiteLLM standard SearchResponse.
    Maps:
      - result.article.title -> SearchResult.title
      - result.article.url -> SearchResult.url
      - result.chunk.text -> SearchResult.snippet
      - result.article.published_at -> SearchResult.date
      - result.score -> SearchResult.score
    """
    raw_results = response_data.get("results", [])
    results: list[SearchResult] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue
        article = item.get("article") or {}
        chunk = item.get("chunk") or {}

        title = article.get("title") or item.get("title", "")
        url = article.get("url") or item.get("url", "")
        snippet = chunk.get("text") or item.get("snippet", "")
        date = article.get("published_at") or item.get("date")
        score = item.get("score")

        result_dict: dict[str, Any] = {
            "title": str(title) if title is not None else "",
            "url": str(url) if url is not None else "",
            "snippet": str(snippet) if snippet is not None else "",
            "date": str(date) if date is not None else None,
        }
        if score is not None:
            result_dict["score"] = score

        results.append(SearchResult.model_validate(result_dict))

    return SearchResponse(
        results=results,
        object="search",
    )


async def async_search(
    query: str | list[str],
    api_key: str | None = None,
    max_results: int = 5,
    country: str | None = None,
    language: str | None = None,
    domain: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    extra_headers: dict[str, Any] | None = None,
    client: AsyncHTTPHandler | None = None,
    **kwargs: Any,
) -> SearchResponse:
    """
    Asynchronously search Webz.io News Search API.

    POST https://api.webz.io/api/news/context
    Headers:
      Authorization: Bearer <api_key>
      Content-Type: application/json
    """
    resolved_api_key = (
        api_key
        or os.environ.get("WEBZ_API_TOKEN")
        or get_secret_str("WEBZ_API_TOKEN")
        or os.environ.get("WEBZ_API_KEY")
        or get_secret_str("WEBZ_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError("WEBZ_API_TOKEN is not set. Set `WEBZ_API_TOKEN` environment variable or pass `api_key`.")

    headers: dict[str, str] = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update({k: str(v) for k, v in extra_headers.items()})

    url = api_base or os.environ.get("WEBZ_API_BASE") or get_secret_str("WEBZ_API_BASE") or WEBZ_API_URL
    payload = _build_webz_payload(
        query=query,
        max_results=max_results,
        country=country,
        language=language,
        domain=domain,
        **kwargs,
    )

    http_client = client or AsyncHTTPHandler(timeout=timeout)
    response = await http_client.post(
        url=url,
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        raise BaseLLMException(
            status_code=response.status_code,
            message=f"Webz.io API returned {response.status_code}: {response.text}",
        )

    response_json = response.json()
    return _parse_webz_response(response_json)


def search(
    query: str | list[str],
    api_key: str | None = None,
    max_results: int = 5,
    country: str | None = None,
    language: str | None = None,
    domain: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    extra_headers: dict[str, Any] | None = None,
    client: HTTPHandler | None = None,
    **kwargs: Any,
) -> SearchResponse:
    """
    Synchronously search Webz.io News Search API.
    """
    resolved_api_key = (
        api_key
        or os.environ.get("WEBZ_API_TOKEN")
        or get_secret_str("WEBZ_API_TOKEN")
        or os.environ.get("WEBZ_API_KEY")
        or get_secret_str("WEBZ_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError("WEBZ_API_TOKEN is not set. Set `WEBZ_API_TOKEN` environment variable or pass `api_key`.")

    headers: dict[str, str] = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update({k: str(v) for k, v in extra_headers.items()})

    url = api_base or os.environ.get("WEBZ_API_BASE") or get_secret_str("WEBZ_API_BASE") or WEBZ_API_URL
    payload = _build_webz_payload(
        query=query,
        max_results=max_results,
        country=country,
        language=language,
        domain=domain,
        **kwargs,
    )

    http_client = client or HTTPHandler(timeout=timeout)
    response = http_client.post(
        url=url,
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        raise BaseLLMException(
            status_code=response.status_code,
            message=f"Webz.io API returned {response.status_code}: {response.text}",
        )

    response_json = response.json()
    return _parse_webz_response(response_json)
