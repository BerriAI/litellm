"""Xquik X post search adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
    SearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_XQUIK_DOCS_URL: Final = "https://docs.xquik.com/api-reference/x/search-tweets"
_XQUIK_PARAMS_KEY: Final = "_xquik_params"
_DomainListAdapter: Final = TypeAdapter(tuple[str, ...])
_QueryParamsAdapter: Final = TypeAdapter(dict[str, str | int | float])
_AUTH_HEADER_NAMES: Final = frozenset(("authorization", "x-api-key"))
_REQUEST_PARAM_NAMES: Final = frozenset(("q", "limit", "placeCountry"))
_NO_QUERY_PARAMS: Final[Mapping[str, str | int | float]] = MappingProxyType({})


class _XquikAuthor(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str | None = None
    username: str | None = None
    name: str | None = None
    followers: int | None = None
    verified: bool | None = None
    profile_picture: str | None = Field(default=None, alias="profilePicture")


class _XquikTweet(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str | None = None
    text: str | None = None
    created_at: str | None = Field(default=None, alias="createdAt")
    url: str | None = None
    author: _XquikAuthor | None = None
    like_count: int | None = Field(default=None, alias="likeCount")
    retweet_count: int | None = Field(default=None, alias="retweetCount")
    reply_count: int | None = Field(default=None, alias="replyCount")
    quote_count: int | None = Field(default=None, alias="quoteCount")
    view_count: int | None = Field(default=None, alias="viewCount")
    bookmark_count: int | None = Field(default=None, alias="bookmarkCount")
    lang: str | None = None


class _XquikSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    tweets: tuple[_XquikTweet, ...]
    has_next_page: bool
    next_cursor: str


class _XquikErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    error: str | None = None
    message: str | None = None


class XquikSearchConfig(BaseSearchConfig):
    XQUIK_API_BASE = "https://xquik.com/api/v1"

    @staticmethod
    def ui_friendly_name() -> str:
        return "Xquik"

    def get_http_method(self) -> Literal["GET", "POST"]:
        return "GET"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.validate_environment passes mutable HTTP headers
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig forwards provider-specific validation arguments
    ) -> dict[str, str]:  # mutable-ok: the HTTP handler requires a mutable header dictionary
        if api_key:
            sanitized_headers: Final = MappingProxyType(
                {key: value for key, value in headers.items() if key.lower() != "x-api-key"}
            )
            return {  # mutable-ok: the HTTP handler passes this dictionary directly to httpx
                **sanitized_headers,
                "x-api-key": api_key,
                "Accept": "application/json",
            }
        if not _has_auth_header(headers):
            raise ValueError("Xquik Search requires api_key or an authentication header.")
        return {**headers, "Accept": "application/json"}  # mutable-ok: httpx requires mutable request headers

    def get_complete_url(
        self,
        api_base: str | None,
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig.get_complete_url passes mutable options
        data: dict[str, object] | list[dict[str, object]] | None = None,  # mutable-ok: inherited request-body contract
        **kwargs: object,  # kwargs-ok: BaseSearchConfig forwards provider-specific URL arguments
    ) -> str:
        resolved_base: Final = (api_base or self.XQUIK_API_BASE).rstrip("/")
        endpoint: Final = (
            resolved_base if resolved_base.endswith("/x/tweets/search") else f"{resolved_base}/x/tweets/search"
        )
        if isinstance(data, dict) and _XQUIK_PARAMS_KEY in data:
            try:
                params: Final = _QueryParamsAdapter.validate_python(data[_XQUIK_PARAMS_KEY])
            except ValidationError as error:
                raise ValueError("Xquik Search request parameters must be a mapping.") from error
            return f"{endpoint}?{urlencode(params)}"
        return endpoint

    def transform_search_request(
        self,
        query: str | list[str],  # mutable-ok: BaseSearchConfig accepts mutable query lists
        optional_params: dict[str, object],  # mutable-ok: BaseSearchConfig passes mutable provider options
        **kwargs: object,  # kwargs-ok: BaseSearchConfig forwards provider-specific request arguments
    ) -> dict[str, object]:  # mutable-ok: the search handler requires a JSON-compatible request dictionary
        resolved_query: Final = " ".join(query) if isinstance(query, list) else query
        unified_params: Final = self.get_supported_perplexity_optional_params()
        passthrough: Final = MappingProxyType(
            {
                key: _query_value(value)
                for key, value in optional_params.items()
                if key not in unified_params and key not in _REQUEST_PARAM_NAMES and value is not None
            }
        )
        country: Final = optional_params.get("country")
        request_params: Final = MappingProxyType(
            {
                "q": _append_domain_filters(resolved_query, optional_params.get("search_domain_filter")),
                **_optional_query_param("limit", optional_params.get("max_results")),
                **_optional_query_param("placeCountry", country.upper() if isinstance(country, str) else None),
                **passthrough,
            }
        )
        return {_XQUIK_PARAMS_KEY: request_params}  # mutable-ok: handler request envelopes are mutable dictionaries

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None,
        **kwargs: object,  # kwargs-ok: BaseSearchConfig forwards provider-specific response arguments
    ) -> SearchResponse:
        headers: Final = dict(raw_response.headers)  # mutable-ok: the base error and metadata contracts require dicts
        if not 200 <= raw_response.status_code < 300:
            raise self.get_error_class(raw_response.text, raw_response.status_code, headers)

        try:
            parsed: Final = _XquikSearchResponse.model_validate_json(raw_response.content)
        except ValidationError as error:
            raise self.get_error_class(
                f"response does not match the documented search schema: {error}",
                raw_response.status_code,
                headers,
            )

        response: Final = SearchResponse.model_validate(
            MappingProxyType(
                {
                    "results": tuple(_search_result(tweet) for tweet in parsed.tweets),
                    "object": "search",
                    "has_next_page": parsed.has_next_page,
                    "next_cursor": parsed.next_cursor,
                }
            )
        )
        response._hidden_params["billed_results"] = len(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # provider cost channel
            parsed.tweets
        )
        response._hidden_params["headers"] = headers  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # provider metadata channel
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str],  # mutable-ok: BaseSearchConfig.get_error_class requires mutable headers
    ) -> Exception:
        detail: Final = _error_detail(error_message).rstrip(". ")
        return BaseLLMException(
            status_code=status_code,
            message=f"Xquik Search: {detail}. See {_XQUIK_DOCS_URL} for details.",
            headers=headers,
        )


def _has_auth_header(headers: Mapping[str, str]) -> bool:
    return any(key.lower() in _AUTH_HEADER_NAMES and bool(value) for key, value in headers.items())


def _optional_query_param(key: str, value: object) -> Mapping[str, str | int | float]:
    return MappingProxyType({key: _query_value(value)}) if value is not None else _NO_QUERY_PARAMS


def _query_value(value: object) -> str | int | float:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, separators=(",", ":"))


def _append_domain_filters(query: str, search_domain_filter: object) -> str:
    try:
        domains: Final = _DomainListAdapter.validate_python(search_domain_filter)
    except ValidationError:
        return query

    included: Final = tuple(domain for domain in domains if domain and not domain.startswith("-"))
    excluded: Final = tuple(domain[1:] for domain in domains if domain.startswith("-") and len(domain) > 1)
    include_clause: Final = f" ({' OR '.join(f'url:{domain}' for domain in included)})" if included else ""
    exclude_clause: Final = "".join(f" -url:{domain}" for domain in excluded)
    return f"({query}){include_clause}{exclude_clause}" if included or excluded else query


def _search_result(tweet: _XquikTweet) -> SearchResult:
    author: Final = tweet.author
    username: Final = author.username if author else None
    title: Final = _result_title(tweet)
    url: Final = tweet.url or (f"https://x.com/{username}/status/{tweet.id}" if username and tweet.id else "")
    return SearchResult.model_validate(
        MappingProxyType(
            {
                "title": title,
                "url": url,
                "snippet": tweet.text or "",
                "date": tweet.created_at,
                "last_updated": None,
                "xquik_tweet": tweet.model_dump(exclude_none=True, by_alias=True),
            }
        )
    )


def _result_title(tweet: _XquikTweet) -> str:
    if tweet.author:
        if tweet.author.name and tweet.author.username:
            return f"{tweet.author.name} (@{tweet.author.username})"
        if tweet.author.name:
            return tweet.author.name
        if tweet.author.username:
            return f"@{tweet.author.username}"
    return f"X post {tweet.id}" if tweet.id else "X post"


def _error_detail(error_message: str) -> str:
    try:
        envelope: Final = _XquikErrorEnvelope.model_validate_json(error_message)
    except ValidationError:
        return error_message
    return envelope.message or envelope.error or error_message
