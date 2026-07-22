"""
Optional MCP server trust scoring via Dominion Observatory.

Queries https://dominionobservatory.com/api/trust for HTTP MCP server URLs and
uses the returned trust score to filter or rank servers before tool listing and
tool calls. Results are cached to respect the free-tier query limit.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Callable, Literal, Protocol, Sequence, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    DEFAULT_MCP_TRUST_SCORING_API_URL,
    DEFAULT_MCP_TRUST_SCORING_CACHE_TTL,
    DEFAULT_MCP_TRUST_SCORING_MIN_SCORE,
    MCP_TRUST_SCORING_CACHE_MAX_SIZE,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

TRUST_SCORE_THRESHOLD_DENIED = (
    "Server does not meet the configured trust score threshold."
)

_SENSITIVE_QUERY_PARAM_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "credential",
        "credentials",
        "key",
        "password",
        "secret",
        "token",
    }
)


class MCPTrustScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    min_trust_score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum normalized trust score (0-1) required to expose or route to a server",
        ),
    ] = DEFAULT_MCP_TRUST_SCORING_MIN_SCORE
    cache_ttl_seconds: Annotated[
        int,
        Field(
            ge=1,
            description="Seconds to cache Dominion Observatory responses per server URL",
        ),
    ] = DEFAULT_MCP_TRUST_SCORING_CACHE_TTL
    api_url: Annotated[
        str,
        Field(description="Dominion Observatory trust endpoint base URL"),
    ] = DEFAULT_MCP_TRUST_SCORING_API_URL
    fail_open: Annotated[
        bool,
        Field(
            description="When True, keep servers reachable if DO is down, rate-limited, or has no score yet"
        ),
    ] = True
    api_key: Annotated[
        str | None,
        Field(description="Optional API key for paid Dominion Observatory tiers"),
    ] = None


class DominionObservatoryTrustResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_url: str  # any-ok: required pydantic field, mypy exports the field symbol as Any but the runtime type is str
    found: bool  # any-ok: required pydantic field, mypy exports the field symbol as Any but the runtime type is bool
    trust_score_normalized: Annotated[
        float | None,
        Field(description="Trust score normalized to 0-1"),
    ] = None
    trust_score_raw: Annotated[
        float | None,
        Field(description="Raw trust score from Dominion Observatory (0-100 scale)"),
    ] = None
    lookup_status: Literal["ok", "skipped", "error"] = "ok"
    message: str | None = None

    def meets_min_trust(self, min_trust_score: float) -> bool:
        if self.lookup_status == "skipped":
            return True
        if self.trust_score_normalized is None:
            return False
        return self.trust_score_normalized >= min_trust_score

    def sort_key(self) -> tuple[int, float]:
        if self.lookup_status == "skipped":
            return (1, 0.0)
        if self.trust_score_normalized is None:
            return (2, 0.0)
        return (0, -self.trust_score_normalized)


class _DominionObservatoryTrustApiResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    found: bool = False
    trust_score: float | None = None
    message: str | None = None


class _SharedTrustCache(Protocol):
    async def async_get_cache(
        self,
        key: str,
        local_only: bool = False,
        **kwargs: object,
    ) -> object | None: ...

    async def async_set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,
    ) -> None: ...


class _LocalTrustCache(Protocol):
    def get_cache(self, key: str, **kwargs: object) -> object | None: ...

    def set_cache(self, key: str, value: object, **kwargs: object) -> None: ...


class _HttpClient(Protocol):
    async def get(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response: ...


@dataclass(frozen=True, slots=True)
class _ScoredServer:
    server: MCPServer  # any-ok: required dataclass field, mypy exports the field symbol as Any but the runtime type is MCPServer
    trust: DominionObservatoryTrustResult  # any-ok: required dataclass field, mypy exports the field symbol as Any but the runtime type is typed


def _normalize_server_url(server_url: str) -> str:
    return server_url.rstrip("/")


def sanitize_url_for_trust_lookup(server_url: str) -> str:
    parsed = urlparse(server_url)
    hostname = parsed.hostname
    if not hostname:
        return _normalize_server_url(server_url)

    port_suffix = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port_suffix}"
    filtered_query = tuple(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _SENSITIVE_QUERY_PARAM_NAMES
    )
    sanitized = parsed._replace(
        netloc=netloc,
        query=urlencode(filtered_query),
        fragment="",
    )
    return urlunparse(sanitized)


def assert_requested_server_passes_trust_filter(
    *,
    filtered_servers: Sequence[MCPServer],
    server_id: str,
) -> None:
    from fastapi import HTTPException

    trusted_server_ids = {server.server_id for server in filtered_servers}
    if server_id not in trusted_server_ids:
        raise HTTPException(
            status_code=403,
            detail=TRUST_SCORE_THRESHOLD_DENIED,
        )


def _cache_key_for_url(server_url: str) -> str:
    digest = hashlib.sha256(
        _normalize_server_url(server_url).encode("utf-8")
    ).hexdigest()[:32]
    return f"mcp:do_trust:{digest}"


def normalize_trust_score(raw_score: object) -> float | None:
    if raw_score is None:
        return None
    if not isinstance(raw_score, (int, float)):
        return None
    score = float(raw_score)
    if score < 0:
        return None
    return min(score / 100.0, 1.0)


def parse_trust_response(
    *,
    server_url: str,
    payload: object,
) -> DominionObservatoryTrustResult:
    try:
        parsed = TypeAdapter(_DominionObservatoryTrustApiResponse).validate_python(
            payload
        )
    except ValidationError:
        return DominionObservatoryTrustResult(
            server_url=server_url,
            found=False,
            lookup_status="error",
            message="Invalid Dominion Observatory response shape",
        )

    return DominionObservatoryTrustResult(
        server_url=server_url,
        found=parsed.found,
        trust_score_normalized=normalize_trust_score(parsed.trust_score),
        trust_score_raw=parsed.trust_score,
        lookup_status="ok",
        message=parsed.message,
    )


def _should_include_server(
    *,
    trust: DominionObservatoryTrustResult,
    config: MCPTrustScoringConfig,
) -> bool:
    if trust.lookup_status == "skipped":
        return True
    if trust.lookup_status == "error":
        return config.fail_open
    if not trust.found or trust.trust_score_normalized is None:
        return config.fail_open
    return trust.meets_min_trust(config.min_trust_score)


class MCPTrustScoringClient:
    def __init__(
        self,
        config: MCPTrustScoringConfig,
        *,
        local_cache: InMemoryCache | None = None,
        cache_getter: Callable[[], _SharedTrustCache] | None = None,
        http_client_getter: Callable[[], _HttpClient] | None = None,
    ) -> None:
        self._config = config
        self._local_cache: _LocalTrustCache = cast(
            _LocalTrustCache,
            local_cache
            or InMemoryCache(
                max_size_in_memory=MCP_TRUST_SCORING_CACHE_MAX_SIZE,
                default_ttl=config.cache_ttl_seconds,
            ),
        )
        self._cache_getter = cache_getter or _default_dual_cache_getter
        self._http_client_getter = http_client_getter or _default_http_client_getter
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def config(self) -> MCPTrustScoringConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @classmethod
    def from_config_dict(
        cls,
        config: dict[str, object] | None,
    ) -> MCPTrustScoringClient | None:
        if not config:
            return None
        parsed = MCPTrustScoringConfig.model_validate(config)
        if not parsed.enabled:
            return None
        return cls(parsed)

    def _lock_for_url(self, server_url: str) -> asyncio.Lock:
        return self._locks.setdefault(_cache_key_for_url(server_url), asyncio.Lock())

    async def _read_cached_result(
        self, server_url: str
    ) -> DominionObservatoryTrustResult | None:
        cache_key = _cache_key_for_url(server_url)
        local_cached = self._local_cache.get_cache(cache_key)
        if isinstance(local_cached, str):
            return DominionObservatoryTrustResult.model_validate_json(local_cached)

        try:
            shared_cache = self._cache_getter()
            shared = await shared_cache.async_get_cache(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is narrowed below
                cache_key
            )
        except Exception:
            shared = None

        if isinstance(shared, str):
            self._local_cache.set_cache(
                cache_key,
                shared,
                ttl=self._config.cache_ttl_seconds,
            )
            return DominionObservatoryTrustResult.model_validate_json(shared)
        return None

    async def _write_cached_result(
        self,
        server_url: str,
        result: DominionObservatoryTrustResult,
    ) -> None:
        cache_key = _cache_key_for_url(server_url)
        payload_json = result.model_dump_json()
        self._local_cache.set_cache(
            cache_key,
            payload_json,
            ttl=self._config.cache_ttl_seconds,
        )
        try:
            shared_cache = self._cache_getter()
            await shared_cache.async_set_cache(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the stored value is a typed JSON string
                cache_key,
                payload_json,
                ttl=self._config.cache_ttl_seconds,
            )
        except Exception as exc:
            verbose_logger.debug(
                "Failed to write MCP trust score to shared cache for %s: %s",
                server_url,
                exc,
            )

    async def get_trust_score(self, server_url: str) -> DominionObservatoryTrustResult:
        sanitized_url = sanitize_url_for_trust_lookup(server_url)
        normalized_url = _normalize_server_url(sanitized_url)
        cached = await self._read_cached_result(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
            normalized_url
        )
        if cached is not None:
            return cached

        async with self._lock_for_url(normalized_url):
            cached = await self._read_cached_result(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
                normalized_url
            )
            if cached is not None:
                return cached

            result = await self._fetch_trust_score(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
                normalized_url
            )
            await self._write_cached_result(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and returns None
                normalized_url, result
            )
            return result

    async def _fetch_trust_score(
        self, server_url: str
    ) -> DominionObservatoryTrustResult:
        headers: dict[str, str] = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        try:
            client = self._http_client_getter()
            response = await client.get(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is an httpx.Response
                self._config.api_url,
                params={"url": server_url},
                headers=headers or None,
            )
            response.raise_for_status()
            parsed_api = TypeAdapter(
                _DominionObservatoryTrustApiResponse
            ).validate_json(response.text)
        except Exception as exc:
            verbose_logger.debug(
                "Dominion Observatory trust lookup failed for %s: %s",
                server_url,
                exc,
            )
            return DominionObservatoryTrustResult(
                server_url=server_url,
                found=False,
                lookup_status="error",
                message=str(exc),
            )

        return DominionObservatoryTrustResult(
            server_url=server_url,
            found=parsed_api.found,
            trust_score_normalized=normalize_trust_score(parsed_api.trust_score),
            trust_score_raw=parsed_api.trust_score,
            lookup_status="ok",
            message=parsed_api.message,
        )

    async def get_trust_score_for_server(
        self,
        server: MCPServer,
    ) -> DominionObservatoryTrustResult:
        if not server.url:
            return DominionObservatoryTrustResult(
                server_url=server.server_id,
                found=False,
                lookup_status="skipped",
                message="Server has no HTTP URL to score",
            )
        return await self.get_trust_score(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
            server.url
        )

    async def score_servers(
        self,
        servers: Sequence[MCPServer],
    ) -> tuple[_ScoredServer, ...]:
        if not servers:
            return ()

        trust_results = await asyncio.gather(  # any-ok: awaited coroutine, gather's element Any is a typing artifact and each result is a typed DominionObservatoryTrustResult
            *(  # any-ok: generator of coroutines, the Coroutine Send/Recv Any is a typing artifact
                self.get_trust_score_for_server(  # any-ok: coroutine expression, Send/Recv Any is a typing artifact and the result is fully typed
                    server
                )
                for server in servers
            )
        )
        return tuple(
            _ScoredServer(server=server, trust=trust)
            for server, trust in zip(servers, trust_results, strict=True)
        )

    async def filter_servers_by_trust(
        self,
        servers: Sequence[MCPServer],
    ) -> tuple[MCPServer, ...]:
        if not self.enabled:
            return tuple(servers)

        scored = await self.score_servers(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
            servers
        )
        return tuple(
            entry.server
            for entry in scored
            if _should_include_server(trust=entry.trust, config=self._config)
        )

    async def rank_servers_by_trust(
        self,
        servers: Sequence[MCPServer],
    ) -> tuple[MCPServer, ...]:
        if not self.enabled:
            return tuple(servers)

        scored = await self.score_servers(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
            servers
        )
        ordered = sorted(scored, key=lambda entry: entry.trust.sort_key())
        return tuple(entry.server for entry in ordered)


def _default_http_client_getter() -> _HttpClient:
    from litellm.llms.custom_httpx import http_handler  # noqa: PLC0415

    return cast(
        _HttpClient,
        http_handler.get_async_httpx_client(  # pyright: ignore[reportUnknownMemberType]
            llm_provider=httpxSpecialProvider.MCP,
        ),
    )


def _default_dual_cache_getter() -> _SharedTrustCache:
    from litellm.proxy.proxy_server import user_api_key_cache  # noqa: PLC0415

    return cast(_SharedTrustCache, user_api_key_cache)


_global_mcp_trust_scoring_client: MCPTrustScoringClient | None = None


def get_mcp_trust_scoring_client() -> MCPTrustScoringClient | None:
    return _global_mcp_trust_scoring_client


def _resolve_trust_scoring_config_values(
    config: dict[str, object] | None,
) -> dict[str, object] | None:
    if not config:
        return None

    resolved = dict(config)
    api_key = resolved.get("api_key")
    if isinstance(api_key, str) and api_key.startswith("os.environ/"):
        from litellm.secret_managers.main import get_secret_str  # noqa: PLC0415

        resolved["api_key"] = get_secret_str(api_key)
    return resolved


def initialize_mcp_trust_scoring_from_config(
    config: dict[str, object] | None,
) -> MCPTrustScoringClient | None:
    global _global_mcp_trust_scoring_client

    client = MCPTrustScoringClient.from_config_dict(
        _resolve_trust_scoring_config_values(config)
    )
    _global_mcp_trust_scoring_client = client

    if client is None:
        verbose_logger.debug("MCP trust scoring disabled")
    else:
        verbose_logger.info(
            "MCP trust scoring enabled (min_trust_score=%s, cache_ttl_seconds=%s, fail_open=%s)",
            client.config.min_trust_score,
            client.config.cache_ttl_seconds,
            client.config.fail_open,
        )

    return client


async def filter_mcp_servers_by_trust(
    servers: Sequence[MCPServer],
    *,
    client: MCPTrustScoringClient | None = None,
) -> tuple[MCPServer, ...]:
    active_client = client or get_mcp_trust_scoring_client()
    if active_client is None:
        return tuple(servers)
    return await active_client.filter_servers_by_trust(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
        servers
    )


async def rank_mcp_servers_by_trust(
    servers: Sequence[MCPServer],
    *,
    client: MCPTrustScoringClient | None = None,
) -> tuple[MCPServer, ...]:
    active_client = client or get_mcp_trust_scoring_client()
    if active_client is None:
        return tuple(servers)
    return await active_client.rank_servers_by_trust(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
        servers
    )


async def apply_trust_filter_to_allowed_mcp_servers(
    servers: Sequence[MCPServer],
    *,
    client: MCPTrustScoringClient | None = None,
) -> tuple[MCPServer, ...]:
    return await filter_mcp_servers_by_trust(  # any-ok: awaited coroutine, Send/Recv Any is a typing artifact and the result is fully typed
        servers, client=client
    )
