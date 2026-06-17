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
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, Sequence, Tuple, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

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


class MCPTrustScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    min_trust_score: float = Field(
        default=DEFAULT_MCP_TRUST_SCORING_MIN_SCORE,
        ge=0.0,
        le=1.0,
        description="Minimum normalized trust score (0-1) required to expose or route to a server",
    )
    cache_ttl_seconds: int = Field(
        default=DEFAULT_MCP_TRUST_SCORING_CACHE_TTL,
        ge=1,
        description="Seconds to cache Dominion Observatory responses per server URL",
    )
    api_url: str = Field(
        default=DEFAULT_MCP_TRUST_SCORING_API_URL,
        description="Dominion Observatory trust endpoint base URL",
    )
    fail_open: bool = Field(
        default=True,
        description="When True, keep servers reachable if DO is down, rate-limited, or has no score yet",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for paid Dominion Observatory tiers",
    )


class DominionObservatoryTrustResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_url: str
    found: bool
    trust_score_normalized: Optional[float] = Field(
        default=None,
        description="Trust score normalized to 0-1",
    )
    trust_score_raw: Optional[float] = Field(
        default=None,
        description="Raw trust score from Dominion Observatory (0-100 scale)",
    )
    lookup_status: Literal["ok", "skipped", "error"] = "ok"
    message: Optional[str] = None

    def meets_min_trust(self, min_trust_score: float) -> bool:
        if self.lookup_status == "skipped":
            return True
        if self.trust_score_normalized is None:
            return False
        return self.trust_score_normalized >= min_trust_score

    def sort_key(self) -> Tuple[int, float]:
        if self.lookup_status == "skipped":
            return (1, 0.0)
        if self.trust_score_normalized is None:
            return (2, 0.0)
        return (0, -self.trust_score_normalized)


class _TrustCacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload_json: str


class _DominionObservatoryTrustApiResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    found: bool = False
    trust_score: Optional[float] = None
    message: Optional[str] = None


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
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response: ...


@dataclass(frozen=True, slots=True)
class _ScoredServer:
    server: MCPServer
    trust: DominionObservatoryTrustResult


def _normalize_server_url(server_url: str) -> str:
    return server_url.rstrip("/")


def _cache_key_for_url(server_url: str) -> str:
    digest = hashlib.sha256(_normalize_server_url(server_url).encode("utf-8")).hexdigest()[
        :32
    ]
    return f"mcp:do_trust:{digest}"


def normalize_trust_score(raw_score: object) -> Optional[float]:
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
    if not isinstance(payload, dict):
        return DominionObservatoryTrustResult(
            server_url=server_url,
            found=False,
            lookup_status="error",
            message="Invalid Dominion Observatory response shape",
        )

    parsed = TypeAdapter(_DominionObservatoryTrustApiResponse).validate_python(payload)

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
        local_cache: Optional[InMemoryCache] = None,
        cache_getter: Optional[Callable[[], _SharedTrustCache]] = None,
        http_client_getter: Optional[Callable[[], _HttpClient]] = None,
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
        config: Optional[dict[str, object]],
    ) -> Optional[MCPTrustScoringClient]:
        if not config:
            return None
        parsed = MCPTrustScoringConfig.model_validate(config)
        if not parsed.enabled:
            return None
        return cls(parsed)

    def _lock_for_url(self, server_url: str) -> asyncio.Lock:
        return self._locks.setdefault(_cache_key_for_url(server_url), asyncio.Lock())

    async def _read_cached_result(self, server_url: str) -> Optional[DominionObservatoryTrustResult]:
        cache_key = _cache_key_for_url(server_url)
        local_cached = self._local_cache.get_cache(cache_key)
        if isinstance(local_cached, str):
            return DominionObservatoryTrustResult.model_validate_json(local_cached)

        try:
            shared_cache = self._cache_getter()
            shared = await shared_cache.async_get_cache(cache_key)
        except Exception:
            shared = None

        if isinstance(shared, str):
            self._local_cache.set_cache(
                cache_key,
                shared,
                ttl=self._config.cache_ttl_seconds,
            )
            return DominionObservatoryTrustResult.model_validate_json(shared)
        if isinstance(shared, dict):
            entry = _TrustCacheEntry.model_validate(shared)
            self._local_cache.set_cache(
                cache_key,
                entry.payload_json,
                ttl=self._config.cache_ttl_seconds,
            )
            return DominionObservatoryTrustResult.model_validate_json(entry.payload_json)
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
            await shared_cache.async_set_cache(
                cache_key,
                _TrustCacheEntry(payload_json=payload_json).model_dump(),
                ttl=self._config.cache_ttl_seconds,
            )
        except Exception as exc:
            verbose_logger.debug(
                "Failed to write MCP trust score to shared cache for %s: %s",
                server_url,
                exc,
            )

    async def get_trust_score(self, server_url: str) -> DominionObservatoryTrustResult:
        normalized_url = _normalize_server_url(server_url)
        cached = await self._read_cached_result(normalized_url)
        if cached is not None:
            return cached

        async with self._lock_for_url(normalized_url):
            cached = await self._read_cached_result(normalized_url)
            if cached is not None:
                return cached

            result = await self._fetch_trust_score(normalized_url)
            await self._write_cached_result(normalized_url, result)
            return result

    async def _fetch_trust_score(self, server_url: str) -> DominionObservatoryTrustResult:
        headers: dict[str, str] = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        try:
            client = self._http_client_getter()
            response = await client.get(
                self._config.api_url,
                params={"url": server_url},
                headers=headers or None,
            )
            response.raise_for_status()
            parsed_api = TypeAdapter(_DominionObservatoryTrustApiResponse).validate_json(
                response.text
            )
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
        return await self.get_trust_score(server.url)

    async def score_servers(
        self,
        servers: Sequence[MCPServer],
    ) -> Tuple[_ScoredServer, ...]:
        if not servers:
            return ()

        trust_results = await asyncio.gather(
            *(self.get_trust_score_for_server(server) for server in servers)
        )
        return tuple(
            _ScoredServer(server=server, trust=trust)
            for server, trust in zip(servers, trust_results, strict=True)
        )

    async def filter_servers_by_trust(
        self,
        servers: Sequence[MCPServer],
    ) -> Tuple[MCPServer, ...]:
        if not self.enabled:
            return tuple(servers)

        scored = await self.score_servers(servers)
        return tuple(
            entry.server
            for entry in scored
            if _should_include_server(trust=entry.trust, config=self._config)
        )

    async def rank_servers_by_trust(
        self,
        servers: Sequence[MCPServer],
    ) -> Tuple[MCPServer, ...]:
        if not self.enabled:
            return tuple(servers)

        scored = await self.score_servers(servers)
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


_global_mcp_trust_scoring_client: Optional[MCPTrustScoringClient] = None


def get_mcp_trust_scoring_client() -> Optional[MCPTrustScoringClient]:
    return _global_mcp_trust_scoring_client


def _resolve_trust_scoring_config_values(
    config: Optional[dict[str, object]],
) -> Optional[dict[str, object]]:
    if not config:
        return None

    resolved = dict(config)
    api_key = resolved.get("api_key")
    if isinstance(api_key, str) and api_key.startswith("os.environ/"):
        from litellm.secret_managers.main import get_secret  # noqa: PLC0415

        resolved["api_key"] = get_secret(api_key)
    return resolved


def initialize_mcp_trust_scoring_from_config(
    config: Optional[dict[str, object]],
) -> Optional[MCPTrustScoringClient]:
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
    client: Optional[MCPTrustScoringClient] = None,
) -> Tuple[MCPServer, ...]:
    active_client = client or get_mcp_trust_scoring_client()
    if active_client is None:
        return tuple(servers)
    return await active_client.filter_servers_by_trust(servers)


async def rank_mcp_servers_by_trust(
    servers: Sequence[MCPServer],
    *,
    client: Optional[MCPTrustScoringClient] = None,
) -> Tuple[MCPServer, ...]:
    active_client = client or get_mcp_trust_scoring_client()
    if active_client is None:
        return tuple(servers)
    return await active_client.rank_servers_by_trust(servers)


async def apply_trust_filter_to_allowed_mcp_servers(
    servers: Sequence[MCPServer],
    *,
    client: Optional[MCPTrustScoringClient] = None,
) -> Tuple[MCPServer, ...]:
    return await filter_mcp_servers_by_trust(servers, client=client)
