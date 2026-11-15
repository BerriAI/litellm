"""Codex's own catalog for the Codex release asking for `/v1/models`, fetched from that release's tag on GitHub.

Codex fetches its catalog from chatgpt.com, which needs a ChatGPT login the proxy never sees, so the proxy reads
the same two files at the release's public tag instead, and falls back to the copy bundled with LiteLLM when the
fetch fails, the release has no tag (dev builds report 0.0.0) or ``LITELLM_OFFLINE_MODE`` is on.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Final, Protocol, TypeVar

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared factory has untyped optional parameters
)
from litellm.proxy.common_utils.codex_model_catalog import (
    CODEX_BUNDLED_MODELS_PATH,
    CodexStockCatalog,
    CodexStockModel,
    bundled_codex_models,
    codex_base_instructions,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

CODEX_RELEASE_FILE_URL: Final = (
    "https://raw.githubusercontent.com/openai/codex/rust-v{version}/codex-rs/models-manager/{file}"
)
CODEX_FETCH_TIMEOUT_SECONDS: Final = 2.0
CODEX_FETCH_FAILURE_TTL_SECONDS: Final = 3600.0
CODEX_CACHED_VERSIONS: Final = 8
CODEX_REMEMBERED_FAILURES: Final = 256
CODEX_FETCHES_PER_MINUTE: Final = 6

_RELEASE_VERSION: Final = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.]+)?")


@dataclass(frozen=True, slots=True)
class CodexStock:
    models: Mapping[str, CodexStockModel]
    instructions: str


_Value: Final = TypeVar("_Value")


class CodexReleaseClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> Awaitable[httpx.Response]: ...


@cache
def bundled_codex_stock() -> CodexStock:
    return CodexStock(models=bundled_codex_models(), instructions=codex_base_instructions())


async def _release_file(client: CodexReleaseClient, version: str, file: str, timeout: float) -> str:
    response: Final = await client.get(CODEX_RELEASE_FILE_URL.format(version=version, file=file), timeout=timeout)
    return response.raise_for_status().content.decode("utf-8")


async def _release_stock(client: CodexReleaseClient, version: str, timeout: float) -> CodexStock:
    models_json, instructions = await asyncio.wait_for(
        asyncio.gather(
            _release_file(client, version, "models.json", timeout),
            _release_file(client, version, "prompt.md", timeout),
        ),
        timeout,
    )
    catalog: Final = CodexStockCatalog.model_validate_json(models_json)
    return CodexStock(
        models=MappingProxyType({model.slug: model for model in catalog.models}), instructions=instructions
    )


class CodexCatalogSource:
    """Codex's catalog per release: the one fetched from GitHub, else the bundled one.

    A fetched catalog is kept for good, a failed fetch is remembered for `failure_ttl` seconds so a blocked egress
    costs one timeout per worker per version per hour. Only the `max_versions` most recently fetched catalogs and
    the `max_failures` most recent failures are kept, apart, so versions that fail never push out ones that were
    fetched.
    Requests for a version already being fetched wait on that fetch, and a worker starts at most
    `fetches_per_minute` fetches, serving the bundled catalog past that, so a caller cycling made-up versions
    cannot turn every request into GitHub requests."""

    def __init__(
        self,
        client: Callable[[], CodexReleaseClient],
        *,
        offline: Callable[[], bool] = is_offline_mode,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = CODEX_FETCH_TIMEOUT_SECONDS,
        failure_ttl: float = CODEX_FETCH_FAILURE_TTL_SECONDS,
        max_versions: int = CODEX_CACHED_VERSIONS,
        max_failures: int = CODEX_REMEMBERED_FAILURES,
        fetches_per_minute: int = CODEX_FETCHES_PER_MINUTE,
    ) -> None:
        self._client: Final = client
        self._offline: Final = offline
        self._clock: Final = clock
        self._timeout: Final = timeout
        self._failure_ttl: Final = failure_ttl
        self._max_versions: Final = max_versions
        self._max_failures: Final = max_failures
        self._fetches_per_minute: Final = fetches_per_minute
        self._fetched: Mapping[str, CodexStock] = MappingProxyType({})
        self._failed_until: Mapping[str, float] = MappingProxyType({})
        self._in_flight: Mapping[str, asyncio.Future[CodexStock]] = MappingProxyType({})
        self._fetch_starts: tuple[float, ...] = ()

    async def stock(self, client_version: str) -> CodexStock:
        if self._offline() or _RELEASE_VERSION.fullmatch(client_version) is None:
            return bundled_codex_stock()
        known: Final = self._fetched.get(client_version)
        if known is not None:
            return known
        if self._clock() < self._failed_until.get(client_version, float("-inf")):
            return bundled_codex_stock()
        pending: Final = self._in_flight.get(client_version)
        if pending is not None:
            return await asyncio.shield(pending)
        if not self._take_fetch_slot():
            return bundled_codex_stock()
        started: Final = asyncio.ensure_future(self._fetch_and_store(client_version))
        self._in_flight = MappingProxyType({**self._in_flight, client_version: started})
        return await asyncio.shield(started)

    def _take_fetch_slot(self) -> bool:
        now: Final = self._clock()
        recent: Final = tuple(start for start in self._fetch_starts if now - start < 60)
        allowed: Final = len(recent) < self._fetches_per_minute
        self._fetch_starts = (*recent, now) if allowed else recent
        return allowed

    async def _fetch_and_store(self, version: str) -> CodexStock:
        try:
            fetched: Final = await self._fetch(version)
        finally:
            self._in_flight = MappingProxyType({key: task for key, task in self._in_flight.items() if key != version})
        if fetched is None:
            self._failed_until = _latest(
                self._failed_until, version, self._clock() + self._failure_ttl, self._max_failures
            )
            return bundled_codex_stock()
        self._fetched = _latest(self._fetched, version, fetched, self._max_versions)
        return fetched

    async def _fetch(self, version: str) -> CodexStock | None:
        try:
            return await _release_stock(self._client(), version, self._timeout)
        except Exception as e:  # noqa: BLE001  # a failed best-effort fetch must never fail the model listing
            verbose_proxy_logger.warning(
                "Codex %s's model catalog could not be fetched from GitHub (%r), serving the catalog bundled "
                "with LiteLLM (%s) for that version for the next %d seconds",
                version,
                e,
                CODEX_BUNDLED_MODELS_PATH.name,
                self._failure_ttl,
            )
            return None


def _latest(stored: Mapping[str, _Value], version: str, value: _Value, limit: int) -> Mapping[str, _Value]:
    others: Final = tuple((key, kept) for key, kept in stored.items() if key != version)
    return MappingProxyType(dict((*others[max(0, len(others) - limit + 1) :], (version, value))))


def _default_client() -> CodexReleaseClient:
    return get_async_httpx_client(llm_provider=httpxSpecialProvider.CodexModelCatalog)


codex_catalog_source: Final = CodexCatalogSource(_default_client)
