"""Codex's own model catalog for the Codex release that is asking.

Codex pointed at ``/v1/models`` as its ``model_catalog_url`` replaces its bundled catalog with ours, so a
model missing from the stock entries we serve runs on Codex's fallback metadata (no apply_patch tool, no
parallel tool calls, a generic prompt). openai/codex publishes every release's entries at
``https://raw.githubusercontent.com/openai/codex/rust-v<version>/codex-rs/models-manager/models.json``
(Apache-2.0) with its ``prompt.md`` beside them, so the pair for the exact ``client_version`` asking is
fetched once and kept for the life of the process, since a release tag never changes. A version without a
tag (before 0.120, or not released yet), a fetch that fails, a body that does not parse, a request the
fetch outlasts by ``CODEX_CATALOG_RESPONSE_DEADLINE_SECONDS`` (Codex gives up on its catalog after five),
``LITELLM_OFFLINE_MODE`` and ``LITELLM_LOCAL_CODEX_CATALOG=True`` all serve the vendored pair instead. A
failed fetch is tried again in the background after ``CODEX_CATALOG_RETRY_SECONDS`` while its vendored pair
keeps answering, and at most ``CODEX_CATALOG_CACHED_VERSIONS`` fetches start per retry window, so made-up
versions cannot turn requests into outbound calls. ``LITELLM_CODEX_CATALOG_BASE_URL`` points the fetch at a
mirror of the openai/codex raw tree, and an empty ``LITELLM_CODEX_CATALOG_OPENAI_URL`` keeps the OpenAI key
home by skipping OpenAI's catalog.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

import httpx
from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.llms.custom_httpx.http_handler import HTTPResponseLimitError
from litellm.proxy.common_utils.codex_model_catalog import (
    CODEX_BUNDLED_VERSION,
    CodexStock,
    CodexStockCatalog,
    bundled_codex_models,
    codex_base_instructions,
)

CODEX_CATALOG_OPENAI_URL: Final = "https://chatgpt.com/backend-api/codex/models"
CODEX_CATALOG_BASE_URL: Final = "https://raw.githubusercontent.com/openai/codex"
CODEX_CATALOG_RETRY_SECONDS: Final = 300.0
CODEX_CATALOG_FETCH_TIMEOUT_SECONDS: Final = 10.0
CODEX_CATALOG_RESPONSE_DEADLINE_SECONDS: Final = 3.0
CODEX_CATALOG_MAX_BYTES: Final = 16 * 1024 * 1024
CODEX_CATALOG_CACHED_VERSIONS: Final = 16
_RELEASE_VERSION: Final = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?$")
_MODELS_PATH: Final = "codex-rs/models-manager/models.json"
_PROMPT_PATH: Final = "codex-rs/models-manager/prompt.md"

_CacheSlot: TypeAlias = tuple[str, str | None]


class _AsyncGetClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,  # mutable-ok: httpx's own get signature
        timeout: float | None = None,
        max_response_bytes: int | None = None,
    ) -> Awaitable[httpx.Response]: ...


@dataclass(frozen=True, slots=True)
class _Cached:
    stock: CodexStock
    retry_at: float | None


@dataclass(frozen=True, slots=True)
class _FetchFailed:
    reason: str


def _default_client() -> _AsyncGetClient:
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # its `params: dict` is untyped
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider

    return get_async_httpx_client(llm_provider=httpxSpecialProvider.CodexModelCatalog)


def vendored_codex_stock() -> CodexStock:
    """The pair vendored with LiteLLM, served when the requesting release's own cannot be."""
    return CodexStock(
        models=bundled_codex_models(),
        instructions=codex_base_instructions(),
        source=f"vendored {CODEX_BUNDLED_VERSION}",
    )


def uses_vendored_codex_catalog() -> bool:
    return is_offline_mode() or os.getenv("LITELLM_LOCAL_CODEX_CATALOG", "").strip().lower() == "true"


def codex_release_tag(client_version: str | None) -> str | None:
    """The openai/codex tag the Codex sending `client_version` was built from, None when the value is no
    release version (Codex sends its crate version, `0.166.0` or `0.167.0-alpha.4`)."""
    if client_version is None or _RELEASE_VERSION.match(client_version) is None:
        return None
    return f"rust-v{client_version}"


class CodexCatalogSource:
    """Resolves the stock catalog for a `client_version`, one fetch per release tag, concurrent requests
    for the same tag sharing that fetch. `client` is the HTTP client the fetch goes through (LiteLLM's
    shared async client when None), `clock` the monotonic clock that times failed-fetch retries."""

    def __init__(
        self,
        *,
        client: _AsyncGetClient | None = None,
        base_url: str | None = None,
        openai_url: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        retry_seconds: float = CODEX_CATALOG_RETRY_SECONDS,
        timeout: float = CODEX_CATALOG_FETCH_TIMEOUT_SECONDS,
        response_deadline: float = CODEX_CATALOG_RESPONSE_DEADLINE_SECONDS,
        cached_versions: int = CODEX_CATALOG_CACHED_VERSIONS,
    ) -> None:
        self._client = client
        self._base_url = (
            base_url if base_url is not None else os.getenv("LITELLM_CODEX_CATALOG_BASE_URL", CODEX_CATALOG_BASE_URL)
        ).rstrip("/")
        self._openai_url = (
            openai_url
            if openai_url is not None
            else os.getenv("LITELLM_CODEX_CATALOG_OPENAI_URL", CODEX_CATALOG_OPENAI_URL)
        )
        self._clock = clock
        self._retry_seconds = retry_seconds
        self._timeout = timeout
        self._response_deadline = response_deadline
        self._cached_versions = cached_versions
        self._cache: OrderedDict[_CacheSlot, _Cached] = OrderedDict()  # mutable-ok: filled as releases are asked for
        self._pending: dict[_CacheSlot, asyncio.Task[CodexStock]] = {}  # mutable-ok: the fetches in flight
        self._fetch_starts: tuple[float, ...] = ()

    async def stock(self, client_version: str | None, *, openai_api_key: str | None = None) -> CodexStock:
        """The catalog for ``client_version``: OpenAI's live one when an OpenAI key is at hand,
        else the release's models.json on GitHub, else the bundled pair."""
        tag: Final = codex_release_tag(client_version)
        if tag is None or client_version is None or uses_vendored_codex_catalog():
            return vendored_codex_stock()
        slot: Final = (tag, openai_api_key)
        cached: Final = self._cache.get(slot)
        if cached is not None:
            self._cache.move_to_end(slot)
            if cached.retry_at is not None and cached.retry_at <= self._clock():
                self._fetching(slot, tag, client_version, openai_api_key)
            return cached.stock
        fetch: Final = self._fetching(slot, tag, client_version, openai_api_key)
        return vendored_codex_stock() if fetch is None else await self._within_deadline(tag, fetch)

    def _fetching(
        self, slot: _CacheSlot, tag: str, version: str, openai_api_key: str | None
    ) -> asyncio.Task[CodexStock] | None:
        pending: Final = self._pending.get(slot)
        if pending is not None:
            return pending
        now: Final = self._clock()
        recent: Final = tuple(start for start in self._fetch_starts if start > now - self._retry_seconds)
        if len(recent) >= self._cached_versions:
            verbose_proxy_logger.warning(
                "Codex catalog %s not fetched, %d fetches started in the last %.0fs, serving the %s one",
                tag,
                len(recent),
                self._retry_seconds,
                vendored_codex_stock().source,
            )
            self._fetch_starts = recent
            return None
        self._fetch_starts = (*recent, now)
        task: Final = asyncio.ensure_future(self._resolve(slot, tag, version, openai_api_key))
        self._pending[slot] = task
        task.add_done_callback(lambda _: self._pending.pop(slot, None))
        return task

    async def _within_deadline(self, tag: str, fetch: asyncio.Task[CodexStock]) -> CodexStock:
        done, _ = await asyncio.wait((fetch,), timeout=self._response_deadline)
        if fetch in done:
            return fetch.result()
        verbose_proxy_logger.warning(
            "Codex catalog %s still fetching after %.0fs, serving the %s one to this request",
            tag,
            self._response_deadline,
            vendored_codex_stock().source,
        )
        return vendored_codex_stock()

    async def _resolve(self, slot: _CacheSlot, tag: str, version: str, openai_api_key: str | None) -> CodexStock:
        fetched: Final = await self._fetch(tag, version, openai_api_key)
        if isinstance(fetched, _FetchFailed):
            verbose_proxy_logger.warning(
                "Codex catalog %s not fetched (%s), serving the %s one for the next %.0fs",
                tag,
                fetched.reason,
                vendored_codex_stock().source,
                self._retry_seconds,
            )
        else:
            verbose_proxy_logger.info(
                "Codex catalog %s fetched from %s, %d models", tag, fetched.source, len(fetched.models)
            )
        entry: Final = (
            _Cached(stock=vendored_codex_stock(), retry_at=self._clock() + self._retry_seconds)
            if isinstance(fetched, _FetchFailed)
            else _Cached(stock=fetched, retry_at=None)
        )
        self._cache[slot] = entry
        self._cache.move_to_end(slot)
        while len(self._cache) > self._cached_versions:
            self._cache.popitem(last=False)
        return entry.stock

    async def _fetch(self, tag: str, version: str, openai_api_key: str | None) -> CodexStock | _FetchFailed:
        if openai_api_key and self._openai_url:
            fetched: Final = await self._fetch_openai(version, openai_api_key)
            if not isinstance(fetched, _FetchFailed):
                return fetched
            verbose_proxy_logger.info(
                "Codex catalog %s not fetched from OpenAI (%s), trying GitHub", tag, fetched.reason
            )
        return await self._fetch_github(tag)

    async def _fetch_openai(self, version: str, openai_api_key: str) -> CodexStock | _FetchFailed:
        url: Final = f"{self._openai_url}?client_version={version}"
        body: Final = await self._get(url, headers={"Authorization": f"Bearer {openai_api_key}"})
        if isinstance(body, _FetchFailed):
            return body
        try:
            catalog: Final = CodexStockCatalog.model_validate_json(body)
        except ValidationError as error:
            return _FetchFailed(reason=f"{url} body does not parse: {type(error).__name__}")
        if not catalog.models:
            return _FetchFailed(reason=f"{url} catalog is empty")
        return CodexStock(
            models=MappingProxyType({model.slug: model for model in catalog.models}),
            instructions=codex_base_instructions(),
            source=f"openai {version}",
        )

    async def _fetch_github(self, tag: str) -> CodexStock | _FetchFailed:
        models, prompt = await asyncio.gather(
            self._get(f"{self._base_url}/{tag}/{_MODELS_PATH}"), self._get(f"{self._base_url}/{tag}/{_PROMPT_PATH}")
        )
        if isinstance(models, _FetchFailed):
            return models
        if isinstance(prompt, _FetchFailed):
            return prompt
        try:
            catalog: Final = CodexStockCatalog.model_validate_json(models)
            instructions: Final = prompt.decode("utf-8")
        except (ValidationError, UnicodeDecodeError) as error:
            return _FetchFailed(reason=f"{tag} body does not parse: {type(error).__name__}")
        if not catalog.models or not instructions.strip():
            return _FetchFailed(reason=f"{tag} catalog or prompt is empty")
        return CodexStock(
            models=MappingProxyType({model.slug: model for model in catalog.models}),
            instructions=instructions,
            source=f"github {tag}",
        )

    async def _get(
        self,
        url: str,
        headers: dict[str, str] | None = None,  # mutable-ok: httpx's own get signature
    ) -> bytes | _FetchFailed:
        client: Final = self._client if self._client is not None else _default_client()
        try:
            response: Final = await client.get(
                url, headers=headers, timeout=self._timeout, max_response_bytes=CODEX_CATALOG_MAX_BYTES
            )
        except (httpx.HTTPError, HTTPResponseLimitError) as error:
            return _FetchFailed(reason=f"{type(error).__name__} fetching {url}")
        if response.status_code != 200:
            return _FetchFailed(reason=f"HTTP {response.status_code} from {url}")
        if len(response.content) > CODEX_CATALOG_MAX_BYTES:
            return _FetchFailed(reason=f"{url} is over {CODEX_CATALOG_MAX_BYTES} bytes")
        return response.content


@cache
def codex_catalog_source() -> CodexCatalogSource:
    """The process-wide source `/v1/models` resolves its stock catalog through."""
    return CodexCatalogSource()
