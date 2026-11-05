"""Codex's own model catalog for the Codex release that is asking.

Codex pointed at ``/v1/models`` as its ``model_catalog_url`` replaces its bundled catalog with ours, so a
model missing from the stock entries we serve runs on Codex's fallback metadata (no apply_patch tool, no
parallel tool calls, a generic prompt). With an OpenAI key at hand the catalog for the exact
``client_version`` asking comes from OpenAI's ``CODEX_CATALOG_OPENAI_URL``, minus the ``base_instructions``
copy of each entry's ``model_messages.instructions_template`` (every tagged Codex reads the template first,
and the copy doubles a catalog that Codex before 0.166.0 drops past 1 MiB), and is fetched again after
``CODEX_CATALOG_REFRESH_SECONDS``, a failed refresh keeping it. Otherwise, or when OpenAI fails before it has
answered once, it is the release's entries openai/codex publishes at
``https://raw.githubusercontent.com/openai/codex/rust-v<version>/codex-rs/models-manager/models.json``
(Apache-2.0) with its ``prompt.md`` beside them, kept for the life of the process since a release tag never
changes, and OpenAI is asked again after ``CODEX_CATALOG_RETRY_SECONDS``. Either catalog is laid over the
vendored entries, so a model it lacks keeps the entry LiteLLM ships for it. A version without a tag (before
0.120, or not released yet), a fetch that fails, a body that does not parse, a request the fetch outlasts by
``CODEX_CATALOG_RESPONSE_DEADLINE_SECONDS`` (Codex gives up on its catalog after five),
``LITELLM_OFFLINE_MODE`` and ``LITELLM_LOCAL_CODEX_CATALOG=True`` all serve the vendored pair instead. A
failed fetch is tried again in the background after ``CODEX_CATALOG_RETRY_SECONDS`` while the last catalog
fetched for that release, else the vendored pair, keeps answering, and at most
``CODEX_CATALOG_CACHED_VERSIONS`` fetches start per retry window, so made-up versions cannot turn requests
into outbound calls. ``LITELLM_CODEX_CATALOG_BASE_URL`` points the fetch at a mirror of the openai/codex raw
tree, and an empty ``LITELLM_CODEX_CATALOG_OPENAI_URL`` keeps the OpenAI key home by skipping OpenAI's
catalog.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

import httpx
from pydantic import ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import HTTPResponseLimitError
from litellm.proxy.common_utils.codex_model_catalog import (
    CODEX_BUNDLED_VERSION,
    CodexStock,
    CodexStockCatalog,
    CodexStockModel,
    bundled_codex_models,
    codex_base_instructions,
    uses_vendored_codex_catalog,
)
from litellm.types.llms.base_model import LiteLLMBaseModel

CODEX_CATALOG_OPENAI_URL: Final = "https://chatgpt.com/backend-api/codex/models"
CODEX_CATALOG_BASE_URL: Final = "https://raw.githubusercontent.com/openai/codex"
CODEX_CATALOG_RETRY_SECONDS: Final = 300.0
CODEX_CATALOG_REFRESH_SECONDS: Final = 3600.0
CODEX_CATALOG_FETCH_TIMEOUT_SECONDS: Final = 10.0
CODEX_CATALOG_RESPONSE_DEADLINE_SECONDS: Final = 3.0
CODEX_CATALOG_MAX_BYTES: Final = 16 * 1024 * 1024
CODEX_CATALOG_CACHED_VERSIONS: Final = 16
_RELEASE_VERSION: Final = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")
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
    refetch_at: float | None
    from_openai: bool


@dataclass(frozen=True, slots=True)
class _Fetched:
    stock: CodexStock
    refetch_after: float | None
    from_openai: bool


@dataclass(frozen=True, slots=True)
class _FetchFailed:
    reason: str


class _ModelMessages(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    instructions_template: str | None = None


def _without_instructions_copy(model: CodexStockModel) -> CodexStockModel:
    """`model` without the `base_instructions` OpenAI mirrors from its `model_messages.instructions_template`,
    the way openai/codex's own models.json ships it."""
    extra: Final[Mapping[str, object]] = model.model_extra or {}
    try:
        messages: Final = _ModelMessages.model_validate(extra.get("model_messages"))
    except ValidationError:
        return model
    return model.model_copy(update={"base_instructions": ""}) if messages.instructions_template else model


def _over_vendored(models: tuple[CodexStockModel, ...]) -> Mapping[str, CodexStockModel]:
    return MappingProxyType({**bundled_codex_models(), **{model.slug: model for model in models}})


def _default_client() -> _AsyncGetClient:
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # its `params: dict` is untyped
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider

    return get_async_httpx_client(llm_provider=httpxSpecialProvider.CodexModelCatalog)


_VENDORED_SOURCE: Final = f"vendored {CODEX_BUNDLED_VERSION}"


def vendored_codex_stock() -> CodexStock:
    """The pair vendored with LiteLLM, served when the requesting release's own cannot be."""
    return CodexStock(models=bundled_codex_models(), instructions=codex_base_instructions(), source=_VENDORED_SOURCE)


def codex_release_tag(client_version: str | None) -> str | None:
    """The openai/codex tag the Codex sending `client_version` was built from, None when the value is no
    release version. Codex sends its version without a pre-release suffix, so an alpha build of 0.167.0 asks
    for `rust-v0.167.0`, which has no tag until that release ships."""
    if client_version is None or _RELEASE_VERSION.fullmatch(client_version) is None:
        return None
    return f"rust-v{client_version}"


class CodexCatalogSource:
    """Resolves the stock catalog for a `client_version`, one fetch per release tag, concurrent requests
    for the same tag sharing that fetch. `client` is the HTTP client the fetch goes through (LiteLLM's
    shared async client when None), `clock` the monotonic clock that times retries and refreshes."""

    def __init__(
        self,
        *,
        client: _AsyncGetClient | None = None,
        base_url: str | None = None,
        openai_url: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        retry_seconds: float = CODEX_CATALOG_RETRY_SECONDS,
        refresh_seconds: float = CODEX_CATALOG_REFRESH_SECONDS,
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
        self._refresh_seconds = refresh_seconds
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
            if cached.refetch_at is not None and cached.refetch_at <= self._clock():
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
        previous: Final = self._cache.get(slot)
        holds_openai: Final = previous is not None and previous.from_openai
        fetched: Final = await self._fetch(tag, version, openai_api_key, github_fallback=not holds_openai)
        now: Final = self._clock()
        entry: Final = (
            _Cached(
                stock=previous.stock if previous is not None else vendored_codex_stock(),
                refetch_at=now + self._retry_seconds,
                from_openai=holds_openai,
            )
            if isinstance(fetched, _FetchFailed)
            else _Cached(
                stock=fetched.stock,
                refetch_at=None if fetched.refetch_after is None else now + fetched.refetch_after,
                from_openai=fetched.from_openai,
            )
        )
        if isinstance(fetched, _FetchFailed):
            verbose_proxy_logger.warning(
                "Codex catalog %s not fetched (%s), serving the %s one for the next %.0fs",
                tag,
                fetched.reason,
                entry.stock.source,
                self._retry_seconds,
            )
        else:
            verbose_proxy_logger.info(
                "Codex catalog %s fetched from %s, %d models", tag, entry.stock.source, len(entry.stock.models)
            )
        self._cache[slot] = entry
        self._cache.move_to_end(slot)
        while len(self._cache) > self._cached_versions:
            del self._cache[self._eviction_victim(keep=slot)]
        return entry.stock

    def _eviction_victim(self, keep: _CacheSlot) -> _CacheSlot:
        """The least recently asked release other than ``keep`` still on the vendored pair, else the least recently
        asked one, so releases no catalog answered push each other out before a fetched release goes, while the
        release just asked keeps its retry time."""
        vendored: Final = (
            slot for slot, cached in self._cache.items() if slot != keep and cached.stock.source == _VENDORED_SOURCE
        )
        return next(vendored, next(iter(self._cache)))

    async def _fetch(
        self, tag: str, version: str, openai_api_key: str | None, *, github_fallback: bool
    ) -> _Fetched | _FetchFailed:
        if not openai_api_key or not self._openai_url:
            github: Final = await self._fetch_github(tag)
            return (
                github
                if isinstance(github, _FetchFailed)
                else _Fetched(stock=github, refetch_after=None, from_openai=False)
            )
        openai: Final = await self._fetch_openai(version, openai_api_key)
        if not isinstance(openai, _FetchFailed):
            return _Fetched(stock=openai, refetch_after=self._refresh_seconds, from_openai=True)
        if not github_fallback:
            return openai
        verbose_proxy_logger.info("Codex catalog %s not fetched from OpenAI (%s), trying GitHub", tag, openai.reason)
        fallback: Final = await self._fetch_github(tag)
        return (
            fallback
            if isinstance(fallback, _FetchFailed)
            else _Fetched(stock=fallback, refetch_after=self._retry_seconds, from_openai=False)
        )

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
            models=_over_vendored(tuple(map(_without_instructions_copy, catalog.models))),
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
            models=_over_vendored(catalog.models),
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
            response: Final = await asyncio.wait_for(
                client.get(url, headers=headers, timeout=self._timeout, max_response_bytes=CODEX_CATALOG_MAX_BYTES),
                self._timeout,
            )
        except (httpx.HTTPError, httpx.InvalidURL, HTTPResponseLimitError, ValueError, asyncio.TimeoutError) as error:
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
