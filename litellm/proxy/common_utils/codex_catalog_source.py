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
from typing import Final, Protocol

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

_RELEASE_VERSION: Final = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")


@dataclass(frozen=True, slots=True)
class CodexStock:
    models: Mapping[str, CodexStockModel]
    instructions: str


@dataclass(frozen=True, slots=True)
class _Remembered:
    stock: CodexStock | None
    expires_at: float


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
    costs one timeout per worker per version per hour, and only the `max_versions` most recently stored releases
    are kept."""

    def __init__(
        self,
        client: Callable[[], CodexReleaseClient],
        *,
        offline: Callable[[], bool] = is_offline_mode,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = CODEX_FETCH_TIMEOUT_SECONDS,
        failure_ttl: float = CODEX_FETCH_FAILURE_TTL_SECONDS,
        max_versions: int = CODEX_CACHED_VERSIONS,
    ) -> None:
        self._client: Final = client
        self._offline: Final = offline
        self._clock: Final = clock
        self._timeout: Final = timeout
        self._failure_ttl: Final = failure_ttl
        self._max_versions: Final = max_versions
        self._remembered: Mapping[str, _Remembered] = MappingProxyType({})

    async def stock(self, client_version: str) -> CodexStock:
        if self._offline() or _RELEASE_VERSION.fullmatch(client_version) is None:
            return bundled_codex_stock()
        remembered: Final = self._remembered.get(client_version)
        if remembered is not None and self._clock() < remembered.expires_at:
            return bundled_codex_stock() if remembered.stock is None else remembered.stock
        fetched: Final = await self._fetch(client_version)
        self._remember(client_version, fetched)
        return bundled_codex_stock() if fetched is None else fetched

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

    def _remember(self, version: str, stock: CodexStock | None) -> None:
        expires_at: Final = float("inf") if stock is not None else self._clock() + self._failure_ttl
        others: Final = tuple((key, value) for key, value in self._remembered.items() if key != version)
        kept: Final = others[max(0, len(others) - self._max_versions + 1) :]
        self._remembered = MappingProxyType(dict((*kept, (version, _Remembered(stock, expires_at)))))


def _default_client() -> CodexReleaseClient:
    return get_async_httpx_client(llm_provider=httpxSpecialProvider.CodexModelCatalog)


codex_catalog_source: Final = CodexCatalogSource(_default_client)
