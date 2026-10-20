"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
The ``lite`` and ``litellm-proxy`` CLI entry points also use the bundled map without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import asyncio
import hashlib
import os
import random
import sys
import threading
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from itertools import chain
from pathlib import Path
from typing import Final, Protocol, cast  # noqa: TID251  # adapts boto3 and botocore's dynamic client responses

import httpx
from pydantic import JsonValue
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm import verbose_logger
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)
from litellm.rust_bridge._native import ModelCatalog
from litellm.rust_bridge.model_catalog import ModelCostView, RegistryRejected, bundled_registry_bytes, parse_registry

_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


def git_blob_id(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


@cache
def bundled_model_catalog() -> ModelCatalog:
    """The registry bundled with the package, parsed once."""
    body: Final = bundled_registry_bytes()
    return ModelCatalog.parse(body, "local", git_blob_id(body))


def _served_registry() -> ModelCatalog:
    import litellm

    return litellm.model_catalog.registry


_LOADED_CATALOG: Final = ModelCostView(_served_registry)


@cache
def _registry_fields(registry: ModelCatalog) -> frozenset[str]:
    return frozenset(chain.from_iterable(ModelCostView(lambda: registry)[key].keys() for key in registry.keys_list()))


class GetModelCostMap:
    """Reads the bundled registry and the registry the process currently serves."""

    @staticmethod
    def read_local_model_cost_map_bytes() -> bytes:
        return bundled_registry_bytes()

    @staticmethod
    def read_local_model_cost_map_text() -> str:
        return bundled_registry_bytes().decode("utf-8")

    @staticmethod
    def load_local_model_cost_map() -> dict[str, dict[str, JsonValue]]:
        """The bundled registry as a plain dict, aliases expanded."""
        return bundled_model_catalog().to_dict()

    @staticmethod
    def loaded_model_cost_map() -> Mapping[str, Mapping[str, JsonValue]]:
        """The registry as last loaded (bundled or remote), untouched by ``register_model`` or router registrations."""
        return _LOADED_CATALOG

    @staticmethod
    def loaded_model_cost_map_fields() -> frozenset[str]:
        """Every field name that at least one entry of the loaded registry carries."""
        return _registry_fields(_served_registry())

    @staticmethod
    def get_backup_model_count() -> int:
        return bundled_model_catalog().registry_model_count

    @staticmethod
    def integrity_failure(
        catalog: ModelCatalog,
        backup_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
    ) -> str | None:
        """Why a fetched registry is too small to trust against the bundled one, or ``None`` when it is not."""
        return catalog.integrity_failure(backup_model_count, min_model_count, max_shrink_ratio)


RETRYABLE_FETCH_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})
MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS: Final = 30.0
_litellm_import_complete = threading.Event()


def mark_litellm_import_complete() -> None:
    _litellm_import_complete.set()


@dataclass(frozen=True, slots=True)
class ModelCostMapReloaded:
    catalog: ModelCatalog


@dataclass(frozen=True, slots=True)
class ModelCostMapReloadUnavailable:
    reason: str


ModelCostMapReloadResult = ModelCostMapReloaded | ModelCostMapReloadUnavailable


@dataclass(frozen=True, slots=True)
class _FetchAttemptRetryable:
    reason: str
    retry_after_seconds: float | None


def _parse_retry_after_seconds(response: httpx.Response) -> float | None:
    try:
        header: Final = response.headers["Retry-After"]
    except KeyError:
        return None
    try:
        seconds: Final = float(header)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _retry_wait_seconds(outcome: _FetchAttemptRetryable, attempt: int, rng: random.Random) -> float:
    if outcome.retry_after_seconds is not None:
        return min(outcome.retry_after_seconds, MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS)
    return min(2.0**attempt + rng.uniform(0.0, 1.0), MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS)


class _AsyncGetClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> Awaitable[httpx.Response]: ...


class _SyncGetClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> httpx.Response: ...


class _S3Body(Protocol):
    def read(self) -> bytes: ...


class _S3GetObjectResponse(TypedDict):
    Body: ReadOnly[_S3Body]
    ETag: NotRequired[ReadOnly[str]]


class _S3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> _S3GetObjectResponse: ...


_FetchAttemptOutcome = ModelCostMapReloaded | ModelCostMapReloadUnavailable | _FetchAttemptRetryable


def _default_reload_client() -> _AsyncGetClient:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    return get_async_httpx_client(llm_provider=httpxSpecialProvider.ModelCostMap)


def _default_s3_client(timeout: int) -> _S3Client:
    import boto3
    from botocore.config import Config

    return cast(  # cast-ok: boto3's generic client factory does not expose its generated S3 response type
        _S3Client,
        boto3.client(  # pyright: ignore[reportUnknownMemberType]  # generic factory returns Unknown service clients
            "s3",
            region_name=os.getenv("AWS_REGION_NAME") or None,
            config=Config(connect_timeout=timeout, read_timeout=timeout, retries={"total_max_attempts": 1}),
        ),
    )


def _classify_fetch_error(error: httpx.HTTPError | httpx.InvalidURL, url: str) -> _FetchAttemptOutcome:
    reason: Final = f"{type(error).__name__} fetching {url}: {error}"
    if isinstance(error, (httpx.InvalidURL, httpx.UnsupportedProtocol)):
        return ModelCostMapReloadUnavailable(reason=reason)
    return _FetchAttemptRetryable(reason=reason, retry_after_seconds=None)


def _classify_fetch_content(content: bytes, url: str, etag: str | None) -> _FetchAttemptOutcome:
    parsed: Final = parse_registry(content, source="remote", revision=git_blob_id(content), etag=etag)
    if isinstance(parsed, RegistryRejected):
        return ModelCostMapReloadUnavailable(reason=f"invalid model cost map from {url}: {parsed.reason}")
    return ModelCostMapReloaded(catalog=parsed)


def _split_url(url: str) -> urllib.parse.SplitResult | ModelCostMapReloadUnavailable:
    try:
        return urllib.parse.urlsplit(url)
    except ValueError as error:
        return ModelCostMapReloadUnavailable(reason=f"invalid URL {url}: {error}")


def _attempt_fetch_s3(parts: urllib.parse.SplitResult, url: str, timeout: int) -> _FetchAttemptOutcome:
    from botocore.exceptions import BotoCoreError, ClientError

    bucket: Final = parts.netloc
    key: Final = parts.path.lstrip("/")
    if not bucket or not key:
        return ModelCostMapReloadUnavailable(reason=f"invalid S3 URL {url}: expected s3://bucket/key")

    try:
        client: Final = _default_s3_client(timeout)
        response: Final = client.get_object(Bucket=bucket, Key=key)
        content: Final = response["Body"].read()
    except (BotoCoreError, ClientError) as error:
        return ModelCostMapReloadUnavailable(reason=f"S3 {type(error).__name__} fetching {url}: {error}")
    return _classify_fetch_content(content, url, response.get("ETag"))


async def _attempt_fetch(client: _AsyncGetClient, url: str, timeout: int) -> _FetchAttemptOutcome:
    parts: Final = _split_url(url)
    if isinstance(parts, ModelCostMapReloadUnavailable):
        return parts
    if parts.scheme == "s3":
        return await asyncio.to_thread(_attempt_fetch_s3, parts, url, timeout)
    try:
        response: Final = await client.get(url, timeout=timeout)
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        return _classify_fetch_error(e, url)
    return _classify_fetch_response(response, url)


def _attempt_fetch_sync(client: _SyncGetClient, url: str, timeout: int) -> _FetchAttemptOutcome:
    parts: Final = _split_url(url)
    if isinstance(parts, ModelCostMapReloadUnavailable):
        return parts
    if parts.scheme == "s3":
        return _attempt_fetch_s3(parts, url, timeout)
    try:
        response: Final = client.get(url, timeout=timeout)
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        return _classify_fetch_error(e, url)
    return _classify_fetch_response(response, url)


def _classify_fetch_response(response: httpx.Response, url: str) -> _FetchAttemptOutcome:
    if response.status_code in RETRYABLE_FETCH_STATUS_CODES:
        return _FetchAttemptRetryable(
            reason=f"HTTP {response.status_code} from {url}",
            retry_after_seconds=_parse_retry_after_seconds(response),
        )
    if response.is_error:
        return ModelCostMapReloadUnavailable(reason=f"HTTP {response.status_code} from {url}")
    return _classify_fetch_content(response.content, url, response.headers.get("etag"))


def _next_retry_wait(
    outcome: _FetchAttemptRetryable, attempt: int, max_attempts: int, rng: random.Random
) -> float | ModelCostMapReloadUnavailable:
    if attempt == max_attempts:
        return ModelCostMapReloadUnavailable(reason=f"{outcome.reason} (after {max_attempts} attempts)")
    wait_seconds: Final = _retry_wait_seconds(outcome=outcome, attempt=attempt, rng=rng)
    verbose_logger.warning(
        "LiteLLM: model cost map fetch attempt %d/%d failed (%s); retrying in %.1fs",
        attempt,
        max_attempts,
        outcome.reason,
        wait_seconds,
    )
    return wait_seconds


async def _fetch_remote_model_cost_map_with_retry(
    url: str,
    timeout: int,
    max_attempts: int,
    sleep: Callable[[float], Awaitable[None]],
    rng: random.Random,
    client: _AsyncGetClient,
) -> ModelCostMapReloadResult:
    for attempt in range(1, max_attempts + 1):
        outcome = await _attempt_fetch(client=client, url=url, timeout=timeout)
        if not isinstance(outcome, _FetchAttemptRetryable):
            return outcome
        wait_seconds = _next_retry_wait(outcome=outcome, attempt=attempt, max_attempts=max_attempts, rng=rng)
        if isinstance(wait_seconds, ModelCostMapReloadUnavailable):
            return wait_seconds
        await sleep(wait_seconds)
    return ModelCostMapReloadUnavailable(reason="model cost map fetch failed")


def _fetch_remote_model_cost_map_with_retry_sync(
    url: str,
    timeout: int,
    attempts: range,
    sleep: Callable[[float], None],
    rng: random.Random,
    client: _SyncGetClient,
) -> ModelCostMapReloadResult:
    max_attempts: Final = attempts.stop - 1
    for attempt in attempts:
        outcome = _attempt_fetch_sync(client=client, url=url, timeout=timeout)
        if not isinstance(outcome, _FetchAttemptRetryable):
            return outcome
        wait_seconds = _next_retry_wait(outcome=outcome, attempt=attempt, max_attempts=max_attempts, rng=rng)
        if isinstance(wait_seconds, ModelCostMapReloadUnavailable):
            return wait_seconds
        sleep(wait_seconds)
    return ModelCostMapReloadUnavailable(reason="model cost map fetch failed")


async def refetch_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
    client: "_AsyncGetClient | None" = None,
) -> ModelCostMapReloadResult:
    """
    Re-fetch the model cost map for a runtime reload, retrying transient HTTP
    errors (429/5xx/transport) with Retry-After-aware backoff.

    Unlike ``get_model_cost_map`` this never falls back to the packaged backup:
    on failure it returns ``ModelCostMapReloadUnavailable`` so callers keep the
    map they already have.
    """
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return ModelCostMapReloaded(catalog=bundled_model_catalog())

    result: Final = await _fetch_remote_model_cost_map_with_retry(
        url=url,
        timeout=timeout,
        max_attempts=max_attempts,
        sleep=sleep,
        rng=rng if rng is not None else random.Random(),
        client=client if client is not None else _default_reload_client(),
    )
    if isinstance(result, ModelCostMapReloadUnavailable):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.",
            result.reason,
        )
        return result
    integrity_failure: Final = GetModelCostMap.integrity_failure(
        result.catalog, backup_model_count=GetModelCostMap.get_backup_model_count()
    )
    if integrity_failure is not None:
        verbose_logger.warning("LiteLLM: Fetched model cost map failed integrity check: %s", integrity_failure)
        return ModelCostMapReloadUnavailable(reason=f"model cost map from {url} failed integrity validation")
    _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
    _cost_map_source_info.source = "remote"
    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False
    _cost_map_source_info.fallback_reason = None
    return result


class ModelCostMapSourceInfo:
    """Tracks where the currently loaded model cost map came from."""

    source: str = "local"  # "local" or "remote"
    url: str | None = None
    is_env_forced: bool = False
    fallback_reason: str | None = None
    loaded_at: "datetime | None" = None


# Module-level singleton tracking the source of the current cost map
_cost_map_source_info: Final = ModelCostMapSourceInfo()


class CostMapProvenance(TypedDict):
    source_revision: ReadOnly[str | None]
    etag: ReadOnly[str | None]


class CostMapSourceInfo(CostMapProvenance):
    source: ReadOnly[str]
    url: ReadOnly[str | None]
    is_env_forced: ReadOnly[bool]
    fallback_reason: ReadOnly[str | None]
    loaded_at: ReadOnly[str | None]


def get_model_cost_map_provenance() -> CostMapProvenance:
    registry: Final = _served_registry()
    return {"source_revision": registry.revision, "etag": registry.etag}


def get_model_cost_map_source_info() -> CostMapSourceInfo:
    """
    Return metadata about where the current model cost map was loaded from.

    Returns a dict with:
    - source: "local" or "remote"
    - url: the remote URL attempted (or None for local-only)
    - is_env_forced: True if LITELLM_LOCAL_MODEL_COST_MAP=True forced local usage
    - fallback_reason: human-readable reason if remote failed and local was used
    - loaded_at: ISO 8601 time this process last loaded the map
    - source_revision: git blob id of the loaded file's bytes
    - etag: the ETag of the remote fetch (None for the bundled backup)
    """
    loaded_at: Final = _cost_map_source_info.loaded_at
    return {
        "source": _cost_map_source_info.source,
        "url": _cost_map_source_info.url,
        "is_env_forced": _cost_map_source_info.is_env_forced,
        "fallback_reason": _cost_map_source_info.fallback_reason,
        "loaded_at": loaded_at.isoformat() if loaded_at is not None else None,
        **get_model_cost_map_provenance(),
    }


def get_model_cost_map_loaded_at() -> "datetime | None":
    """When this process last loaded its cost map, stamped at the start of every load"""
    return _cost_map_source_info.loaded_at


def adopt_model_cost_map(registry: ModelCatalog) -> int:
    """Serve ``registry`` from now on, underneath every registration made so far."""
    import litellm
    from litellm import utils

    litellm.model_catalog.adopt(registry)
    set_fallback_generalizations(registry.fallback_rules())
    utils.invalidate_model_cost_caches()
    litellm.add_known_models(model_cost_map=litellm.model_cost)
    from litellm.router import catalog_adopted

    catalog_adopted()
    return len(registry)


def _retry_remote_fetch_in_background(
    url: str,
    timeout: int,
    max_attempts: int,
    sleep: Callable[[float], None],
    rng: random.Random,
    client: _SyncGetClient,
    first_outcome: _FetchAttemptRetryable,
) -> None:
    try:
        first_wait: Final = _next_retry_wait(outcome=first_outcome, attempt=1, max_attempts=max_attempts, rng=rng)
        if isinstance(first_wait, ModelCostMapReloadUnavailable):
            return
        sleep(first_wait)
        result: Final = _fetch_remote_model_cost_map_with_retry_sync(
            url=url,
            timeout=timeout,
            attempts=range(2, max_attempts + 1),
            sleep=sleep,
            rng=rng,
            client=client,
        )
        if isinstance(result, ModelCostMapReloadUnavailable):
            verbose_logger.warning(
                "LiteLLM: Failed to fetch remote model cost map from %s after %d attempts; keeping local backup",
                url,
                max_attempts,
            )
            return
        _litellm_import_complete.wait()
        integrity_failure: Final = GetModelCostMap.integrity_failure(
            result.catalog, backup_model_count=GetModelCostMap.get_backup_model_count()
        )
        if integrity_failure is not None:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map failed integrity check (%s). Using local backup instead. url=%s",
                integrity_failure,
                url,
            )
            return
        adopt_model_cost_map(result.catalog)
        _cost_map_source_info.source = "remote"
        _cost_map_source_info.fallback_reason = None
        _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
    except Exception as e:  # noqa: BLE001  # a failed background retry must not kill the task; the backup stays
        verbose_logger.warning("LiteLLM: Background model cost map retry failed: %s", e)


def load_model_catalog(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    client: "_SyncGetClient | None" = None,
) -> ModelCatalog:
    """
    Load the registry the process starts serving.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set or this is a ``lite`` /
       ``litellm-proxy`` CLI process, uses the bundled registry only.
    2. Otherwise fetches from ``url``, retrying transient HTTP errors
       (429/5xx/transport) with Retry-After-aware backoff in a background
       thread, validates integrity, and falls back to the bundled registry on any
       failure.
    """
    _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true" or _is_cli_process():
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return bundled_model_catalog()

    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False

    fetch_client: Final = client if client is not None else httpx
    fetch_rng: Final = rng if rng is not None else random.Random()
    outcome: Final = _attempt_fetch_sync(client=fetch_client, url=url, timeout=timeout)
    if isinstance(outcome, _FetchAttemptRetryable) and max_attempts > 1:
        threading.Thread(
            target=_retry_remote_fetch_in_background,
            kwargs={
                "url": url,
                "timeout": timeout,
                "max_attempts": max_attempts,
                "sleep": sleep,
                "rng": fetch_rng,
                "client": fetch_client,
                "first_outcome": outcome,
            },
            name="litellm-model-cost-map-retry",
            daemon=True,
        ).start()
    if not isinstance(outcome, ModelCostMapReloaded):
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup.",
            url,
            outcome.reason,
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = f"Remote fetch failed: {outcome.reason}"
        return bundled_model_catalog()

    integrity_failure: Final = GetModelCostMap.integrity_failure(
        outcome.catalog, backup_model_count=GetModelCostMap.get_backup_model_count()
    )
    if integrity_failure is not None:
        verbose_logger.warning(
            "LiteLLM: Fetched model cost map failed integrity check (%s). Using local backup instead. url=%s",
            integrity_failure,
            url,
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = "Remote data failed integrity validation"
        return bundled_model_catalog()

    _cost_map_source_info.source = "remote"
    _cost_map_source_info.fallback_reason = None
    return outcome.catalog


def get_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    client: "_SyncGetClient | None" = None,
) -> dict[str, dict[str, JsonValue]]:
    """Public entry point: the registry ``load_model_catalog`` would load, as a plain dict."""
    return load_model_catalog(
        url=url, timeout=timeout, max_attempts=max_attempts, sleep=sleep, rng=rng, client=client
    ).to_dict()
