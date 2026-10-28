"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
The ``lite`` and ``litellm-proxy`` CLI entry points also use the bundled map without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import asyncio
import os
import random
import sys
import threading
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # adapts boto3 and botocore's dynamic client responses

import httpx
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm import verbose_logger
from litellm._catalog import CATALOG, Catalog, Installed, Parsed, Unavailable
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.offline_mode import is_offline_mode

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


class GetModelCostMap:
    """The catalog as last loaded, for readers that want it without later registrations"""

    _loaded_catalog: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    _loaded_catalog_fields: frozenset[str] = frozenset()

    @classmethod
    def loaded_model_cost_map(cls) -> Mapping[str, Mapping[str, object]]:
        """The catalog as last loaded (bundled or remote), untouched by ``register_model`` or router registrations."""
        return cls._loaded_catalog

    @classmethod
    def loaded_model_cost_map_fields(cls) -> frozenset[str]:
        """Every field name that at least one entry of the loaded catalog carries."""
        return cls._loaded_catalog_fields

    @classmethod
    def _snapshot_loaded_catalog(cls, model_cost: Mapping[str, object]) -> None:
        raw: Final = _CATALOG_ADAPTER.validate_python(model_cost)
        cls._loaded_catalog = MappingProxyType({key: MappingProxyType(entry) for key, entry in raw.items()})
        cls._loaded_catalog_fields = frozenset(chain.from_iterable(entry.keys() for entry in raw.values()))


RETRYABLE_FETCH_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})
MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS: Final = 30.0
_litellm_import_complete = threading.Event()


def mark_litellm_import_complete() -> None:
    _litellm_import_complete.set()


@dataclass(frozen=True, slots=True)
class ModelCostMapReloaded:
    parsed: Parsed

    @property
    def revision(self) -> str:
        return self.parsed.revision

    @property
    def etag(self) -> str | None:
        return self.parsed.etag


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
    parsed: Final = CATALOG.parse(
        content,
        etag=etag,
        min_model_count=MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio=MODEL_COST_MAP_MAX_SHRINK_RATIO,
    )
    if isinstance(parsed, Unavailable):
        return ModelCostMapReloadUnavailable(reason=f"model cost map from {url} is unusable: {parsed.reason}")
    return ModelCostMapReloaded(parsed=parsed)


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
    etag_values: Final = response.headers.get_list("etag")
    return _classify_fetch_content(response.content, url, ", ".join(etag_values) if etag_values else None)


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
    if is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return _note_loaded(ModelCostMapReloaded(parsed=CATALOG.bundled()))

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
    _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
    _cost_map_source_info.source = "remote"
    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False
    _cost_map_source_info.fallback_reason = None
    return _note_loaded(result)


class ModelCostMapSourceInfo:
    """Tracks the source of the currently loaded model cost map."""

    source: str = "local"  # "local" or "remote"
    url: str | None = None
    is_env_forced: bool = False
    fallback_reason: str | None = None
    loaded_at: "datetime | None" = None
    source_revision: str | None = None
    etag: str | None = None


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
    return {
        "source_revision": _cost_map_source_info.source_revision,
        "etag": _cost_map_source_info.etag,
    }


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
        "source_revision": _cost_map_source_info.source_revision,
        "etag": _cost_map_source_info.etag,
    }


def get_model_cost_map_loaded_at() -> "datetime | None":
    """When this process last loaded its cost map, stamped at the start of every load"""
    return _cost_map_source_info.loaded_at


class AliasedCostEntry(dict[str, object]):
    __slots__ = ("names", "registered_fields")

    def __init__(
        self, entry: Mapping[str, object], names: tuple[str, ...], registered_fields: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(entry)
        self.names: tuple[str, ...] = names
        self.registered_fields: frozenset[str] = registered_fields


def _note_loaded(loaded: ModelCostMapReloaded) -> ModelCostMapReloaded:
    _cost_map_source_info.source_revision = loaded.parsed.revision
    _cost_map_source_info.etag = loaded.parsed.etag
    set_fallback_generalizations(loaded.parsed.fallback_rules())
    GetModelCostMap._snapshot_loaded_catalog(  # pyright: ignore[reportPrivateUsage]  # same module
        loaded.parsed.served()
    )
    return loaded


class _InstallGeneration:
    """Bumped by every install, so a registration racing a reload into the same dict can rebuild its entry"""

    value: int = 0


def model_cost_install_generation() -> int:
    return _InstallGeneration.value


def _install(parsed: Parsed, catalog: Catalog = CATALOG) -> Installed:
    installed: Final = catalog.install(parsed)
    _InstallGeneration.value += 1
    if installed.rejected:
        verbose_logger.warning(
            "LiteLLM: model cost map rows failed validation and are served from the bundled map or left out: %s",
            "; ".join(f"{error.key}: {error.message}" for error in installed.rejected),
        )
    return installed


def adopt_model_cost_map(parsed: Parsed) -> int:
    import litellm
    from litellm import utils

    installed: Final = _install(parsed)
    utils._invalidate_model_cost_lowercase_map()  # pyright: ignore[reportPrivateUsage]  # required cache invalidation
    litellm.add_known_models(model_cost_map=litellm.model_cost)
    utils.reapply_runtime_model_cost_registrations()
    return installed.count


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
        _note_loaded(result)
        _cost_map_source_info.source = "remote"
        _cost_map_source_info.fallback_reason = None
        _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
        adopt_model_cost_map(result.parsed)
    except Exception as e:  # noqa: BLE001  # a failed background retry must not kill the task; the backup stays
        verbose_logger.warning("LiteLLM: Background model cost map retry failed: %s", e)


def _pick_model_cost_map(
    url: str,
    timeout: int,
    max_attempts: int,
    sleep: Callable[[float], None],
    rng: random.Random | None,
    client: "_SyncGetClient | None",
) -> Parsed:
    """
    The map main loads at import. The bundled map when ``LITELLM_LOCAL_MODEL_COST_MAP`` is set, in offline mode or in a
    ``lite`` / ``litellm-proxy`` CLI process. Otherwise one remote attempt, with a background retry on a transient
    failure, and the bundled map on any failure
    """
    _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    if is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true" or _is_cli_process():
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return _note_loaded(ModelCostMapReloaded(parsed=CATALOG.bundled())).parsed

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
        return _note_loaded(ModelCostMapReloaded(parsed=CATALOG.bundled())).parsed

    _cost_map_source_info.source = "remote"
    _cost_map_source_info.fallback_reason = None
    return _note_loaded(outcome).parsed


def install_initial_model_cost_map(url: str) -> None:
    """Fills ``litellm.model_cost`` at import with the map main's rules pick"""
    _install(
        _pick_model_cost_map(
            url=url,
            timeout=5,
            max_attempts=MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
            sleep=time.sleep,
            rng=None,
            client=None,
        )
    )


def get_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    client: "_SyncGetClient | None" = None,
) -> dict[str, dict[str, JsonValue]]:
    """
    Public entry point, returns a fresh model cost map dict picked by main's rules. A remote body is parsed and
    validated by the Rust catalog, and the bundled map is returned on any failure
    """
    return _pick_model_cost_map(
        url=url, timeout=timeout, max_attempts=max_attempts, sleep=sleep, rng=rng, client=client
    ).served()
