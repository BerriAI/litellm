"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True, which serves the copy
embedded in the native extension. The ``lite`` and ``litellm-proxy`` CLI entry points also use it without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import asyncio
import hashlib
import itertools
import json
import os
import random
import sys
import threading
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    Protocol,
    TypeAlias,
    cast,  # noqa: TID251  # adapts boto3's dynamic client response type
)

from pydantic import TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm import verbose_logger
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.model_cost_dict import (
    ModelCostDict,
    ModelCostLayers,
    RegistrationIdentity,
    ServedRow,
)
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import (
    AdoptStep,
    DoneStep,
    FetchedRegistry,
    FetchStep,
    ModelRegistry,
    RegistryNotModified,
    RegistryUnavailable,
    Revalidation,
    RevalidationState,
    SleepStep,
    fetch_model_registry,
)

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
_MODEL_COST_REGISTRATION_LOCK: Final = threading.RLock()
_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})
_NativeServedRow: TypeAlias = tuple[str, Mapping[str, object], frozenset[str]]
_MODEL_COST_LAYERS: Final = ModelCostLayers()


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


def _has_valid_url(url: str) -> bool:
    try:
        urllib.parse.urlsplit(url)
    except ValueError:
        return False
    return True


RESERVED_TOP_LEVEL_KEYS: Final = frozenset({"sample_spec", FALLBACK_GENERALIZATIONS_KEY})


def git_blob_id(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


def _report_errors(registry: ModelRegistry, source: str) -> ModelRegistry:
    problems: Final = tuple(
        f"{error.key}.{error.field}: {error.message}" if error.field else f"{error.key}: {error.message}"
        for error in registry.errors()
    )
    if problems:
        verbose_logger.warning(
            "LiteLLM: %d validation or alias conflict issues in the model cost map from %s, serving every entry as written: %s",
            len(problems),
            source,
            "; ".join(problems),
        )
    return registry


class GetModelCostMap:
    """
    Reads the bundled registry and retains the installed base snapshot.
    """

    _base_registry: ModelRegistry = ModelRegistry.bundled()

    @staticmethod
    def read_local_model_cost_map_bytes() -> bytes:
        return ModelRegistry.bundled_bytes()

    @staticmethod
    def read_local_model_cost_map_text() -> str:
        return GetModelCostMap.read_local_model_cost_map_bytes().decode("utf-8")

    @staticmethod
    def load_local_model_cost_map_with_revision() -> "ModelCostMapReloaded":
        return ModelCostMapReloaded(
            model_cost_map=ModelRegistry.bundled().to_dict(),
            revision=git_blob_id(GetModelCostMap.read_local_model_cost_map_bytes()),
        )

    @staticmethod
    def load_local_model_cost_map() -> dict:
        """Load the model cost map embedded in the native extension."""
        return GetModelCostMap.load_local_model_cost_map_with_revision().model_cost_map

    _loaded_catalog: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    _loaded_catalog_fields: frozenset[str] = frozenset()

    @classmethod
    def loaded_registry(cls) -> ModelRegistry:
        return cls._base_registry

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


MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
_litellm_import_complete = threading.Event()


def mark_litellm_import_complete() -> None:
    _litellm_import_complete.set()


@dataclass(frozen=True, slots=True)
class ModelCostMapReloaded:
    model_cost_map: dict  # mutable-ok: adopted as litellm.model_cost, whose consumer contract is a plain mutable dict
    revision: str | None = None
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class ModelCostMapReloadUnavailable:
    reason: str


ModelCostMapReloadResult = ModelCostMapReloaded | ModelCostMapReloadUnavailable


@dataclass(frozen=True, slots=True)
class NativeSource:
    kind: Literal["bundled", "file", "remote", "s3"]
    value: str | None = None
    timeout_seconds: float = 5.0
    installed: ModelRegistry | None = None
    revision: str | None = None
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedCatalog:
    registry: ModelRegistry
    revision: str
    etag: str | None


@dataclass(frozen=True, slots=True)
class ModelCostMapUnavailable:
    reason: str
    retry_after: float | None


def _load(source: NativeSource) -> LoadedCatalog | ModelCostMapUnavailable:
    if source.kind != "bundled" and not source.value:
        return ModelCostMapUnavailable("model cost map source is missing", None)
    if source.kind in {"remote", "s3"} and is_offline_mode():
        return ModelCostMapUnavailable("remote model cost map fetch disabled in offline mode", None)
    fetched: Final = fetch_model_registry(
        source.value if source.kind in {"remote", "s3"} else None,
        file=source.value if source.kind == "file" else None,
        if_none_match=source.etag,
        timeout_seconds=source.timeout_seconds,
    )
    if isinstance(fetched, FetchedRegistry):
        return LoadedCatalog(fetched.registry, fetched.revision, fetched.etag)
    if isinstance(fetched, RegistryNotModified) and source.installed is not None and source.revision is not None:
        return LoadedCatalog(source.installed, source.revision, source.etag)
    if isinstance(fetched, RegistryUnavailable) and fetched.reason == "s3 is fetched by the host" and source.value:
        parts: Final = _split_url(source.value)
        if isinstance(parts, ModelCostMapReloadUnavailable):
            return ModelCostMapUnavailable(parts.reason, None)
        outcome: Final = _attempt_fetch_s3(parts, source.value, int(source.timeout_seconds))
        if isinstance(outcome, ModelCostMapUnavailable):
            return outcome
        return outcome
    if isinstance(fetched, RegistryUnavailable):
        return ModelCostMapUnavailable(fetched.reason, fetched.retry_after)
    return ModelCostMapUnavailable("model cost map was not modified without an installed catalog", None)


class _FetchResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    content: bytes


class _AsyncGetClient(Protocol):
    def get(
        self,
        url: str,
        *,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Awaitable[_FetchResponse]: ...


class _SyncGetClient(Protocol):
    def get(
        self,
        url: str,
        *,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> _FetchResponse: ...


class _S3Body(Protocol):
    def read(self) -> bytes: ...


class _S3GetObjectResponse(TypedDict):
    Body: ReadOnly[_S3Body]
    ETag: NotRequired[ReadOnly[str]]


class _S3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> _S3GetObjectResponse: ...


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


def _classify_fetch_content(content: bytes, url: str, etag: str | None) -> LoadedCatalog | ModelCostMapUnavailable:
    try:
        registry: Final = ModelRegistry.parse(content)
    except ValueError:
        return ModelCostMapUnavailable("invalid model cost map", None)
    if not registry.without_rejected()[0].check_size():
        return ModelCostMapUnavailable("model cost map failed integrity validation", None)
    return LoadedCatalog(registry, git_blob_id(content), etag)


def _split_url(url: str) -> urllib.parse.SplitResult | ModelCostMapReloadUnavailable:
    try:
        return urllib.parse.urlsplit(url)
    except ValueError:
        return ModelCostMapReloadUnavailable("invalid model cost map URL")


def _attempt_fetch_s3(
    parts: urllib.parse.SplitResult, url: str, timeout: int
) -> LoadedCatalog | ModelCostMapUnavailable:
    from botocore.exceptions import BotoCoreError, ClientError

    bucket: Final = parts.netloc
    key: Final = parts.path.lstrip("/")
    if not bucket or not key:
        return ModelCostMapUnavailable(
            reason="invalid S3 model cost map URL, expected s3://bucket/key",
            retry_after=None,
        )

    try:
        client: Final = _default_s3_client(timeout)
        response: Final = client.get_object(Bucket=bucket, Key=key)
        content: Final = response["Body"].read()
    except (BotoCoreError, ClientError):
        return ModelCostMapUnavailable(
            reason="S3 model cost map fetch failed",
            retry_after=None,
        )
    return _classify_fetch_content(content, url, response.get("ETag"))


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
    local_source: Final = is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
    loaded: Final = await _load_refetch_catalog(
        local_source=local_source,
        url=url,
        timeout=timeout,
        max_attempts=max_attempts,
        sleep=sleep,
        rng=rng,
        client=client,
    )
    if isinstance(loaded, ModelCostMapUnavailable):
        if not local_source and client is None and not url.startswith("s3://"):
            verbose_logger.warning(
                "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.",
                loaded.reason,
            )
        return ModelCostMapReloadUnavailable(loaded.reason)
    provenance: Final = _source_info(
        loaded,
        "local" if local_source else "remote",
        None if local_source else url,
        local_source,
        None,
    )
    model_cost: Final = install_loaded_catalog(loaded, provenance=provenance)
    return ModelCostMapReloaded(model_cost, loaded.revision, loaded.etag)


async def _load_refetch_catalog(
    *,
    local_source: bool,
    url: str,
    timeout: int,
    max_attempts: int,
    sleep: Callable[[float], Awaitable[None]],
    rng: random.Random | None,
    client: _AsyncGetClient | None,
) -> LoadedCatalog | ModelCostMapUnavailable:
    if local_source:
        return _load(NativeSource("bundled"))
    if client is not None:
        return await _native_revalidation_async(
            url=url,
            timeout_seconds=float(timeout),
            max_attempts=max_attempts,
            sleep=sleep,
            installed=GetModelCostMap.loaded_registry(),
            revision=_cost_map_source_info.source_revision,
            etag=_cost_map_source_info.etag,
            client=client,
            seed=rng.getrandbits(64) if rng is not None else None,
        )
    if url.startswith("s3://"):
        return await asyncio.to_thread(_load, NativeSource("s3", url, float(timeout)))
    return await _native_revalidation_async(
        url=url,
        timeout_seconds=float(timeout),
        max_attempts=max_attempts,
        sleep=sleep,
        installed=GetModelCostMap.loaded_registry(),
        revision=_cost_map_source_info.source_revision,
        etag=_cost_map_source_info.etag,
    )


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


def _native_served_rows(registry: ModelRegistry) -> tuple[_NativeServedRow, ...]:
    return tuple((key, fields, frozenset[str]()) for key, fields in registry.served().items())


def _reresolve_saved_registration(
    registry: ModelRegistry,
    folded: Mapping[str, str],
    identity: RegistrationIdentity,
) -> None:
    from litellm import utils

    prefix: Final = _MODEL_COST_LAYERS.compose_prefix(registry, identity.handle)
    view: Final = MappingProxyType({key: fields for key, fields, _ in prefix})
    write: Final = utils._registration_write(  # pyright: ignore[reportPrivateUsage]  # resolves saved writes against this prefix
        identity.original_key,
        identity.fields,
        identity.custom_llm_provider,
        identity.original_key,
        model_cost_view=view,
        warn_missing_cache=False,
    )
    destination_row: Final = view.get(write.key)
    aliases: Final = (
        tuple(key for key, fields in view.items() if fields is destination_row and key != write.key)
        if destination_row is not None
        else ()
    )
    destination: Final = folded.get(write.key.casefold(), write.key)
    _MODEL_COST_LAYERS.rebind_registration(
        identity,
        destination,
        write.inherits_from,
        alias_keys=aliases,
        defaults=write.defaults,
        cleanup_overrides=write.cleanup_overrides,
        fallback=write.fallback,
        names_the_row=write.names_the_row,
    )


def _reresolve_saved_registrations(registry: ModelRegistry) -> None:
    if not _MODEL_COST_LAYERS.identities:
        return

    served: Final = registry.served()
    folded: Final = MappingProxyType({key.casefold(): key for key in served})
    for identity in _MODEL_COST_LAYERS.identities:
        _reresolve_saved_registration(registry, folded, identity)


def install_loaded_catalog(
    catalog: LoadedCatalog,
    *,
    provenance: CostMapSourceInfo,
) -> ModelCostDict:
    registry: Final = _report_errors(catalog.registry, provenance["source"])
    base_rows: Final = _model_cost_dict_from_native_rows(_native_served_rows(registry))
    with _MODEL_COST_REGISTRATION_LOCK:
        set_fallback_generalizations(registry.fallback_rules())
        GetModelCostMap._base_registry = registry  # pyright: ignore[reportPrivateUsage]  # this installer alone owns catalog adoption
        GetModelCostMap._snapshot_loaded_catalog(base_rows)  # pyright: ignore[reportPrivateUsage]  # this installer alone owns catalog adoption
        _reresolve_saved_registrations(registry)
        composed: Final = _model_cost_dict_from_native_rows(_MODEL_COST_LAYERS.compose(registry))
        _cost_map_source_info.source = provenance["source"]
        _cost_map_source_info.url = provenance["url"]
        _cost_map_source_info.is_env_forced = provenance["is_env_forced"]
        _cost_map_source_info.fallback_reason = provenance["fallback_reason"]
        _cost_map_source_info.loaded_at = (
            datetime.fromisoformat(provenance["loaded_at"]) if provenance["loaded_at"] is not None else None
        )
        _cost_map_source_info.source_revision = catalog.revision
        _cost_map_source_info.etag = catalog.etag
        if _MODEL_COST_LAYERS.identities:
            from litellm import utils

            utils.restore_model_cost_registration_metadata(composed)
        return composed


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


AliasedCostEntry = ServedRow


def _expand_model_aliases(model_cost: Mapping[str, Mapping[str, object]]) -> ModelCostDict:  # pyright: ignore[reportUnusedFunction]  # compatibility entry point imported by callers
    if not model_cost:
        return ModelCostDict()
    registry: Final = _report_errors(ModelRegistry.parse(json.dumps(model_cost).encode()), "entries")
    return _model_cost_dict_from_native_rows(_native_served_rows(registry))


def _finalize_model_cost_map(model_cost: Mapping[str, object]) -> ModelCostDict:  # pyright: ignore[reportUnusedFunction]  # compatibility entry point imported by callers
    registry: Final = ModelRegistry.parse(json.dumps(model_cost).encode())
    return _model_cost_dict_from_native_rows(_native_served_rows(registry))


def _model_cost_dict_from_native_rows(rows: Sequence[_NativeServedRow]) -> ModelCostDict:
    rows_by_identity: Final = {
        identity: tuple(group)
        for identity, group in itertools.groupby(
            sorted(rows, key=lambda row: id(row[1])),
            key=lambda row: id(row[1]),
        )
    }
    served_rows: Final = {
        identity: ServedRow(
            identity_rows[0][1],
            tuple(key for key, _, _ in identity_rows),
            identity_rows[0][2],
        )
        for identity, identity_rows in rows_by_identity.items()
    }
    return ModelCostDict({key: served_rows[id(fields)] for key, fields, _ in rows})


def _source_info(
    catalog: LoadedCatalog,
    source: str,
    url: str | None,
    is_env_forced: bool,
    fallback_reason: str | None,
) -> CostMapSourceInfo:
    return {
        "source": source,
        "url": url,
        "is_env_forced": is_env_forced,
        "fallback_reason": fallback_reason,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": catalog.revision,
        "etag": catalog.etag,
    }


def _continue_native_revalidation(
    state: RevalidationState,
    step: FetchStep | SleepStep | AdoptStep | DoneStep,
    sleep: Callable[[float], None],
    installed: ModelRegistry | None,
    revision: str | None,
    etag: str | None,
    client: _SyncGetClient | None = None,
) -> LoadedCatalog | ModelCostMapUnavailable:
    match step:
        case FetchStep():
            next_step: Final = _injected_sync_reply(state, step, client) if client is not None else state.perform(step)
            return _continue_native_revalidation(state, next_step, sleep, installed, revision, etag, client)
        case SleepStep(seconds=seconds):
            sleep(seconds)
            return _continue_native_revalidation(state, state.slept(), sleep, installed, revision, etag, client)
        case AdoptStep(registry=registry, revision=adopted_revision, etag=adopted_etag):
            return LoadedCatalog(registry, adopted_revision, adopted_etag)
        case DoneStep(reason=None):
            if installed is not None and revision is not None:
                return LoadedCatalog(installed, revision, etag)
            return ModelCostMapUnavailable("model cost map was not modified without an installed catalog", None)
        case DoneStep(reason=reason):
            return ModelCostMapUnavailable(reason or "model cost map fetch failed", None)


def _native_revalidation(
    url: str,
    timeout_seconds: float,
    max_attempts: int,
    sleep: Callable[[float], None],
    installed: ModelRegistry | None,
    revision: str | None,
    etag: str | None,
    client: _SyncGetClient | None = None,
    seed: int | None = None,
) -> LoadedCatalog | ModelCostMapUnavailable:
    state, step = Revalidation(
        url,
        installed_revision=revision,
        installed_etag=etag,
        installed_url=_cost_map_source_info.url,
        seed=seed,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    return _continue_native_revalidation(state, step, sleep, installed, revision, etag, client)


def _injected_sync_reply(
    state: RevalidationState,
    step: FetchStep,
    client: _SyncGetClient,
) -> FetchStep | SleepStep | AdoptStep | DoneStep:
    try:
        response: Final = (
            client.get(step.url, timeout=int(step.timeout_seconds))
            if step.if_none_match is None
            else client.get(
                step.url,
                timeout=int(step.timeout_seconds),
                headers={"If-None-Match": step.if_none_match},
            )
        )
    except Exception:  # noqa: BLE001  # injected clients may raise arbitrary transport errors
        return state.transport_error("model cost map transport failed")
    return state.reply(
        response.status_code,
        response.headers.get("ETag"),
        response.headers.get("Retry-After"),
        response.content,
    )


async def _continue_native_revalidation_async(
    state: RevalidationState,
    step: FetchStep | SleepStep | AdoptStep | DoneStep,
    sleep: Callable[[float], Awaitable[None]],
    installed: ModelRegistry | None,
    revision: str | None,
    etag: str | None,
    client: _AsyncGetClient | None = None,
) -> LoadedCatalog | ModelCostMapUnavailable:
    match step:
        case FetchStep():
            if client is None:
                return await _continue_native_revalidation_async(
                    state,
                    await asyncio.to_thread(state.perform, step),
                    sleep,
                    installed,
                    revision,
                    etag,
                    client,
                )
            try:
                response: Final = (
                    await client.get(step.url, timeout=int(step.timeout_seconds))
                    if step.if_none_match is None
                    else await client.get(
                        step.url,
                        timeout=int(step.timeout_seconds),
                        headers={"If-None-Match": step.if_none_match},
                    )
                )
            except Exception:  # noqa: BLE001  # injected clients expose different transport exception classes
                return await _continue_native_revalidation_async(
                    state,
                    state.transport_error("model cost map transport failed"),
                    sleep,
                    installed,
                    revision,
                    etag,
                    client,
                )
            return await _continue_native_revalidation_async(
                state,
                state.reply(
                    response.status_code,
                    response.headers.get("ETag"),
                    response.headers.get("Retry-After"),
                    response.content,
                ),
                sleep,
                installed,
                revision,
                etag,
                client,
            )
        case SleepStep(seconds=seconds):
            await sleep(seconds)
            return await _continue_native_revalidation_async(
                state, state.slept(), sleep, installed, revision, etag, client
            )
        case AdoptStep(registry=registry, revision=adopted_revision, etag=adopted_etag):
            return LoadedCatalog(registry, adopted_revision, adopted_etag)
        case DoneStep(reason=None):
            if installed is not None and revision is not None:
                return LoadedCatalog(installed, revision, etag)
            return ModelCostMapUnavailable("model cost map was not modified without an installed catalog", None)
        case DoneStep(reason=reason):
            return ModelCostMapUnavailable(reason or "model cost map fetch failed", None)


async def _native_revalidation_async(
    url: str,
    timeout_seconds: float,
    max_attempts: int,
    sleep: Callable[[float], Awaitable[None]],
    installed: ModelRegistry | None,
    revision: str | None,
    etag: str | None,
    client: _AsyncGetClient | None = None,
    seed: int | None = None,
) -> LoadedCatalog | ModelCostMapUnavailable:
    state, step = Revalidation(
        url,
        installed_revision=revision,
        installed_etag=etag,
        installed_url=_cost_map_source_info.url,
        seed=seed,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    return await _continue_native_revalidation_async(state, step, sleep, installed, revision, etag, client)


def adopt_model_cost_map(
    new_model_cost_map: dict,  # mutable-ok: public API preserves the mutable cost-map contract
) -> int:
    body: Final = json.dumps(new_model_cost_map).encode()
    catalog: Final = LoadedCatalog(ModelRegistry.parse(body), git_blob_id(body), None)
    provenance: Final = _source_info(
        catalog,
        _cost_map_source_info.source,
        _cost_map_source_info.url,
        _cost_map_source_info.is_env_forced,
        _cost_map_source_info.fallback_reason,
    )
    composed_model_cost: Final = install_loaded_catalog(catalog, provenance=provenance)
    _adopt_installed_model_cost_map(composed_model_cost)
    fetched_model_count: Final = len(composed_model_cost)
    return fetched_model_count


def _adopt_installed_model_cost_map(model_cost: ModelCostDict) -> None:
    import litellm
    from litellm import utils

    with _MODEL_COST_REGISTRATION_LOCK:
        latest: Final = _model_cost_dict_from_native_rows(_MODEL_COST_LAYERS.compose(GetModelCostMap.loaded_registry()))
        utils.restore_model_cost_registration_metadata(latest)
        published: Final = model_cost if latest == model_cost else latest
        litellm.model_cost = published
        utils.restore_model_cost_registration_metadata(published)
        utils._invalidate_model_cost_lowercase_map()  # pyright: ignore[reportPrivateUsage]  # required cache invalidation
        litellm.add_known_models(model_cost_map=published)
        if utils._LiveDeploymentReplay.callback is not None:  # pyright: ignore[reportPrivateUsage]  # installation replays live router metadata
            utils._LiveDeploymentReplay.callback()  # pyright: ignore[reportPrivateUsage]  # installation replays live router metadata


def _retry_native_fetch_in_background(
    url: str,
    timeout: int,
    max_attempts: int,
    sleep: Callable[[float], None],
    client: _SyncGetClient | None = None,
    seed: int | None = None,
) -> None:
    try:
        result: Final = _native_revalidation(
            url=url,
            timeout_seconds=float(timeout),
            max_attempts=max_attempts - 1,
            sleep=sleep,
            installed=GetModelCostMap.loaded_registry(),
            revision=_cost_map_source_info.source_revision,
            etag=_cost_map_source_info.etag,
            client=client,
            seed=seed,
        )
        if isinstance(result, ModelCostMapUnavailable):
            verbose_logger.warning(
                "LiteLLM: Failed to fetch remote model cost map from %s after %d attempts; keeping local backup",
                url,
                max_attempts,
            )
            return
        _litellm_import_complete.wait()
        model_cost: Final = install_loaded_catalog(
            result,
            provenance=_source_info(result, "remote", url, False, None),
        )
        _adopt_installed_model_cost_map(model_cost)
    except Exception as e:  # noqa: BLE001  # a failed background retry must not kill the task; the backup stays
        verbose_logger.warning("LiteLLM: Background model cost map retry failed: %s", e)


def _get_native_model_cost_map(
    url: str,
    timeout: int,
    max_attempts: int,
    sleep: Callable[[float], None],
    client: _SyncGetClient | None,
    seed: int | None,
) -> ModelCostDict:
    local_source: Final = (
        is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true" or _is_cli_process()
    )
    if local_source:
        local_catalog: Final = _load(NativeSource("bundled"))
        if isinstance(local_catalog, ModelCostMapUnavailable):
            verbose_logger.warning("LiteLLM: Failed to load the bundled model cost map: %s", local_catalog.reason)
            return ModelCostDict()
        return install_loaded_catalog(local_catalog, provenance=_source_info(local_catalog, "local", None, True, None))

    loaded: Final = (
        _load(NativeSource("remote", url, float(timeout)))
        if client is None
        else (
            ModelCostMapUnavailable("invalid URL", None)
            if not _has_valid_url(url)
            else _native_revalidation(
                url=url,
                timeout_seconds=float(timeout),
                max_attempts=1,
                sleep=sleep,
                installed=GetModelCostMap.loaded_registry(),
                revision=_cost_map_source_info.source_revision,
                etag=_cost_map_source_info.etag,
                client=client,
                seed=seed,
            )
        )
    )
    if isinstance(loaded, LoadedCatalog):
        return install_loaded_catalog(loaded, provenance=_source_info(loaded, "remote", url, False, None))
    verbose_logger.warning(
        "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup.",
        url,
        loaded.reason,
    )
    if (
        max_attempts > 1
        and not url.startswith("s3://")
        and (loaded.retry_after is not None or loaded.reason == "model cost map transport failed")
    ):
        threading.Thread(
            target=_retry_native_fetch_in_background,
            args=(url, timeout, max_attempts, sleep, client, seed),
            name="litellm-model-cost-map-retry",
            daemon=True,
        ).start()
    bundled: Final = _load(NativeSource("bundled"))
    if isinstance(bundled, ModelCostMapUnavailable):
        verbose_logger.warning("LiteLLM: Failed to load the bundled model cost map: %s", bundled.reason)
        return ModelCostDict()
    return install_loaded_catalog(
        bundled,
        provenance=_source_info(bundled, "local", None, False, f"Remote fetch failed: {loaded.reason}"),
    )


def get_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    client: "_SyncGetClient | None" = None,
) -> dict:
    """
    Public entry point — returns the model cost map dict.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set or this is a ``lite`` /
       ``litellm-proxy`` CLI process, uses the local backup only.
    2. Otherwise fetches from ``url``, retrying transient HTTP errors
       (429/5xx/transport) with Retry-After-aware backoff in a background
       thread, validates integrity, and falls back to the local backup on any
       failure.

    """
    fetch_seed: Final = rng.getrandbits(64) if rng is not None else None
    return _get_native_model_cost_map(url, timeout, max_attempts, sleep, client, fetch_seed)
