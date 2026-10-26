"""Import-only retry scheduling, retained until CatalogRefresher takes ownership."""

from __future__ import annotations

import random
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from litellm._catalog import reload_source
from litellm._catalog._fetch import Bundled, CatalogUnavailable
from litellm._catalog._install import Provenance, install
from litellm._catalog._load import LoadedCatalog
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_model_cost_map import (
    MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS,
    _litellm_import_complete,  # pyright: ignore[reportPrivateUsage]  # retained import-only scheduler boundary until M7
    git_blob_id,
)
from litellm.rust_bridge._native import FetchedRegistry, ModelRegistry, RegistryUnavailable, fetch_once


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in {"lite", "litellm-proxy"}


@dataclass(frozen=True, slots=True)
class _FetchAttemptRetryable:
    reason: str
    retry_after_seconds: float | None


def _retry_wait_seconds(outcome: _FetchAttemptRetryable, attempt: int, rng: random.Random) -> float:
    if outcome.retry_after_seconds is not None:
        return min(outcome.retry_after_seconds, MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS)
    return min(2.0**attempt + rng.uniform(0.0, 1.0), MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS)


def _attempt(url: str, timeout: float) -> LoadedCatalog | _FetchAttemptRetryable | CatalogUnavailable:
    try:
        result: Final = fetch_once(url, if_none_match=None, timeout_seconds=timeout)
    except ValueError as error:
        return CatalogUnavailable(str(error))
    if isinstance(result, FetchedRegistry):
        return LoadedCatalog(result.registry)
    if isinstance(result, RegistryUnavailable):
        return (
            _FetchAttemptRetryable(result.reason, result.retry_after)
            if result.retryable
            else CatalogUnavailable(result.reason)
        )
    return CatalogUnavailable("Model cost map returned not-modified without an installed validator")


def _next_retry_wait(
    outcome: _FetchAttemptRetryable, attempt: int, max_attempts: int, rng: random.Random
) -> float | CatalogUnavailable:
    if attempt == max_attempts:
        return CatalogUnavailable(reason=f"{outcome.reason} (after {max_attempts} attempts)")
    wait_seconds: Final = _retry_wait_seconds(outcome=outcome, attempt=attempt, rng=rng)
    verbose_logger.warning(
        "LiteLLM: model cost map fetch attempt %d/%d failed (%s); retrying in %.1fs",
        attempt,
        max_attempts,
        outcome.reason,
        wait_seconds,
    )
    return wait_seconds


def _retry(
    url: str,
    timeout: float,
    max_attempts: int,
    sleep: Callable[[float], None],
    rng: random.Random,
    first_outcome: _FetchAttemptRetryable,
) -> None:
    try:
        first_wait: Final = _next_retry_wait(first_outcome, 1, max_attempts, rng)
        if isinstance(first_wait, CatalogUnavailable):
            return
        sleep(first_wait)
        for attempt in range(2, max_attempts + 1):
            outcome = _attempt(url, timeout)
            if isinstance(outcome, LoadedCatalog):
                _litellm_import_complete.wait()
                install(outcome, provenance=_remote_provenance(url, outcome))
                return
            if isinstance(outcome, CatalogUnavailable):
                verbose_logger.warning("LiteLLM: model cost map background fetch failed: %s", outcome.reason)
                return
            wait_seconds = _next_retry_wait(outcome, attempt, max_attempts, rng)
            if isinstance(wait_seconds, CatalogUnavailable):
                return
            sleep(wait_seconds)
    except Exception as error:  # noqa: BLE001  # failed background refresh must retain the installed catalog
        verbose_logger.warning("LiteLLM: Background model cost map retry failed: %s", error)


def _remote_provenance(url: str, catalog: LoadedCatalog) -> Provenance:
    return Provenance(
        source="remote",
        url=url,
        revision=catalog.registry.source_revision,
        etag=catalog.registry.source_validator,
    )


def initialize(
    url: str,
    *,
    timeout: float = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> None:
    forced: Final = isinstance(reload_source(url), Bundled) or _is_cli_process()
    outcome: Final = None if forced else _attempt(url, timeout)
    if isinstance(outcome, LoadedCatalog) and install(outcome, provenance=_remote_provenance(url, outcome)):
        return
    fallback: Final = (
        None
        if outcome is None
        else ("Remote data failed integrity validation" if isinstance(outcome, LoadedCatalog) else outcome.reason)
    )
    install(
        LoadedCatalog(ModelRegistry.bundled()),
        provenance=Provenance(
            source="local",
            url=None if forced else url,
            is_env_forced=forced,
            fallback_reason=fallback,
            revision=git_blob_id(ModelRegistry.bundled_bytes()),
        ),
    )
    if isinstance(outcome, _FetchAttemptRetryable) and max_attempts > 1:
        threading.Thread(
            target=_retry,
            args=(url, timeout, max_attempts, sleep, rng if rng is not None else random.Random(), outcome),
            name="litellm-model-cost-map-retry",
            daemon=True,
        ).start()
