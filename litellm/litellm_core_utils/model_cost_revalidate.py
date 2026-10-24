from __future__ import annotations

import os
import random
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

from litellm import verbose_logger
from litellm.litellm_core_utils import get_model_cost_map as loader
from litellm.litellm_core_utils.get_model_cost_map import LoadedCatalog, ModelCostMapUnavailable
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import AdoptStep, DoneStep, FetchStep, Revalidation, RevalidationState, SleepStep
from litellm.rust_bridge.fork_guard import is_process_reserved_for_forking

if TYPE_CHECKING:
    from litellm.litellm_core_utils.model_cost_api import Remote


_PRELOAD_MASTER_PID: Final = (
    os.getpid()
    if ("gunicorn" in Path(sys.argv[0]).name or "gunicorn" in sys.orig_argv) and "--preload" in sys.argv
    else None
)


def _revalidation_allowed() -> bool:
    return not (
        is_offline_mode()
        or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
        or loader._is_cli_process()  # pyright: ignore[reportPrivateUsage]  # revalidation reuses the loader CLI gate
        or is_process_reserved_for_forking()
        or _PRELOAD_MASTER_PID == os.getpid()
    )


def _record_check(reason: str | None) -> None:
    loader._cost_map_source_info.checked_at = datetime.now(timezone.utc)  # pyright: ignore[reportPrivateUsage]  # shared source metadata
    if reason is not None:
        loader._cost_map_source_info.fallback_reason = reason  # pyright: ignore[reportPrivateUsage]  # shared source metadata
        verbose_logger.warning("LiteLLM: model cost map revalidation failed: %s", reason)


def _adopt_catalog(catalog: LoadedCatalog, url: str) -> int:
    rows: Final = loader.install_loaded_catalog(
        catalog,
        provenance=loader._source_info(  # pyright: ignore[reportPrivateUsage]  # use loader provenance construction
            catalog, "remote", url, False, None
        ),
    )
    if isinstance(rows, ModelCostMapUnavailable):
        _record_check(rows.reason)
        return 0
    loader._adopt_installed_model_cost_map(rows)  # pyright: ignore[reportPrivateUsage]  # preserve loader adoption semantics
    _record_check(None)
    return len(rows)


class Revalidator:
    def __init__(
        self,
        *,
        adopt: Callable[[LoadedCatalog], int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        allowed: Callable[[], bool] = _revalidation_allowed,
        max_attempts: int = loader.MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
        timeout: float = 5,
    ) -> None:
        self._adopt: Final = adopt
        self._sleep: Final = sleep
        self._allowed: Final = allowed
        self._max_attempts: Final = max_attempts
        self._timeout: Final = timeout
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    def start(self, source: Remote | str, *, jitter: bool = False) -> None:
        url: Final = source if isinstance(source, str) else source.url
        with self._lock:
            self._url = url
            if not self._allowed() or (self._thread is not None and self._thread.is_alive()):
                return
            with loader._MODEL_COST_REGISTRATION_LOCK:  # pyright: ignore[reportPrivateUsage]  # fence reloads before worker startup
                initial: Final = (
                    loader.GetModelCostMap.loaded_registry(),
                    loader.get_model_cost_map_source_info()["loaded_at"],
                )
            self._thread = threading.Thread(
                target=self._worker,
                args=(url, jitter, initial),
                name="litellm-model-cost-map-revalidate",
                daemon=True,
            )
            self._thread.start()

    def _worker(self, url: str, jitter: bool, initial: tuple[loader.ModelRegistry, str | None]) -> None:
        loader._litellm_import_complete.wait()  # pyright: ignore[reportPrivateUsage]  # wait until import state is ready
        if not self._allowed():
            return
        if jitter:
            self._sleep(random.uniform(0, 1))
        try:
            self._run(url, initial=initial)
        except Exception as error:  # noqa: BLE001  # background failures retain installed prices
            with loader._MODEL_COST_REGISTRATION_LOCK:  # pyright: ignore[reportPrivateUsage]  # serialize loader state
                _record_check(str(error))

    def _run(self, source: Remote | str, *, initial: tuple[loader.ModelRegistry, str | None] | None = None) -> None:
        url: Final = source if isinstance(source, str) else source.url
        if not self._allowed():
            return
        with loader._MODEL_COST_REGISTRATION_LOCK:  # pyright: ignore[reportPrivateUsage]  # serialize loader state
            registry: Final = loader.GetModelCostMap.loaded_registry()
            provenance: Final = loader.get_model_cost_map_source_info()
            if initial is not None and (registry is not initial[0] or provenance["loaded_at"] != initial[1]):
                return
        state, step = Revalidation(
            url,
            installed_revision=provenance["source_revision"],
            installed_etag=provenance["etag"],
            installed_url=provenance["url"] if provenance["source"] == "remote" else None,
            max_attempts=self._max_attempts,
            timeout_seconds=self._timeout,
        )
        self._drive(state, step, url, registry, provenance)

    def _drive(
        self,
        state: RevalidationState,
        step: FetchStep | SleepStep | AdoptStep | DoneStep,
        url: str,
        registry: loader.ModelRegistry,
        provenance: loader.CostMapSourceInfo,
    ) -> None:
        if not self._allowed():
            return
        match step:
            case FetchStep():
                self._drive(state, state.perform(step), url, registry, provenance)
            case SleepStep(seconds=seconds):
                self._sleep(seconds)
                self._drive(state, state.slept(), url, registry, provenance)
            case AdoptStep() | DoneStep():
                with loader._MODEL_COST_REGISTRATION_LOCK:  # pyright: ignore[reportPrivateUsage]  # serialize loader state
                    current: Final = loader.get_model_cost_map_source_info()
                    if (
                        not self._allowed()
                        or loader.GetModelCostMap.loaded_registry() is not registry
                        or current["url"] != provenance["url"]
                        or current["loaded_at"] != provenance["loaded_at"]
                    ):
                        return
                    if isinstance(step, DoneStep):
                        _record_check(step.reason)
                        return
                    catalog: Final = LoadedCatalog(step.registry, step.revision, step.etag)
                    if self._adopt is None:
                        _adopt_catalog(catalog, url)
                    else:
                        self._adopt(catalog)

    def after_fork_in_child(self) -> None:
        self._lock = threading.Lock()
        self._thread = None
        completed: Final = loader._litellm_import_complete.is_set()  # pyright: ignore[reportPrivateUsage]  # reset after fork
        loader._litellm_import_complete = threading.Event()  # pyright: ignore[reportPrivateUsage]  # reset after fork
        if completed:
            loader._litellm_import_complete.set()  # pyright: ignore[reportPrivateUsage]  # restore import readiness
        if self._url is not None and loader._cost_map_source_info.source != "remote":  # pyright: ignore[reportPrivateUsage]  # inherited provenance
            self.start(self._url, jitter=True)


REVALIDATOR: Final = Revalidator()


def _after_fork_in_child() -> None:
    loader._MODEL_COST_REGISTRATION_LOCK.release()  # pyright: ignore[reportPrivateUsage]  # release before resetting child locks
    REVALIDATOR.after_fork_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=loader._MODEL_COST_REGISTRATION_LOCK.acquire,  # pyright: ignore[reportPrivateUsage]  # coordinate fork with loader writes
        after_in_parent=loader._MODEL_COST_REGISTRATION_LOCK.release,  # pyright: ignore[reportPrivateUsage]  # release parent lock
        after_in_child=_after_fork_in_child,
    )
