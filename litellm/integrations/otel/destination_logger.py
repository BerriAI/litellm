"""Export to admin-owned destinations, independently of which logger owns a backend.

A destination is a sink, never a reason to change ownership. Whether ``OpenTelemetryV2``
or a legacy logger owns a backend is decided by the operator's own configuration; this
logger delivers the gen-AI span to whichever destinations the request resolved, so
registering a destination for one team cannot change any other tenant's pipeline.

The span's vocabulary comes from the backend's preset called purely for its mappers and
semconv, with its exporters stripped: the preset's own exporter belongs to the backend's
owning logger, and including it here would export the same call twice.
"""

from collections.abc import Mapping
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.context import request_destinations
from litellm.integrations.otel.presets import PRESET_BY_CALLBACK

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination
    from litellm.types.utils import StandardCallbackDynamicParams, StandardLoggingPayload


def _vocabulary_config(backend: str) -> OpenTelemetryV2Config:
    """The backend's span vocabulary with no exporter of its own."""
    preset_fn = PRESET_BY_CALLBACK.get(backend)
    if preset_fn is None:
        return OpenTelemetryV2Config()
    try:
        config = preset_fn(allow_missing_credentials=True)
    except Exception:  # noqa: BLE001  # an unbuildable preset still has a usable default vocabulary
        return OpenTelemetryV2Config()
    return config.model_copy(update={"exporters": ()})


class _DestinationOnlyOtel(OpenTelemetryV2):
    """An ``OpenTelemetryV2`` that reaches admin destinations and nothing else.

    It stays out of the proxy's global callback lists, and drops the request's own
    ``callback_vars`` credentials: those name the tenant's account, which the backend's
    owning logger already exports to.
    """

    def _init_otel_logger_on_litellm_proxy(self) -> None:
        return None

    def _emit_deferred_llm_call(
        self,
        payload: "StandardLoggingPayload",
        destinations: "tuple[OtelDestination, ...]",
        start_time_ns: int | None,
        end_time_ns: int | None,
        time_to_first_chunk_seconds: float | None = None,
        dynamic_params: "StandardCallbackDynamicParams | None" = None,
    ) -> Span | None:
        return super()._emit_deferred_llm_call(
            payload,
            destinations,
            start_time_ns,
            end_time_ns,
            time_to_first_chunk_seconds,
            None,
        )

    def export_to_destinations(
        self,
        kwargs: "Mapping[str, Any]",
        start_time: "datetime | float | None",
        end_time: "datetime | float | None",
    ) -> None:
        self._close_llm_call(kwargs, start_time, end_time)


class AdminDestinationLogger(CustomLogger):
    """Delivers each request's gen-AI span to the destinations its identity resolved."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._emitters: dict[str, _DestinationOnlyOtel] = {}  # mutable-ok: bounded per-backend emitter cache

    def _emitter_for(self, backend: str) -> _DestinationOnlyOtel:
        existing = self._emitters.get(backend)
        if existing is not None:
            return existing
        emitter = _DestinationOnlyOtel(config=_vocabulary_config(backend), callback_name=backend)
        self._emitters[backend] = emitter
        return emitter

    def _export(
        self,
        kwargs: "Mapping[str, Any]",
        start_time: "datetime | float | None",
        end_time: "datetime | float | None",
    ) -> None:
        for backend in sorted({d.callback_name for d in request_destinations() if d.callback_name}):
            try:
                self._emitter_for(backend).export_to_destinations(kwargs, start_time, end_time)
            except Exception as exc:  # noqa: BLE001  # one destination's failure must not break the request or the others
                litellm.verbose_logger.debug("OTel V2 destination export for %s failed: %s", backend, exc)

    async def async_log_success_event(
        self,
        kwargs: "Mapping[str, Any]",
        response_obj: Any,
        start_time: "datetime | float | None",
        end_time: "datetime | float | None",
    ) -> None:
        self._export(kwargs, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: "Mapping[str, Any]",
        response_obj: Any,
        start_time: "datetime | float | None",
        end_time: "datetime | float | None",
    ) -> None:
        self._export(kwargs, start_time, end_time)


@lru_cache(maxsize=1)
def admin_destination_logger() -> AdminDestinationLogger:
    return AdminDestinationLogger()


def register_admin_destination_logger() -> None:
    """Put the destination sink on the proxy's async callback lists, once."""
    sink = admin_destination_logger()
    for bucket in (litellm._async_success_callback, litellm._async_failure_callback):
        if not any(callback is sink for callback in bucket):
            bucket.append(sink)
