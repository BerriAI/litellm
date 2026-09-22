"""
Opt-in, anonymous OSS usage telemetry for the proxy.

Off unless ``LITELLM_TELEMETRY=true``. When enabled it owns a dedicated
OpenTelemetry meter provider with an OTLP/HTTP exporter, deliberately isolated
from the global meter provider so a deployment's own OTEL metrics are
untouched. The OSS twin of litellm.proxy.enterprise_billing.billing_metrics:
same exporter shape, no license key, no client certificate, no hostname.

Only aggregate counts leave the process: request totals by endpoint category
and status class, plus per-call provider, call type, token counts, and cost
from the standard logging payload. Model names are exported only when they are
present in the pricing map shipped with the package; anything else is
reported as "other". No API keys, teams, users, model groups, or api_base
values are ever exported.
"""

import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.proxy.middleware.billable_request_metrics_middleware import (
    BillableCategory,
    GatewayRequestSink,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

ENABLED_ENV: Final = "LITELLM_TELEMETRY"
ENDPOINT_ENV: Final = "LITELLM_TELEMETRY_ENDPOINT"
EXPORT_INTERVAL_ENV: Final = "LITELLM_TELEMETRY_EXPORT_INTERVAL_MS"
DEFAULT_ENDPOINT: Final = "https://telemetry.litellm.ai/v1/metrics"
DEFAULT_EXPORT_INTERVAL_MS: Final = 60_000
EXPORT_TIMEOUT_S: Final = 5
SHUTDOWN_FLUSH_TIMEOUT_MS: Final = 5_000
_METRICS_PATH: Final = "/v1/metrics"

METER_NAME: Final = "litellm.usage"
INSTANCE_ID_CONFIG_KEY: Final = "usage_telemetry_instance_id"
_TRUE_VALUES: Final = frozenset({"true", "1"})

AttributeValue: TypeAlias = str | int


@dataclass(frozen=True, slots=True)
class UsageTelemetryConfig:
    endpoint: str
    export_interval_ms: int
    litellm_version: str
    instance_id: str


def usage_telemetry_enabled() -> bool:
    return os.getenv(ENABLED_ENV, "").strip().lower() in _TRUE_VALUES


def _metrics_endpoint(endpoint: str) -> str:
    """The OTLP/HTTP metric exporter wants the full URL including the signal path."""
    trimmed: Final = endpoint.rstrip("/")
    return trimmed if trimmed.endswith(_METRICS_PATH) else f"{trimmed}{_METRICS_PATH}"


def _export_interval_ms() -> int:
    raw: Final = os.getenv(EXPORT_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_EXPORT_INTERVAL_MS
    try:
        return int(raw)
    except ValueError:
        verbose_proxy_logger.warning(
            "Invalid %s=%r, falling back to %d ms", EXPORT_INTERVAL_ENV, raw, DEFAULT_EXPORT_INTERVAL_MS
        )
        return DEFAULT_EXPORT_INTERVAL_MS


def load_usage_telemetry_config(*, litellm_version: str, instance_id: str) -> UsageTelemetryConfig:
    endpoint: Final = os.getenv(ENDPOINT_ENV, DEFAULT_ENDPOINT)
    return UsageTelemetryConfig(
        endpoint=endpoint,
        export_interval_ms=_export_interval_ms(),
        litellm_version=litellm_version,
        instance_id=instance_id,
    )


def _build_exporter(config: UsageTelemetryConfig) -> OTLPMetricExporter:
    """``headers`` must be non-empty: with a falsy value OTLPMetricExporter
    reads OTEL_EXPORTER_OTLP_*_HEADERS from the environment, which would
    forward a deployment's own collector auth header to telemetry.litellm.ai."""
    return OTLPMetricExporter(
        endpoint=_metrics_endpoint(config.endpoint),
        timeout=EXPORT_TIMEOUT_S,
        headers={"User-Agent": f"litellm-proxy/{config.litellm_version}"},
    )


def build_usage_meter_provider(config: UsageTelemetryConfig) -> MeterProvider:
    exporter: Final = _build_exporter(config)
    reader: Final = PeriodicExportingMetricReader(exporter, export_interval_millis=config.export_interval_ms)
    resource: Final = Resource(
        MappingProxyType(
            {
                "service.name": "litellm-proxy",
                "litellm.version": config.litellm_version,
                "litellm.instance.id": config.instance_id,
            }
        )
    )
    return MeterProvider(metric_readers=(reader,), resource=resource)


class _UsageTelemetryLogFields(BaseModel):
    """The only fields telemetry reads out of the standard logging payload."""

    model_config = ConfigDict(extra="ignore")

    model: str
    custom_llm_provider: str | None
    call_type: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    response_cost: float = Field(ge=0)


class UsageTelemetryRecorder(CustomLogger):
    """
    Both the middleware sink (the ``GatewayRequestSink`` protocol: every
    classified request, any status) and a success callback (per-call token and
    spend counts off the standard logging payload). The meter provider is
    injected; see ``build_usage_telemetry_recorder``.
    """

    def __init__(self, provider: MeterProvider) -> None:
        super().__init__()
        self._provider = provider
        meter: Final = provider.get_meter(METER_NAME)
        self._requests = meter.create_counter(
            name="litellm.usage.requests",
            unit="{request}",
            description="Count of requests to LLM/MCP/A2A endpoints by category, route, and status class",
        )
        self._llm_requests = meter.create_counter(
            name="litellm.usage.llm_requests",
            unit="{request}",
            description="Count of successful LLM calls by model, provider, and call type",
        )
        self._tokens = meter.create_counter(
            name="litellm.usage.tokens",
            unit="{token}",
            description="Count of prompt and completion tokens by model, provider, and call type",
        )
        self._spend = meter.create_counter(
            name="litellm.usage.spend_usd",
            unit="USD",
            description="Estimated spend in USD by model, provider, and call type",
        )

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None:
        self._requests.add(
            1,
            MappingProxyType(
                {
                    "litellm.endpoint.category": category.value,
                    "http.route": route,
                    "http.response.status_class": f"{status_code // 100}xx",
                }
            ),
        )

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        raw_payload: Final = kwargs.get("standard_logging_object")
        if not isinstance(raw_payload, dict):
            return
        try:
            payload: Final = _UsageTelemetryLogFields.model_validate(raw_payload)
        except ValidationError:
            verbose_proxy_logger.debug("Usage telemetry: standard_logging_object missing expected fields, skipping")
            return
        attrs: Final[Mapping[str, str]] = MappingProxyType(
            {
                "litellm.model": _public_model_label(payload.model, payload.custom_llm_provider),
                "litellm.provider": payload.custom_llm_provider or "unknown",
                "litellm.call_type": payload.call_type,
            }
        )
        self._llm_requests.add(1, attrs)
        self._tokens.add(payload.prompt_tokens, MappingProxyType({**attrs, "litellm.token.kind": "prompt"}))
        self._tokens.add(payload.completion_tokens, MappingProxyType({**attrs, "litellm.token.kind": "completion"}))
        self._spend.add(payload.response_cost, attrs)

    def shutdown(self) -> None:
        """Final flush + exporter-thread stop. Without this, up to one export
        interval of counts is dropped on every proxy restart."""
        self._provider.shutdown(timeout_millis=SHUTDOWN_FLUSH_TIMEOUT_MS)


_PUBLIC_MODELS: Final[frozenset[str]] = frozenset(GetModelCostMap.load_local_model_cost_map())


def _public_model_label(model: str, provider: str | None) -> str:
    """Only names shipped in the packaged pricing map are exported; anything
    else, including names operators registered for billing, becomes "other"."""
    candidates: Final = (
        (model, f"{provider}/{model}", model.removeprefix(f"{provider}/")) if provider is not None else (model,)
    )
    return next((candidate for candidate in candidates if candidate in _PUBLIC_MODELS), "other")


class _ConfigParamWhere(TypedDict):
    param_name: ReadOnly[str]


class _InstanceIdValue(TypedDict):
    instance_id: ReadOnly[str]


class _ConfigParamCreate(TypedDict):
    param_name: ReadOnly[str]
    param_value: ReadOnly[str]


class _ConfigParamRow(Protocol):
    param_value: object


def _instance_id_from_row(row: "_ConfigParamRow | None") -> str | None:
    param_value: Final[object] = row.param_value if row is not None else None
    parsed: Final[object] = json.loads(param_value) if isinstance(param_value, str) else param_value
    if isinstance(parsed, dict) and isinstance(parsed.get("instance_id"), str):
        return parsed["instance_id"]
    return None


async def resolve_instance_id(prisma_client: "PrismaClient | None") -> str:
    """
    A stable anonymous deployment id persisted in ``LiteLLM_Config``, so counts
    from one deployment aggregate across restarts. Without a database each
    process gets a fresh uuid.
    """
    if prisma_client is None:
        return str(uuid.uuid4())
    try:
        where: Final[_ConfigParamWhere] = {"param_name": INSTANCE_ID_CONFIG_KEY}
        existing: Final = _instance_id_from_row(await prisma_client.writer_db.litellm_config.find_unique(where=where))
        if existing is not None:
            return existing
        instance_id: Final = str(uuid.uuid4())
        instance_value: Final[_InstanceIdValue] = {"instance_id": instance_id}
        data: Final[_ConfigParamCreate] = {
            "param_name": INSTANCE_ID_CONFIG_KEY,
            "param_value": json.dumps(instance_value),
        }
        try:
            await prisma_client.writer_db.litellm_config.create(data=data)
        except Exception:  # noqa: BLE001 -- a concurrent worker won the insert; read its row instead
            raced: Final = _instance_id_from_row(await prisma_client.writer_db.litellm_config.find_unique(where=where))
            if raced is not None:
                return raced
            raise
        return instance_id
    except Exception as exc:  # noqa: BLE001 -- telemetry must never break proxy startup
        verbose_proxy_logger.warning(
            "Usage telemetry: could not resolve a persistent instance id (%s); using a per-process id", exc
        )
        return str(uuid.uuid4())


class FanOutGatewayRequestSink:
    """Fans one middleware record() out to several sinks; a raising sink never stops the others."""

    def __init__(self, sinks: tuple[GatewayRequestSink, ...]) -> None:
        self._sinks: Final = sinks

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None:
        for sink in self._sinks:
            try:
                sink.record(category=category, route=route, status_code=status_code)
            except Exception:  # noqa: BLE001 -- metering must never fail a request that was already served
                verbose_proxy_logger.warning("usage telemetry sink failed for %s", route, exc_info=True)


def compose_gateway_request_sinks(*sinks: GatewayRequestSink | None) -> GatewayRequestSink | None:
    active: Final = tuple(sink for sink in sinks if sink is not None)
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return FanOutGatewayRequestSink(active)


class _ActiveRecorderRegistry:
    """One-slot registry linking the factory-built recorder to the shutdown
    hook; the middleware instance holding the recorder is not reachable from
    proxy_shutdown_event."""

    def __init__(self) -> None:
        self._recorder: UsageTelemetryRecorder | None = None

    def set(self, recorder: UsageTelemetryRecorder) -> None:
        self._recorder = recorder

    def pop(self) -> UsageTelemetryRecorder | None:
        recorder: Final = self._recorder
        self._recorder = None
        return recorder


_ACTIVE_RECORDER: Final = _ActiveRecorderRegistry()


def build_usage_telemetry_recorder(
    *,
    litellm_version: str,
    instance_id: str,
    provider_factory: Callable[[UsageTelemetryConfig], MeterProvider] = build_usage_meter_provider,
) -> UsageTelemetryRecorder | None:
    """Build the recorder, or None when telemetry is not opted in or the exporter cannot initialize."""
    if not usage_telemetry_enabled():
        return None

    config: Final = load_usage_telemetry_config(litellm_version=litellm_version, instance_id=instance_id)
    try:
        recorder: Final = UsageTelemetryRecorder(provider_factory(config))
    except Exception as exc:  # noqa: BLE001 -- telemetry must never break proxy startup
        verbose_proxy_logger.warning("Usage telemetry disabled: failed to initialize exporter: %s", exc)
        return None
    _ACTIVE_RECORDER.set(recorder)
    verbose_proxy_logger.info(
        "Usage telemetry enabled: exporting anonymous usage metrics to %s every %d ms; "
        "set LITELLM_TELEMETRY=false to disable",
        config.endpoint,
        config.export_interval_ms,
    )
    return recorder


def shutdown_usage_telemetry_recorder() -> None:
    """Flush and stop the active recorder, if any. Idempotent; never raises."""
    recorder: Final = _ACTIVE_RECORDER.pop()
    if recorder is None:
        return
    try:
        recorder.shutdown()
    except Exception as exc:  # noqa: BLE001 -- shutdown must never block or fail proxy exit
        verbose_proxy_logger.warning("Usage telemetry: final flush failed: %s", exc)
