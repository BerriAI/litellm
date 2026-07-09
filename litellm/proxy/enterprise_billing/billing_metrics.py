"""
Push-based OTLP metering for enterprise litellm deployments.

Owns a dedicated OpenTelemetry meter provider and an OTLP/HTTP exporter
authenticated to our global collector with a TLS client certificate. The
collector front end terminates mutual TLS: the client certificate presented
here is validated against our CA at the edge, and the verified subject is
what identifies the deployment. It is intentionally isolated from the global
meter provider so the customer's own OTEL metrics are untouched and ours
never leak into their backend.

The deployment's identity rides on the TLS client certificate, not on the
payload; the secret license key is never sent as an attribute or header.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Counter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.billable_request_metrics_middleware import (
    BillableCategory,
)

if TYPE_CHECKING:
    from litellm.proxy._types import EnterpriseLicenseData

ENDPOINT_ENV = "LITELLM_BILLING_METRICS_ENDPOINT"
CLIENT_CERT_ENV = "LITELLM_BILLING_METRICS_CLIENT_CERT"
CLIENT_KEY_ENV = "LITELLM_BILLING_METRICS_CLIENT_KEY"
CA_CERT_ENV = "LITELLM_BILLING_METRICS_CA_CERT"
EXPORT_INTERVAL_ENV = "LITELLM_BILLING_METRICS_EXPORT_INTERVAL_MS"
DEFAULT_EXPORT_INTERVAL_MS = 60_000
_METRICS_PATH = "/v1/metrics"

METRIC_NAME = "litellm.enterprise.billable_requests"
METER_NAME = "litellm.enterprise.billing"

AttributeValue = Union[str, int]


@dataclass(frozen=True, slots=True)
class BillingMetricsConfig:
    endpoint: str
    client_cert_path: str
    client_key_path: str
    ca_cert_path: Optional[str]
    export_interval_ms: int
    litellm_version: str
    license_id: Optional[str]


def _metrics_endpoint(endpoint: str) -> str:
    """The OTLP/HTTP metric exporter wants the full URL including the signal path."""
    trimmed = endpoint.rstrip("/")
    return trimmed if trimmed.endswith(_METRICS_PATH) else f"{trimmed}{_METRICS_PATH}"


def _resource_attributes(config: BillingMetricsConfig) -> dict[str, AttributeValue]:
    base: dict[str, AttributeValue] = {
        "service.name": "litellm-proxy",
        "litellm.version": config.litellm_version,
    }
    license_attr: dict[str, AttributeValue] = {"litellm.license.id": config.license_id} if config.license_id else {}
    return {**base, **license_attr}


def _billable_attributes(
    category: BillableCategory, route: str, status_code: int, model_id: Optional[str]
) -> dict[str, AttributeValue]:
    base: dict[str, AttributeValue] = {
        "litellm.endpoint.category": category.value,
        "http.route": route,
        "http.response.status_code": status_code,
    }
    model_attr: dict[str, AttributeValue] = {"litellm.model_id": model_id} if model_id else {}
    return {**base, **model_attr}


def build_mtls_meter_provider(config: BillingMetricsConfig) -> MeterProvider:
    """OTLP/HTTP exporter presenting a TLS client certificate.

    The collector's load balancer terminates mutual TLS and validates the client
    certificate against our CA. Server verification uses the system trust store
    (the collector presents a public web-PKI certificate); ca_cert_path overrides
    it only for private/test collectors.
    """
    exporter = OTLPMetricExporter(
        endpoint=_metrics_endpoint(config.endpoint),
        # None -> exporter falls back to the system trust store.
        certificate_file=config.ca_cert_path,
        client_certificate_file=config.client_cert_path,
        client_key_file=config.client_key_path,
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=config.export_interval_ms)
    return MeterProvider(metric_readers=[reader], resource=Resource.create(_resource_attributes(config)))


class BillingMetricsRecorder:
    """Increments one OTLP counter per billable request. The meter provider is injected (see the factory)."""

    def __init__(self, provider: MeterProvider) -> None:
        self._provider = provider
        self._counter: Counter = provider.get_meter(METER_NAME).create_counter(
            name=METRIC_NAME,
            unit="{request}",
            description="Count of 2xx HTTP requests to billable LLM/MCP/A2A endpoints",
        )

    def record(self, *, category: BillableCategory, route: str, status_code: int, model_id: Optional[str]) -> None:
        self._counter.add(1, _billable_attributes(category, route, status_code, model_id))


def _export_interval_ms() -> int:
    raw = os.getenv(EXPORT_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_EXPORT_INTERVAL_MS
    try:
        return int(raw)
    except ValueError:
        verbose_proxy_logger.warning(
            "Invalid %s=%r, falling back to %d ms", EXPORT_INTERVAL_ENV, raw, DEFAULT_EXPORT_INTERVAL_MS
        )
        return DEFAULT_EXPORT_INTERVAL_MS


def load_billing_metrics_config(
    *, license_data: Optional["EnterpriseLicenseData"], litellm_version: str
) -> Optional[BillingMetricsConfig]:
    endpoint = os.getenv(ENDPOINT_ENV)
    client_cert = os.getenv(CLIENT_CERT_ENV)
    client_key = os.getenv(CLIENT_KEY_ENV)
    # Optional: only for private/test collectors whose server cert is not on the
    # public web PKI. The production collector needs no CA override.
    ca_cert = os.getenv(CA_CERT_ENV)

    missing = [
        name
        for name, value in (
            (ENDPOINT_ENV, endpoint),
            (CLIENT_CERT_ENV, client_cert),
            (CLIENT_KEY_ENV, client_key),
        )
        if not value
    ]
    if not endpoint or not client_cert or not client_key:
        verbose_proxy_logger.warning(
            "Enterprise billing metrics disabled: licensed deployment missing config (%s)",
            ", ".join(missing),
        )
        return None

    required_paths = [client_cert, client_key] + ([ca_cert] if ca_cert else [])
    unreadable = [path for path in required_paths if not os.path.isfile(path)]
    if unreadable:
        verbose_proxy_logger.warning(
            "Enterprise billing metrics disabled: certificate file(s) not found: %s",
            ", ".join(unreadable),
        )
        return None

    return BillingMetricsConfig(
        endpoint=endpoint,
        client_cert_path=client_cert,
        client_key_path=client_key,
        ca_cert_path=ca_cert,
        export_interval_ms=_export_interval_ms(),
        litellm_version=litellm_version,
        license_id=(license_data or {}).get("user_id"),
    )


def build_billing_metrics_recorder(
    *, premium: bool, license_data: Optional["EnterpriseLicenseData"], litellm_version: str
) -> Optional[BillingMetricsRecorder]:
    """Build the recorder, or None when the deployment is not licensed or metering is unconfigured."""
    if not premium:
        return None

    config = load_billing_metrics_config(license_data=license_data, litellm_version=litellm_version)
    if config is None:
        return None

    try:
        return BillingMetricsRecorder(build_mtls_meter_provider(config))
    except Exception as exc:  # noqa: BLE001 -- metering must never break proxy startup
        verbose_proxy_logger.warning("Enterprise billing metrics disabled: failed to initialize exporter: %s", exc)
        return None
