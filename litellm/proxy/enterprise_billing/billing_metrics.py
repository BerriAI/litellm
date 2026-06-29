"""
Push-based OTLP metering for enterprise litellm deployments.

Owns a dedicated OpenTelemetry meter provider and an OTLP/gRPC exporter
authenticated to our global collector with mutual TLS. It is intentionally
isolated from the global meter provider so the customer's own OTEL metrics are
untouched and ours never leak into their backend.

The deployment's identity rides on the mTLS client certificate, not on the
payload; the secret license key is never sent as an attribute or header.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Union

import grpc
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Counter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory

if TYPE_CHECKING:
    from litellm.proxy._types import EnterpriseLicenseData

ENDPOINT_ENV = "LITELLM_BILLING_METRICS_ENDPOINT"
CLIENT_CERT_ENV = "LITELLM_BILLING_METRICS_CLIENT_CERT"
CLIENT_KEY_ENV = "LITELLM_BILLING_METRICS_CLIENT_KEY"
CA_CERT_ENV = "LITELLM_BILLING_METRICS_CA_CERT"
EXPORT_INTERVAL_ENV = "LITELLM_BILLING_METRICS_EXPORT_INTERVAL_MS"
DEFAULT_EXPORT_INTERVAL_MS = 60_000

METRIC_NAME = "litellm.enterprise.billable_requests"
METER_NAME = "litellm.enterprise.billing"

AttributeValue = Union[str, int]


@dataclass(frozen=True, slots=True)
class BillingMetricsConfig:
    endpoint: str
    client_cert_path: str
    client_key_path: str
    ca_cert_path: str
    export_interval_ms: int
    litellm_version: str
    license_id: Optional[str]


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _mtls_credentials_args(config: BillingMetricsConfig) -> Dict[str, bytes]:
    """Map cert files to grpc.ssl_channel_credentials kwargs: CA verifies the server, cert+key authenticate us."""
    return {
        "root_certificates": _read_bytes(config.ca_cert_path),
        "private_key": _read_bytes(config.client_key_path),
        "certificate_chain": _read_bytes(config.client_cert_path),
    }


def _resource_attributes(config: BillingMetricsConfig) -> Dict[str, AttributeValue]:
    base: Dict[str, AttributeValue] = {
        "service.name": "litellm-proxy",
        "litellm.version": config.litellm_version,
    }
    license_attr: Dict[str, AttributeValue] = {"litellm.license.id": config.license_id} if config.license_id else {}
    return {**base, **license_attr}


def _billable_attributes(
    category: BillableCategory, route: str, status_code: int, model_id: Optional[str]
) -> Dict[str, AttributeValue]:
    base: Dict[str, AttributeValue] = {
        "litellm.endpoint.category": category.value,
        "http.route": route,
        "http.response.status_code": status_code,
    }
    model_attr: Dict[str, AttributeValue] = {"litellm.model_id": model_id} if model_id else {}
    return {**base, **model_attr}


def build_mtls_meter_provider(config: BillingMetricsConfig) -> MeterProvider:
    credentials: grpc.ChannelCredentials = grpc.ssl_channel_credentials(**_mtls_credentials_args(config))
    exporter = OTLPMetricExporter(endpoint=config.endpoint, credentials=credentials)
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
    ca_cert = os.getenv(CA_CERT_ENV)

    missing = [
        name
        for name, value in (
            (ENDPOINT_ENV, endpoint),
            (CLIENT_CERT_ENV, client_cert),
            (CLIENT_KEY_ENV, client_key),
            (CA_CERT_ENV, ca_cert),
        )
        if not value
    ]
    if endpoint is None or client_cert is None or client_key is None or ca_cert is None:
        verbose_proxy_logger.warning(
            "Enterprise billing metrics disabled: licensed deployment missing config (%s)",
            ", ".join(missing),
        )
        return None

    unreadable = [path for path in (client_cert, client_key, ca_cert) if not os.path.isfile(path)]
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
    except Exception as exc:
        verbose_proxy_logger.warning("Enterprise billing metrics disabled: failed to initialize exporter: %s", exc)
        return None
