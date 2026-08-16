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
import tempfile
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
SHUTDOWN_FLUSH_TIMEOUT_MS = 5_000
_METRICS_PATH = "/v1/metrics"

# The cert env vars take a path or the PEM itself. Secret stores that inject
# values as env content cannot mount them as files, so inline PEM is written out.
_PEM_PREFIX = "-----BEGIN"
_PEM_DIR_PREFIX = "litellm-billing-mtls-"
_PEM_FILE_MODE = 0o600
_CLIENT_CERT_FILENAME = "client.crt"
_CLIENT_KEY_FILENAME = "client.key"
_CA_CERT_FILENAME = "ca.crt"

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

    def shutdown(self) -> None:
        """Final flush + exporter-thread stop. Without this, up to one export
        interval of billable counts is dropped on every proxy restart."""
        self._provider.shutdown(timeout_millis=SHUTDOWN_FLUSH_TIMEOUT_MS)


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


@dataclass(frozen=True, slots=True)
class _CredentialPaths:
    client_cert_path: str
    client_key_path: str
    ca_cert_path: Optional[str]


def _is_pem_content(value: str) -> bool:
    return value.lstrip().startswith(_PEM_PREFIX)


def _write_pem(directory: str, filename: str, pem: str) -> str:
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(pem if pem.endswith("\n") else f"{pem}\n")
    os.chmod(path, _PEM_FILE_MODE)
    return path


def _resolve_credential_paths(*, client_cert: str, client_key: str, ca_cert: Optional[str]) -> _CredentialPaths:
    """
    Accept either a filesystem path or inline PEM content for each credential.

    Secret stores that inject values as environment content rather than mounted
    files (ECS tasks reading AWS Secrets Manager, Cloud Run reading Secret
    Manager) can only deliver the certificate as a string. The OTLP exporter
    takes paths, so inline PEM is written to a private directory once, when the
    recorder is built. Raises OSError if that write fails; the caller disables
    metering rather than propagating.
    """
    inline = tuple(value for value in (client_cert, client_key, ca_cert) if value and _is_pem_content(value))
    if not inline:
        return _CredentialPaths(client_cert, client_key, ca_cert)

    # mkdtemp is 0o700, so the 0o600 key file it holds is unreachable by other users.
    directory = tempfile.mkdtemp(prefix=_PEM_DIR_PREFIX)
    return _CredentialPaths(
        client_cert_path=(
            _write_pem(directory, _CLIENT_CERT_FILENAME, client_cert) if _is_pem_content(client_cert) else client_cert
        ),
        client_key_path=(
            _write_pem(directory, _CLIENT_KEY_FILENAME, client_key) if _is_pem_content(client_key) else client_key
        ),
        ca_cert_path=(
            _write_pem(directory, _CA_CERT_FILENAME, ca_cert) if ca_cert and _is_pem_content(ca_cert) else ca_cert
        ),
    )


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

    try:
        paths = _resolve_credential_paths(client_cert=client_cert, client_key=client_key, ca_cert=ca_cert)
    except OSError as exc:
        verbose_proxy_logger.warning(
            "Enterprise billing metrics disabled: could not write inline certificate content to disk: %s", exc
        )
        return None

    # Report the variable names, never their values. A value that is neither a
    # readable path nor recognizable PEM is still secret material, and this
    # warning would otherwise copy a client key straight into the proxy logs.
    unreadable = [
        env_name
        for env_name, path in (
            (CLIENT_CERT_ENV, paths.client_cert_path),
            (CLIENT_KEY_ENV, paths.client_key_path),
            (CA_CERT_ENV, paths.ca_cert_path),
        )
        if path and not os.path.isfile(path)
    ]
    if unreadable:
        verbose_proxy_logger.warning(
            "Enterprise billing metrics disabled: %s did not resolve to a readable certificate file. "
            "Set each to a file path, or to inline PEM content beginning with '%s'.",
            ", ".join(unreadable),
            _PEM_PREFIX,
        )
        return None

    return BillingMetricsConfig(
        endpoint=endpoint,
        client_cert_path=paths.client_cert_path,
        client_key_path=paths.client_key_path,
        ca_cert_path=paths.ca_cert_path,
        export_interval_ms=_export_interval_ms(),
        litellm_version=litellm_version,
        license_id=(license_data or {}).get("user_id"),
    )


class _ActiveRecorderRegistry:
    """One-slot registry linking the factory-built recorder to the shutdown
    hook; the middleware instance holding the recorder is not reachable from
    proxy_shutdown_event."""

    def __init__(self) -> None:
        self._recorder: Optional[BillingMetricsRecorder] = None

    def set(self, recorder: BillingMetricsRecorder) -> None:
        self._recorder = recorder

    def pop(self) -> Optional[BillingMetricsRecorder]:
        recorder = self._recorder
        self._recorder = None
        return recorder


_ACTIVE_RECORDER = _ActiveRecorderRegistry()


def build_billing_metrics_recorder(
    *, premium: bool, license_data: Optional["EnterpriseLicenseData"], litellm_version: str
) -> Optional[BillingMetricsRecorder]:
    """Build the recorder, or None when the deployment is not licensed or metering is unconfigured."""
    if not premium:
        # Debug, not warning: unlicensed is the common case and a warning here
        # would be noise on every OSS proxy. Every other disable path warns.
        verbose_proxy_logger.debug("Enterprise billing metrics disabled: deployment is not licensed")
        return None

    config = load_billing_metrics_config(license_data=license_data, litellm_version=litellm_version)
    if config is None:
        return None

    try:
        recorder = BillingMetricsRecorder(build_mtls_meter_provider(config))
    except Exception as exc:  # noqa: BLE001 -- metering must never break proxy startup
        verbose_proxy_logger.warning("Enterprise billing metrics disabled: failed to initialize exporter: %s", exc)
        return None
    _ACTIVE_RECORDER.set(recorder)
    # The only positive signal that this component meters. Without it, a silent
    # return above is indistinguishable from a working exporter in the logs, and
    # a component that carries the cert but no license would look healthy.
    verbose_proxy_logger.info(
        "Enterprise billing metrics enabled: exporting to %s every %d ms",
        config.endpoint,
        config.export_interval_ms,
    )
    return recorder


def shutdown_billing_metrics_recorder() -> None:
    """Flush and stop the active recorder, if any. Idempotent; never raises."""
    recorder = _ACTIVE_RECORDER.pop()
    if recorder is None:
        return
    try:
        recorder.shutdown()
    except Exception as exc:  # noqa: BLE001 -- shutdown must never block or fail proxy exit
        verbose_proxy_logger.warning("Enterprise billing metrics: final flush failed: %s", exc)
