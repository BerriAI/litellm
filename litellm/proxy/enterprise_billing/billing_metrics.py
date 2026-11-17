"""
Push-based OTLP metering for enterprise litellm deployments.

Owns a dedicated OpenTelemetry meter provider and an OTLP/HTTP exporter
authenticated to our global collector with a Cognito bearer token. It is
intentionally isolated from the global meter provider so the customer's own
OTEL metrics are untouched and ours never leak into their backend.

The secret license key is never sent as an attribute or header.
"""

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypedDict

import requests
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Counter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from pydantic import BaseModel, ConfigDict
from requests.auth import AuthBase
from typing_extensions import ReadOnly, Unpack

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.url_utils import is_https_url_with_host
from litellm.proxy.middleware.billable_request_metrics_middleware import (
    BillableCategory,
)

if TYPE_CHECKING:
    from litellm.proxy._types import EnterpriseLicenseData

ENDPOINT_ENV: Final = "LITELLM_BILLING_METRICS_ENDPOINT"
CLIENT_ID_ENV: Final = "LITELLM_BILLING_METRICS_CLIENT_ID"
CLIENT_SECRET_ENV: Final = "LITELLM_BILLING_METRICS_CLIENT_SECRET"
TOKEN_URL_ENV: Final = "LITELLM_BILLING_METRICS_TOKEN_URL"
EXPORT_INTERVAL_ENV: Final = "LITELLM_BILLING_METRICS_EXPORT_INTERVAL_MS"
DEFAULT_EXPORT_INTERVAL_MS: Final = 60_000
SHUTDOWN_FLUSH_TIMEOUT_MS: Final = 5_000
_TOKEN_REFRESH_SKEW_SECONDS: Final = 60
_METRICS_PATH: Final = "/v1/metrics"

METRIC_NAME: Final = "litellm.enterprise.billable_requests"
METER_NAME: Final = "litellm.enterprise.billing"

AttributeValue = str | int


@dataclass(frozen=True, slots=True)
class BillingMetricsConfig:
    endpoint: str
    token_url: str
    client_id: str
    client_secret: str = field(repr=False)
    export_interval_ms: int
    litellm_version: str
    license_id: str | None


class _BillingMetricsOAuthToken(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    access_token: str
    expires_in: int


def _metrics_endpoint(endpoint: str) -> str:
    """The OTLP/HTTP metric exporter wants the full URL including the signal path."""
    trimmed: Final = endpoint.rstrip("/")
    return trimmed if trimmed.endswith(_METRICS_PATH) else f"{trimmed}{_METRICS_PATH}"


def resource_attributes(config: BillingMetricsConfig) -> dict[str, AttributeValue]:
    base: Final[dict[str, AttributeValue]] = {
        "service.name": "litellm-proxy",
        "litellm.version": config.litellm_version,
    }
    license_attr: dict[str, AttributeValue] = {"litellm.license.id": config.license_id} if config.license_id else {}
    return {**base, **license_attr}


def billable_attributes(
    category: BillableCategory, route: str, status_code: int, model_id: str | None
) -> dict[str, AttributeValue]:
    base: Final[dict[str, AttributeValue]] = {
        "litellm.endpoint.category": category.value,
        "http.route": route,
        "http.response.status_code": status_code,
    }
    model_attr: Final[dict[str, AttributeValue]] = {"litellm.model_id": model_id} if model_id else {}
    return {**base, **model_attr}


class BillingMetricsBearerAuth(AuthBase):
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        token_url: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_id: Final = client_id
        self._client_secret: Final = client_secret
        self._token_url: Final = token_url
        self._clock: Final = clock
        self._lock: Final = threading.Lock()
        self._cached_token: str | None = None
        self._refresh_at: float = 0.0

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        token: Final = self._get_token()
        authenticated_request: Final = request.copy()
        authenticated_request.headers["Authorization"] = f"Bearer {token}"
        return authenticated_request

    def invalidate(self) -> None:
        with self._lock:
            self._cached_token = None
            self._refresh_at = 0.0

    def _get_token(self) -> str:
        with self._lock:
            now: Final = self._clock()
            if self._cached_token is not None and now < self._refresh_at:
                return self._cached_token
            token, cache_seconds = self._fetch_token()
            self._cached_token = token
            self._refresh_at = now + cache_seconds
            return token

    def _fetch_token(self) -> tuple[str, float]:
        try:
            # The OTel periodic exporter's background thread makes blocking token I/O safe here.
            response: Final = requests.post(
                self._token_url,
                auth=(self._client_id, self._client_secret),
                data={"grant_type": "client_credentials", "scope": "telemetry/write"},
                timeout=10,
            )
            response.raise_for_status()
            token_response: Final = _BillingMetricsOAuthToken.model_validate(response.json())
            return token_response.access_token, max(0.0, float(token_response.expires_in) - _TOKEN_REFRESH_SKEW_SECONDS)
        except (requests.RequestException, ValueError):
            raise requests.RequestException("Billing metrics token request failed") from None


class _SessionHookKwargs(TypedDict, total=False):
    stream: ReadOnly[object]
    timeout: ReadOnly[object]
    verify: ReadOnly[object]
    cert: ReadOnly[object]
    proxies: ReadOnly[object]


class _BillingMetricsBearerSession(requests.Session):
    def __init__(self, auth: BillingMetricsBearerAuth) -> None:
        super().__init__()
        self._bearer_auth: Final = auth
        self.auth = auth
        self.hooks = {"response": [self._invalidate_token_on_unauthorized]}

    def _invalidate_token_on_unauthorized(
        self,
        response: requests.Response,
        **kwargs: Unpack[_SessionHookKwargs],  # kwargs-ok: requests hooks forward send options
    ) -> requests.Response:
        if response.status_code == 401:
            self._bearer_auth.invalidate()
        return response


def build_bearer_meter_provider(config: BillingMetricsConfig) -> MeterProvider:
    auth: Final = BillingMetricsBearerAuth(
        client_id=config.client_id,
        client_secret=config.client_secret,
        token_url=config.token_url,
    )
    session: Final = _BillingMetricsBearerSession(auth)
    exporter: Final = OTLPMetricExporter(endpoint=_metrics_endpoint(config.endpoint), session=session)
    reader: Final = PeriodicExportingMetricReader(exporter, export_interval_millis=config.export_interval_ms)
    return MeterProvider(metric_readers=[reader], resource=Resource.create(resource_attributes(config)))


class BillingMetricsRecorder:
    """Increments one OTLP counter per billable request. The meter provider is injected (see the factory)."""

    def __init__(self, provider: MeterProvider) -> None:
        self._provider = provider
        self._counter: Counter = provider.get_meter(METER_NAME).create_counter(
            name=METRIC_NAME,
            unit="{request}",
            description="Count of 2xx HTTP requests to billable LLM/MCP/A2A endpoints",
        )

    def record(self, *, category: BillableCategory, route: str, status_code: int, model_id: str | None) -> None:
        self._counter.add(1, billable_attributes(category, route, status_code, model_id))

    def shutdown(self) -> None:
        """Final flush + exporter-thread stop. Without this, up to one export
        interval of billable counts is dropped on every proxy restart."""
        self._provider.shutdown(timeout_millis=SHUTDOWN_FLUSH_TIMEOUT_MS)


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


def load_billing_metrics_config(
    *, license_data: "EnterpriseLicenseData | None", litellm_version: str
) -> BillingMetricsConfig | None:
    env: Final = MappingProxyType(
        {name: os.getenv(name) or "" for name in (ENDPOINT_ENV, CLIENT_ID_ENV, CLIENT_SECRET_ENV, TOKEN_URL_ENV)}
    )
    missing: Final = tuple(name for name, value in env.items() if not value)
    if missing:
        verbose_proxy_logger.warning(
            "Enterprise billing metrics disabled: licensed deployment missing config (%s)",
            ", ".join(missing),
        )
        return None

    insecure_urls: Final = tuple(
        env_name
        for env_name, url in ((ENDPOINT_ENV, env[ENDPOINT_ENV]), (TOKEN_URL_ENV, env[TOKEN_URL_ENV]))
        if not is_https_url_with_host(url)
    )
    if insecure_urls:
        verbose_proxy_logger.warning("Enterprise billing metrics disabled: %s must use HTTPS", insecure_urls[0])
        return None

    return BillingMetricsConfig(
        endpoint=env[ENDPOINT_ENV],
        client_id=env[CLIENT_ID_ENV],
        client_secret=env[CLIENT_SECRET_ENV],
        token_url=env[TOKEN_URL_ENV],
        export_interval_ms=_export_interval_ms(),
        litellm_version=litellm_version,
        license_id=(license_data or {}).get("user_id"),
    )


class _ActiveRecorderRegistry:
    """One-slot registry linking the factory-built recorder to the shutdown
    hook; the middleware instance holding the recorder is not reachable from
    proxy_shutdown_event."""

    def __init__(self) -> None:
        self._recorder: BillingMetricsRecorder | None = None

    def set(self, recorder: BillingMetricsRecorder) -> None:
        self._recorder = recorder

    def pop(self) -> BillingMetricsRecorder | None:
        recorder: Final = self._recorder
        self._recorder = None
        return recorder


_ACTIVE_RECORDER: Final = _ActiveRecorderRegistry()


def build_billing_metrics_recorder(
    *, premium: bool, license_data: "EnterpriseLicenseData | None", litellm_version: str
) -> BillingMetricsRecorder | None:
    """Build the recorder, or None when the deployment is not licensed or metering is unconfigured."""
    if not premium:
        # Debug, not warning: unlicensed is the common case and a warning here
        # would be noise on every OSS proxy. Every other disable path warns.
        verbose_proxy_logger.debug("Enterprise billing metrics disabled: deployment is not licensed")
        return None

    config: Final = load_billing_metrics_config(license_data=license_data, litellm_version=litellm_version)
    if config is None:
        return None

    try:
        provider: Final = build_bearer_meter_provider(config)
        recorder: Final = BillingMetricsRecorder(provider)
    except Exception as exc:  # noqa: BLE001 -- metering must never break proxy startup
        verbose_proxy_logger.warning("Enterprise billing metrics disabled: failed to initialize exporter: %s", exc)
        return None
    _ACTIVE_RECORDER.set(recorder)
    verbose_proxy_logger.info(
        "Enterprise billing metrics enabled: exporting to %s every %d ms",
        config.endpoint,
        config.export_interval_ms,
    )
    return recorder


def shutdown_billing_metrics_recorder() -> None:
    """Flush and stop the active recorder, if any. Idempotent; never raises."""
    recorder: Final = _ACTIVE_RECORDER.pop()
    if recorder is None:
        return
    try:
        recorder.shutdown()
    except Exception as exc:  # noqa: BLE001 -- shutdown must never block or fail proxy exit
        verbose_proxy_logger.warning("Enterprise billing metrics: final flush failed: %s", exc)
