"""Keep explicit collector TLS settings authoritative at request time."""

import os
from collections.abc import Mapping
from typing import Final, Literal

import requests
from opentelemetry.exporter.http.transport._requests import RequestsHTTPResult, RequestsHTTPTransport
from opentelemetry.exporter.otlp.proto.http._common import (
    _load_session_from_envvar,  # pyright: ignore[reportPrivateUsage]  # pinned SDK loader preserves operator auth
)

from litellm.integrations.otel.plumbing.otlp_tls import OtlpHttpTls


class _OtlpHTTPTransport(RequestsHTTPTransport):
    def __init__(
        self,
        verify: bool | str,
        cert: str | tuple[str, str] | None,
        session: requests.Session | None,
    ) -> None:
        self._request_session: Final = session if session is not None else requests.Session()
        self._request_verify: Final = verify
        super().__init__(verify=verify, cert=cert, session=self._request_session)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        data: bytes | None = None,
    ) -> RequestsHTTPResult:
        try:
            response: Final = self._request_session.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=timeout,
                verify=self._request_verify,
                allow_redirects=False,
            )
        except Exception as error:  # noqa: BLE001  # SDK transport returns all session and auth failures as values
            return RequestsHTTPResult(error=error)
        return RequestsHTTPResult(status_code=response.status_code, reason=response.reason, response=response)


def build_otlp_http_transport(
    signal: Literal["TRACES", "LOGS", "METRICS"],
    tls: OtlpHttpTls,
    *,
    client_key_file: str | None = None,
    client_certificate_file: str | None = None,
) -> RequestsHTTPTransport:
    verify: Final = (
        tls.certificate_file
        or os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_CERTIFICATE")
        or os.getenv("OTEL_EXPORTER_OTLP_CERTIFICATE")
        or True
    )
    client_key: Final = (
        client_key_file
        or os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_CLIENT_KEY")
        or os.getenv("OTEL_EXPORTER_OTLP_CLIENT_KEY")
    )
    client_certificate: Final = (
        client_certificate_file
        or os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_CLIENT_CERTIFICATE")
        or os.getenv("OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE")
    )
    credential_variable: Final[
        Literal[
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER",
            "OTEL_PYTHON_EXPORTER_OTLP_HTTP_METRICS_CREDENTIAL_PROVIDER",
        ]
    ] = (
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER"
        if signal == "TRACES"
        else "OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER"
        if signal == "LOGS"
        else "OTEL_PYTHON_EXPORTER_OTLP_HTTP_METRICS_CREDENTIAL_PROVIDER"
    )
    session: Final = tls.session or _load_session_from_envvar(credential_variable)
    return _OtlpHTTPTransport(
        verify=verify,
        cert=(client_certificate, client_key) if client_certificate and client_key else client_certificate,
        session=session,
    )
