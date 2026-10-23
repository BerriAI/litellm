"""OTLP/JSON export with the same requests and TLS configuration as OTLP/protobuf."""

import os
from collections.abc import Mapping
from typing import Final

import requests
from opentelemetry.exporter.http.transport._requests import RequestsHTTPTransport
from opentelemetry.exporter.otlp.json.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_CERTIFICATE,
    OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE,
    OTEL_EXPORTER_OTLP_CLIENT_KEY,
    OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE,
    OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE,
    OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY,
)


class OTLPJsonSpanExporter(OTLPSpanExporter):
    def __init__(
        self,
        endpoint: str | None,
        headers: Mapping[str, str],
        certificate_file: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        verify: Final = (
            certificate_file
            or os.getenv(OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE)
            or os.getenv(OTEL_EXPORTER_OTLP_CERTIFICATE)
            or True
        )
        client_key: Final = os.getenv(OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY) or os.getenv(OTEL_EXPORTER_OTLP_CLIENT_KEY)
        client_certificate: Final = os.getenv(OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE) or os.getenv(
            OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE
        )
        # The JSON SDK has no session argument. Its transport injection retains
        # requests proxies, session authentication and the existing TLS adapters.
        transport: Final = RequestsHTTPTransport(
            verify=verify,
            cert=(client_certificate, client_key) if client_certificate and client_key else client_certificate,
            session=session,
        )
        super().__init__(
            endpoint=endpoint, headers={**headers, "content-type": "application/json"}, _transport=transport
        )
