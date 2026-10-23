"""OTLP/JSON export with the same requests and TLS configuration as OTLP/protobuf."""

from collections.abc import Mapping
from typing import Final

import requests
from opentelemetry.exporter.otlp.json.http.trace_exporter import OTLPSpanExporter

from litellm.integrations.otel.plumbing.otlp_tls import OtlpHttpTls
from litellm.integrations.otel.plumbing.otlp_transport import build_otlp_http_transport


class OTLPJsonSpanExporter(OTLPSpanExporter):
    def __init__(
        self,
        endpoint: str | None,
        headers: Mapping[str, str],
        certificate_file: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        transport: Final = build_otlp_http_transport(
            "TRACES", OtlpHttpTls(certificate_file=certificate_file, session=session)
        )
        super().__init__(
            endpoint=endpoint, headers={**headers, "content-type": "application/json"}, _transport=transport
        )
