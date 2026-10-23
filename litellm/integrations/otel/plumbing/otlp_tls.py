import os
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from typing import Final, Literal

import requests
from pydantic import BaseModel, ConfigDict
from requests.adapters import HTTPAdapter


class _RequestHeaders(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    headers: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class OtlpHttpTls:
    certificate_file: str | None
    session: requests.Session | None


class _NoVerifyAdapter(HTTPAdapter):
    def cert_verify(
        self,
        conn: object,
        url: str,
        verify: bool | str,
        cert: str | tuple[str, str] | None,
    ) -> None:
        super().cert_verify(  # pyright: ignore[reportUnknownMemberType]  # requests stubs omit HTTPAdapter.cert_verify
            conn, url, False, cert
        )


class _IsolatedHeadersSession(requests.Session):
    def __init__(self, signal: str, headers: Mapping[str, str], content_type: str) -> None:
        from opentelemetry.util.re import parse_env_headers

        super().__init__()
        environment: Final = parse_env_headers(
            os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_HEADERS") or os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""),
            liberal=True,
        )
        explicit: Final = frozenset(key.lower() for key in headers)
        self._excluded_headers: Final = frozenset(key.lower() for key in environment) - explicit
        self._content_type: Final = content_type
        self._compression: Final = (
            (
                os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_COMPRESSION")
                or os.getenv("OTEL_EXPORTER_OTLP_COMPRESSION", "none")
            )
            .strip()
            .lower()
        )

    def prepare_request(self, request: requests.Request) -> requests.PreparedRequest:
        isolated: Final = copy(request)
        source: Final[Mapping[str, str]] = _RequestHeaders.model_validate(request).headers or {}
        kept: Final = {key: value for key, value in source.items() if key.lower() not in self._excluded_headers}
        content_type: Final = {"Content-Type": self._content_type} if "content-type" in self._excluded_headers else {}
        compression: Final = (
            {"Content-Encoding": self._compression}
            if "content-encoding" in self._excluded_headers and self._compression in ("gzip", "deflate")
            else {}
        )
        isolated.headers = {**kept, **content_type, **compression}
        session: Final = copy(self)
        session.headers = {
            key: value for key, value in self.headers.items() if key.lower() not in self._excluded_headers
        }
        return requests.Session.prepare_request(session, isolated)


def resolve_otlp_http_tls(
    signal: Literal["TRACES", "METRICS", "LOGS"],
    headers: Mapping[str, str] | None = None,
    *,
    content_type: str = "application/x-protobuf",
    isolate_headers: bool = False,
) -> OtlpHttpTls:
    session: Final = (
        _IsolatedHeadersSession(signal, headers or {}, content_type) if headers or isolate_headers else None
    )
    if os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_CERTIFICATE") or os.getenv("OTEL_EXPORTER_OTLP_CERTIFICATE"):
        return OtlpHttpTls(certificate_file=None, session=session)

    from litellm.llms.custom_httpx.http_handler import get_ssl_verify

    verify: Final = get_ssl_verify()
    if verify is False:
        unverified: Final = session or requests.Session()
        unverified.mount("https://", _NoVerifyAdapter())
        return OtlpHttpTls(certificate_file=None, session=unverified)
    if isinstance(verify, str):
        return OtlpHttpTls(certificate_file=verify, session=session)
    return OtlpHttpTls(certificate_file=None, session=session)
