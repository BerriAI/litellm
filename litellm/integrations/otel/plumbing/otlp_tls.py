import os
from dataclasses import dataclass
from typing import Final, Literal

import requests
from requests.adapters import HTTPAdapter


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


def resolve_otlp_http_tls(signal: Literal["TRACES", "METRICS", "LOGS"]) -> OtlpHttpTls:
    if os.getenv(f"OTEL_EXPORTER_OTLP_{signal}_CERTIFICATE") or os.getenv("OTEL_EXPORTER_OTLP_CERTIFICATE"):
        return OtlpHttpTls(certificate_file=None, session=None)

    from litellm.llms.custom_httpx.http_handler import get_ssl_verify

    verify: Final = get_ssl_verify()
    if verify is False:
        session: Final = requests.Session()
        session.mount("https://", _NoVerifyAdapter())
        return OtlpHttpTls(certificate_file=None, session=session)
    if isinstance(verify, str):
        return OtlpHttpTls(certificate_file=verify, session=None)
    return OtlpHttpTls(certificate_file=None, session=None)
