"""azure-identity's `ClientSecretCredential` over a token endpoint that gives one recorded answer, so
Python reads a native flow's final token response exactly as it reads one off the wire."""

import io
import json
import time
from dataclasses import dataclass
from typing import Final

import requests
from azure.core.pipeline.transport import HttpRequest, RequestsTransport, RequestsTransportResponse
from azure.identity import ClientSecretCredential
from requests.structures import CaseInsensitiveDict


@dataclass(frozen=True, slots=True)
class _TokenEndpointAnswer:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


class _ReplayedTokenEndpoint(RequestsTransport):
    """Answers every token POST with one recorded response and every discovery GET with the endpoints
    MSAL needs. It does not wait between retries, but a wait `time.sleep` refuses still raises its error."""

    def __init__(self, answer: _TokenEndpointAnswer) -> None:
        super().__init__()
        self._answer: Final = answer

    def send(self, request: HttpRequest, **kwargs: object) -> RequestsTransportResponse:  # pyright: ignore[reportIncompatibleMethodOverride]  # azure-core types the overloads loosely  # kwargs-ok: azure-core passes its pipeline options through
        answer: Final = self._answer if request.method == "POST" else _DISCOVERY
        response: Final = requests.Response()
        response.status_code = answer.status
        response.headers = CaseInsensitiveDict(dict(answer.headers))
        response.raw = io.BytesIO(answer.body)
        response.url = request.url
        return RequestsTransportResponse(request, response)

    def sleep(self, duration: float) -> None:
        if duration * 1e9 >= _SLEEP_LIMIT_NANOS:
            time.sleep(duration)


def replayed_credential(status: int, headers: tuple[tuple[str, str], ...], body: bytes) -> ClientSecretCredential:
    return ClientSecretCredential(
        _TENANT,
        "replay",
        "replay",
        transport=_ReplayedTokenEndpoint(_TokenEndpointAnswer(status, headers, body)),
        disable_instance_discovery=True,
    )


_TENANT: Final = "00000000-0000-0000-0000-000000000000"
_SLEEP_LIMIT_NANOS: Final = 2.0**63
_DISCOVERY: Final = _TokenEndpointAnswer(
    200,
    (("content-type", "application/json"),),
    json.dumps(
        {
            "token_endpoint": f"https://login.microsoftonline.com/{_TENANT}/oauth2/v2.0/token",
            "authorization_endpoint": f"https://login.microsoftonline.com/{_TENANT}/oauth2/v2.0/authorize",
            "issuer": f"https://login.microsoftonline.com/{_TENANT}/v2.0",
        }
    ).encode(),
)
