import json
from typing import Final

from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import TranslationTestCase

TRANSPORT_HEADERS: Final = frozenset({"host", "accept", "accept-encoding", "connection", "content-length", "user-agent"})


def run(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    provider.expect(Reply(body=json.dumps(case.provider_response).encode()))
    response: Final = gateway.request("POST", case.client_path, case.client_request)
    received: Final = provider.received()
    assert [(request.method, request.target) for request in received] == [("POST", case.provider_path)]
    sent: Final = received[0]
    assert {name: value for name, value in sent.headers.items() if name not in TRANSPORT_HEADERS} == dict(case.provider_headers)
    assert json.loads(sent.body) == case.provider_request
    assert response.status_code == case.client_status, response.text
    assert response.json() == case.client_response
