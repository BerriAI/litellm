import json
from typing import Final

from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import TranslationTestCase


def run(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    provider.expect(Reply(body=json.dumps(case.provider_response).encode()))
    response: Final = gateway.request("POST", case.client_path, case.client_request)
    received: Final = provider.received()
    assert [(request.method, request.target) for request in received] == [("POST", case.provider_path)]
    sent: Final = received[0]
    assert {name: sent.headers.get(name) for name in case.provider_headers} == dict(case.provider_headers)
    assert json.loads(sent.body) == case.provider_request
    assert response.status_code == case.client_status, response.text
    assert response.json() == case.client_response
