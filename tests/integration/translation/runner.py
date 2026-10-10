import json
from typing import Final

from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import TranslationTestCase

TRANSPORT_HEADERS: Final = frozenset({"host", "accept", "accept-encoding", "connection", "content-length", "user-agent"})


def assert_translation(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    provider.expect(Reply(status=case.mock_provider_status_code, body=json.dumps(case.mock_provider_response).encode()))
    response: Final = gateway.request("POST", case.litellm_endpoint, case.litellm_request)
    received: Final = provider.received()
    assert [(request.method, request.target) for request in received] == [("POST", case.expected_provider_endpoint)]
    sent: Final = received[0]
    assert {
        name: value
        for name, value in sent.headers.items()
        if name not in TRANSPORT_HEADERS and not name.startswith("x-stainless-")
    } == dict(case.expected_provider_headers)
    assert json.loads(sent.body) == case.expected_provider_request
    assert response.status_code == case.expected_litellm_status_code, response.text
    assert response.json() == case.expected_litellm_response
