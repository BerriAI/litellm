import json
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import TranslationTestCase
from litellm.responses.utils import ResponsesAPIRequestUtils

TRANSPORT_HEADERS: Final = frozenset(
    {"host", "accept", "accept-encoding", "connection", "content-length", "user-agent"}
)


def expected_response_for_deployment(
    expected: Mapping[str, JsonValue], model_id: str | None
) -> Mapping[str, JsonValue]:
    identifier: Final = expected.get("id")
    if not isinstance(identifier, str) or not ResponsesAPIRequestUtils._is_litellm_encoded_response_id(identifier):
        return expected
    assert model_id, "An encoded response ID must identify the selected deployment"
    decoded: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(identifier)
    return {
        **expected,
        "id": ResponsesAPIRequestUtils._build_responses_api_response_id(
            decoded["custom_llm_provider"], model_id, decoded["response_id"] or identifier
        ),
    }


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
    model_id: Final = response.headers.get("x-litellm-model-id")
    expected: Final = expected_response_for_deployment(case.expected_litellm_response, model_id)
    if expected is not case.expected_litellm_response:
        deployments: Final = gateway.get("/model/info")["data"]
        assert isinstance(deployments, list), deployments
        assert any(
            isinstance(deployment, dict)
            and deployment.get("model_name") == case.litellm_request["model"]
            and isinstance(deployment.get("model_info"), dict)
            and deployment["model_info"].get("id") == model_id
            for deployment in deployments
        ), "The encoded response ID must match the configured deployment"
    assert response.json() == expected
