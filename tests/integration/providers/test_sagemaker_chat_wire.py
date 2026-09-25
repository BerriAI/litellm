import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_ENDPOINT: Final = "integration-vllm-endpoint"
_INFERENCE_COMPONENT: Final = "integration-vllm-component"
_SERVED_MODEL: Final = "integration-org/served-chat-model"
_ACCESS_KEY: Final = "AKIAINTEGRATION000003"
_PROMPT: Final = "synthetic inference component request"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _SERVED_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "sagemaker wire control"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }
    ).encode()


@pytest.mark.covers(
    "providers.sagemaker_chat_wire.inference_component_header_is_signed_and_hf_model_name_is_the_body_model"
)
def test_sagemaker_chat_signs_the_inference_component_header_and_sends_hf_model_name_as_the_body_model(
    gateway: Gateway,
) -> None:
    identity: Final = f"sagemaker-chat-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        assert request.target == "/", request
        assert request.headers["x-amzn-sagemaker-inference-component"] == _INFERENCE_COMPONENT, dict(request.headers)
        authorization: Final = request.headers["authorization"]
        assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={_ACCESS_KEY}/"), authorization
        signed_headers: Final = next(part for part in authorization.split(", ") if part.startswith("SignedHeaders="))
        assert "x-amzn-sagemaker-inference-component" in signed_headers.removeprefix("SignedHeaders=").split(";"), (
            authorization
        )
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _SERVED_MODEL, body
        assert body["messages"] == [{"role": "user", "content": _PROMPT}], body
        assert body["max_tokens"] == 16, body
        assert "hf_model_name" not in body, body
        return Reply(body=_completion(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"sagemaker_chat/{_ENDPOINT}",
            api_key=None,
            api_base=None,
            model_id=_INFERENCE_COMPONENT,
            hf_model_name=_SERVED_MODEL,
            aws_access_key_id=_ACCESS_KEY,
            aws_secret_access_key="synthetic-secret-key-for-testing",
            aws_region_name="us-east-1",
            sagemaker_base_url=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "max_tokens": 16},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity, response.text
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "sagemaker wire control"},
                "provider_specific_fields": {},
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/")], response.text
