import json
import uuid
from typing import Final

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.wire import Reply, Request, wire_server


def _delete_model_if_present(gateway: Gateway, identity: str) -> None:
    if gateway.request("GET", "/model/info", params={"litellm_model_id": identity}).status_code == 200:
        gateway.post("/model/delete", {"id": identity})


def test_model_deployment_can_be_used_and_deleted(gateway: Gateway) -> None:
    model: Final = f"lifecycle-model-{uuid.uuid4().hex}"
    identity: Final = f"lifecycle-model-id-{uuid.uuid4().hex}"

    def upstream(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert body["messages"] == [{"role": "user", "content": "model lifecycle"}]
        return Reply(
            body=json.dumps(
                {
                    "id": "model-lifecycle-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "available"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
            ).encode(),
        )

    with gateway.scenario() as scenario:
        with wire_server(upstream) as provider:
            created: Final = gateway.post(
                "/model/new",
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": f"{provider.url}/v1",
                        "api_key": "synthetic-model-key",
                    },
                    "model_info": {"id": identity},
                },
            )
            assert string_value(object_value(created["model_info"])["id"]) == identity
            scenario.cleanups.callback(_delete_model_if_present, gateway, identity)

            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "model lifecycle"}]},
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "available"

            deleted: Final = gateway.request("POST", "/model/delete", {"id": identity})
            assert deleted.status_code == 200, deleted.text
            remaining: Final = gateway.request("GET", "/model/info", params={"litellm_model_id": identity})
            assert remaining.status_code == 400, remaining.text
            assert f"Model id = {identity} not found" in remaining.text

            unavailable: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "model lifecycle"}]},
            )
            assert unavailable.status_code == 400, unavailable.text
            assert f"Invalid model name passed in model={model}" in unavailable.text
