import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

ROUTER_DEPLOYMENT: Final = "router-deploy"
SELECTED_MODEL: Final = "grok-4-1-fast-reasoning"
SELECTED_MODEL_WITH_PROVIDER: Final = f"azure_ai/{SELECTED_MODEL}"


@pytest.mark.covers("spend.model_router.selected_model_is_returned_and_persisted_for_plain_alias")
def test_model_router_alias_without_router_in_name_keeps_selected_model_in_response_and_spend_log(
    gateway: Gateway,
) -> None:
    prompt: Final = uuid.uuid4().hex

    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/chat/completions", request.target
        assert json.loads(request.body) == {
            "model": ROUTER_DEPLOYMENT,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }, request.body
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": 1,
                    "model": SELECTED_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "routed answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
                }
            ).encode()
        )

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model=f"azure_ai/model_router/{ROUTER_DEPLOYMENT}", api_base=wire.url, num_retries=0
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": alias, "messages": [{"role": "user", "content": prompt}]},
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["model"] == SELECTED_MODEL_WITH_PROVIDER, response.text
        assert body["choices"][0]["message"]["content"] == "routed answer", response.text
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT model, model_group, status FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (body["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows == [{"model": SELECTED_MODEL_WITH_PROVIDER, "model_group": alias, "status": "success"}]
        logs: Final = gateway.request("GET", "/spend/logs", params={"request_id": body["id"]})
        assert logs.status_code == 200, logs.text
        assert [(row["model"], row["model_group"]) for row in logs.json()] == [(SELECTED_MODEL_WITH_PROVIDER, alias)], (
            logs.text
        )
