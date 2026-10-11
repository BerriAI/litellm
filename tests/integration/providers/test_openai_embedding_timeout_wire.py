from __future__ import annotations

import threading
from typing import Final

from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.wire import Reply, Request, wire_server

_API_KEY: Final = "synthetic-openai-key"


def test_openai_embedding_timeouts(gateway: Gateway) -> None:
    release: Final = threading.Event()

    def respond(request: Request) -> Reply:
        assert request.target == "/v1/embeddings"
        assert JSON_OBJECT.validate_json(request.body)["input"] == ["good morning from litellm"]
        release.wait(timeout=30)
        return Reply(drop_connection=True)

    with wire_server(respond) as wire:
        try:
            with gateway.scenario() as scenario:
                model: Final = scenario.model(
                    model="openai/slow-endpoint",
                    api_base=f"{wire.url}/v1",
                    api_key=_API_KEY,
                    timeout=0.5,
                )
                response: Final = gateway.request(
                    "POST",
                    "/v1/embeddings",
                    {"model": model, "input": ["good morning from litellm"]},
                )
        finally:
            release.set()
        received: Final = wire.drain()

    assert response.status_code == 408, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["code"] == "408"
    assert received
    assert {request.headers["x-stainless-read-timeout"] for request in received} == {"0.5"}
