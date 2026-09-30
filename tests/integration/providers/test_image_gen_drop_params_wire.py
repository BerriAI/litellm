import json
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def test_image_generation_additional_drop_params_reaches_provider_body(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/images/generations"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "style" not in body, body
        assert body["model"] == "dall-e-3"
        assert body["prompt"] == "a scripted cat"
        assert body["size"] == "1024x1024"
        return Reply(
            body=json.dumps(
                {
                    "created": 1700000000,
                    "data": [{"b64_json": "aW1n", "revised_prompt": None, "url": None}],
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3",
            api_base=wire.url,
            api_key="synthetic-image-key",
            additional_drop_params=["style"],
        )
        response: Final = gateway.client.post(
            "/v1/images/generations",
            json={
                "model": model,
                "prompt": "a scripted cat",
                "size": "1024x1024",
                "style": "vivid",
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == "aW1n"
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/images/generations")]
