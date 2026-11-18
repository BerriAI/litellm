import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_FLEX_MODEL: Final = "azure_ai/FLUX.2-flex"
_PROMPT: Final = "a red fox in the snow"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@pytest.mark.covers("other.provider_wire.azure_ai.flux2_flex_generation_targets_flex_path_with_bfl_body")
def test_azure_flux2_flex_generation_hits_flex_provider_path_not_pro(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/providers/blackforestlabs/v1/flux-2-flex?api-version=preview"
        assert request.headers["api-key"] == "synthetic-azure-key"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "FLUX.2-flex",
            "prompt": _PROMPT,
            "num_images": 2,
            "width": 1536,
            "height": 1024,
            "guidance": 4.5,
            "steps": 32,
        }
        return Reply(body=json.dumps({"data": [{"b64_json": "aW1n"}, {"b64_json": "aW1n"}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_FLEX_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT, "n": 2, "size": "1536x1024", "guidance": 4.5, "steps": 32},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["data"] == [
            {"url": None, "b64_json": "aW1n", "revised_prompt": None, "provider_specific_fields": None},
            {"url": None, "b64_json": "aW1n", "revised_prompt": None, "provider_specific_fields": None},
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/providers/blackforestlabs/v1/flux-2-flex?api-version=preview")
        ]
