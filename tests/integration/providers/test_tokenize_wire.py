import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server


def _tokenizer(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/tokenize")
    assert request.headers["authorization"] == "Bearer deployment-token"
    assert json.loads(request.body) == {
        "model": "acme/model",
        "prompt": "Hello world",
        "return_token_strs": True,
        "add_special_tokens": True,
    }
    return Reply(body=json.dumps({"count": 2, "tokens": [101, 102], "max_model_len": 4096}).encode())


@pytest.mark.parametrize("path", ["/tokenize", "/v1/tokenize"])
def test_tokenization_routes_model_alias_with_inference_auth(gateway: Gateway, path: str) -> None:
    with wire_server(_tokenizer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="hosted_vllm/acme/model",
            api_base=wire.url,
            api_key="deployment-token",
        )
        denied_key: Final = scenario.key(models=["another-model"])
        denied: Final = gateway.request("POST", path, {"model": model, "prompt": "Hello world"}, key=denied_key)
        assert denied.status_code in (401, 403), denied.text
        allowed_key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", path, {"model": model, "prompt": "Hello world"}, key=allowed_key)
        assert response.status_code == 200, response.text
        payload: Final = response.json()
        assert payload["model"] == model
        assert payload["count"] == 2
        assert payload["token_ids"] == [101, 102]
        assert payload["tokens"] is None
        assert float(response.headers["x-litellm-response-cost"]) == 0
        assert response.headers["x-litellm-rust"] == "true"
        assert len(wire.drain()) == 1
