import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import openai
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types import Moderation
from pydantic import BaseModel

_OPENAI_KEY: Final = "synthetic-openai-moderation-key"
_UPSTREAM_MODEL: Final = "omni-moderation-latest"
_PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_CATEGORIES: Final = (
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
)
_RESULT: Final = {
    "flagged": True,
    "categories": {category: category == "violence" for category in _CATEGORIES},
    "category_scores": {category: 0.93 if category == "violence" else 0.01 for category in _CATEGORIES},
    "category_applied_input_types": {
        category: ["text", "image"] if category == "violence" else ["text"] for category in _CATEGORIES
    },
}
_INPUTS: Final = (
    "a single moderation string",
    ["first moderation string", "second moderation string"],
    [
        {"type": "text", "text": "describe this picture"},
        {"type": "image_url", "image_url": {"url": "https://image.invalid/picture.png"}},
    ],
)


def _openai_peer() -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if (request.method, request.target) == ("GET", "/v1/models"):
            return Reply(
                body=json.dumps({"object": "list", "data": [{"id": _UPSTREAM_MODEL, "object": "model"}]}).encode()
            )
        assert (request.method, request.target) == ("POST", "/v1/moderations"), request.target
        assert request.headers["authorization"] == f"Bearer {_OPENAI_KEY}", request.headers
        return Reply(
            body=json.dumps({"id": "modr-integration", "model": _UPSTREAM_MODEL, "results": [_RESULT]}).encode()
        )

    return respond


def _spec_view(result: Moderation) -> dict[str, object]:
    def by_alias(section: BaseModel) -> dict[str, object]:
        return {field.alias or name: getattr(section, name) for name, field in type(section).model_fields.items()}

    return {
        "flagged": result.flagged,
        "categories": by_alias(result.categories),
        "category_scores": by_alias(result.category_scores),
        "category_applied_input_types": by_alias(result.category_applied_input_types),
    }


def _moderation_bodies(wire: Wire) -> list[object]:
    received: Final = wire.drain()
    calls: Final = [(request.method, request.target) for request in received]
    assert set(calls) <= {("GET", "/v1/models"), ("POST", "/v1/moderations")}, calls
    return [json.loads(request.body) for request in received if request.method == "POST"]


def _sdk(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def test_openai_sdk_moderation_inputs_reach_upstream_as_exact_model_and_input(gateway: Gateway) -> None:
    with wire_server(_openai_peer()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_UPSTREAM_MODEL}", api_base=f"{wire.url}/v1", api_key=_OPENAI_KEY)
        client: Final = _sdk(gateway)
        for moderation_input in _INPUTS:
            response = client.moderations.create(model=model, input=moderation_input)
            assert [_spec_view(result) for result in response.results] == [_RESULT], response
        alias: Final = gateway.request("POST", "/moderations", {"model": model, "input": _INPUTS[0]})
        assert alias.status_code == 200, alias.text
        assert [_spec_view(Moderation.model_validate(result)) for result in alias.json()["results"]] == [_RESULT], (
            alias.text
        )
        assert _moderation_bodies(wire) == [
            {"model": _UPSTREAM_MODEL, "input": moderation_input} for moderation_input in (*_INPUTS, _INPUTS[0])
        ]


def test_model_less_moderation_resolves_to_configured_moderation_model(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_openai_peer()) as wire:
        base: Final = yaml.safe_load(_PROXY_CONFIG.read_text())
        config: Final = {
            **base,
            "model_list": [
                {
                    "model_name": "default-moderation",
                    "litellm_params": {
                        "model": f"openai/{_UPSTREAM_MODEL}",
                        "api_base": f"{wire.url}/v1",
                        "api_key": _OPENAI_KEY,
                    },
                }
            ],
            "general_settings": {**base["general_settings"], "moderation_model": "default-moderation"},
        }
        path: Final = tmp_path / "moderation-default.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            for moderation_input in (_INPUTS[0], _INPUTS[2]):
                response = candidate.request("POST", "/v1/moderations", {"input": moderation_input})
                assert response.status_code == 200, response.text
                results = [_spec_view(Moderation.model_validate(result)) for result in response.json()["results"]]
                assert results == [_RESULT], response.text
        assert _moderation_bodies(wire) == [
            {"model": _UPSTREAM_MODEL, "input": moderation_input} for moderation_input in (_INPUTS[0], _INPUTS[2])
        ]
