import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL_NAME: Final = "labelled-gemini"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_ROUTER_LABELS: Final = {"env": "prod"}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
    }
).encode()


def _gemini_peer(request: Request) -> Reply:
    if request.target.endswith(":generateContent"):
        return Reply(body=_REPLY)
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _config(directory: Path, gateway: Gateway, api_base: str) -> Path:
    config: Final = directory / "router-default-requester-metadata.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": _MODEL_NAME,
                        "litellm_params": {
                            "model": "vertex_ai/gemini-3.8-flash",
                            "api_base": api_base,
                            "vertex_project": _PROJECT,
                            "vertex_location": _LOCATION,
                            "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
                        },
                    }
                ],
                "router_settings": {"default_litellm_params": {"metadata": {"requester_metadata": _ROUTER_LABELS}}},
            }
        ),
        encoding="utf-8",
    )
    return config


def _labels_sent(proxy: Gateway, wire: Wire, metadata: Mapping[str, JsonValue] | None) -> JsonValue:
    response: Final = proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _MODEL_NAME,
            "messages": [{"role": "user", "content": "Return exactly OK."}],
            **({"metadata": dict(metadata)} if metadata is not None else {}),
        },
    )
    assert response.status_code == 200, response.text
    generations: Final = tuple(request for request in wire.drain() if request.target.endswith(":generateContent"))
    assert len(generations) == 1, generations
    return _JSON_OBJECT.validate_json(generations[0].body).get("labels")


def test_router_default_requester_metadata_merges_under_each_callers_vertex_labels(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        wire_server(_gemini_peer) as wire,
        owned_proxy_process(gateway, tmp_path, {}, config=_config(tmp_path, gateway, wire.url)) as owned,
    ):
        proxy: Final = Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)
        assert _labels_sent(proxy, wire, {"team": "ai-platform"}) == {**_ROUTER_LABELS, "team": "ai-platform"}
        assert _labels_sent(proxy, wire, {"env": "staging"}) == {"env": "staging"}
        assert _labels_sent(proxy, wire, None) == _ROUTER_LABELS
