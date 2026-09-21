import json
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "fal-ai/trellis-2"
_REQUEST_BODY: Final = {"image_url": "https://example.com/in.png", "resolution": 1536}
_UPSTREAM_BODY: Final = {
    "model_glb": {
        "url": "https://fal.media/model.glb",
        "content_type": "model/gltf-binary",
        "file_name": "model.glb",
        "file_size": 123,
    }
}
_EXPECTED_SPEND: Final = 0.35


@pytest.mark.covers("other.provider_wire.fal_ai.passthrough_body_forwarding_and_resolution_keyed_spend")
def test_fal_passthrough_forwards_body_and_charges_resolution_tier(gateway: Gateway, tmp_path) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == f"/{_MODEL}"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert json.loads(request.body) == _REQUEST_BODY
        return Reply(body=json.dumps(_UPSTREAM_BODY).encode())

    config: Final = tmp_path / "proxy_config.yaml"
    config.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  store_model_in_db: true\n"
        "  disable_spend_logs: false\n"
        "  proxy_batch_write_at: 1\n"
        "router_settings:\n"
        "  disable_cooldowns: true\n"
    )
    with wire_server(respond) as wire:
        with owned_proxy(
            gateway,
            tmp_path,
            {"FAL_AI_API_BASE": wire.url, "FAL_AI_API_KEY": "synthetic-fal-key"},
            config=config,
        ) as candidate:
            response: Final = candidate.request(
                "POST",
                "/fal_ai/fal-ai/trellis-2",
                _REQUEST_BODY,
            )
            assert response.status_code == 200, response.text
            assert json.loads(response.content) == _UPSTREAM_BODY
            call_id: Final = response.headers["x-litellm-call-id"]
            rows: Final = eventually(
                lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert float(rows[0]["spend"]) == pytest.approx(_EXPECTED_SPEND)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", f"/{_MODEL}")]
