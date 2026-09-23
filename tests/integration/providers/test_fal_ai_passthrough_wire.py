import json
from pathlib import Path
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


@pytest.mark.covers("other.provider_wire.fal_ai.passthrough_queue_submit_charges_and_polls_do_not")
def test_fal_queue_submit_charges_and_polls_pass_through_free(gateway: Gateway, tmp_path) -> None:
    def respond(request: Request) -> Reply:
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_MODEL}"
            assert json.loads(request.body) == _REQUEST_BODY
            return Reply(body=json.dumps({"request_id": "req-1", "status": "IN_QUEUE"}).encode())
        if request.target == f"/{_MODEL}/requests/req-1/status":
            return Reply(body=json.dumps({"status": "COMPLETED"}).encode())
        assert request.target == f"/{_MODEL}/requests/req-1"
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
            {"FAL_AI_QUEUE_API_BASE": wire.url, "FAL_AI_API_KEY": "synthetic-fal-key"},
            config=config,
        ) as candidate:
            submit: Final = candidate.request("POST", f"/fal_ai/{_MODEL}", _REQUEST_BODY)
            assert submit.status_code == 200, submit.text
            assert json.loads(submit.content) == {"request_id": "req-1", "status": "IN_QUEUE"}
            status_response: Final = candidate.request("GET", f"/fal_ai/{_MODEL}/requests/req-1/status")
            assert status_response.status_code == 200, status_response.text
            assert json.loads(status_response.content) == {"status": "COMPLETED"}
            result_response: Final = candidate.request("GET", f"/fal_ai/{_MODEL}/requests/req-1")
            assert result_response.status_code == 200, result_response.text
            assert json.loads(result_response.content) == _UPSTREAM_BODY
            submit_spend: Final = eventually(
                lambda: read_rows(
                    'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (submit.headers["x-litellm-call-id"],),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert float(submit_spend[0]["spend"]) == pytest.approx(_EXPECTED_SPEND)
            poll_rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=ANY(%s)',
                    ([status_response.headers["x-litellm-call-id"], result_response.headers["x-litellm-call-id"]],),
                ),
                lambda values: len(values) == 2,
                seconds=70,
            )
            assert sorted(float(row["spend"]) for row in poll_rows) == [0.0, 0.0]
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"/{_MODEL}"),
            ("GET", f"/{_MODEL}/requests/req-1/status"),
            ("GET", f"/{_MODEL}/requests/req-1"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.passthrough_queue_submit_rejects_unpriceable_catalog_key")
def test_fal_queue_submit_to_catalog_key_the_pricer_cannot_price_is_rejected_not_forwarded(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps({"request_id": "req-1", "status": "IN_QUEUE"}).encode())

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
            {"FAL_AI_QUEUE_API_BASE": wire.url, "FAL_AI_API_KEY": "synthetic-fal-key"},
            config=config,
        ) as candidate:
            submit: Final = candidate.request(
                "POST",
                "/fal_ai/fal-ai/moondream3-preview/query",
                {"image_url": "https://example.com/in.png", "prompt": "one word"},
            )
            assert submit.status_code == 400, submit.text
            assert "pricing" in submit.text
        assert wire.drain() == ()


@pytest.mark.covers("other.provider_wire.fal_ai.passthrough_queue_submit_prices_string_resolution_like_integer")
def test_fal_queue_submit_prices_string_resolution_like_the_integer(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        return Reply(body=json.dumps({"request_id": "req-1", "status": "IN_QUEUE"}).encode())

    numeric_body: Final = {"image_url": "https://example.com/in.png", "resolution": 512}
    string_body: Final = {"image_url": "https://example.com/in.png", "resolution": "512"}
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
            {"FAL_AI_QUEUE_API_BASE": wire.url, "FAL_AI_API_KEY": "synthetic-fal-key"},
            config=config,
        ) as candidate:
            numeric: Final = candidate.request("POST", f"/fal_ai/{_MODEL}", numeric_body)
            assert numeric.status_code == 200, numeric.text
            assert json.loads(numeric.content) == {"request_id": "req-1", "status": "IN_QUEUE"}
            string: Final = candidate.request("POST", f"/fal_ai/{_MODEL}", string_body)
            assert string.status_code == 200, string.text
            assert json.loads(string.content) == {"request_id": "req-1", "status": "IN_QUEUE"}
            numeric_rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (numeric.headers["x-litellm-call-id"],),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            string_rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (string.headers["x-litellm-call-id"],),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            numeric_spend_value: Final = numeric_rows[0]["spend"]
            string_spend_value: Final = string_rows[0]["spend"]
            assert isinstance(numeric_spend_value, (int, float))
            assert isinstance(string_spend_value, (int, float))
            numeric_spend: Final = float(numeric_spend_value)
            string_spend: Final = float(string_spend_value)
            assert numeric_spend > 0, f"resolution 512 logged {numeric_spend} spend"
            assert string_spend > 0, f'resolution "512" logged {string_spend} spend'
            assert numeric_spend == string_spend, (
                f'resolution 512 was billed {numeric_spend} but resolution "512" was billed {string_spend}'
            )
        forwarded: Final = wire.drain()
        assert [(request.method, request.target) for request in forwarded] == [
            ("POST", f"/{_MODEL}"),
            ("POST", f"/{_MODEL}"),
        ]
        assert [json.loads(request.body) for request in forwarded] == [numeric_body, string_body]
