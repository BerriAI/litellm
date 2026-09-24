import json
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("quota_management.spend_tracking.pass_through_failure.failure_row_written")
@pytest.mark.timeout(180)
def test_config_pass_through_endpoint_failure_writes_a_spend_row(gateway: Gateway, tmp_path) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == "/"
        assert json.loads(request.body) == {"search": "hello"}
        return Reply(status=500, body=json.dumps({"error": "upstream search failure"}).encode())

    with wire_server(respond) as wire:
        config: Final = tmp_path / "proxy_config.yaml"
        config.write_text(
            "model_list: []\n"
            "general_settings:\n"
            "  master_key: os.environ/LITELLM_MASTER_KEY\n"
            "  database_url: os.environ/DATABASE_URL\n"
            "  store_model_in_db: true\n"
            "  disable_spend_logs: false\n"
            "  proxy_batch_write_at: 1\n"
            "  pass_through_endpoints:\n"
            "    - path: /azure-search\n"
            f"      target: {wire.url}\n"
            "      headers:\n"
            "        Authorization: Bearer synthetic-search-key\n"
            "router_settings:\n"
            "  disable_cooldowns: true\n"
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request("POST", "/azure-search", {"search": "hello"})
            assert response.status_code == 500, response.text
            call_id: Final = response.headers["x-litellm-call-id"]
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT status, call_type, api_base, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (call_id,),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert rows[0]["status"] == "failure", rows
            assert rows[0]["call_type"] == "pass_through_endpoint", rows
            assert "127.0.0.1" in rows[0]["api_base"], rows
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/")]
