import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from integration._support import claude_code as cc


def _error_529() -> Reply:
    return Reply(
        status=529,
        body=json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}).encode(),
    )


def test_overloaded_primary_falls_back_to_second_deployment(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = f"msg_fb_{uuid.uuid4().hex}"
    request_body: Final = cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}")

    def respond_fallback(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(identity, cc.SONNET, "PONG", {"input_tokens": 12, "output_tokens": 4}),
        )

    with (
        wire_server(lambda request: _error_529()) as primary,
        wire_server(respond_fallback) as fallback,
    ):
        config: Final = {
            **yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()),
            "model_list": [
                {
                    "model_name": "cc-primary",
                    "litellm_params": {
                        "model": f"anthropic/{cc.SONNET}",
                        "api_key": cc.ANTHROPIC_API_KEY,
                        "api_base": primary.url,
                        "model_info": {"id": "primary-cc"},
                    },
                },
                {
                    "model_name": "cc-fallback-group",
                    "litellm_params": {
                        "model": f"anthropic/{cc.SONNET}",
                        "api_key": cc.ANTHROPIC_API_KEY,
                        "api_base": fallback.url,
                        "model_info": {"id": "fallback-cc"},
                    },
                },
            ],
            "router_settings": {
                "num_retries": 0,
                "disable_cooldowns": True,
                "fallbacks": [{"cc-primary": ["cc-fallback-group"]}],
            },
        }
        path: Final = tmp_path / "fallbacks.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(
            gateway, tmp_path, {"REDIS_HOST": "127.0.0.1", "REDIS_PORT": "6379"}, config=path
        ) as candidate:
            with candidate.client.stream(
                "POST",
                "/v1/messages",
                params={"beta": "true"},
                json={**request_body, "model": "cc-primary"},
                headers={**cc.cli_headers(candidate.key), "authorization": f"Bearer {candidate.key}"},
            ) as response:
                assert response.status_code == 200, response.status_code
                body: Final = "".join(response.iter_text())
            assert "message_stop" in body, body
            assert "PONG" in body, body
            deployments: Final = candidate.get("/model/info")["data"]
            fallback_id: Final = next(
                entry["model_info"]["id"]
                for entry in deployments
                if entry["litellm_params"]["api_base"] == fallback.url
            )
            assert response.headers.get("x-litellm-model-id") == fallback_id, dict(response.headers)
            assert len(primary.drain()) == 1
            assert len(fallback.drain()) == 1
            rows: Final = eventually(
                lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert len(rows) == 1
