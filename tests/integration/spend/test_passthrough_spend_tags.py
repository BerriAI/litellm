import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "claude-sonnet-4-5-20250929"
SENT_HEADERS: Final = {"user-agent": "claude-cli/2.0.0", "x-tenant-id": "tenant-a"}
EXPECTED_TAGS: Final = ["User-Agent: claude-cli", "User-Agent: claude-cli/2.0.0", "x-tenant-id: tenant-a"]


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/messages", request.target
    return Reply(
        body=json.dumps(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "tagged"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ).encode()
    )


def _request_tags(key: str) -> list[list[str]]:
    rows: Final = read_rows(
        'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (sha256(key.encode()).hexdigest(),)
    )
    return [
        json.loads(row["request_tags"]) if isinstance(row["request_tags"], str) else row["request_tags"] for row in rows
    ]


@pytest.mark.parametrize(
    "route", [pytest.param("/anthropic/v1/messages", id="passthrough"), pytest.param("/v1/messages", id="unified")]
)
def test_header_derived_spend_tags_are_recorded_on_anthropic_messages_routes(
    gateway: Gateway, tmp_path: Path, route: str
) -> None:
    with wire_server(_respond) as wire:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"]["extra_spend_tag_headers"] = ["x-tenant-id"]
        path: Final = tmp_path / "spend-tag-headers.yaml"
        path.write_text(yaml.safe_dump(config))
        environment: Final = {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": "synthetic-anthropic-key"}
        with owned_proxy(gateway, tmp_path, environment, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model=f"anthropic/{MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            key: Final = scenario.key()
            response: Final = candidate.request(
                "POST",
                route,
                {
                    "model": MODEL if route == "/anthropic/v1/messages" else model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "tag me"}],
                },
                key=key,
                headers={**SENT_HEADERS, "anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert len(wire.drain()) == 1
            assert eventually(lambda: _request_tags(key), lambda tags: len(tags) == 1, seconds=70) == [EXPECTED_TAGS]
