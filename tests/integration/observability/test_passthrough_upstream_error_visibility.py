import json
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

_GEMINI_STREAM_PATH: Final = "/gemini/v1beta/models/claude-nope-9:streamGenerateContent"

def _gemini_config(path: Path, wire_url: str) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["environment_variables"] = {"GEMINI_API_BASE": wire_url, "GEMINI_API_KEY": "scripted"}
    path.write_text(yaml.safe_dump(config))

def _gemini_headers(candidate: Gateway) -> dict[str, str]:
    return {"Authorization": f"Bearer {candidate.key}", "x-goog-api-key": candidate.key}

@pytest.mark.covers("other.observability.passthrough.streaming_tags_reach_spend")
def test_gemini_streaming_passthrough_request_tags_reach_spend_log(gateway: Gateway, tmp_path: Path) -> None:
    expected_body: Final = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    chunks: Final = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"hello"}],"role":"model"},"index":0}]}\n\n',
        b'data: {"candidates":[{"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}\n\n',
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        assert request.target.split("?", 1)[0] == "/v1beta/models/claude-nope-9:streamGenerateContent", request
        assert json.loads(request.body) == expected_body, request
        return Reply(status=200, chunks=chunks, content_type="text/event-stream")

    config_path: Final = tmp_path / "gemini-stream-tags.yaml"
    with wire_server(respond) as wire:
        _gemini_config(config_path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=config_path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=expected_body,
                headers={**_gemini_headers(candidate), "tags": "gemini-js-sdk,pass-through-endpoint"},
            ) as response:
                assert response.status_code == 200
                streamed: Final = response.read()
                call_id: Final = response.headers["x-litellm-call-id"]
            assert streamed == b"".join(chunks)

            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT request_tags, call_type, status FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (call_id,),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            row: Final = rows[0]
            raw_tags: Final = row["request_tags"]
            request_tags: Final = json.loads(raw_tags) if isinstance(raw_tags, str) else raw_tags
            policy_tags: Final = [
                tag
                for tag in request_tags
                if not (isinstance(tag, str) and tag.startswith("User-Agent: "))
            ]
            assert policy_tags == ["gemini-js-sdk", "pass-through-endpoint"], row
            assert row["call_type"] == "pass_through_endpoint", row
            assert row["status"] == "success", row
