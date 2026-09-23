import json
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_UPSTREAM_ERROR: Final[dict[str, JsonValue]] = {
    "error": {
        "code": 404,
        "message": "Publisher Model `publishers/anthropic/models/claude-nope-9` was not found or your project does not have access to it. Please ensure you are using a valid model version.",
        "status": "NOT_FOUND",
    }
}


@pytest.mark.covers("observability.passthrough.upstream_error_body_logged_and_in_spend_log")
def test_gemini_passthrough_upstream_error_body_reaches_proxy_log_and_spend_row(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=404, body=json.dumps(_UPSTREAM_ERROR).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "gemini-passthrough.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST",
                "/gemini/v1beta/models/claude-nope-9:generateContent",
                {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                headers={"x-goog-api-key": candidate.key},
            )
            assert response.status_code == 404, response.text
            assert response.json() == _UPSTREAM_ERROR, response.text
            try:
                eventually(
                    lambda: owned.log.read_text(),
                    lambda text: "was not found or your project" in text,
                    seconds=30,
                )
            except AssertionError:
                pytest.fail(
                    f"upstream 404 body never reached the proxy log after {response.status_code} passthrough; "
                    f"log tail: {owned.log.read_text()[-2000:]}"
                )
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (response.headers["x-litellm-call-id"],),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            metadata: Final = rows[0]["metadata"]
            parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
            error_information: Final = object_value(parsed["error_information"])
            assert "was not found or your project" in str(error_information["error_message"]), response.text
            assert error_information["error_code"] == "404", response.text
