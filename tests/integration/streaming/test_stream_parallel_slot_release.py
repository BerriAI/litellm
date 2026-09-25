import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


def frame(identity: str, delta: dict[str, str], *, finish: str | None = None) -> bytes:
    event: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(event).encode() + b"\n\n"


@pytest.mark.covers("streaming.max_parallel_requests.slot_released_when_stream_logging_callback_fails")
def test_failing_stream_logging_callback_does_not_leak_max_parallel_requests_slot(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "stream-slot-" + uuid.uuid4().hex
    prompt: Final = "slot release control " + identity

    def analyzer(request: Request) -> Reply:
        assert request.target == "/analyze"
        assert json.loads(request.body)["text"] == prompt
        return Reply(status=500, body=json.dumps({"error": "synthetic analyzer outage"}).encode())

    def provider(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions"
        body: Final = json.loads(request.body)
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert body["stream"] is True
        return Reply(
            content_type="text/event-stream",
            chunks=(
                frame(identity, {"role": "assistant", "content": "Hello"}),
                frame(identity, {"content": " slot"}),
                frame(identity, {}, finish="stop"),
                b"data: [DONE]\n\n",
            ),
        )

    with wire_server(analyzer) as policy, wire_server(provider) as upstream:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "presidio",
                    "mode": "logging_only",
                    "default_on": True,
                    "presidio_filter_scope": "input",
                    "pii_entities_config": {"EMAIL_ADDRESS": "MASK"},
                    "presidio_analyzer_api_base": policy.url + "/",
                    "presidio_anonymizer_api_base": policy.url + "/",
                },
            }
        ]
        path: Final = tmp_path / "failing_logging_guardrail.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(api_base=upstream.url + "/v1")
            key: Final = scenario.key(max_parallel_requests=1)
            body: Final = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
            first: Final = candidate.request("POST", "/v1/chat/completions", body, key=key)
            assert first.status_code == 200, first.text
            assert first.text.endswith("data: [DONE]\n\n"), first.text
            assert len(upstream.drain()) == 1
            eventually(lambda: policy.received.qsize(), lambda count: count >= 1)
            assert {scan.target for scan in policy.drain()} == {"/analyze"}
            second: Final = eventually(
                lambda: candidate.request("POST", "/v1/chat/completions", body, key=key),
                lambda response: response.status_code == 200,
                seconds=20,
                return_last_on_timeout=True,
            )
            assert second.status_code == 200, second.text
            assert second.text.endswith("data: [DONE]\n\n"), second.text
