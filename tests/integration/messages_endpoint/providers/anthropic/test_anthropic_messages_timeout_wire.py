import json
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.process import owned_proxy
from integration._support.timing import assert_cut_short, timed
from integration._support.wire import Reply, Request, wire_server

_UPSTREAM_STALL_SECONDS: Final = 4.0
_CONFIGURED_TIMEOUT_SECONDS: Final = 1.0


@pytest.mark.covers("providers.anthropic_messages.configured_timeout_aborts_stalled_upstream")
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("timeout_source", ("deployment", "request_timeout"))
def test_messages_endpoint_honors_configured_timeout_against_stalled_upstream(
    gateway: Gateway, tmp_path: Path, stream: bool, timeout_source: str
) -> None:
    prompt: Final = "stall-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-anthropic-key"
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body == {
            "model": "claude-sonnet-5-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "stream": stream,
        }
        time.sleep(_UPSTREAM_STALL_SECONDS)
        return Reply(
            body=json.dumps(
                {
                    "id": "msg_stalled",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [{"type": "text", "text": "too late"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            ).encode()
        )

    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = JSON_OBJECT.validate_python(config["litellm_settings"])
    config_path: Final = tmp_path / "request-timeout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                **config,
                "litellm_settings": {**settings, "request_timeout": _CONFIGURED_TIMEOUT_SECONDS},
            }
        )
    )
    context: Final = (
        owned_proxy(gateway, tmp_path, {}, config=config_path)
        if timeout_source == "request_timeout"
        else nullcontext(gateway)
    )
    with context as proxy, wire_server(respond) as wire, proxy.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5-5",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            model_info={},
            **({"timeout": _CONFIGURED_TIMEOUT_SECONDS} if timeout_source == "deployment" else {}),
        )
        response: Final = timed(
            lambda: proxy.request(
                "POST",
                "/v1/messages",
                {"model": model, "max_tokens": 16, "stream": stream, "messages": [{"role": "user", "content": prompt}]},
            )
        )
        assert response.value.status_code == 408, response.value.text
        assert_cut_short(response.seconds, injected_delay=_UPSTREAM_STALL_SECONDS)
        assert len(wire.drain()) == 1
