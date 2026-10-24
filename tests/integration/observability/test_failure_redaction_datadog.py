import gzip
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, JsonValue, eventually, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server


def _body(batch: Request) -> bytes:
    return gzip.decompress(batch.body) if batch.body[:2] == b"\x1f\x8b" else batch.body


def _provider(request: Request) -> Reply:
    try:
        body: Final = json.loads(request.body)
    except json.JSONDecodeError:
        return Reply(status=404, body=b"{}")
    text: Final = body["messages"][-1]["content"]
    return Reply(
        status=400,
        body=json.dumps(
            {"error": {"type": "invalid_request_error", "message": f"Unsupported content: {text}"}}
        ).encode(),
    )


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    sink: Wire

    def log_entries(self, model: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            object_value(entry)
            for batch in self.sink.drain()
            for entry in json.loads(_body(batch))
            if model in json.dumps(entry)
        )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("failure_redaction_datadog")
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update(
        {"callbacks": ["datadog"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1, "turn_off_message_logging": True}
    )
    path: Final = root / "datadog_failure.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(_provider) as provider,
        wire_server(lambda _: Reply()) as sink,
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway,
            root,
            {"DD_API_KEY": "synthetic-dd-key", "DD_BASE_URL": sink.url, "DD_SITE": "localhost"},
            config=path,
            workers=2,
        ) as proxy,
    ):
        yield Rig(proxy, provider, sink)


def test_c5_datadog_failure_log_redacted_keeps_status(rig: Rig) -> None:
    secret: Final = "dd-secret-" + uuid.uuid4().hex
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": secret}]},
        )
        assert response.status_code == 400, response.text
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        entries: Final = eventually(lambda: rig.log_entries(model), lambda values: len(values) >= 1, seconds=30)
        entry: Final = entries[0]
        assert secret not in json.dumps(entry), json.dumps(entry)[:2000]
        message: Final = object_value(json.loads(str(entry.get("message", "{}"))))
        assert secret not in json.dumps(message), json.dumps(message)[:2000]
        error_information: Final = object_value(message.get("error_information"))
        assert error_information.get("error_class"), error_information
