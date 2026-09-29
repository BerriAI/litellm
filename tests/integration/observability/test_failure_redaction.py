import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, JsonValue, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    sink: Wire
    batches: list[Request]

    def failure_events(self, model: str) -> tuple[dict[str, JsonValue], ...]:
        self.batches.extend(self.sink.drain())
        return tuple(
            object_value(event)
            for batch in self.batches
            for event in json.loads(batch.body)
            if model in json.dumps(event)
        )


def _provider(request: Request) -> Reply:
    text: Final = json.loads(request.body)["messages"][-1]["content"]
    return Reply(
        status=400,
        body=json.dumps(
            {"error": {"type": "invalid_request_error", "message": f"Unsupported content: {text}"}}
        ).encode(),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("failure_redaction")
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update(
        {
            "callbacks": ["generic_api"],
            "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
            "turn_off_message_logging": True,
            "standard_logging_payload_excluded_fields": ["hidden_params"],
        }
    )
    path: Final = root / "failure_redaction.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        gateway_from_environment() as gateway,
        wire_server(_provider) as provider,
        wire_server(lambda _: Reply()) as sink,
        owned_proxy(gateway, root, {"GENERIC_LOGGER_ENDPOINT": sink.url}, config=path) as proxy,
    ):
        yield Rig(proxy, provider, sink, [])  # mutable-ok: sink drain consumes batches, later polls keep earlier ones


def _secret_prompt() -> str:
    return "confidential-prompt-" + uuid.uuid4().hex


def _single_failure_event(rig: Rig, model: str) -> dict[str, JsonValue]:
    events: Final = eventually(lambda: rig.failure_events(model), lambda values: len(values) >= 1, seconds=20)
    assert len(events) == 1, events
    assert events[0]["status"] == "failure", events[0]
    return events[0]


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT status, messages, response, proxy_server_request, metadata FROM "LiteLLM_SpendLogs" '
            "WHERE request_id=%s",
            (call_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _chat(rig: Rig, model: str, messages: list[JsonValue], key: str | None = None) -> httpx.Response:
    return rig.proxy.request("POST", "/v1/chat/completions", {"model": model, "messages": messages}, key=key)


def _error_information(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(event["error_information"])


def test_provider_error_echoing_the_prompt_is_redacted_in_callbacks_and_spend_logs(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(
            api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key", num_retries=0
        )
        response: Final = _chat(rig, model, [{"role": "user", "content": secret}])
        assert response.status_code == 400, response.text
        assert any(secret.encode() in request.body for request in rig.provider.drain())
        event: Final = _single_failure_event(rig, model)
        assert secret not in json.dumps(event), json.dumps(event)
        error_information: Final = _error_information(event)
        assert error_information["error_class"] == "BadRequestError", error_information
        assert error_information["error_code"] == "400", error_information
        assert error_information["llm_provider"] == "openai", error_information
        assert "hidden_params" not in event, sorted(event)
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert row["status"] == "failure", row
        assert secret not in json.dumps(row, default=str), row
        persisted: Final = object_value(
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        )
        assert object_value(persisted["error_information"])["error_class"] == "BadRequestError", persisted


def test_transformation_error_quoting_the_prompt_is_redacted_in_callbacks(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5", api_base=rig.provider.url, api_key="synthetic-anthropic-key"
        )
        response: Final = _chat(
            rig,
            model,
            [
                {"role": "user", "content": secret},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": 12345, "type": "function", "function": {"name": None, "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": 12345, "content": secret},
            ],
        )
        assert response.status_code == 400, response.text
        assert rig.provider.drain() == ()
        event: Final = _single_failure_event(rig, model)
        assert secret not in json.dumps(event), json.dumps(event)
        error_information: Final = _error_information(event)
        assert error_information["error_code"] == "400", error_information
        assert error_information["error_class"], error_information


def test_proxy_only_rejection_does_not_leak_the_prompt_to_callbacks(rig: Rig) -> None:
    secret: Final = _secret_prompt()
    with rig.proxy.scenario() as scenario:
        allowed: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        denied: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[allowed])
        response: Final = _chat(rig, denied, [{"role": "user", "content": secret}], key=key)
        assert response.status_code in (401, 403), response.text
        assert rig.provider.drain() == ()
        event: Final = _single_failure_event(rig, denied)
        assert secret not in json.dumps(event), json.dumps(event)
        assert "hidden_params" not in event, sorted(event)
        assert _error_information(event)["error_code"] == str(response.status_code), event
