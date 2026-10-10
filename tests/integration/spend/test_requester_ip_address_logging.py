import json
import re
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

FORWARDED_IP: Final = "203.0.113.74"
PROXY_HOP: Final = "198.51.100.23"
REAL_IP: Final = "203.0.113.75"
SETTING: Final = "disable_requester_ip_address_logging"
AKTO_KEY: Final = "synthetic-akto-key"
AKTO_GUARDRAIL: Final = "akto-requester-ip"
PROVIDER_FAILURE: Final = "synthetic-requester-ip-provider-failure"
LONG_PROVIDER_FAILURE: Final = "synthetic-requester-ip-long-provider-failure"
UPSTREAM_IP: Final = "192.0.2.61"
CALLBACK_SECRET: Final = "synthetic-callback-sink-secret"
_ROW_COLUMNS: Final = (
    "SELECT s.request_id, s.metadata, s.requester_ip_address, row_to_json(s)::text AS row_text "
    'FROM "LiteLLM_SpendLogs" s'
)


def _config(tmp_path: Path, general_settings: Mapping[str, JsonValue], **sections: JsonValue) -> Path:
    config: Final = tmp_path / f"requester-ip-{uuid4()}.json"
    config.write_text(
        json.dumps(
            {
                **sections,
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "use_x_forwarded_for": True,
                    **general_settings,
                },
            }
        )
    )
    return config


def _caller(scenario: Scenario) -> tuple[str, str, str]:
    model: Final = scenario.model()
    user_id: Final = scenario.user(user_role="internal_user")
    return model, user_id, scenario.key(user_id=user_id, models=[model])


def _chat(proxy: Gateway, model: str, key: str) -> str:
    response: Final = proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "requester ip"}]},
        key=key,
        headers={"x-forwarded-for": FORWARDED_IP},
    )
    assert response.status_code == 200, response.text
    return string_value(object_value(response.json())["id"])


def _stored_row(request_id: str, database_url: str | None = None) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(f"{_ROW_COLUMNS} WHERE s.request_id=%s", (request_id,), database_url=database_url),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _served_row(proxy: Gateway, key: str, request_id: str, since: datetime) -> tuple[dict[str, JsonValue], str]:
    response: Final = proxy.request(
        "GET",
        "/spend/logs/ui",
        key=key,
        params={
            "request_id": request_id,
            "start_date": (since - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": (since + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    assert response.status_code == 200, response.text
    served: Final = tuple(
        object_value(row)
        for row in list_value(object_value(response.json())["data"])
        if object_value(row)["request_id"] == request_id
    )
    assert len(served) == 1, response.text
    return served[0], response.text


def _assert_recorded(row: Mapping[str, JsonValue], user_id: str) -> None:
    metadata: Final = object_value(row["metadata"])
    assert metadata["user_api_key_user_id"] == user_id, metadata
    assert row["requester_ip_address"] == FORWARDED_IP, row
    assert metadata["requester_ip_address"] == FORWARDED_IP, metadata


def _assert_not_recorded(row: Mapping[str, JsonValue], text: str, user_id: str) -> None:
    metadata: Final = object_value(row["metadata"])
    assert metadata["user_api_key_user_id"] == user_id, metadata
    assert row["requester_ip_address"] is None, row
    assert metadata.get("requester_ip_address") is None, metadata
    assert FORWARDED_IP not in text, text


def test_allowed_ips_denial_row_masks_the_ip_while_the_caller_still_sees_it(gateway: Gateway, tmp_path: Path) -> None:
    user_agent: Final = f"requester-ip-allowed-ips-{uuid4()}"
    config: Final = _config(tmp_path, {SETTING: True, "allowed_ips": ["127.0.0.1"]})
    with owned_proxy(gateway, tmp_path, {}, config=config) as proxy, proxy.scenario() as scenario:
        model, _, key = _caller(scenario)
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "requester ip"}]},
            key=key,
            headers={"x-forwarded-for": FORWARDED_IP, "user-agent": user_agent},
        )
        assert response.status_code == 403, response.text
        assert FORWARDED_IP in response.text, response.text
        rows: Final = eventually(
            lambda: read_rows(f"{_ROW_COLUMNS} WHERE s.metadata->>'user_agent' = %s", (user_agent,)),
            lambda values: len(values) == 1,
            seconds=70,
        )

    denied: Final = rows[0]
    error_information: Final = object_value(object_value(denied["metadata"])["error_information"])
    assert object_value(denied["metadata"])["status"] == "failure", denied
    assert denied["requester_ip_address"] is None, denied
    assert "IP address REDACTED_BY_LITELM not allowed" in string_value(error_information["error_message"]), denied
    assert FORWARDED_IP not in string_value(denied["row_text"]), denied


def test_admin_setting_change_applies_to_the_next_request(gateway: Gateway, tmp_path: Path) -> None:
    since: Final = datetime.now(UTC)
    with (
        scratch_database() as database_url,
        owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=_config(tmp_path, {}),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as proxy,
        proxy.scenario() as scenario,
    ):
        model, user_id, key = _caller(scenario)
        before: Final = _chat(proxy, model, key)
        _assert_recorded(_stored_row(before, database_url), user_id)

        proxy.post(
            "/config/field/update",
            {"field_name": SETTING, "field_value": True, "config_type": "general_settings"},
        )
        listed: Final = tuple(
            object_value(field)
            for field in list_value(
                proxy.request("GET", "/config/list", params={"config_type": "general_settings"}).json()
            )
            if object_value(field)["field_name"] == SETTING
        )
        assert len(listed) == 1, listed
        assert listed[0]["field_type"] == "Boolean" and listed[0]["field_value"] is True, listed

        disabled: Final = _chat(proxy, model, key)
        stored_disabled: Final = _stored_row(disabled, database_url)
        served_disabled, served_disabled_text = _served_row(proxy, key, disabled, since)

        proxy.post(
            "/config/field/update",
            {"field_name": SETTING, "field_value": False, "config_type": "general_settings"},
        )
        enabled: Final = _chat(proxy, model, key)
        stored_enabled: Final = _stored_row(enabled, database_url)

    _assert_not_recorded(stored_disabled, string_value(stored_disabled["row_text"]), user_id)
    _assert_not_recorded(served_disabled, served_disabled_text, user_id)
    _assert_recorded(stored_enabled, user_id)


def _completion_reply(marker: str, streaming: bool) -> Reply:
    if not streaming:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + marker,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )
    chunks: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": marker}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )
    frames: Final = tuple(
        (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-" + marker,
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    **chunk,
                }
            )
            + "\n\n"
        ).encode()
        for chunk in chunks
    )
    return Reply(chunks=(*frames, b"data: [DONE]\n\n"), content_type="text/event-stream")


def _provider(request: Request) -> Reply:
    if request.method == "GET" and request.target.endswith("/v1/models"):
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    body: Final = object_value(json.loads(request.body))
    if request.target.endswith("/v1/embeddings"):
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                }
            ).encode()
        )
    if request.target.endswith("/v1/messages"):
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_{uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "content": [{"type": "text", "text": "requester ip"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ).encode()
        )
    if request.target.endswith("/v1/responses"):
        identity: Final = uuid4().hex
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg_{identity}",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "requester ip", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )
    assert request.target.endswith("/v1/chat/completions"), request.target
    if LONG_PROVIDER_FAILURE.encode() in request.body:
        return Reply(
            status=400,
            body=json.dumps(
                {"error": {"message": f"upstream {UPSTREAM_IP} " * 400, "type": "invalid_request_error"}}
            ).encode(),
        )
    if PROVIDER_FAILURE.encode() in request.body:
        return Reply(
            status=400,
            body=json.dumps({"error": {"message": PROVIDER_FAILURE, "type": "invalid_request_error"}}).encode(),
        )
    return _completion_reply(uuid4().hex, body.get("stream") is True)


def _akto_allows(_: Request) -> Reply:
    return Reply(body=json.dumps({"data": {"guardrailsResult": {"Allowed": True}}}).encode())


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    akto: Wire
    callbacks: Wire
    key: str
    chat_model: str
    claude_model: str
    embedding_model: str

    def send(self, path: str, body: Mapping[str, JsonValue], forwarded_for: str) -> str:
        response: Final = self.proxy.request(
            "POST", path, body, key=self.key, headers={"x-forwarded-for": forwarded_for, "x-real-ip": REAL_IP}
        )
        assert response.status_code == 200, response.text
        return response.headers["x-litellm-call-id"]

    def callback_metadata(self, call_id: str) -> dict[str, JsonValue]:
        received: Final[
            list[Request]
        ] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

        def delivered() -> tuple[dict[str, JsonValue], ...]:
            received.extend(self.callbacks.drain())
            return tuple(
                event
                for batch in received
                for event in (object_value(item) for item in list_value(json.loads(batch.body)))
                if event.get("litellm_call_id") == call_id
            )

        events: Final = eventually(delivered, lambda values: len(values) == 1, seconds=20)
        return object_value(events[0]["metadata"])


def _row_for_call(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(f"{_ROW_COLUMNS} WHERE s.litellm_call_id=%s", (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _rig(stack: ExitStack, gateway: Gateway, root: Path, provider: Wire, disabled: bool) -> Rig:
    akto: Final = stack.enter_context(wire_server(_akto_allows))
    callbacks: Final = stack.enter_context(wire_server(lambda _: Reply()))
    config: Final = _config(
        root,
        {SETTING: True, "store_prompts_in_spend_logs": True} if disabled else {"store_prompts_in_spend_logs": True},
        litellm_settings={"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1},
        guardrails=[
            {
                "guardrail_name": AKTO_GUARDRAIL,
                "litellm_params": {
                    "guardrail": "akto",
                    "mode": "pre_call",
                    "default_on": False,
                    "akto_base_url": akto.url,
                    "akto_api_key": AKTO_KEY,
                },
            }
        ],
    )
    proxy: Final = stack.enter_context(
        owned_proxy(
            gateway,
            root,
            {
                "GENERIC_LOGGER_ENDPOINT": callbacks.url,
                "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {CALLBACK_SECRET}",
            },
            config=config,
            workers=2,
        )
    )
    scenario: Final = stack.enter_context(proxy.scenario())
    chat: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-openai-key")
    claude: Final = scenario.model(
        model="anthropic/claude-haiku-5-5", api_base=provider.url, api_key="synthetic-anthropic-key"
    )
    embedding: Final = scenario.model(
        model="openai/text-embedding-3-small",
        api_base=provider.url + "/v1",
        api_key="synthetic-openai-key",
        model_info={"mode": "embedding"},
    )
    return Rig(proxy, akto, callbacks, scenario.key(models=[chat, claude, embedding]), chat, claude, embedding)


@pytest.fixture(scope="module")
def rigs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Mapping[bool, Rig]]:
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        provider: Final = stack.enter_context(wire_server(_provider))
        yield MappingProxyType(
            {
                disabled: _rig(stack, gateway, tmp_path_factory.mktemp("requester-ip"), provider, disabled)
                for disabled in (False, True)
            }
        )


def _endpoint_request(rig: Rig, endpoint: str, marker: str) -> tuple[str, dict[str, JsonValue]]:
    if endpoint == "embeddings":
        return "/v1/embeddings", {"model": rig.embedding_model, "input": marker}
    if endpoint == "responses":
        return "/v1/responses", {"model": rig.chat_model, "input": marker}
    if endpoint == "messages":
        return "/v1/messages", {
            "model": rig.claude_model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": marker}],
        }
    return "/v1/chat/completions", {
        "model": rig.chat_model,
        "messages": [{"role": "user", "content": marker}],
        "stream": endpoint == "chat-stream",
    }


def _ips_in_row(row: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(ip for ip in (FORWARDED_IP, PROXY_HOP, REAL_IP) if ip in string_value(row["row_text"]))


DISABLED: Final = pytest.mark.parametrize("disabled", [False, True], ids=["setting-unset", "setting-on"])


@pytest.mark.parametrize("forwarded_for", [FORWARDED_IP, f"{FORWARDED_IP}, {PROXY_HOP}"], ids=["single", "chain"])
@pytest.mark.parametrize("endpoint", ["chat", "chat-stream", "messages", "responses", "embeddings"])
@DISABLED
def test_spend_logs_follow_the_setting_on_every_endpoint_while_callbacks_keep_the_ip(
    rigs: Mapping[bool, Rig], disabled: bool, endpoint: str, forwarded_for: str
) -> None:
    rig: Final = rigs[disabled]
    path, body = _endpoint_request(rig, endpoint, f"requester-ip-{uuid4().hex}")
    call_id: Final = rig.send(path, body, forwarded_for)

    row: Final = _row_for_call(call_id)
    callback_metadata: Final = rig.callback_metadata(call_id)

    recorded_ip: Final = None if disabled else forwarded_for
    assert callback_metadata["requester_ip_address"] == forwarded_for, callback_metadata
    assert row["requester_ip_address"] == recorded_ip, row
    assert object_value(row["metadata"])["requester_ip_address"] == recorded_ip, row
    assert (REAL_IP in _ips_in_row(row)) is not disabled, row
    assert (_ips_in_row(row) == ()) is disabled, row


@DISABLED
def test_logs_page_serves_no_ip_when_the_setting_is_on(rigs: Mapping[bool, Rig], disabled: bool) -> None:
    rig: Final = rigs[disabled]
    since: Final = datetime.now(UTC)
    call_id: Final = rig.send(
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": f"requester-ip-{uuid4().hex}"}]},
        FORWARDED_IP,
    )
    request_id: Final = string_value(_row_for_call(call_id)["request_id"])
    window: Final = {
        "start_date": (since - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": (since + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
    }

    listed: Final = rig.proxy.request("GET", "/spend/logs/ui", params={"request_id": request_id, **window})
    detail: Final = rig.proxy.request("GET", f"/spend/logs/ui/{request_id}", params=window)

    assert listed.status_code == 200 and detail.status_code == 200, (listed.text, detail.text)
    assert request_id in listed.text and request_id in detail.text, (listed.text, detail.text)
    served: Final = listed.text + detail.text
    assert (FORWARDED_IP in served) is not disabled, served
    assert (REAL_IP in served) is not disabled, served


@DISABLED
def test_auth_failure_row_follows_the_setting(rigs: Mapping[bool, Rig], disabled: bool) -> None:
    rig: Final = rigs[disabled]
    user_agent: Final = f"requester-ip-auth-failure-{uuid4()}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": "requester ip"}]},
        key=f"sk-requester-ip-{uuid4().hex}",
        headers={"x-forwarded-for": FORWARDED_IP, "user-agent": user_agent},
    )
    assert response.status_code == 401, response.text

    rows: Final = eventually(
        lambda: read_rows(f"{_ROW_COLUMNS} WHERE s.metadata->>'user_agent' = %s", (user_agent,)),
        lambda values: len(values) == 1,
        seconds=70,
    )

    assert object_value(rows[0]["metadata"])["status"] == "failure", rows[0]
    assert rows[0]["requester_ip_address"] == (None if disabled else FORWARDED_IP), rows[0]
    assert _ips_in_row(rows[0]) == (() if disabled else (FORWARDED_IP,)), rows[0]


@DISABLED
def test_provider_failure_row_follows_the_setting(rigs: Mapping[bool, Rig], disabled: bool) -> None:
    rig: Final = rigs[disabled]
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": PROVIDER_FAILURE}]},
        key=rig.key,
        headers={"x-forwarded-for": FORWARDED_IP, "x-real-ip": REAL_IP},
    )
    assert response.status_code == 400, response.text

    row: Final = _row_for_call(response.headers["x-litellm-call-id"])

    assert object_value(row["metadata"])["status"] == "failure", row
    assert row["requester_ip_address"] == (None if disabled else FORWARDED_IP), row
    assert _ips_in_row(row) == (() if disabled else (FORWARDED_IP, REAL_IP)), row


def test_long_provider_error_row_keeps_no_ip_fragment_at_the_truncation_cut(rigs: Mapping[bool, Rig]) -> None:
    rig: Final = rigs[True]
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": LONG_PROVIDER_FAILURE}]},
        key=rig.key,
        headers={"x-forwarded-for": FORWARDED_IP},
    )
    assert response.status_code == 400, response.text

    row: Final = _row_for_call(response.headers["x-litellm-call-id"])
    error_message: Final = string_value(
        object_value(object_value(row["metadata"])["error_information"])["error_message"]
    )
    cut: Final = re.search(r"\.\.\. \(litellm_truncated .*?\) \.\.\.", error_message)

    assert cut is not None, error_message
    assert re.search(r"[\d.]$", error_message[: cut.start()]) is None, error_message[: cut.start()][-60:]
    assert re.search(r"^[\d.]", error_message[cut.end() :]) is None, error_message[cut.end() :][:60]
    assert UPSTREAM_IP not in string_value(row["row_text"]), row


@DISABLED
def test_pre_call_guardrail_still_receives_the_ip(rigs: Mapping[bool, Rig], disabled: bool) -> None:
    rig: Final = rigs[disabled]
    marker: Final = f"requester-ip-{uuid4().hex}"
    call_id: Final = rig.send(
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": marker}], "guardrails": [AKTO_GUARDRAIL]},
        f"{FORWARDED_IP}, {PROXY_HOP}",
    )

    row: Final = _row_for_call(call_id)
    akto_calls: Final = tuple(
        object_value(json.loads(call.body)) for call in rig.akto.drain() if marker.encode() in call.body
    )

    assert [call["ip"] for call in akto_calls] == [FORWARDED_IP], akto_calls
    assert object_value(row["metadata"])["applied_guardrails"] == [AKTO_GUARDRAIL], row
