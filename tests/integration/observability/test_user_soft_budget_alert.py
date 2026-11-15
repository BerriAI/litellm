import asyncio
import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from anthropic import Anthropic, AsyncAnthropic
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    held,
    object_value,
    string_value,
)
from integration._support.mail import Mailbox, smtp_sink
from integration._support.ports import allocate_port
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 240), pytest.mark.shares_worker_state]

ALERT_SECONDS: Final = 40
QUIET_SECONDS: Final = 3
SOFT_BUDGET: Final = 0.5
SPEND: Final = 1.0
REPLY: Final = "soft budget reply"
RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "id": "msg_$UNIQUE_ID",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": REPLY, "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40},
}
RESPONSE_EVENTS: Final[tuple[Mapping[str, JsonValue], ...]] = (
    {"type": "response.created", "sequence_number": 0, "response": {**RESPONSE, "status": "in_progress", "output": []}},
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_$UNIQUE_ID",
        "output_index": 0,
        "content_index": 0,
        "delta": REPLY,
    },
    {"type": "response.completed", "sequence_number": 2, "response": RESPONSE},
)


class AlertLog:
    """Every body the alert sink received so far; reading it never loses an earlier alert."""

    def __init__(self, wire: Wire) -> None:
        self.wire: Final = wire
        self._bodies: tuple[str, ...] = ()

    def bodies(self) -> tuple[str, ...]:
        self._bodies = (*self._bodies, *(request.body.decode() for request in self.wire.drain()))
        return self._bodies

    def naming(self, marker: str) -> tuple[str, ...]:
        return tuple(body for body in self.bodies() if marker in body)


@dataclass(frozen=True, slots=True)
class SinkControls:
    refusing: threading.Event
    stalling: threading.Event
    release: threading.Event


@dataclass(frozen=True, slots=True)
class AlertingProxy:
    gateway: Gateway
    alerts: AlertLog
    sink: SinkControls


@dataclass(frozen=True, slots=True)
class MailingProxy:
    gateway: Gateway
    webhooks: AlertLog
    mailbox: Mailbox


@dataclass(frozen=True, slots=True)
class ChaosProxy:
    proxy: OwnedProxy
    sink_port: int


def alert_reply(controls: SinkControls) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if controls.refusing.is_set():
            return Reply(status=403, body=b"invalid_token", content_type="text/plain")
        if controls.stalling.is_set():
            return Reply(chunks=(b"ok",), content_type="text/plain", gate_before_first=controls.release)
        return Reply(body=b"ok", content_type="text/plain")

    return respond


def accept_alert(request: Request) -> Reply:
    return Reply(body=b"ok", content_type="text/plain")


def alerting_config(directory: Path, name: str, **general_settings: JsonValue) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **object_value(base["general_settings"]),
            "alerting": ["slack"],
            "alert_types": ["budget_alerts"],
            **general_settings,
        },
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def slack_alerting(directory: Path, name: str, **general_settings: JsonValue) -> Iterator[AlertingProxy]:
    controls: Final = SinkControls(threading.Event(), threading.Event(), threading.Event())
    with gateway_from_environment() as shared, wire_server(alert_reply(controls)) as alerts:
        with owned_proxy(
            shared,
            directory,
            {"SLACK_WEBHOOK_URL": alerts.url, "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"},
            config=alerting_config(directory, name, **general_settings),
        ) as gateway:
            yield AlertingProxy(gateway, AlertLog(alerts), controls)


@pytest.fixture(scope="module")
def alerting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AlertingProxy]:
    with slack_alerting(tmp_path_factory.mktemp("user-soft-budget-alerts"), "alerting") as proxy:
        yield proxy


@pytest.fixture(scope="module")
def team_key_alerting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AlertingProxy]:
    directory: Final = tmp_path_factory.mktemp("user-soft-budget-team-keys")
    with slack_alerting(directory, "team-keys", apply_user_budget_to_team_keys=True) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def fail_closed_alerting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AlertingProxy]:
    directory: Final = tmp_path_factory.mktemp("user-soft-budget-fail-closed")
    with slack_alerting(directory, "fail-closed", fail_closed_budget_enforcement=True) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def mailing(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MailingProxy]:
    directory: Final = tmp_path_factory.mktemp("user-soft-budget-mail")
    with (
        gateway_from_environment() as shared,
        wire_server(accept_alert) as slack,
        wire_server(accept_alert) as webhooks,
        smtp_sink() as mailbox,
    ):
        overrides: Final = {
            "SLACK_WEBHOOK_URL": slack.url,
            "WEBHOOK_URL": webhooks.url,
            "SMTP_HOST": mailbox.host,
            "SMTP_PORT": str(mailbox.port),
            "SMTP_TLS": "False",
            "SMTP_SENDER_EMAIL": "alerts@integration.test",
        }
        config: Final = alerting_config(directory, "mailing", alerting=["slack", "webhook", "email"])
        with owned_proxy(shared, directory, overrides, config=config) as gateway:
            yield MailingProxy(gateway, AlertLog(webhooks), mailbox)


@pytest.fixture(scope="module")
def chaos(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ChaosProxy]:
    directory: Final = tmp_path_factory.mktemp("user-soft-budget-chaos")
    sink_port: Final = allocate_port()
    with gateway_from_environment() as shared:
        with owned_proxy_process(
            shared,
            directory,
            {"SLACK_WEBHOOK_URL": f"http://127.0.0.1:{sink_port}", "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"},
            config=alerting_config(directory, "chaos"),
            workers=2,
        ) as proxy:
            yield ChaosProxy(proxy, sink_port)


def slack_text(body: str) -> str:
    return string_value(JSON_OBJECT.validate_json(body)["text"])


def alerts_naming(alerts: AlertLog, marker: str, *, seconds: float = ALERT_SECONDS) -> tuple[str, ...]:
    return eventually(lambda: alerts.naming(marker), lambda found: len(found) > 0, seconds=seconds)


def user_info(gateway: Gateway, user_id: str) -> dict[str, JsonValue]:
    return object_value(gateway.get("/user/info", {"user_id": user_id})["user_info"])


def crossing_user(scenario: Scenario, **fields: JsonValue) -> str:
    return scenario.user(soft_budget=SOFT_BUDGET, spend=SPEND, **fields)


def alerts_once_a_later_key_alert_lands(
    gateway: Gateway, alerts: AlertLog, scenario: Scenario, model: str, watched: tuple[str, ...]
) -> tuple[str, ...]:
    """Alerts naming a watched id, read after a key soft budget alert sent later has reached the sink."""
    alias: Final = f"sentinel-{uuid.uuid4().hex}"
    gateway.chat(model, key=scenario.key(key_alias=alias, soft_budget=SOFT_BUDGET, spend=SPEND))
    eventually(
        lambda: alerts.naming(alias),
        lambda found: any("Soft Budget Crossed" in slack_text(body) for body in found),
        seconds=ALERT_SECONDS,
    )
    return tuple(body for body in alerts.bodies() if any(marker in body for marker in watched))


def base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def upstream_requests_carrying(gateway: Gateway, marker: str) -> tuple[str, ...]:
    observed: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    assert observed.status_code == 200, observed.text
    requests: Final = JSON_OBJECT.validate_json(observed.content)["requests"]
    assert isinstance(requests, list)
    return tuple(
        string_value(object_value(request)["path"])
        for request in requests
        if marker in json.dumps(object_value(request)["body"])
    )


def responses_model(scenario: Scenario, gateway: Gateway, response: JsonResponse | SseResponse) -> str:
    handle: Final = register_scenario(
        f"user-soft-budget-{uuid.uuid4().hex}", response, control_url=gateway.upstream_url
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(api_base=f"{handle.api_base()}/v1")


def json_responses_model(scenario: Scenario, gateway: Gateway) -> str:
    return responses_model(scenario, gateway, JsonResponse(content_type="application/json", body=RESPONSE))


def sse_responses_model(scenario: Scenario, gateway: Gateway) -> str:
    frames: Final = tuple(f"event: {event['type']}\ndata: {json.dumps(event)}" for event in RESPONSE_EVENTS)
    return responses_model(scenario, gateway, SseResponse(content_type="text/event-stream", frames=frames))


def chat_model(scenario: Scenario, gateway: Gateway) -> str:
    return scenario.model()


def raw_post(url: str, key: str, path: str, body: Mapping[str, JsonValue]) -> str:
    response: Final = httpx.post(
        f"{url}{path}", json=dict(body), headers={"Authorization": f"Bearer {key}"}, timeout=60, trust_env=False
    )
    assert response.status_code == 200, response.text
    return response.text


def raw_stream(url: str, key: str, path: str, body: Mapping[str, JsonValue]) -> str:
    with httpx.stream(
        "POST",
        f"{url}{path}",
        json={**body, "stream": True},
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
        trust_env=False,
    ) as response:
        text: Final = "".join(response.iter_text())
        assert response.status_code == 200, text
        return text


def chat_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}]}


def messages_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker}]}


def responses_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "input": marker}


def openai_client(url: str, key: str) -> OpenAI:
    return OpenAI(
        api_key=key, base_url=f"{url}/v1", max_retries=0, http_client=httpx.Client(timeout=60, trust_env=False)
    )


def chat_raw(url: str, key: str, model: str, marker: str) -> str:
    return raw_post(url, key, "/v1/chat/completions", chat_body(model, marker))


def chat_raw_stream(url: str, key: str, model: str, marker: str) -> str:
    return raw_stream(url, key, "/v1/chat/completions", chat_body(model, marker))


def chat_openai_sdk(url: str, key: str, model: str, marker: str) -> str:
    completion: Final = openai_client(url, key).chat.completions.create(
        model=model, messages=[{"role": "user", "content": marker}]
    )
    return completion.model_dump_json()


def chat_openai_sdk_async_stream(url: str, key: str, model: str, marker: str) -> str:
    async def consume() -> str:
        client: Final = AsyncOpenAI(
            api_key=key,
            base_url=f"{url}/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=60, trust_env=False),
        )
        stream: Final = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": marker}], stream=True
        )
        return "".join([chunk.model_dump_json() async for chunk in stream])

    return asyncio.run(consume())


def messages_raw(url: str, key: str, model: str, marker: str) -> str:
    return raw_post(url, key, "/v1/messages", messages_body(model, marker))


def messages_raw_stream(url: str, key: str, model: str, marker: str) -> str:
    return raw_stream(url, key, "/v1/messages", messages_body(model, marker))


def messages_anthropic_sdk(url: str, key: str, model: str, marker: str) -> str:
    client: Final = Anthropic(
        api_key=key, base_url=url, max_retries=0, http_client=httpx.Client(timeout=60, trust_env=False)
    )
    message: Final = client.messages.create(model=model, max_tokens=16, messages=[{"role": "user", "content": marker}])
    return message.model_dump_json()


def messages_anthropic_sdk_async_stream(url: str, key: str, model: str, marker: str) -> str:
    async def consume() -> str:
        client: Final = AsyncAnthropic(
            api_key=key, base_url=url, max_retries=0, http_client=httpx.AsyncClient(timeout=60, trust_env=False)
        )
        async with client.messages.stream(
            model=model, max_tokens=16, messages=[{"role": "user", "content": marker}]
        ) as stream:
            return "".join([event.model_dump_json() async for event in stream])

    return asyncio.run(consume())


def responses_raw(url: str, key: str, model: str, marker: str) -> str:
    return raw_post(url, key, "/v1/responses", responses_body(model, marker))


def responses_raw_stream(url: str, key: str, model: str, marker: str) -> str:
    return raw_stream(url, key, "/v1/responses", responses_body(model, marker))


def responses_openai_sdk(url: str, key: str, model: str, marker: str) -> str:
    return openai_client(url, key).responses.create(model=model, input=marker).model_dump_json()


def responses_openai_sdk_async_stream(url: str, key: str, model: str, marker: str) -> str:
    async def consume() -> str:
        client: Final = AsyncOpenAI(
            api_key=key,
            base_url=f"{url}/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=60, trust_env=False),
        )
        stream: Final = await client.responses.create(model=model, input=marker, stream=True)
        return "".join([event.model_dump_json() async for event in stream])

    return asyncio.run(consume())


@dataclass(frozen=True, slots=True)
class Call:
    model: Callable[[Scenario, Gateway], str]
    send: Callable[[str, str, str, str], str]
    reply: str


CALLS: Final[Mapping[str, Call]] = {
    "chat-httpx": Call(chat_model, chat_raw, "chat.completion"),
    "chat-httpx-stream": Call(chat_model, chat_raw_stream, "chat.completion.chunk"),
    "chat-openai-sdk": Call(chat_model, chat_openai_sdk, "chat.completion"),
    "chat-openai-sdk-async-stream": Call(chat_model, chat_openai_sdk_async_stream, "chat.completion.chunk"),
    "messages-httpx": Call(json_responses_model, messages_raw, REPLY),
    "messages-httpx-stream": Call(sse_responses_model, messages_raw_stream, "message_stop"),
    "messages-anthropic-sdk": Call(json_responses_model, messages_anthropic_sdk, REPLY),
    "messages-anthropic-sdk-async-stream": Call(
        sse_responses_model, messages_anthropic_sdk_async_stream, "message_stop"
    ),
    "responses-httpx": Call(json_responses_model, responses_raw, REPLY),
    "responses-httpx-stream": Call(sse_responses_model, responses_raw_stream, "response.completed"),
    "responses-openai-sdk": Call(json_responses_model, responses_openai_sdk, REPLY),
    "responses-openai-sdk-async-stream": Call(
        sse_responses_model, responses_openai_sdk_async_stream, "response.completed"
    ),
}


def soft_budget_alerts_naming(alerts: AlertLog, marker: str) -> tuple[str, ...]:
    return tuple(body for body in alerts.naming(marker) if "Message: Soft Budget Crossed" in slack_text(body))


def exactly_one_alert_naming(alerts: AlertLog, marker: str) -> str:
    (alert,) = held(
        lambda: alerts.naming(marker), lambda found: len(found) == 1, holding=QUIET_SECONDS, seconds=ALERT_SECONDS
    )
    return alert


def assert_user_soft_budget_alert(alert: str, user_id: str, soft_budget: float, spend: float | None = None) -> None:
    text: Final = slack_text(alert)
    assert f"Message: Soft Budget Crossed: Total Soft Budget:`{soft_budget}`" in text, text
    assert f"*user_id:* `{user_id}`" in text, text
    assert f"*soft_budget:* `{soft_budget}`" in text, text
    assert spend is None or f"*spend:* `{spend}`" in text, text
    assert "*event_group:* `user`" in text, text


def test_user_soft_budget_round_trips_and_stays_below_max_budget(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=100, soft_budget=50)
        assert user_info(gateway, user_id)["soft_budget"] == 50

        gateway.post("/user/update", {"user_id": user_id, "soft_budget": 75})
        assert user_info(gateway, user_id)["soft_budget"] == 75

        rejected: Final = gateway.request("POST", "/user/update", {"user_id": user_id, "soft_budget": 100})
        assert rejected.status_code == 400, rejected.text
        assert user_info(gateway, user_id)["soft_budget"] == 75

        gateway.post("/user/update", {"user_id": user_id, "soft_budget": None})
        assert user_info(gateway, user_id)["soft_budget"] is None


def test_bulk_created_user_keeps_a_valid_soft_budget_and_a_row_above_its_max_budget_is_refused(
    alerting: AlertingProxy,
) -> None:
    gateway: Final = alerting.gateway
    kept, refused = (f"integration-{uuid.uuid4().hex}" for _ in range(2))
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(gateway.request, "POST", "/user/delete", {"user_ids": [kept, refused]})

        created: Final = gateway.post(
            "/management/v1/users/bulk",
            {
                "users": [
                    {"user_id": kept, "max_budget": 100, "soft_budget": 20},
                    {"user_id": refused, "max_budget": 100, "soft_budget": 150},
                ]
            },
        )

        assert created["meta"] == {"total_requested": 2, "created": 1, "failed": 1}, created
        assert user_info(gateway, kept)["soft_budget"] == 20
        assert gateway.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (refused,)) == []


def test_user_info_v2_reports_the_soft_budget(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=100, soft_budget=50)

        info: Final = gateway.get("/v2/user/info", {"user_id": user_id})

        assert (info["soft_budget"], info["max_budget"]) == (50, 100), info


@pytest.mark.parametrize(
    "budgets",
    [
        {"soft_budget": 100, "max_budget": 100},
        {"soft_budget": 150, "max_budget": 100},
        {"soft_budget": -1},
        {"soft_budget": "inf"},
        {"soft_budget": "nan"},
    ],
    ids=["equal-to-max", "above-max", "negative", "infinite", "not-a-number"],
)
def test_user_new_rejects_a_soft_budget_that_could_never_warn_before_the_max(
    alerting: AlertingProxy, budgets: dict[str, JsonValue]
) -> None:
    gateway: Final = alerting.gateway
    user_id: Final = f"integration-{uuid.uuid4().hex}"
    try:
        rejected: Final = gateway.request(
            "POST", "/user/new", {"user_id": user_id, "auto_create_key": False, **budgets}
        )
        assert rejected.status_code == 400, rejected.text
        assert "soft_budget" in rejected.text, rejected.text
        assert gateway.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)) == []
    finally:
        gateway.request("POST", "/user/delete", {"user_ids": [user_id]})


@pytest.mark.parametrize(
    "soft_budget", ["not-a-number", [1], "", "9" * 5000 + "x"], ids=["word", "list", "empty-string", "5kb-string"]
)
def test_user_new_rejects_a_malformed_soft_budget_and_keeps_serving(
    alerting: AlertingProxy, soft_budget: JsonValue
) -> None:
    gateway: Final = alerting.gateway
    user_id: Final = f"integration-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        bystander: Final = scenario.user(soft_budget=50)
        try:
            rejected: Final = gateway.request(
                "POST", "/user/new", {"user_id": user_id, "auto_create_key": False, "soft_budget": soft_budget}
            )
            assert 400 <= rejected.status_code < 500, rejected.text
            assert "soft_budget" in rejected.text, rejected.text
            assert gateway.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)) == []
        finally:
            gateway.request("POST", "/user/delete", {"user_ids": [user_id]})
        assert user_info(gateway, bystander)["soft_budget"] == 50


def test_user_update_rejects_a_max_budget_lowered_to_the_stored_soft_budget(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=100, soft_budget=50)

        lowered: Final = gateway.request("POST", "/user/update", {"user_id": user_id, "max_budget": 50})
        assert lowered.status_code == 400, lowered.text
        crossed: Final = gateway.request(
            "POST", "/user/update", {"user_id": user_id, "max_budget": 40, "soft_budget": 45}
        )
        assert crossed.status_code == 400, crossed.text
        info: Final = user_info(gateway, user_id)
        assert (info["soft_budget"], info["max_budget"]) == (50, 100), info

        gateway.post("/user/update", {"user_id": user_id, "max_budget": 40, "soft_budget": 30})
        moved: Final = user_info(gateway, user_id)
        assert (moved["soft_budget"], moved["max_budget"]) == (30, 40), moved


def test_user_bulk_update_stores_a_valid_soft_budget_and_rejects_one_above_a_max_budget(
    alerting: AlertingProxy,
) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        roomy: Final = scenario.user(max_budget=100)
        tight: Final = scenario.user(max_budget=10)

        gateway.post("/user/bulk_update", {"users": [{"user_id": roomy, "soft_budget": 20}]})
        assert user_info(gateway, roomy)["soft_budget"] == 20

        gateway.request("POST", "/user/bulk_update", {"users": [{"user_id": tight, "soft_budget": 10}]})
        assert user_info(gateway, tight)["soft_budget"] is None

        everyone: Final = gateway.request(
            "POST", "/user/bulk_update", {"all_users": True, "user_updates": {"soft_budget": 10}}
        )
        assert everyone.status_code == 400, everyone.text
        assert (user_info(gateway, roomy)["soft_budget"], user_info(gateway, tight)["soft_budget"]) == (20, None)


@pytest.mark.parametrize("max_budget", [None, 100.0])
def test_user_over_soft_budget_gets_alert_and_request_still_succeeds(
    alerting: AlertingProxy, max_budget: float | None
) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario, max_budget=max_budget)
        key: Final = scenario.key(user_id=user_id)

        response: Final = gateway.chat(model, key=key)

        assert response["choices"], response
        assert_user_soft_budget_alert(exactly_one_alert_naming(alerting.alerts, user_id), user_id, SOFT_BUDGET, SPEND)


@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
def test_user_soft_budget_alert_fires_on_every_llm_endpoint_and_client(alerting: AlertingProxy, call: Call) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = call.model(scenario, gateway)
        user_id: Final = crossing_user(scenario)
        key: Final = scenario.key(user_id=user_id)
        marker: Final = f"soft-budget-{uuid.uuid4().hex}"

        received: Final = call.send(base_url(gateway), key, model, marker)

        assert call.reply in received, received
        assert len(upstream_requests_carrying(gateway, marker)) == 1
        assert_user_soft_budget_alert(exactly_one_alert_naming(alerting.alerts, user_id), user_id, SOFT_BUDGET, SPEND)
        logged: Final = eventually(
            lambda: gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE "user" = %s', (user_id,)),
            lambda rows: len(rows) > 0,
            seconds=70,
        )
        assert len(logged) == 1, logged


@pytest.mark.parametrize(("soft_budget", "spend"), [(1.0, 1.0), (0.0, 0.0)], ids=["spend-equals-soft", "zero-soft"])
def test_user_soft_budget_alert_fires_when_spend_reaches_it_exactly(
    alerting: AlertingProxy, soft_budget: float, spend: float
) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = scenario.user(soft_budget=soft_budget, spend=spend)

        gateway.chat(model, key=scenario.key(user_id=user_id))

        assert_user_soft_budget_alert(exactly_one_alert_naming(alerting.alerts, user_id), user_id, soft_budget, spend)


def test_user_under_soft_budget_gets_no_alert(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        under_id: Final = scenario.user(soft_budget=50.0, spend=SPEND)

        gateway.chat(model, key=scenario.key(user_id=under_id))

        assert alerts_once_a_later_key_alert_lands(gateway, alerting.alerts, scenario, model, (under_id,)) == ()


@pytest.mark.parametrize("budgets", [{}, {"max_budget": 100.0}], ids=["no-budget", "max-budget-only"])
def test_user_without_a_soft_budget_gets_no_alert(alerting: AlertingProxy, budgets: dict[str, JsonValue]) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = scenario.user(spend=SPEND, **budgets)

        response: Final = gateway.chat(model, key=scenario.key(user_id=user_id))

        assert response["choices"], response
        assert alerts_once_a_later_key_alert_lands(gateway, alerting.alerts, scenario, model, (user_id,)) == ()


@pytest.mark.parametrize("soft_budget", [None, SOFT_BUDGET], ids=["max-budget-only", "soft-and-max-budget"])
def test_user_over_max_budget_is_blocked_whether_or_not_a_soft_budget_is_set(
    alerting: AlertingProxy, soft_budget: float | None
) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = scenario.user(max_budget=0.8, soft_budget=soft_budget, spend=SPEND)

        blocked: Final = gateway.request(
            "POST", "/v1/chat/completions", chat_body(model, "over max budget"), key=scenario.key(user_id=user_id)
        )

        assert blocked.status_code == 422, blocked.text
        assert f"ExceededBudget: User={user_id} over budget. Spend=1.0, Budget=0.8" in blocked.text, blocked.text
        expected: Final = 1 if soft_budget is None else 2
        alerts: Final = held(
            lambda: alerting.alerts.naming(user_id),
            lambda found: len(found) == expected,
            holding=QUIET_SECONDS,
            seconds=ALERT_SECONDS,
        )
        texts: Final = tuple(slack_text(alert) for alert in alerts)
        assert len([text for text in texts if "Budget Crossed\n Total Budget:`0.8`" in text]) == 1, texts
        assert len([text for text in texts if "Total Soft Budget:`0.5`" in text]) == expected - 1, texts


def test_team_key_does_not_alert_on_its_owners_soft_budget_by_default(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario)
        key: Final = scenario.key(team_id=scenario.team(), user_id=user_id)

        response: Final = gateway.chat(model, key=key)

        assert response["choices"], response
        assert alerts_once_a_later_key_alert_lands(gateway, alerting.alerts, scenario, model, (user_id,)) == ()


def test_team_key_alerts_on_its_owners_soft_budget_when_user_budgets_apply_to_team_keys(
    team_key_alerting: AlertingProxy,
) -> None:
    gateway: Final = team_key_alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario)
        key: Final = scenario.key(team_id=scenario.team(), user_id=user_id)

        response: Final = gateway.chat(model, key=key)

        assert response["choices"], response
        alert: Final = exactly_one_alert_naming(team_key_alerting.alerts, user_id)
        assert_user_soft_budget_alert(alert, user_id, SOFT_BUDGET, SPEND)


def test_fail_closed_budget_enforcement_still_serves_a_user_over_only_a_soft_budget(
    fail_closed_alerting: AlertingProxy,
) -> None:
    gateway: Final = fail_closed_alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario)

        response: Final = gateway.chat(model, key=scenario.key(user_id=user_id))

        assert response["choices"], response
        alert: Final = exactly_one_alert_naming(fail_closed_alerting.alerts, user_id)
        assert_user_soft_budget_alert(alert, user_id, SOFT_BUDGET, SPEND)


def test_key_soft_budget_alert_is_unchanged_for_an_owner_without_a_soft_budget(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = f"soft-key-{uuid.uuid4().hex}"
        user_id: Final = scenario.user(spend=SPEND)
        key: Final = scenario.key(user_id=user_id, key_alias=alias, soft_budget=SOFT_BUDGET, spend=SPEND)

        gateway.chat(model, key=key)
        gateway.chat(model, key=key)

        (alert,) = held(
            lambda: soft_budget_alerts_naming(alerting.alerts, user_id),
            lambda found: len(found) == 1,
            holding=QUIET_SECONDS,
            seconds=ALERT_SECONDS,
        )
        text: Final = slack_text(alert)
        assert "Total Soft Budget:`0.5`" in text and f"*key_alias:* `{alias}`" in text, text
        assert "*event_group:* `key`" in text, text


def test_team_soft_budget_alert_is_unchanged(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    call_cost: Final = 20 * 0.001 + 20 * 0.002
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_id: Final = scenario.team(
            soft_budget=0.05, metadata={"soft_budget_alerting_emails": ["team-owner@integration.test"]}
        )
        key: Final = scenario.key(team_id=team_id)
        gateway.chat(model, key=key)
        eventually(
            lambda: gateway.rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,)),
            lambda rows: float(str(rows[0]["spend"])) == pytest.approx(call_cost),
            seconds=70,
        )

        def alerts_after_another_call() -> tuple[str, ...]:
            gateway.chat(model, key=key)
            return soft_budget_alerts_naming(alerting.alerts, team_id)

        eventually(alerts_after_another_call, lambda found: len(found) > 0, seconds=ALERT_SECONDS)
        (alert,) = held(
            lambda: soft_budget_alerts_naming(alerting.alerts, team_id),
            lambda found: len(found) == 1,
            holding=QUIET_SECONDS,
            seconds=ALERT_SECONDS,
        )
        text: Final = slack_text(alert)
        assert "Total Soft Budget:`0.05`" in text and "*event_group:* `team`" in text, text


def test_key_and_its_owner_each_get_their_own_soft_budget_alert(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = f"soft-key-{uuid.uuid4().hex}"
        user_id: Final = crossing_user(scenario)
        key: Final = scenario.key(user_id=user_id, key_alias=alias, soft_budget=SOFT_BUDGET, spend=SPEND)

        gateway.chat(model, key=key)
        gateway.chat(model, key=key)

        alerts: Final = held(
            lambda: soft_budget_alerts_naming(alerting.alerts, user_id),
            lambda found: len(found) == 2,
            holding=QUIET_SECONDS,
            seconds=ALERT_SECONDS,
        )
        texts: Final = tuple(slack_text(alert) for alert in alerts)
        assert sorted("key" if "*event_group:* `key`" in text else "user" for text in texts) == ["key", "user"], texts


def test_user_soft_budget_alerts_once_across_requests_and_keys(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario)
        first_key: Final = scenario.key(user_id=user_id)
        second_key: Final = scenario.key(user_id=user_id)

        for key in (first_key, second_key, first_key, second_key):
            gateway.chat(model, key=key)

        assert_user_soft_budget_alert(exactly_one_alert_naming(alerting.alerts, user_id), user_id, SOFT_BUDGET, SPEND)


def test_user_soft_budget_alert_fires_once_real_spend_crosses_it(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    call_cost: Final = 20 * 0.001 + 20 * 0.002
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        user_id: Final = scenario.user(soft_budget=0.05)
        key: Final = scenario.key(user_id=user_id)

        first: Final = gateway.request("POST", "/v1/chat/completions", chat_body(model, "first call"), key=key)
        assert first.status_code == 200, first.text
        assert float(first.headers["x-litellm-response-cost"]) == pytest.approx(call_cost)
        eventually(
            lambda: gateway.rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)),
            lambda rows: float(str(rows[0]["spend"])) == pytest.approx(call_cost),
            seconds=70,
        )
        assert alerts_once_a_later_key_alert_lands(gateway, alerting.alerts, scenario, model, (user_id,)) == ()

        def alerts_after_another_call() -> tuple[str, ...]:
            gateway.chat(model, key=key)
            return alerting.alerts.naming(user_id)

        eventually(alerts_after_another_call, lambda found: len(found) > 0, seconds=ALERT_SECONDS)
        text: Final = slack_text(exactly_one_alert_naming(alerting.alerts, user_id))
        assert "Total Soft Budget:`0.05`" in text and "*event_group:* `user`" in text, text


def test_lowering_the_soft_budget_below_spend_starts_alerting(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = scenario.user(soft_budget=50.0, spend=SPEND)
        key: Final = scenario.key(user_id=user_id)
        gateway.chat(model, key=key)
        assert alerts_once_a_later_key_alert_lands(gateway, alerting.alerts, scenario, model, (user_id,)) == ()

        gateway.post("/user/update", {"user_id": user_id, "soft_budget": SOFT_BUDGET})

        def alerts_after_another_call() -> tuple[str, ...]:
            gateway.chat(model, key=key)
            return alerting.alerts.naming(user_id)

        eventually(alerts_after_another_call, lambda found: len(found) > 0, seconds=ALERT_SECONDS)
        assert_user_soft_budget_alert(exactly_one_alert_naming(alerting.alerts, user_id), user_id, SOFT_BUDGET)


def test_rejected_key_of_a_user_over_soft_budget_sends_no_alert(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario)
        revoked: Final = gateway.post("/key/generate", {"user_id": user_id})["key"]
        gateway.post("/key/delete", {"keys": [revoked]})

        rejected: Final = gateway.request(
            "POST", "/v1/chat/completions", chat_body(model, "revoked key"), key=string_value(revoked)
        )

        assert rejected.status_code == 401, rejected.text
        assert alerts_once_a_later_key_alert_lands(gateway, alerting.alerts, scenario, model, (user_id,)) == ()


def test_alert_sink_refusing_the_alert_never_fails_the_request(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario)
        alerting.sink.refusing.set()
        try:
            response: Final = gateway.chat(model, key=scenario.key(user_id=user_id))
            assert response["choices"], response
            assert_user_soft_budget_alert(
                exactly_one_alert_naming(alerting.alerts, user_id), user_id, SOFT_BUDGET, SPEND
            )
            assert gateway.chat(model, key=scenario.key())["choices"]
        finally:
            alerting.sink.refusing.clear()
        other_id: Final = crossing_user(scenario)
        gateway.chat(model, key=scenario.key(user_id=other_id))
        assert_user_soft_budget_alert(exactly_one_alert_naming(alerting.alerts, other_id), other_id, SOFT_BUDGET, SPEND)


def test_alert_queued_while_the_sink_stalls_is_delivered_once_the_sink_answers(alerting: AlertingProxy) -> None:
    pytest.skip("BUG: LIT-5231 a Slack alert queued while an earlier alert is still being posted is dropped")
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        stalled, queued = (f"soft-key-{uuid.uuid4().hex}" for _ in range(2))
        alerting.sink.stalling.set()
        try:
            gateway.chat(model, key=scenario.key(key_alias=stalled, soft_budget=SOFT_BUDGET, spend=SPEND))
            alerts_naming(alerting.alerts, stalled)
            gateway.chat(model, key=scenario.key(key_alias=queued, soft_budget=SOFT_BUDGET, spend=SPEND))
        finally:
            alerting.sink.stalling.clear()
            alerting.sink.release.set()
        try:
            assert len(alerts_naming(alerting.alerts, queued)) == 1
        finally:
            alerting.sink.release.clear()


def test_stalled_alert_sink_does_not_hold_the_request_or_duplicate_the_alert(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        stalled_id: Final = crossing_user(scenario)
        alerting.sink.stalling.set()
        try:
            assert gateway.chat(model, key=scenario.key(user_id=stalled_id))["choices"]
            alerts_naming(alerting.alerts, stalled_id)
            assert gateway.chat(model, key=scenario.key())["choices"]
            assert not alerting.sink.release.is_set()
        finally:
            alerting.sink.stalling.clear()
            alerting.sink.release.set()
        try:
            eventually(alerting.alerts.wire.open_connections, lambda still_open: still_open == 0, seconds=ALERT_SECONDS)
            later_id: Final = crossing_user(scenario)
            gateway.chat(model, key=scenario.key(user_id=later_id))
            for user_id in (later_id, stalled_id):
                assert_user_soft_budget_alert(
                    exactly_one_alert_naming(alerting.alerts, user_id), user_id, SOFT_BUDGET, SPEND
                )
        finally:
            alerting.sink.release.clear()


def mail_to(mailbox: Mailbox, recipient: str) -> tuple[str, ...]:
    return tuple(delivery.html for delivery in mailbox.deliveries() if recipient in delivery.recipients)


def test_user_soft_budget_webhook_event_and_email_name_the_user_and_budget(mailing: MailingProxy) -> None:
    gateway: Final = mailing.gateway
    email: Final = f"owner-{uuid.uuid4().hex}@integration.test"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario, user_email=email)

        key: Final = scenario.key(user_id=user_id)

        response: Final = gateway.chat(model, key=key)
        gateway.chat(model, key=key)

        assert response["choices"], response
        event: Final = JSON_OBJECT.validate_json(exactly_one_alert_naming(mailing.webhooks, user_id))
        assert {
            name: event[name]
            for name in ("event", "event_group", "user_id", "user_email", "soft_budget", "spend", "max_budget", "token")
        } == {
            "event": "soft_budget_crossed",
            "event_group": "user",
            "user_id": user_id,
            "user_email": email,
            "soft_budget": SOFT_BUDGET,
            "spend": SPEND,
            "max_budget": None,
            "token": None,
        }, event
        mails: Final = held(
            lambda: mail_to(mailing.mailbox, email),
            lambda found: len(found) == 1,
            holding=QUIET_SECONDS,
            seconds=ALERT_SECONDS,
        )
        assert len(mails) == 1, mails
        assert "Your LiteLLM account has crossed its <b>soft budget limit of $0.5</b>" in mails[0], mails
        assert "<b>Current Spend:</b> $1.0" in mails[0], mails


def test_key_soft_budget_still_sends_the_generic_and_the_api_key_email(mailing: MailingProxy) -> None:
    gateway: Final = mailing.gateway
    email: Final = f"owner-{uuid.uuid4().hex}@integration.test"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = scenario.user(user_email=email, spend=SPEND)
        key: Final = scenario.key(user_id=user_id, soft_budget=SOFT_BUDGET, spend=SPEND)

        gateway.chat(model, key=key)

        mails: Final = held(
            lambda: mail_to(mailing.mailbox, email),
            lambda found: len(found) == 2,
            holding=QUIET_SECONDS,
            seconds=ALERT_SECONDS,
        )
        assert sorted("generic" if mail == "Alert from LiteLLM Server" else "soft budget" for mail in mails) == [
            "generic",
            "soft budget",
        ], mails
        assert any("Your LiteLLM API key has crossed its <b>soft budget limit of $0.5</b>" in mail for mail in mails)


def test_key_soft_budget_of_an_owner_without_an_email_reaches_the_webhook_once(mailing: MailingProxy) -> None:
    pytest.skip("BUG: LIT-10808 a key soft budget crossing whose owner has no email re-sends the webhook each request")
    gateway: Final = mailing.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = f"soft-key-{uuid.uuid4().hex}"
        key: Final = scenario.key(
            user_id=scenario.user(spend=SPEND), key_alias=alias, soft_budget=SOFT_BUDGET, spend=SPEND
        )

        for _ in range(3):
            assert gateway.chat(model, key=key)["choices"]

        events: Final = held(
            lambda: tuple(body for body in mailing.webhooks.naming(alias) if '"soft_budget_crossed"' in body),
            lambda found: len(found) == 1,
            holding=QUIET_SECONDS,
            seconds=ALERT_SECONDS,
        )
        assert JSON_OBJECT.validate_json(events[0])["event_group"] == "key", events


def test_user_without_an_email_is_served_and_still_reaches_the_webhook(mailing: MailingProxy) -> None:
    gateway: Final = mailing.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = crossing_user(scenario)
        key: Final = scenario.key(user_id=user_id)

        for _ in range(3):
            assert gateway.chat(model, key=key)["choices"]

        event: Final = JSON_OBJECT.validate_json(exactly_one_alert_naming(mailing.webhooks, user_id))
        assert (event["event"], event["event_group"], event["user_email"]) == ("soft_budget_crossed", "user", None)
        assert [html for delivery in mailing.mailbox.deliveries() if user_id in (html := delivery.html)] == []


@dataclass(frozen=True, slots=True)
class Job:
    send: Callable[[str, str, str, str], str]
    model: str
    key: str
    user_id: str


def burst_jobs(scenario: Scenario, gateway: Gateway, users: int, requests: int) -> tuple[Job, ...]:
    chat: Final = scenario.model()
    json_model: Final = json_responses_model(scenario, gateway)
    sse_model: Final = sse_responses_model(scenario, gateway)
    routes: Final = (
        (chat_raw, chat),
        (chat_raw_stream, chat),
        (messages_raw, json_model),
        (messages_raw_stream, sse_model),
        (responses_raw, json_model),
        (responses_raw_stream, sse_model),
    )
    user_ids: Final = tuple(crossing_user(scenario) for _ in range(users))
    keys: Final = tuple(scenario.key(user_id=user_id) for user_id in user_ids)
    return tuple(
        Job(
            routes[index % len(routes)][0], routes[index % len(routes)][1], keys[index % users], user_ids[index % users]
        )
        for index in range(requests)
    )


def alerts_per_user(bodies: tuple[str, ...], user_ids: frozenset[str]) -> Mapping[str, int]:
    return {user_id: len([body for body in bodies if user_id in body]) for user_id in user_ids}


def test_alert_sink_outage_mid_burst_fails_no_request_and_alerting_resumes(
    chaos: ChaosProxy, record_property: Callable[[str, object], None]
) -> None:
    gateway: Final = chaos.proxy.gateway
    url: Final = base_url(gateway)
    with gateway.scenario() as scenario:
        jobs: Final = burst_jobs(scenario, gateway, users=10, requests=30)
        user_ids: Final = frozenset(job.user_id for job in jobs)
        with ThreadPoolExecutor(max_workers=10) as pool:
            with wire_server(accept_alert, port=chaos.sink_port) as before:
                sent: Final = tuple(
                    pool.submit(job.send, url, job.key, job.model, f"burst-{uuid.uuid4().hex}") for job in jobs
                )
                wait(sent, return_when=FIRST_COMPLETED)
                before_outage: Final = AlertLog(before)
            received: Final = tuple(future.result() for future in sent)
        assert len(received) == 30 and all(received), received

        with wire_server(accept_alert, port=chaos.sink_port) as after:
            after_outage: Final = AlertLog(after)
            late_id: Final = crossing_user(scenario)
            assert gateway.chat(scenario.model(), key=scenario.key(user_id=late_id))["choices"]
            assert_user_soft_budget_alert(exactly_one_alert_naming(after_outage, late_id), late_id, SOFT_BUDGET, SPEND)
            delivered: Final = alerts_per_user((*before_outage.bodies(), *after_outage.bodies()), user_ids)
        record_property("burst_users_alerted", sum(delivered.values()))
        assert all(count <= 1 for count in delivered.values()), delivered


def status_or_disconnect(url: str, key: str, model: str) -> int:
    try:
        return httpx.post(
            f"{url}/v1/chat/completions",
            json=chat_body(model, "worker kill burst"),
            headers={"Authorization": f"Bearer {key}", "connection": "close"},
            timeout=60,
            trust_env=False,
        ).status_code
    except httpx.TransportError:
        return 0


def test_user_soft_budget_alert_still_fires_after_a_worker_is_killed_mid_burst(chaos: ChaosProxy) -> None:
    proxy: Final = chaos.proxy
    gateway: Final = proxy.gateway
    url: Final = base_url(gateway)
    with wire_server(accept_alert, port=chaos.sink_port) as sink, gateway.scenario() as scenario:
        alerts: Final = AlertLog(sink)
        model: Final = scenario.model()
        user_ids: Final = tuple(crossing_user(scenario) for _ in range(6))
        keys: Final = tuple(scenario.key(user_id=user_id) for user_id in user_ids)
        victim: Final = min(proxy.worker_pids())
        with ThreadPoolExecutor(max_workers=8) as pool:
            sent: Final = tuple(pool.submit(status_or_disconnect, url, keys[index % 6], model) for index in range(24))
            proxy.kill_worker(victim)
            statuses: Final = tuple(future.result() for future in sent)
        assert set(statuses) <= {200, 0} and 200 in statuses, statuses
        assert victim not in proxy.worker_pids()

        late_id: Final = crossing_user(scenario)
        assert gateway.chat(model, key=scenario.key(user_id=late_id))["choices"]
        assert_user_soft_budget_alert(exactly_one_alert_naming(alerts, late_id), late_id, SOFT_BUDGET, SPEND)
        delivered: Final = alerts_per_user(alerts.bodies(), frozenset(user_ids))
        assert all(count <= 1 for count in delivered.values()), delivered
