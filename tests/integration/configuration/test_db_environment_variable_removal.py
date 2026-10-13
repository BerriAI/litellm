from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import anthropic
import httpx
import openai
import pytest
import yaml
from anthropic.types import Message, RawContentBlockDeltaEvent, TextBlock, TextDelta
from integration._support.anthropic_sse import delta_text, message_json, message_stream, parse_sse, stream_reply
from integration._support.budget import readiness_seconds
from integration._support.client import Gateway, eventually, gateway_from_environment, held, string_value
from integration._support.database import scratch_database, write_rows
from integration._support.database_relay import database_relay
from integration._support.environment_probe import PROBE_HOOK, PROBE_ROUTE
from integration._support.process import (
    DB_PUSH,
    OwnedProxy,
    WorkerHealthcheck,
    graceful_stop_seconds,
    owned_proxy_process,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

MODEL: Final = "db-env-removal-claude"
UPSTREAM_MODEL: Final = "claude-sonnet-5-5"
REPLY_TEXT: Final = "ok"
PROCESS_KEY: Final = "sk-ant-from-the-process-environment"
DB_KEY: Final = "sk-ant-from-the-database-row"
LOGO_URL: Final = "https://example.com/db-env-removal-logo.png"
CONVERGE_SECONDS: Final = 60
HOLD_SECONDS: Final = 5
PROBES_PER_READ: Final = 12
WORKERS: Final = 2
BURST: Final = 20
SCRATCH_ONLY: Final = ("DATABASE_URL_READ_REPLICA",)
CLI_ONLY: Final = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "UI_USERNAME", "UI_PASSWORD")
CLOSE: Final[Mapping[str, str]] = MappingProxyType({"connection": "close"})
YAML_LITERAL: Final = "LIT9489_YAML_LITERAL"
YAML_REFERENCE: Final = "LIT9489_YAML_REFERENCE"
YAML_SOURCE: Final = "LIT9489_YAML_SOURCE"
YAML_LITERAL_VALUE: Final = "literal-from-yaml"
YAML_SOURCE_VALUE: Final = "reference-from-the-process"
PROCESS_ONLY: Final = "LIT9489_PROCESS_ONLY"
PROCESS_ONLY_VALUE: Final = "process-only-value"
DB_ONLY: Final = "LIT9489_DB_ONLY"
NUMBER_KEY: Final = "LIT9489_NUMBER"
EMPTY_KEY: Final = "LIT9489_EMPTY"
LONG_KEY: Final = "LIT9489_LONG"
LONG_VALUE: Final = "v" * 5120
NULLED_KEY: Final = "LIT9489_NULLED"
SYNC_MARKER: Final = "LIT9489_SYNC_MARKER"
CLI_MARKER: Final = "LIT9489_CLI_MARKER"
UNTOUCHED: Final[Mapping[str, str | None]] = MappingProxyType(
    {YAML_LITERAL: YAML_LITERAL_VALUE, YAML_REFERENCE: YAML_SOURCE_VALUE, PROCESS_ONLY: PROCESS_ONLY_VALUE}
)
REQUEST_BODY: Final = TypeAdapter(dict[str, JsonValue])

pytestmark: Final = pytest.mark.timeout(
    2 * readiness_seconds(WORKERS) + 4 * graceful_stop_seconds() + 6 * CONVERGE_SECONDS
)


class WorkerEnvironment(BaseModel):
    model_config = ConfigDict(frozen=True)

    pid: int
    environment: Mapping[str, str | None]


class Endpoint(Enum):
    CHAT = "chat"
    MESSAGES = "messages"
    RESPONSES = "responses"
    ANTHROPIC_PASS_THROUGH = "anthropic_pass_through"


@dataclass(frozen=True, slots=True)
class Pods:
    writer: OwnedProxy
    reader: OwnedProxy
    provider: Wire
    database_url: str


def _anthropic_message(request: Request) -> Reply:
    message_id: Final = f"msg_{uuid.uuid4().hex}"
    streams: Final = bool(request.body) and REQUEST_BODY.validate_json(request.body).get("stream") is True
    if streams:
        return stream_reply(message_stream(message_id, UPSTREAM_MODEL, REPLY_TEXT))
    return Reply(body=message_json(message_id, UPSTREAM_MODEL, REPLY_TEXT))


def _config(directory: Path, general_settings: Mapping[str, str] = MappingProxyType({})) -> Path:
    path: Final = directory / f"db_env_removal_{uuid.uuid4().hex}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [{"model_name": MODEL, "litellm_params": {"model": f"anthropic/{UPSTREAM_MODEL}"}}],
                "environment_variables": {
                    YAML_LITERAL: YAML_LITERAL_VALUE,
                    YAML_REFERENCE: f"os.environ/{YAML_SOURCE}",
                },
                **({"general_settings": dict(general_settings)} if general_settings else {}),
            }
        )
    )
    return path


def _process_environment(provider: Wire, database_url: str | None) -> Mapping[str, str]:
    return MappingProxyType(
        {
            **({"DATABASE_URL": database_url} if database_url else {}),
            "ANTHROPIC_API_KEY": PROCESS_KEY,
            "ANTHROPIC_API_BASE": provider.url,
            YAML_LITERAL: "literal-from-the-process",
            YAML_SOURCE: YAML_SOURCE_VALUE,
            PROCESS_ONLY: PROCESS_ONLY_VALUE,
            "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2",
            "LITELLM_CONFIG_PARAM_CACHE_TTL_SECONDS": "1",
            "LITELLM_WORKER_STARTUP_HOOKS": PROBE_HOOK,
        }
    )


def _pod(
    gateway: Gateway,
    directory: Path,
    provider: Wire,
    database_url: str | None,
    *,
    workers: int = WORKERS,
    general_settings: Mapping[str, str] = MappingProxyType({}),
    remove_environment: tuple[str, ...] = SCRATCH_ONLY,
) -> AbstractContextManager[OwnedProxy]:
    directory.mkdir(parents=True, exist_ok=True)
    return owned_proxy_process(
        gateway,
        directory,
        _process_environment(provider, database_url),
        config=_config(directory, general_settings),
        remove_environment=remove_environment,
        workers=workers,
        database_setup=DB_PUSH,
        healthcheck=WorkerHealthcheck.LENIENT,
    )


@pytest.fixture(scope="module")
def pods(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pods]:
    directory: Final = tmp_path_factory.mktemp("db-env-removal")
    with (
        gateway_from_environment() as gateway,
        wire_server(_anthropic_message) as provider,
        scratch_database() as database_url,
        _pod(gateway, directory / "writer", provider, database_url) as writer,
        _pod(gateway, directory / "reader", provider, database_url) as reader,
    ):
        yield Pods(writer, reader, provider, database_url)


def _worker_environment(proxy: OwnedProxy, names: tuple[str, ...]) -> WorkerEnvironment:
    response: Final = proxy.gateway.request("GET", PROBE_ROUTE, params={"names": ",".join(names)}, headers=CLOSE)
    assert response.status_code == 200, response.text
    return WorkerEnvironment.model_validate_json(response.content)


def _environments(proxy: OwnedProxy, names: tuple[str, ...]) -> Mapping[int, Mapping[str, str | None]]:
    observed: Final = tuple(_worker_environment(proxy, names) for _ in range(PROBES_PER_READ))
    return MappingProxyType({seen.pid: dict(seen.environment) for seen in observed})


def _settled(proxy: OwnedProxy, expected: Mapping[str, str | None]) -> None:
    names: Final = tuple(expected)
    wanted: Final = dict(expected)
    serving: Final = proxy.worker_pids() or frozenset({proxy.process.pid})

    def everywhere(seen: Mapping[int, Mapping[str, str | None]]) -> bool:
        return seen.keys() == serving and all(environment == wanted for environment in seen.values())

    def wherever_seen(seen: Mapping[int, Mapping[str, str | None]]) -> bool:
        return all(environment == wanted for environment in seen.values())

    eventually(lambda: _environments(proxy, names), everywhere, seconds=CONVERGE_SECONDS)
    held(lambda: _environments(proxy, names), wherever_seen, holding=HOLD_SECONDS, seconds=CONVERGE_SECONDS)


def _settled_on_both(pods: Pods, expected: Mapping[str, str | None]) -> None:
    _settled(pods.writer, expected)
    _settled(pods.reader, expected)


def _merge_into_row(database_url: str, values: Mapping[str, JsonValue]) -> None:
    write_rows(
        'INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES (%s, %s::jsonb) '
        'ON CONFLICT (param_name) DO UPDATE SET param_value = "LiteLLM_Config".param_value || EXCLUDED.param_value',
        ("environment_variables", json.dumps(dict(values))),
        database_url=database_url,
    )


def _remove_from_row(database_url: str, keys: tuple[str, ...]) -> None:
    write_rows(
        'UPDATE "LiteLLM_Config" SET param_value = param_value - %s::text[] WHERE param_name = %s',
        ("{" + ",".join(keys) + "}", "environment_variables"),
        database_url=database_url,
    )


def _set_through_the_api(proxy: OwnedProxy, values: Mapping[str, JsonValue]) -> None:
    proxy.gateway.post("/config/update", {"environment_variables": dict(values)})


def _upstream_key(provider: Wire, marker: str) -> str:
    sent: Final = tuple(request for request in provider.drain() if marker.encode() in request.body)
    assert len(sent) == 1, [(request.method, request.target) for request in sent]
    return sent[0].headers["x-api-key"]


def _marker() -> str:
    return f"db env removal {uuid.uuid4().hex}"


def _key_received(gateway: Gateway, provider: Wire) -> str:
    marker: Final = _marker()
    provider.drain()
    gateway.chat(MODEL, text=marker)
    return _upstream_key(provider, marker)


def _key_the_provider_received(pods: Pods) -> str:
    return _key_received(pods.reader.gateway, pods.provider)


def _provider_key_converges(pods: Pods, expected: str) -> None:
    eventually(lambda: _key_the_provider_received(pods), lambda key: key == expected, seconds=CONVERGE_SECONDS)


def _origin(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _openai(gateway: Gateway, transport: httpx.Client) -> openai.OpenAI:
    return openai.OpenAI(api_key=gateway.key, base_url=f"{_origin(gateway)}/v1", max_retries=0, http_client=transport)


def _async_openai(gateway: Gateway, transport: httpx.AsyncClient) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=gateway.key, base_url=f"{_origin(gateway)}/v1", max_retries=0, http_client=transport
    )


def _anthropic(gateway: Gateway, transport: httpx.Client) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=gateway.key, base_url=_origin(gateway), max_retries=0, http_client=transport)


def _async_anthropic(gateway: Gateway, transport: httpx.AsyncClient) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=gateway.key, base_url=_origin(gateway), max_retries=0, http_client=transport
    )


def _message_text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _chat(gateway: Gateway, marker: str) -> str:
    with httpx.Client(timeout=15, trust_env=False) as transport, _openai(gateway, transport) as client:
        completion: Final = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": marker}])
    return completion.choices[0].message.content or ""


async def _chat_stream(gateway: Gateway, marker: str) -> str:
    async with httpx.AsyncClient(timeout=15, trust_env=False) as transport, _async_openai(gateway, transport) as client:
        stream: Final = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": marker}], stream=True
        )
        return "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])


def _messages(gateway: Gateway, marker: str) -> str:
    with httpx.Client(timeout=15, trust_env=False) as transport, _anthropic(gateway, transport) as client:
        message: Final = client.messages.create(
            model=MODEL, max_tokens=16, messages=[{"role": "user", "content": marker}]
        )
    return _message_text(message)


def _text_delta(event: object) -> str:
    if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, TextDelta):
        return event.delta.text
    return ""


async def _messages_stream(gateway: Gateway, marker: str) -> str:
    async with (
        httpx.AsyncClient(timeout=15, trust_env=False) as transport,
        _async_anthropic(gateway, transport) as client,
    ):
        stream: Final = await client.messages.create(
            model=MODEL, max_tokens=16, messages=[{"role": "user", "content": marker}], stream=True
        )
        return "".join([_text_delta(event) async for event in stream])


def _responses(gateway: Gateway, marker: str) -> str:
    with httpx.Client(timeout=15, trust_env=False) as transport, _openai(gateway, transport) as client:
        response: Final = client.responses.create(model=MODEL, input=marker)
    return response.output_text


async def _responses_stream(gateway: Gateway, marker: str) -> str:
    async with httpx.AsyncClient(timeout=15, trust_env=False) as transport, _async_openai(gateway, transport) as client:
        stream: Final = await client.responses.create(model=MODEL, input=marker, stream=True)
        return "".join([event.delta async for event in stream if isinstance(event, ResponseTextDeltaEvent)])


def _pass_through_body(marker: str, stream: bool) -> dict[str, JsonValue]:
    return {
        "model": UPSTREAM_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
    }


def _pass_through(gateway: Gateway, marker: str) -> str:
    response: Final = gateway.request("POST", "/anthropic/v1/messages", _pass_through_body(marker, False))
    assert response.status_code == 200, response.text
    return _message_text(Message.model_validate_json(response.content))


def _pass_through_stream(gateway: Gateway, marker: str) -> str:
    with gateway.client.stream(
        "POST",
        "/anthropic/v1/messages",
        json=_pass_through_body(marker, True),
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        body: Final = response.read()
    assert response.status_code == 200, body
    return delta_text(parse_sse(body.decode()))


ASKS: Final[Mapping[tuple[Endpoint, bool], Callable[[Gateway, str], str]]] = MappingProxyType(
    {
        (Endpoint.CHAT, False): _chat,
        (Endpoint.CHAT, True): lambda gateway, marker: asyncio.run(_chat_stream(gateway, marker)),
        (Endpoint.MESSAGES, False): _messages,
        (Endpoint.MESSAGES, True): lambda gateway, marker: asyncio.run(_messages_stream(gateway, marker)),
        (Endpoint.RESPONSES, False): _responses,
        (Endpoint.RESPONSES, True): lambda gateway, marker: asyncio.run(_responses_stream(gateway, marker)),
        (Endpoint.ANTHROPIC_PASS_THROUGH, False): _pass_through,
        (Endpoint.ANTHROPIC_PASS_THROUGH, True): _pass_through_stream,
    }
)


def _received_through(pods: Pods, endpoint: Endpoint, stream: bool) -> tuple[str, str]:
    marker: Final = _marker()
    pods.provider.drain()
    text: Final = ASKS[endpoint, stream](pods.reader.gateway, marker)
    return text, _upstream_key(pods.provider, marker)


def _logo_url(gateway: Gateway) -> str:
    return string_value(gateway.get("/get_logo_url")["logo_url"])


def _update_theme(gateway: Gateway, logo_url: str | None) -> None:
    response: Final = gateway.request(
        "PATCH",
        "/update/ui_theme_settings",
        {"logo_url": logo_url, "logo_url_dark": None, "favicon_url": None},
    )
    assert response.status_code == 200, response.text


def _null_in_the_sso_table(pods: Pods) -> str:
    client_id: Final = f"lit9489-google-{uuid.uuid4().hex}"
    response: Final = pods.writer.gateway.request(
        "PATCH",
        "/update/sso_settings",
        {"google_client_id": client_id, "google_client_secret": "lit9489-google-secret"},
    )
    assert response.status_code == 200, response.text
    _settled(pods.reader, {"GOOGLE_CLIENT_ID": client_id})
    return "MICROSOFT_CLIENT_ID"


def _null_in_the_environment_row(pods: Pods) -> str:
    _set_through_the_api(pods.writer, {NULLED_KEY: "set-before-the-null"})
    _settled(pods.reader, {NULLED_KEY: "set-before-the-null"})
    _set_through_the_api(pods.writer, {NULLED_KEY: None})
    return NULLED_KEY


NULL_WRITERS: Final[Mapping[str, Callable[[Pods], str]]] = MappingProxyType(
    {"sso_table": _null_in_the_sso_table, "environment_row": _null_in_the_environment_row}
)


def test_provider_key_removed_from_the_database_falls_back_to_the_process_key(pods: Pods) -> None:
    assert _key_the_provider_received(pods) == PROCESS_KEY

    _set_through_the_api(pods.writer, {"ANTHROPIC_API_KEY": DB_KEY})
    _provider_key_converges(pods, DB_KEY)
    _settled_on_both(pods, {"ANTHROPIC_API_KEY": DB_KEY})

    write_rows(
        'UPDATE "LiteLLM_Config" SET param_value = param_value - %s WHERE param_name = %s',
        ("ANTHROPIC_API_KEY", "environment_variables"),
        database_url=pods.database_url,
    )

    _provider_key_converges(pods, PROCESS_KEY)
    _settled_on_both(pods, {"ANTHROPIC_API_KEY": PROCESS_KEY})


def test_logo_reset_on_one_pod_is_reset_on_the_other(pods: Pods) -> None:
    assert _logo_url(pods.reader.gateway) == ""

    _update_theme(pods.writer.gateway, LOGO_URL)
    eventually(lambda: _logo_url(pods.reader.gateway), lambda url: url == LOGO_URL, seconds=CONVERGE_SECONDS)
    _settled_on_both(pods, {"UI_LOGO_PATH": LOGO_URL})

    _update_theme(pods.writer.gateway, None)

    _settled_on_both(pods, {"UI_LOGO_PATH": None})
    assert _logo_url(pods.writer.gateway) == ""
    assert _logo_url(pods.reader.gateway) == ""


def test_key_only_the_database_set_is_unset_on_every_worker(pods: Pods) -> None:
    _set_through_the_api(pods.writer, {DB_ONLY: "db-only-value"})
    _settled_on_both(pods, {**UNTOUCHED, DB_ONLY: "db-only-value"})

    _remove_from_row(pods.database_url, (DB_ONLY,))

    _settled_on_both(pods, {**UNTOUCHED, DB_ONLY: None})


def test_yaml_owned_keys_fall_back_to_their_yaml_values(pods: Pods) -> None:
    _settled_on_both(pods, UNTOUCHED)
    overridden: Final = {YAML_LITERAL: "literal-from-the-database", YAML_REFERENCE: "reference-from-the-database"}

    _merge_into_row(pods.database_url, overridden)
    _settled_on_both(pods, {**UNTOUCHED, **overridden})

    _remove_from_row(pods.database_url, (YAML_LITERAL, YAML_REFERENCE))

    _settled_on_both(pods, UNTOUCHED)


def test_malformed_neighbours_do_not_block_a_restore(pods: Pods) -> None:
    neighbours_key: Final = "sk-ant-db-malformed-neighbours"
    _set_through_the_api(pods.writer, {"ANTHROPIC_API_KEY": neighbours_key})
    _provider_key_converges(pods, neighbours_key)

    _merge_into_row(pods.database_url, {"ANTHROPIC_API_KEY": None, NUMBER_KEY: 5, EMPTY_KEY: "", LONG_KEY: LONG_VALUE})

    _settled_on_both(pods, {"ANTHROPIC_API_KEY": PROCESS_KEY, NUMBER_KEY: None, EMPTY_KEY: "", LONG_KEY: LONG_VALUE})
    assert _key_the_provider_received(pods) == PROCESS_KEY

    _remove_from_row(pods.database_url, ("ANTHROPIC_API_KEY", NUMBER_KEY, EMPTY_KEY, LONG_KEY))

    _settled_on_both(pods, {EMPTY_KEY: None, LONG_KEY: None})


@pytest.mark.parametrize("database_key", ("sk-ant-db-first-add", "sk-ant-db-second-add"))
def test_re_added_key_is_applied_and_removed_again(pods: Pods, database_key: str) -> None:
    _set_through_the_api(pods.writer, {"ANTHROPIC_API_KEY": database_key})
    _provider_key_converges(pods, database_key)

    _set_through_the_api(pods.writer, {"ANTHROPIC_API_KEY": None})

    _provider_key_converges(pods, PROCESS_KEY)
    _settled_on_both(pods, {"ANTHROPIC_API_KEY": PROCESS_KEY})


def test_lowercase_key_removal_restores_the_uppercase_process_value(pods: Pods) -> None:
    lowercase: Final = PROCESS_ONLY.lower()
    _set_through_the_api(pods.writer, {lowercase: "lowercase-from-the-database"})
    _settled_on_both(pods, {lowercase: "lowercase-from-the-database", PROCESS_ONLY: "lowercase-from-the-database"})

    _remove_from_row(pods.database_url, (lowercase,))

    _settled_on_both(pods, {lowercase: None, PROCESS_ONLY: PROCESS_ONLY_VALUE})


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.parametrize("endpoint", tuple(Endpoint), ids=lambda endpoint: endpoint.value)
def test_removed_provider_key_reaches_every_endpoint_as_the_process_key(
    pods: Pods, endpoint: Endpoint, stream: bool
) -> None:
    database_key: Final = f"sk-ant-db-{endpoint.value}-{'stream' if stream else 'json'}"
    _set_through_the_api(pods.writer, {"ANTHROPIC_API_KEY": database_key})
    eventually(
        lambda: _received_through(pods, endpoint, stream),
        lambda received: received == (REPLY_TEXT, database_key),
        seconds=CONVERGE_SECONDS,
    )

    _remove_from_row(pods.database_url, ("ANTHROPIC_API_KEY",))

    eventually(
        lambda: _received_through(pods, endpoint, stream),
        lambda received: received == (REPLY_TEXT, PROCESS_KEY),
        seconds=CONVERGE_SECONDS,
    )
    _settled(pods.reader, {"ANTHROPIC_API_KEY": PROCESS_KEY})


@pytest.mark.parametrize("store", tuple(NULL_WRITERS))
def test_null_values_do_not_log_an_error_on_every_sync(pods: Pods, store: str) -> None:
    nulled: Final = NULL_WRITERS[store](pods)
    markers: Final = tuple(f"sync-{uuid.uuid4().hex}" for _ in range(2))

    for marker in markers:
        _merge_into_row(pods.database_url, {SYNC_MARKER: marker})
        _settled(pods.reader, {SYNC_MARKER: marker})

    errors: Final = pods.reader.log.read_text(errors="replace").count(f"Error setting env variable: {nulled} ")
    _remove_from_row(pods.database_url, (SYNC_MARKER, NULLED_KEY))
    assert errors == 0


def test_respawned_worker_restores_the_process_key_after_removal(tmp_path: Path) -> None:
    worker_kill_key: Final = "sk-ant-db-worker-kill"
    with (
        gateway_from_environment() as gateway,
        wire_server(_anthropic_message) as provider,
        scratch_database() as database_url,
        _pod(gateway, tmp_path / "pod", provider, database_url) as proxy,
    ):
        _set_through_the_api(proxy, {"ANTHROPIC_API_KEY": worker_kill_key})
        _settled(proxy, {"ANTHROPIC_API_KEY": worker_kill_key})

        replacement: Final = proxy.kill_worker(min(proxy.worker_pids()))

        assert replacement in proxy.worker_pids()
        _settled(proxy, {"ANTHROPIC_API_KEY": worker_kill_key})

        _remove_from_row(database_url, ("ANTHROPIC_API_KEY",))

        _settled(proxy, {"ANTHROPIC_API_KEY": PROCESS_KEY})
        assert _key_received(proxy.gateway, provider) == PROCESS_KEY


def _plaintext(database_url: str) -> str:
    parts: Final = urlsplit(database_url)
    return urlunsplit(parts._replace(query="&".join(part for part in (parts.query, "sslmode=disable") if part)))


def _burst_status(gateway: Gateway, marker: str) -> int:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": marker}]}
    ).status_code


def test_database_outage_holds_the_key_and_applies_the_removal_after_reconnect(tmp_path: Path) -> None:
    outage_key: Final = "sk-ant-db-outage"
    with (
        gateway_from_environment() as gateway,
        wire_server(_anthropic_message) as provider,
        scratch_database() as database_url,
        database_relay(database_url, b"environment_variables") as (relay, relayed_url),
        _pod(gateway, tmp_path / "pod", provider, _plaintext(relayed_url)) as proxy,
    ):
        _set_through_the_api(proxy, {"ANTHROPIC_API_KEY": outage_key})
        _settled(proxy, {"ANTHROPIC_API_KEY": outage_key})

        relay.arm()
        assert relay.tripped.wait(CONVERGE_SECONDS), "No config sync read the environment_variables row"
        _remove_from_row(database_url, ("ANTHROPIC_API_KEY",))
        markers: Final = tuple(_marker() for _ in range(BURST))
        provider.drain()
        with ThreadPoolExecutor(max_workers=BURST) as pool:
            statuses: Final = tuple(pool.map(lambda marker: _burst_status(proxy.gateway, marker), markers))
        received: Final = provider.drain()
        during_outage: Final = not relay.reconnected.is_set()

        assert during_outage, "The burst outlasted the database outage"
        assert statuses == (200,) * BURST
        assert {
            marker: tuple(request.headers["x-api-key"] for request in received if marker.encode() in request.body)
            for marker in markers
        } == {marker: (outage_key,) for marker in markers}

        assert relay.reconnected.wait(CONVERGE_SECONDS), "The proxy never reconnected after the outage"
        _settled(proxy, {"ANTHROPIC_API_KEY": PROCESS_KEY})
        assert _key_received(proxy.gateway, provider) == PROCESS_KEY


def test_single_worker_keeps_the_database_url_the_cli_set_from_the_config(tmp_path: Path) -> None:
    with (
        gateway_from_environment() as gateway,
        wire_server(_anthropic_message) as provider,
        scratch_database() as database_url,
        _pod(
            gateway,
            tmp_path / "pod",
            provider,
            None,
            workers=1,
            general_settings={"database_url": database_url},
            remove_environment=CLI_ONLY,
        ) as proxy,
    ):
        cli_url: Final = _worker_environment(proxy, ("DATABASE_URL",)).environment["DATABASE_URL"]
        assert cli_url is not None

        _set_through_the_api(proxy, {"DATABASE_URL": cli_url, CLI_MARKER: "marker-from-the-database"})
        _settled(proxy, {CLI_MARKER: "marker-from-the-database"})

        _remove_from_row(database_url, ("DATABASE_URL", CLI_MARKER))

        _settled(proxy, {CLI_MARKER: None})
        keeps_the_cli_url: Final = _worker_environment(proxy, ("DATABASE_URL",)).environment["DATABASE_URL"] == cli_url
        assert keeps_the_cli_url, "DATABASE_URL no longer holds the value proxy_cli set"
        login: Final = proxy.gateway.request("POST", "/v2/login", {"username": "admin", "password": proxy.gateway.key})
        assert login.status_code == 200, login.text
