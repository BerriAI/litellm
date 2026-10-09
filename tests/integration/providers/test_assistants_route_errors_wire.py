import asyncio
import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import httpx
import openai
import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request
from integration.providers._assistants_route_errors_support import (
    ASSISTANT_MODEL,
    BODY_ROUTES,
    CONTROL_MODEL,
    MAPPED_BY_STATUS,
    MARKED_ROUTES,
    MARKER,
    ROUTE_BY_NAME,
    ROUTE_IDS,
    ROUTES,
    SESSION_HEADER,
    STATUS_TABLE,
    UNMAPPED_500,
    Call,
    Mapped,
    Route,
    answer_all,
    assert_answered_with_its_own_marker,
    assert_clean_message,
    assert_mapped_upstream_error,
    assert_openai_error,
    assert_reached_upstream_once,
    assistants_wire,
    error_reply,
    free_port,
    held_error_peer,
    held_one_by_one,
    held_upstream_connections,
    live_worker_pids,
    new_marker,
    openai_assistants_config,
    owned_config,
    provider_requests,
    proxy_path,
    request_body,
    scripted_message,
    success_peer,
    success_trail,
    upstream_trail,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))

_DEPLOYMENT_TIMEOUT_SECONDS: Final = 15
_AZURE_CREDENTIALS: Final = (
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_AD_TOKEN",
    "AZURE_OPENAI_AD_TOKEN",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
)
_PRIVATE_DETAILS: Final = ("10.20.30.40", "/etc/litellm/secrets/db.yaml", "sk-proj-" + "a1B2c3D4" * 5)
_URL_MODEL: Final = "http://169.254.169.254/latest"
_BURST_STATUSES: Final = (400, 401, 404, 429, 500, 503)
_SDK_ERRORS: Final = MappingProxyType({404: openai.NotFoundError, 429: openai.RateLimitError})
_BODY_VARIANTS: Final = ("text_502", "empty_503", "hostile_400", "null_message_404", "garbage_200")


@dataclass(frozen=True, slots=True)
class _Scripted:
    gateway: Gateway
    port: int
    log: Path


@pytest.fixture(scope="module")
def scripted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Scripted]:
    directory: Final = tmp_path_factory.mktemp("assistants-route-errors-scripted")
    port: Final = free_port()
    config: Final = openai_assistants_config(directory, port, _DEPLOYMENT_TIMEOUT_SECONDS)
    with (
        gateway_from_environment() as shared,
        owned_proxy_process(shared, directory, {}, config=config, workers=2) as owned,
    ):
        yield _Scripted(owned.gateway, port, owned.log)


@pytest.fixture(scope="module")
def azure_without_credentials(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("assistants-route-errors-azure")
    settings: Final[dict[str, JsonValue]] = {
        "custom_llm_provider": "azure",
        "litellm_params": {"api_base": "http://127.0.0.1:9/", "api_version": "2024-05-01-preview", "max_retries": 0},
    }
    config: Final = owned_config(directory, (), settings, {"missing_session_id": "reject"})
    with (
        gateway_from_environment() as shared,
        owned_proxy_process(
            shared, directory, {}, config=config, remove_environment=_AZURE_CREDENTIALS, workers=2
        ) as owned,
    ):
        yield owned.gateway


def _send(
    gateway: Gateway, route: Route, marker: str, headers: Mapping[str, str] = MappingProxyType({})
) -> httpx.Response:
    return gateway.request(route.method, proxy_path(route, marker), request_body(route, marker), headers=headers)


def _send_raw(gateway: Gateway, route: Route, marker: str, content: bytes) -> httpx.Response:
    return gateway.client.request(
        route.method,
        proxy_path(route, marker),
        content=content,
        headers={"Authorization": f"Bearer {gateway.key}", "content-type": "application/json"},
    )


def _scripted_error(status: int, marker: str) -> Reply:
    return error_reply(status, scripted_message(marker, status))


@pytest.mark.parametrize("status", [mapped.status for mapped in STATUS_TABLE])
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_route_keeps_the_mapped_upstream_status(scripted: _Scripted, route: Route, status: int) -> None:
    marker: Final = new_marker()
    with assistants_wire(lambda _: _scripted_error(status, marker), port=scripted.port) as wire:
        response: Final = _send(scripted.gateway, route, marker)
        assert_mapped_upstream_error(response, MAPPED_BY_STATUS[status], marker)
        assert_reached_upstream_once(wire.drain(), route, marker)


def _sdk_call(client: openai.OpenAI, route: Route, marker: str) -> object:
    match route.name:
        case "get_assistants":
            return client.beta.assistants.list()
        case "create_assistant":
            return client.beta.assistants.create(model=ASSISTANT_MODEL, name=marker)
        case "delete_assistant":
            return client.beta.assistants.delete(f"asst_{marker}")
        case "create_thread":
            return client.beta.threads.create(messages=[{"role": "user", "content": marker}])
        case "get_thread":
            return client.beta.threads.retrieve(f"thread_{marker}")
        case "add_message":
            return client.beta.threads.messages.create(f"thread_{marker}", role="user", content=marker)
        case "get_messages":
            return client.beta.threads.messages.list(f"thread_{marker}")
        case _:
            return client.beta.threads.runs.create(f"thread_{marker}", assistant_id=f"asst_{marker}")


async def _async_sdk_call(client: openai.AsyncOpenAI, route: Route, marker: str) -> object:
    match route.name:
        case "get_assistants":
            return await client.beta.assistants.list()
        case "create_assistant":
            return await client.beta.assistants.create(model=ASSISTANT_MODEL, name=marker)
        case "delete_assistant":
            return await client.beta.assistants.delete(f"asst_{marker}")
        case "create_thread":
            return await client.beta.threads.create(messages=[{"role": "user", "content": marker}])
        case "get_thread":
            return await client.beta.threads.retrieve(f"thread_{marker}")
        case "add_message":
            return await client.beta.threads.messages.create(f"thread_{marker}", role="user", content=marker)
        case "get_messages":
            return await client.beta.threads.messages.list(f"thread_{marker}")
        case _:
            return await client.beta.threads.runs.create(f"thread_{marker}", assistant_id=f"asst_{marker}")


def _sdk_base_url(gateway: Gateway) -> str:
    return f"{str(gateway.client.base_url).rstrip('/')}/v1"


def _assert_sdk_error(error: openai.APIStatusError, mapped: Mapped, marker: str) -> None:
    assert (error.status_code, error.code, error.type, error.param) == (
        mapped.status,
        str(mapped.status),
        mapped.type,
        mapped.param,
    ), error.message
    assert marker in error.message, error.message
    assert_clean_message(error.message)


@pytest.mark.parametrize("status", sorted(_SDK_ERRORS))
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_sync_sdk_raises_the_mapped_error(scripted: _Scripted, route: Route, status: int) -> None:
    marker: Final = new_marker()
    with (
        assistants_wire(lambda _: _scripted_error(status, marker), port=scripted.port) as wire,
        openai.OpenAI(
            base_url=_sdk_base_url(scripted.gateway), api_key=scripted.gateway.key, max_retries=0, timeout=30
        ) as client,
    ):
        with pytest.raises(_SDK_ERRORS[status]) as raised:
            _sdk_call(client, route, marker)
        assert_reached_upstream_once(wire.drain(), route, marker)
    _assert_sdk_error(raised.value, MAPPED_BY_STATUS[status], marker)


@pytest.mark.parametrize("status", sorted(_SDK_ERRORS))
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_async_sdk_raises_the_mapped_error(scripted: _Scripted, route: Route, status: int) -> None:
    marker: Final = new_marker()
    with assistants_wire(lambda _: _scripted_error(status, marker), port=scripted.port) as wire:
        async with openai.AsyncOpenAI(
            base_url=_sdk_base_url(scripted.gateway), api_key=scripted.gateway.key, max_retries=0, timeout=30
        ) as client:
            with pytest.raises(_SDK_ERRORS[status]) as raised:
                await _async_sdk_call(client, route, marker)
        assert_reached_upstream_once(wire.drain(), route, marker)
    _assert_sdk_error(raised.value, MAPPED_BY_STATUS[status], marker)


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_internal_details_in_an_upstream_message_are_redacted(scripted: _Scripted, route: Route) -> None:
    marker: Final = new_marker()
    upstream_message: Final = f"{scripted_message(marker, 500)} at {' '.join(_PRIVATE_DETAILS)}"
    with assistants_wire(lambda _: error_reply(500, upstream_message), port=scripted.port) as wire:
        response: Final = _send(scripted.gateway, route, marker)
        assert_reached_upstream_once(wire.drain(), route, marker)
    message: Final = assert_openai_error(response, MAPPED_BY_STATUS[500])
    assert marker in message and "REDACTED" in message, response.text
    assert [detail for detail in _PRIVATE_DETAILS if detail in response.text] == [], response.text


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_azure_without_credentials_answers_a_clean_connection_error(
    azure_without_credentials: Gateway, route: Route
) -> None:
    response: Final = _send(azure_without_credentials, route, new_marker(), SESSION_HEADER)
    message: Final = assert_openai_error(response, UNMAPPED_500)
    assert "AzureException" in message, response.text


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_refused_upstream_answers_a_connection_error(scripted: _Scripted, route: Route) -> None:
    response: Final = _send(scripted.gateway, route, new_marker())
    message: Final = assert_openai_error(response, UNMAPPED_500)
    assert "litellm.APIConnectionError" in message, response.text


async def test_upstream_held_past_the_deployment_timeout_answers_408_on_every_route(scripted: _Scripted) -> None:
    marker: Final = new_marker()
    release: Final = threading.Event()
    arrived: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        arrived.put(request.target)
        assert release.wait(timeout=60), "The held requests were never released"
        return _scripted_error(404, marker)

    with assistants_wire(held, port=scripted.port) as wire:
        try:
            answers: Final = await answer_all(scripted.gateway, tuple(Call(route, marker) for route in ROUTES))
        finally:
            release.set()
        received: Final = wire.drain()
    assert len(answers) == len(ROUTES)
    for answer in answers:
        assert "litellm.Timeout" in assert_openai_error(answer.response, MAPPED_BY_STATUS[408]), answer.response.text
    assert sorted(upstream_trail(received)) == sorted((route.method, proxy_path(route, marker)) for route in ROUTES)


def _plain_exception_cases() -> tuple[tuple[str, str, bytes, str | None], ...]:
    malformed: Final = tuple((f"malformed_{route.name}", route.name, b"{not json", None) for route in BODY_ROUTES)
    return (
        ("run_without_assistant", "run_thread", b"{}", "assistant_id"),
        ("message_without_role", "add_message", b"{}", "role"),
        *malformed,
    )


@pytest.mark.parametrize(
    ("route_name", "content", "named"),
    [case[1:] for case in _plain_exception_cases()],
    ids=[case[0] for case in _plain_exception_cases()],
)
def test_plain_exception_answers_an_openai_shaped_500(
    scripted: _Scripted, route_name: str, content: bytes, named: str | None
) -> None:
    marker: Final = new_marker()
    with assistants_wire(lambda _: _scripted_error(404, marker), port=scripted.port) as wire:
        response: Final = _send_raw(scripted.gateway, ROUTE_BY_NAME[route_name], marker, content)
        assert provider_requests(wire.drain()) == ()
    message: Final = assert_openai_error(response, UNMAPPED_500)
    assert message, response.text
    assert named is None or named in message, response.text


@pytest.mark.parametrize("route", BODY_ROUTES, ids=[route.name for route in BODY_ROUTES])
def test_url_valued_model_answers_like_chat_completions(scripted: _Scripted, route: Route) -> None:
    marker: Final = new_marker()
    body: Final[dict[str, JsonValue]] = {**(request_body(route, marker) or {}), "model": _URL_MODEL}
    with assistants_wire(lambda _: _scripted_error(404, marker), port=scripted.port) as wire:
        assistants: Final = scripted.gateway.request(route.method, proxy_path(route, marker), body)
        chat: Final = scripted.gateway.request(
            "POST", "/v1/chat/completions", {"model": _URL_MODEL, "messages": [{"role": "user", "content": marker}]}
        )
        assert provider_requests(wire.drain()) == ()
    assert chat.status_code == 400, chat.text
    assert (assistants.status_code, assistants.json()) == (chat.status_code, chat.json()), assistants.text


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_missing_session_id_rejection_keeps_its_400(azure_without_credentials: Gateway, route: Route) -> None:
    response: Final = _send(azure_without_credentials, route, new_marker())
    message: Final = assert_openai_error(response, Mapped(400, "bad_request_error", "session_id"))
    assert "session id" in message, response.text


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_missing_assistant_settings_answers_an_openai_shaped_500(gateway: Gateway, route: Route) -> None:
    response: Final = _send(gateway, route, new_marker())
    message: Final = assert_openai_error(response, UNMAPPED_500)
    assert "custom_llm_provider" in message, response.text


def _variant_reply(variant: str, marker: str) -> Reply:
    match variant:
        case "text_502":
            return Reply(
                status=502, content_type="text/plain", body=f"upstream proxy 10.0.0.7 failed {marker}".encode()
            )
        case "empty_503":
            return Reply(status=503, body=b"")
        case "hostile_400":
            hostile: Final = {"message": marker + "y" * 5000, "type": ["scripted"], "param": {"field": 1}, "code": 123}
            return Reply(status=400, body=json.dumps({"error": hostile}).encode())
        case "null_message_404":
            nulls: Final = {"message": None, "type": "scripted_type", "param": None, "code": None, "detail": marker}
            return Reply(status=404, body=json.dumps({"error": nulls}).encode())
        case _:
            return Reply(status=200, body=f"not json {marker}".encode())


_VARIANT_MAPPED: Final = MappingProxyType(
    {
        "text_502": Mapped(502, "internal_server_error", None),
        "empty_503": Mapped(503, "internal_server_error", None),
        "hostile_400": Mapped(400, "invalid_request_error", None),
        "null_message_404": Mapped(404, "invalid_request_error", None),
        "garbage_200": UNMAPPED_500,
    }
)
_VARIANTS_CARRYING_THE_MARKER: Final = frozenset(("text_502", "hostile_400", "null_message_404"))


@pytest.mark.parametrize("variant", _BODY_VARIANTS)
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_unusual_upstream_bodies_keep_status_and_string_fields(scripted: _Scripted, route: Route, variant: str) -> None:
    marker: Final = new_marker()
    with assistants_wire(lambda _: _variant_reply(variant, marker), port=scripted.port) as wire:
        response: Final = _send(scripted.gateway, route, marker)
        assert_reached_upstream_once(wire.drain(), route, marker)
    message: Final = assert_openai_error(response, _VARIANT_MAPPED[variant])
    assert variant not in _VARIANTS_CARRYING_THE_MARKER or marker in message, response.text
    assert "10.0.0.7" not in response.text, response.text


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_repeated_failing_request_answers_the_same_each_time(scripted: _Scripted, route: Route) -> None:
    marker: Final = new_marker()
    with assistants_wire(lambda _: _scripted_error(404, marker), port=scripted.port) as wire:
        first: Final = _send(scripted.gateway, route, marker)
        second: Final = _send(scripted.gateway, route, marker)
        received: Final = wire.drain()
    assert_mapped_upstream_error(first, MAPPED_BY_STATUS[404], marker)
    assert first.json() == second.json(), (first.text, second.text)
    assert upstream_trail(received) == ((route.method, proxy_path(route, marker)),) * 2, received


async def test_mixed_status_burst_answers_each_request_with_its_own_status_and_marker(scripted: _Scripted) -> None:
    calls: Final = tuple(Call(MARKED_ROUTES[index % len(MARKED_ROUTES)], new_marker()) for index in range(48))
    status_by_marker: Final = MappingProxyType(
        {
            call.marker: _BURST_STATUSES[(index // len(MARKED_ROUTES)) % len(_BURST_STATUSES)]
            for index, call in enumerate(calls)
        }
    )
    release: Final = threading.Event()
    arrived: Final[SimpleQueue[str]] = SimpleQueue()
    workers: Final = live_worker_pids(scripted.log)
    assert len(workers) == 2, workers
    with assistants_wire(held_error_peer(status_by_marker, arrived, release), port=scripted.port) as wire:
        try:
            async with held_one_by_one(scripted.gateway, calls, arrived, lambda: False) as held:
                held_by: Final = await asyncio.to_thread(held_upstream_connections, workers, scripted.port, len(calls))
                release.set()
                answers: Final = await held.answers()
        finally:
            release.set()
        received: Final = wire.drain()
    assert sum(held_by.values()) == len(calls) and min(held_by.values()) > 0, held_by
    assert len(answers) == len(calls)
    for answer in answers:
        assert_answered_with_its_own_marker(answer, MAPPED_BY_STATUS[status_by_marker[answer.call.marker]])
    assert sorted(upstream_trail(received)) == sorted(
        (call.route.method, proxy_path(call.route, call.marker)) for call in calls
    )


async def test_outage_then_recovery_answers_connection_errors_then_upstream_bodies(scripted: _Scripted) -> None:
    marker: Final = new_marker()
    refused: Final = await answer_all(scripted.gateway, tuple(Call(route, marker) for route in ROUTES))
    for answer in refused:
        assert "litellm.APIConnectionError" in assert_openai_error(answer.response, UNMAPPED_500), answer.response.text
    with assistants_wire(success_peer(marker), port=scripted.port) as wire:
        recovered: Final = tuple(_send(scripted.gateway, route, marker) for route in ROUTES)
        chat: Final = scripted.gateway.request(
            "POST", "/v1/chat/completions", {"model": CONTROL_MODEL, "messages": [{"role": "user", "content": marker}]}
        )
        received: Final = wire.drain()
    for route, response in zip(ROUTES, recovered, strict=True):
        assert response.status_code == 200, (route.name, response.text)
        assert marker in response.text, (route.name, response.text)
    assert chat.status_code == 200 and marker in chat.text, chat.text
    assert upstream_trail(received) == (
        *(step for route in ROUTES for step in success_trail(route, marker)),
        ("POST", "/v1/chat/completions"),
    ), received


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_control_upstream_success_answers_200_with_the_upstream_body(scripted: _Scripted, route: Route) -> None:
    marker: Final = new_marker()
    with assistants_wire(success_peer(marker), port=scripted.port) as wire:
        response: Final = _send(scripted.gateway, route, marker)
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert marker in response.text, response.text
    assert upstream_trail(received) == success_trail(route, marker), received


def test_control_streamed_run_keeps_the_upstream_404(scripted: _Scripted) -> None:
    marker: Final = new_marker()
    route: Final = ROUTE_BY_NAME["run_thread"]
    with assistants_wire(lambda _: _scripted_error(404, marker), port=scripted.port) as wire:
        response: Final = scripted.gateway.request(
            "POST", proxy_path(route, marker), {"assistant_id": f"asst_{marker}", "stream": True}
        )
        assert_reached_upstream_once(wire.drain(), route, marker)
    assert response.status_code == 404, response.text
    assert MARKER.search(response.text) is not None, response.text
