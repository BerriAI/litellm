from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, assert_never
from urllib.parse import urlsplit

import httpx
import psycopg
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.database import scratch_database
from integration._support.process import graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.worker_readiness import wait_for_worker_readiness
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_UPSTREAM_KEY: Final = "synthetic-openai-key"
_MODEL: Final = "gpt-5"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MANAGED: Final = "general_settings:\n  passthrough_managed_object_ids: true\n"
_ALLOW_UNMANAGED: Final = (
    "general_settings:\n  passthrough_managed_object_ids: true\n  allow_unmanaged_response_ids: true\n"
)
_UNMANAGED_DETAIL: Final = (
    "Forbidden. This response id was not issued by this proxy, so the proxy cannot tell who owns it. "
    "To let keys address responses this proxy did not issue, set "
    "general_settings::allow_unmanaged_response_ids to True in the config.yaml file."
)
_LOOKUP_FAILED_DETAIL: Final = (
    "The proxy could not check who owns this response id because its database lookup failed. Retry the request."
)
_ACCESS_DENIED: Final = "Access denied to managed resource."
_NOT_FOUND: Final = "Managed resource not found."
_TAKE_OWNER_TABLE_OFFLINE: Final = (
    'ALTER TABLE "LiteLLM_ManagedObjectTable" RENAME TO "LiteLLM_ManagedObjectTable_offline"'
)
_BRING_OWNER_TABLE_BACK: Final = (
    'ALTER TABLE "LiteLLM_ManagedObjectTable_offline" RENAME TO "LiteLLM_ManagedObjectTable"'
)

Route = Literal[
    "retrieve",
    "input_items",
    "delete",
    "cancel",
    "create",
    "compact",
    "input_tokens",
    "openai_passthrough",
    "percent_encoded",
]


def _vendor_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _response(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": _MODEL,
            "output": [],
        }
    ).encode()


def _requested_vendor_id(request: Request) -> str | None:
    if not request.body.startswith(b"{"):
        return None
    requested: Final = _JSON_OBJECT.validate_json(request.body).get("input")
    return requested if isinstance(requested, str) and requested.startswith("resp_") else None


def _respond(request: Request) -> Reply:
    addressed: Final = next(
        (segment for segment in urlsplit(request.target).path.split("/") if segment.startswith("resp_")), None
    )
    return Reply(body=_response(addressed or _requested_vendor_id(request) or _vendor_id()))


def _forwarded(wire: Wire) -> tuple[tuple[str, str], ...]:
    return tuple((request.method, urlsplit(request.target).path) for request in wire.drain())


def _send(candidate: Gateway, key: str, method: str, path: str, body: bytes | None = None) -> httpx.Response:
    json_header: Final = {"Content-Type": "application/json"} if body is not None else {}
    return candidate.client.request(
        method, path, content=body, headers={"Authorization": f"Bearer {key}", **json_header}
    )


def _continue(previous_response_id: str) -> bytes:
    return json.dumps({"model": _MODEL, "input": "continue", "previous_response_id": previous_response_id}).encode()


def _user_key(scenario: Scenario) -> str:
    return scenario.key(user_id=scenario.user())


def _create(candidate: Gateway, key: str, vendor_id: str, prefix: str = "/openai") -> str:
    created: Final = _send(
        candidate, key, "POST", f"{prefix}/responses", json.dumps({"model": _MODEL, "input": vendor_id}).encode()
    )
    assert created.status_code == 200, created.text
    issued: Final = created.json()["id"]
    assert isinstance(issued, str) and issued != vendor_id, created.text
    return issued


def _assert_unmanaged_refusal(response: httpx.Response) -> None:
    assert response.status_code == 403, response.text
    assert response.json() == {
        "error": {"message": _UNMANAGED_DETAIL, "type": "permission_error", "param": None, "code": "403"}
    }, response.text


@contextmanager
def _proxy(
    gateway: Gateway,
    directory: Path,
    upstream: Wire,
    settings: str = _MANAGED,
    overrides: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[Gateway]:
    config: Final = directory / f"passthrough-{uuid.uuid4().hex}.yaml"
    config.write_text(settings)
    with owned_proxy(
        gateway,
        directory,
        {"OPENAI_API_BASE": upstream.url, "OPENAI_API_KEY": _UPSTREAM_KEY, **overrides},
        config=config,
    ) as candidate:
        yield candidate


@contextmanager
def _owner_table_offline(database_url: str) -> Iterator[None]:
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute(_TAKE_OWNER_TABLE_OFFLINE)
        try:
            yield
        finally:
            admin.execute(_BRING_OWNER_TABLE_BACK)


def _call(route: Route, vendor_id: str) -> tuple[str, str, bytes | None]:
    match route:
        case "retrieve":
            return ("GET", f"/openai/responses/{vendor_id}", None)
        case "input_items":
            return ("GET", f"/openai/responses/{vendor_id}/input_items", None)
        case "delete":
            return ("DELETE", f"/openai/responses/{vendor_id}", None)
        case "cancel":
            return ("POST", f"/openai/responses/{vendor_id}/cancel", b"{}")
        case "create":
            return ("POST", "/openai/responses", _continue(vendor_id))
        case "compact":
            return ("POST", "/openai/responses/compact", _continue(vendor_id))
        case "input_tokens":
            return ("POST", "/openai/responses/input_tokens", _continue(vendor_id))
        case "openai_passthrough":
            return ("GET", f"/openai_passthrough/v1/responses/{vendor_id}", None)
        case "percent_encoded":
            return ("GET", f"/openai/responses/%72{vendor_id[1:]}", None)
        case _:
            assert_never(route)


def test_the_owner_keeps_using_the_gateway_id_and_another_key_cannot(gateway: Gateway, tmp_path: Path) -> None:
    vendor_id: Final = _vendor_id()
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        owner: Final = _user_key(scenario)
        other: Final = _user_key(scenario)
        with _proxy(gateway, tmp_path, upstream) as candidate:
            issued: Final = _create(candidate, owner, vendor_id)
            assert _forwarded(upstream) == (("POST", "/responses"),)

            retrieved: Final = _send(candidate, owner, "GET", f"/openai/responses/{issued}")
            assert retrieved.status_code == 200, retrieved.text
            assert retrieved.json()["id"] == issued, retrieved.text
            assert _forwarded(upstream) == (("GET", f"/responses/{vendor_id}"),)

            continued: Final = _send(candidate, owner, "POST", "/openai/responses", _continue(issued))
            assert continued.status_code == 200, continued.text
            sent: Final = upstream.drain()
            assert [(request.method, request.target) for request in sent] == [("POST", "/responses")], sent
            assert _JSON_OBJECT.validate_json(sent[0].body)["previous_response_id"] == vendor_id, sent[0].body

            refused: Final = _send(candidate, other, "GET", f"/openai/responses/{issued}")
            assert refused.status_code == 403, refused.text
            assert refused.json()["error"]["message"] == _ACCESS_DENIED, refused.text
            assert _forwarded(upstream) == ()


@pytest.mark.parametrize(
    "route",
    (
        "retrieve",
        "input_items",
        "delete",
        "cancel",
        "create",
        "compact",
        "input_tokens",
        "openai_passthrough",
        "percent_encoded",
    ),
)
def test_a_vendor_id_the_gateway_never_issued_is_refused_before_it_reaches_openai(
    gateway: Gateway, tmp_path: Path, route: Route
) -> None:
    method, path, body = _call(route, _vendor_id())
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        key: Final = _user_key(scenario)
        with _proxy(gateway, tmp_path, upstream) as candidate:
            _assert_unmanaged_refusal(_send(candidate, key, method, path, body))
        assert _forwarded(upstream) == ()


def test_another_keys_vendor_id_is_not_found_and_never_forwarded(gateway: Gateway, tmp_path: Path) -> None:
    vendor_id: Final = _vendor_id()
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        owner: Final = _user_key(scenario)
        other: Final = _user_key(scenario)
        with _proxy(gateway, tmp_path, upstream) as candidate:
            _create(candidate, owner, vendor_id)
            assert _forwarded(upstream) == (("POST", "/responses"),)
            refusals: Final = tuple(
                _send(candidate, other, *_call(route, vendor_id)) for route in ("retrieve", "create")
            )
            assert [response.status_code for response in refusals] == [404, 404], [r.text for r in refusals]
            assert _forwarded(upstream) == ()


@pytest.mark.parametrize("exemption", ("admin", "allow_unmanaged", "flag_off"))
def test_exempt_callers_reach_openai_with_the_vendor_id(
    gateway: Gateway, tmp_path: Path, exemption: Literal["admin", "allow_unmanaged", "flag_off"]
) -> None:
    vendor_id: Final = _vendor_id()
    settings: Final = {"admin": _MANAGED, "allow_unmanaged": _ALLOW_UNMANAGED, "flag_off": "general_settings: {}\n"}[
        exemption
    ]
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        key: Final = gateway.key if exemption == "admin" else _user_key(scenario)
        with _proxy(gateway, tmp_path, upstream, settings) as candidate:
            retrieved: Final = _send(candidate, key, "GET", f"/openai/responses/{vendor_id}")
            assert retrieved.status_code == 200, retrieved.text
            continued: Final = _send(candidate, key, "POST", "/openai/responses", _continue(vendor_id))
            assert continued.status_code == 200, continued.text
        sent: Final = upstream.drain()
        assert [(request.method, request.target) for request in sent] == [
            ("GET", f"/responses/{vendor_id}"),
            ("POST", "/responses"),
        ], sent
        assert _JSON_OBJECT.validate_json(sent[1].body)["previous_response_id"] == vendor_id, sent[1].body


@pytest.mark.parametrize(
    "value",
    (123, ["resp_listed"], "", None),
    ids=("int", "list", "empty", "null"),
)
def test_a_previous_response_id_that_is_not_an_id_string_is_forwarded_unchanged(
    gateway: Gateway, tmp_path: Path, value: JsonValue
) -> None:
    body: Final = {"model": _MODEL, "input": "hostile", "previous_response_id": value}
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        key: Final = _user_key(scenario)
        with _proxy(gateway, tmp_path, upstream) as candidate:
            response: Final = _send(candidate, key, "POST", "/openai/responses", json.dumps(body).encode())
            assert response.status_code == 200, response.text
        sent: Final = upstream.drain()
        assert [(request.method, request.target) for request in sent] == [("POST", "/responses")], sent
        assert _JSON_OBJECT.validate_json(sent[0].body) == body, sent[0].body


def test_an_oversized_vendor_id_is_refused_before_it_reaches_openai(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        key: Final = _user_key(scenario)
        with _proxy(gateway, tmp_path, upstream) as candidate:
            _assert_unmanaged_refusal(
                _send(candidate, key, "POST", "/openai/responses", _continue("resp_" + "a" * 5000))
            )
        assert _forwarded(upstream) == ()


def test_a_duplicated_previous_response_id_is_judged_and_forwarded_as_one_value(
    gateway: Gateway, tmp_path: Path
) -> None:
    owned_vendor_id: Final = _vendor_id()
    foreign_vendor_id: Final = _vendor_id()
    with wire_server(_respond) as upstream, gateway.scenario() as scenario:
        key: Final = _user_key(scenario)
        with _proxy(gateway, tmp_path, upstream) as candidate:
            issued: Final = _create(candidate, key, owned_vendor_id)
            upstream.drain()
            vendor_last: Final = (
                '{"model":"%s","input":"twice","previous_response_id":"%s","previous_response_id":"%s"}'
                % (_MODEL, issued, foreign_vendor_id)
            ).encode()
            _assert_unmanaged_refusal(_send(candidate, key, "POST", "/openai/responses", vendor_last))
            assert _forwarded(upstream) == ()

            issued_last: Final = (
                '{"model":"%s","input":"twice","previous_response_id":"%s","previous_response_id":"%s"}'
                % (_MODEL, foreign_vendor_id, issued)
            ).encode()
            continued: Final = _send(candidate, key, "POST", "/openai/responses", issued_last)
            assert continued.status_code == 200, continued.text
        sent: Final = upstream.drain()
        assert [(request.method, request.target) for request in sent] == [("POST", "/responses")], sent
        assert sent[0].body.count(b"previous_response_id") == 1, sent[0].body
        assert _JSON_OBJECT.validate_json(sent[0].body)["previous_response_id"] == owned_vendor_id, sent[0].body


def test_a_failed_owner_lookup_answers_a_retryable_503_and_forwards_nothing(gateway: Gateway, tmp_path: Path) -> None:
    vendor_id: Final = _vendor_id()
    with (
        scratch_database("passthrough_lookup") as database_url,
        wire_server(_respond) as upstream,
        _proxy(gateway, tmp_path, upstream, overrides={"DATABASE_URL": database_url}) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = _user_key(scenario)
        with _owner_table_offline(database_url):
            failed: Final = _send(candidate, key, "GET", f"/openai/responses/{vendor_id}")
        assert failed.status_code == 503, failed.text
        assert failed.json()["error"]["message"] == _LOOKUP_FAILED_DETAIL, failed.text
        assert _forwarded(upstream) == ()
        _assert_unmanaged_refusal(_send(candidate, key, "GET", f"/openai/responses/{vendor_id}"))
        assert _forwarded(upstream) == ()


def test_the_model_list_relay_hands_out_and_checks_gateway_ids(gateway: Gateway, tmp_path: Path) -> None:
    vendor_id: Final = _vendor_id()
    with (
        wire_server(_respond) as azure,
        wire_server(_respond) as openai_upstream,
        _proxy(gateway, tmp_path, openai_upstream) as candidate,
        candidate.scenario() as scenario,
    ):
        group: Final = scenario.model(model=f"azure/{_MODEL}", api_base=azure.url, api_key=_UPSTREAM_KEY)
        wait_for_worker_readiness(candidate, group, 1, 1)
        owner: Final = _user_key(scenario)
        other: Final = _user_key(scenario)
        relay: Final = f"/azure/{group}/openai/v1"
        issued: Final = _create(candidate, owner, vendor_id, relay)
        assert _forwarded(azure) == (("POST", "/openai/v1/responses"),)

        retrieved: Final = _send(candidate, owner, "GET", f"{relay}/responses/{issued}")
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["id"] == issued, retrieved.text
        assert _forwarded(azure) == (("GET", f"/openai/v1/responses/{vendor_id}"),)

        refusals: Final = tuple(
            _send(candidate, other, "GET", path)
            for path in (
                f"/azure/{group}/openai/v1/responses/{_vendor_id()}",
                f"/azure_ai/{group}/openai/v1/responses/{_vendor_id()}",
                f"{relay}/responses/{issued}",
                f"{relay}/responses/{vendor_id}",
            )
        )
        assert [(response.status_code, response.json()) for response in refusals] == [
            (403, {"detail": _UNMANAGED_DETAIL}),
            (403, {"detail": _UNMANAGED_DETAIL}),
            (403, {"detail": _ACCESS_DENIED}),
            (404, {"detail": _NOT_FOUND}),
        ], [response.text for response in refusals]
        assert _forwarded(azure) == ()
        assert _forwarded(openai_upstream) == ()


def test_every_verdict_holds_through_a_burst_after_a_worker_is_replaced(gateway: Gateway, tmp_path: Path) -> None:
    vendor_ids: Final = tuple(_vendor_id() for _ in range(10))
    config: Final = tmp_path / "passthrough-burst.yaml"
    config.write_text(_MANAGED)
    with (
        wire_server(_respond) as upstream,
        gateway.scenario() as scenario,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"OPENAI_API_BASE": upstream.url, "OPENAI_API_KEY": _UPSTREAM_KEY},
            config=config,
            workers=2,
        ) as owned,
    ):
        owner: Final = _user_key(scenario)
        other: Final = _user_key(scenario)
        issued: Final = tuple(_create(owned.gateway, owner, vendor_id) for vendor_id in vendor_ids)
        upstream.drain()
        workers: Final = owned.worker_pids()
        assert len(workers) >= 2, workers
        owned.kill_worker(min(workers))

        calls: Final = (
            *((owner, f"/openai/responses/{identity}", 200) for identity in issued),
            *((other, f"/openai/responses/{_vendor_id()}", 403) for _ in range(10)),
            *((other, f"/openai/responses/{vendor_id}", 404) for vendor_id in vendor_ids),
        )

        def call(entry: tuple[str, str, int]) -> tuple[int, int]:
            key, path, expected = entry
            return expected, _send(owned.gateway, key, "GET", path).status_code

        with ThreadPoolExecutor(max_workers=10) as pool:
            outcomes: Final = tuple(pool.map(call, calls))
        assert all(expected == observed for expected, observed in outcomes), outcomes
        assert sorted(_forwarded(upstream)) == sorted(("GET", f"/responses/{vendor_id}") for vendor_id in vendor_ids)
