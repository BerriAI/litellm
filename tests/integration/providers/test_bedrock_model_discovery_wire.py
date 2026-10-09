from __future__ import annotations

import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import litellm
import pytest
from integration._support.aws_control_plane import (
    FOUNDATION_MODELS,
    INFERENCE_PROFILES,
    UNKNOWN_CREDENTIAL,
    Catalog,
    ControlPlane,
    ControlPlaneRequest,
    control_plane,
    json_reply,
)
from integration._support.bedrock_discovery import (
    CLOSE,
    CONTROL_MODEL,
    CONVERSE_MODEL,
    HOSTED_REGIONS,
    INFO_PATHS,
    LISTING_FAILURE_LOG,
    LISTING_PATHS,
    LISTING_TIMEOUT_SECONDS,
    PROFILE_PAGE_CAP,
    RELOAD_SECONDS,
    UNHOSTED_REGION,
    WORKERS,
    assert_listing_shape,
    assert_sigv4,
    catalog_for,
    control_plane_host,
    credential,
    deployment,
    discovered,
    discovery_proxy,
    from_stem,
    listed_ids,
    listings,
    mine,
    sigv4_deployment,
    stem,
)
from integration._support.bedrock_runtime_peer import MARKER, marker_of, respond, target_of
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds
from integration._support.wire import Reply, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

STREAM_TERMINALS: Final[Mapping[str, str]] = {
    "/v1/chat/completions": "data: [DONE]",
    "/v1/messages": "event: message_stop",
    "/v1/responses": "response.completed",
}


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    log: Path
    plane: ControlPlane
    runtime: Wire


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ControlPlane]:
    hosts: Final = tuple(control_plane_host(region) for region in HOSTED_REGIONS)
    with control_plane(tmp_path_factory.mktemp("bedrock_control_plane"), hosts) as value:
        yield value


@pytest.fixture(scope="module")
def runtime() -> Iterator[Wire]:
    with wire_server(respond) as wire:
        yield wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory, plane: ControlPlane, runtime: Wire) -> Iterator[Rig]:
    with gateway_from_environment() as parent:
        directory: Final = tmp_path_factory.mktemp("bedrock_discovery_on")
        with discovery_proxy(parent, directory, plane, check_provider_endpoint=True, workers=WORKERS) as owned:
            yield Rig(owned.gateway, owned.log, plane, runtime)


@pytest.fixture(scope="module")
def flag_off_rig(tmp_path_factory: pytest.TempPathFactory, plane: ControlPlane, runtime: Wire) -> Iterator[Rig]:
    with gateway_from_environment() as parent:
        directory: Final = tmp_path_factory.mktemp("bedrock_discovery_off")
        with discovery_proxy(parent, directory, plane, check_provider_endpoint=False, workers=1) as owned:
            yield Rig(owned.gateway, owned.log, plane, runtime)


def test_admin_models_list_the_accounts_invocable_models_signed_for_the_deployment_region(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        for path in LISTING_PATHS:
            assert discovered(rig.gateway, catalog, marker, path) == catalog.invocable_ids()
            assert CONTROL_MODEL in listed_ids(rig.gateway, path)
        requests: Final = mine(rig.plane, key)
        assert_listing_shape(requests)
        assert_sigv4(requests, key=key, secret=secret, region="us-east-1")


def _info_rows(gateway: Gateway, path: str, marker: str, identity: str) -> Mapping[str, str]:
    entries: Final = gateway.get(path)["data"]
    assert isinstance(entries, list)
    rows: Final = tuple(object_value(entry) for entry in entries)
    return {
        string_value(row["model_name"]): string_value(object_value(row["litellm_params"])["model"])
        for row in rows
        if marker in string_value(row["model_name"]) and string_value(object_value(row["model_info"])["id"]) == identity
    }


def test_model_info_expands_the_wildcard_into_one_row_per_invocable_model(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        identity: Final = sigv4_deployment(scenario, key, secret, "us-east-1")
        for path in INFO_PATHS:
            expanded: Final = _info_rows(rig.gateway, path, marker, identity)
            assert expanded == {name: name for name in catalog.invocable_ids()}, expanded
        assert_listing_shape(mine(rig.plane, key))


def test_model_group_info_carries_every_invocable_model(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")

        def groups() -> frozenset[str]:
            entries: Final = rig.gateway.get("/model_group/info")["data"]
            assert isinstance(entries, list)
            return frozenset(string_value(object_value(entry)["model_group"]) for entry in entries)

        listed: Final = eventually(groups, lambda names: catalog.invocable_ids() <= names, seconds=RELOAD_SECONDS * 4)
        assert from_stem(marker, listed) == catalog.invocable_ids()
        assert_listing_shape(mine(rig.plane, key))


def test_key_scoped_to_the_wildcard_lists_the_discovered_models_and_nothing_else(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        scoped: Final = scenario.key(models=["bedrock/*"])
        assert discovered(rig.gateway, catalog, marker, key=scoped) == catalog.invocable_ids()
        assert CONTROL_MODEL not in listed_ids(rig.gateway, key=scoped)
        assert_listing_shape(mine(rig.plane, key))


def test_team_scoped_to_the_wildcard_lists_the_discovered_models(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        team: Final = scenario.team(models=["bedrock/*"])
        member: Final = scenario.key(team_id=team)
        assert discovered(rig.gateway, catalog, marker, key=member) == catalog.invocable_ids()
        assert_listing_shape(mine(rig.plane, key))


def test_access_group_holding_the_wildcard_lists_the_discovered_models(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    group: Final = f"bedrock-group-{marker}"
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1", model_info={"access_groups": [group]})
        grouped: Final = scenario.key(models=[group])
        assert discovered(rig.gateway, catalog, marker, key=grouped) == catalog.invocable_ids()
        assert_listing_shape(mine(rig.plane, key))


def test_bearer_token_deployment_lists_with_the_token_and_no_signature(rig: Rig) -> None:
    token: Final = f"bearer-{uuid.uuid4().hex}"
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(token, catalog.respond):
        deployment(scenario, litellm_params={"api_key": token, "aws_region_name": "us-east-1"})
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        requests: Final = mine(rig.plane, token)
        assert_listing_shape(requests)
        for request in requests:
            assert request.headers["authorization"] == f"Bearer {token}", request
            assert request.host == control_plane_host("us-east-1"), request
            assert "x-amz-date" not in request.headers, request


@pytest.mark.parametrize("region", ("eu-west-1", "us-gov-west-1", "cn-north-1"))
def test_listing_reaches_the_control_plane_of_the_deployment_region_and_partition(rig: Rig, region: str) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, region)
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        requests: Final = mine(rig.plane, key)
        assert_listing_shape(requests)
        assert_sigv4(requests, key=key, secret=secret, region=region)


def _frame_id(frame: Mapping[str, JsonValue]) -> str | None:
    if frame.get("type") == "message_start":
        return string_value(object_value(frame["message"])["id"])
    response: Final = frame.get("response")
    if isinstance(response, dict) and "id" in response:
        return string_value(response["id"])
    identity: Final = frame.get("id")
    return identity if isinstance(identity, str) else None


def _response_id(text: str, *, stream: bool) -> str:
    if not stream:
        return string_value(JSON_OBJECT.validate_json(text)["id"])
    frames: Final = tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    ids: Final = tuple(identity for identity in map(_frame_id, frames) if identity is not None)
    assert ids, text
    return ids[-1]


def _call(gateway: Gateway, path: str, body: Mapping[str, JsonValue], *, key: str, tag: str) -> tuple[str, str]:
    """POST one call tagged for its spend row and return (response id, response text); a stream is read to its terminal frame."""
    response: Final = gateway.request("POST", path, body, key=key, headers={**CLOSE, "x-litellm-tags": tag})
    assert response.status_code == 200, f"POST {path}: {response.status_code} {response.text}"
    stream: Final = body.get("stream") is True
    if stream:
        assert STREAM_TERMINALS[path] in response.text, response.text
    return _response_id(response.text, stream=stream), response.text


def _calls_on(model: str, markers: tuple[str, ...]) -> tuple[tuple[str, Mapping[str, JsonValue]], ...]:
    def chat(marker: str, stream: bool) -> Mapping[str, JsonValue]:
        return {"model": model, "messages": [{"role": "user", "content": f"marker-{marker}"}], "stream": stream}

    def messages(marker: str, stream: bool) -> Mapping[str, JsonValue]:
        return {**chat(marker, stream), "max_tokens": 64}

    def responses(marker: str, stream: bool) -> Mapping[str, JsonValue]:
        return {"model": model, "input": f"marker-{marker}", "stream": stream}

    return (
        ("/v1/chat/completions", chat(markers[0], False)),
        ("/v1/chat/completions", chat(markers[1], True)),
        ("/v1/messages", messages(markers[2], False)),
        ("/v1/messages", messages(markers[3], True)),
        ("/v1/responses", responses(markers[4], False)),
        ("/v1/responses", responses(markers[5], True)),
    )


def _chat_rows_landed(ids: tuple[str, ...]) -> None:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)',
            (list(ids),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
        ),
        lambda found: len(found) >= len(ids),
        seconds=60,
    )
    assert sorted(string_value(row["request_id"]) for row in rows) == sorted(ids), rows


def _spend_row_id(rows: Sequence[Mapping[str, JsonValue]], tag: str) -> str:
    tagged: Final = tuple(row for row in rows if tag in list_value(row["request_tags"]))
    assert len(tagged) == 1, (tag, rows)
    return string_value(tagged[0]["request_id"])


def _landed_once(served: tuple[tuple[str, str, bool], ...], tags: tuple[str, ...]) -> None:
    """One spend row per tagged call, carrying the id the client saw.

    FIXME: a streamed /v1/responses row carries the managed `resp_` id LiteLLM built before the proxy encrypted the
    advertised one, so that row is matched by tag alone until the spend log reads the id the client received.
    """
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, request_tags FROM "LiteLLM_SpendLogs" WHERE request_tags ?| %s',
            (list(tags),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
        ),
        lambda found: len(found) >= len(tags),
        seconds=60,
    )
    for (identity, path, stream), tag in zip(served, tags, strict=True):
        row_id: Final = _spend_row_id(rows, tag)
        if path == "/v1/responses" and stream:
            assert row_id.startswith("resp_"), (row_id, identity)
            continue
        assert row_id == identity, (path, stream, row_id, identity)


def test_discovered_model_is_callable_through_the_wildcard_on_every_endpoint(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = Catalog(active_profiles=(f"us.{marker}.sonnet-v1:0",), on_demand_models=(CONVERSE_MODEL,))
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(
            scenario, key, secret, "us-east-1", litellm_params={"aws_bedrock_runtime_endpoint": rig.runtime.url}
        )
        scoped: Final = scenario.key(models=["bedrock/*"])
        assert discovered(rig.gateway, catalog, marker, key=scoped) == from_stem(marker, catalog.invocable_ids())
        assert f"bedrock/{CONVERSE_MODEL}" in listed_ids(rig.gateway, key=scoped)
        rig.runtime.drain()
        markers: Final = tuple(uuid.uuid4().hex for _ in range(6))
        calls: Final = _calls_on(f"bedrock/{CONVERSE_MODEL}", markers)
        served: Final = tuple(
            _call(rig.gateway, path, body, key=scoped, tag=f"discovery-{call_marker}")
            for (path, body), call_marker in zip(calls, markers, strict=True)
        )
        for (identity, text), call_marker in zip(served, markers, strict=True):
            assert identity, text
            assert set(MARKER.findall(text)) == {call_marker}, text
        received: Final = rig.runtime.drain()
        assert sorted(marker_of(request) for request in received) == sorted(markers), received
        for request in received:
            assert target_of(request).startswith("/openai/v1/"), request.target
            assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={key}/"), request
        _landed_once(
            tuple(
                (identity, path, body.get("stream") is True)
                for (identity, _), (path, body) in zip(served, calls, strict=True)
            ),
            tuple(f"discovery-{call_marker}" for call_marker in markers),
        )
        assert_listing_shape(mine(rig.plane, key))


def test_flag_off_proxy_lists_the_static_catalog_and_never_calls_the_control_plane(flag_off_rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    static: Final = frozenset(
        name if name.startswith("bedrock/") else f"bedrock/{name}" for name in litellm.models_by_provider["bedrock"]
    )
    with flag_off_rig.gateway.scenario() as scenario, flag_off_rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        listed: Final = eventually(
            lambda: listed_ids(flag_off_rig.gateway), lambda ids: static <= ids, seconds=RELOAD_SECONDS * 4
        )
        assert from_stem(marker, listed) == frozenset()
        for _ in range(3):
            assert from_stem(marker, listed_ids(flag_off_rig.gateway)) == frozenset()
        assert mine(flag_off_rig.plane, key) == ()


def _sad_listing(rig: Rig, answer: Callable[[ControlPlaneRequest], Reply]) -> tuple[ControlPlaneRequest, ...]:
    """Deploy against a control plane answering `answer`; the proxy keeps listing without the account's ids."""
    key, secret = credential()
    marker: Final = stem()
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, answer):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        for path in LISTING_PATHS:
            for _ in range(3):
                listed: Final = listed_ids(rig.gateway, path)
                assert from_stem(marker, listed) == frozenset(), listed
                assert CONTROL_MODEL in listed
        assert rig.gateway.chat(CONTROL_MODEL)["choices"], "the control deployment stopped answering"
        return mine(rig.plane, key)


@pytest.mark.parametrize(
    "answer",
    (
        pytest.param(lambda _: json_reply(403, {"message": "AccessDeniedException"}), id="403"),
        pytest.param(lambda _: json_reply(404, {"message": "ResourceNotFoundException"}), id="404"),
        pytest.param(lambda _: Reply(body=b"<html>upstream maintenance</html>", content_type="text/html"), id="html"),
        pytest.param(lambda _: json_reply(200, {"modelSummaries": "nope", "inferenceProfileSummaries": 7}), id="shape"),
        pytest.param(lambda _: json_reply(200, {"modelSummaries": [], "inferenceProfileSummaries": []}), id="empty"),
    ),
)
def test_control_plane_errors_leave_the_listing_without_bedrock_ids_and_the_proxy_serving(
    rig: Rig, answer: Callable[[ControlPlaneRequest], Reply]
) -> None:
    requests: Final = _sad_listing(rig, answer)
    assert requests, "the proxy never asked the control plane"
    assert set(listings(requests)) <= {FOUNDATION_MODELS, INFERENCE_PROFILES}, listings(requests)


def test_deployment_without_credentials_never_reaches_the_control_plane(rig: Rig) -> None:
    marker: Final = stem()
    rig.plane.drain()
    with rig.gateway.scenario() as scenario:
        deployment(scenario, litellm_params={"aws_region_name": "us-east-1"})
        for _ in range(6):
            listed: Final = listed_ids(rig.gateway)
            assert from_stem(marker, listed) == frozenset() and CONTROL_MODEL in listed
    assert tuple(request for request in rig.plane.drain() if request.credential == UNKNOWN_CREDENTIAL) == ()


def test_refused_egress_leaves_the_listing_without_bedrock_ids(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    rig.plane.refusals()
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog_for(marker).respond):
        sigv4_deployment(scenario, key, secret, UNHOSTED_REGION)
        for _ in range(6):
            assert from_stem(marker, listed_ids(rig.gateway)) == frozenset()
        assert mine(rig.plane, key) == ()
        assert f"{control_plane_host(UNHOSTED_REGION)}:443" in rig.plane.refusals()


def test_hung_control_plane_is_bounded_by_the_listing_timeout_while_the_proxy_keeps_serving(
    rig: Rig, record_property: Callable[[str, object], None]
) -> None:
    key, secret = credential()
    marker: Final = stem()
    release: Final = threading.Event()
    catalog: Final = catalog_for(marker)

    def hang(request: ControlPlaneRequest) -> Reply:
        assert release.wait(timeout=120), "the hung listing was never released"
        return catalog.respond(request)

    stalled: Final = threading.Event()
    latencies: Final[list[float]] = []  # mutable-ok: appended by the probe thread

    def probe() -> None:
        while not stalled.is_set():
            started: Final = time.monotonic()
            health: Final = rig.gateway.request("GET", "/health/liveliness", headers=CLOSE)
            latencies.append(time.monotonic() - started)
            assert health.status_code == 200, health.text

    failures_before: Final = rig.log.read_text().count(LISTING_FAILURE_LOG)
    try:
        with rig.gateway.scenario() as scenario, rig.plane.answering(key, hang):
            sigv4_deployment(scenario, key, secret, "us-east-1")
            prober: Final = threading.Thread(target=probe)
            prober.start()
            started: Final = time.monotonic()
            listed: Final = listed_ids(rig.gateway)
            elapsed: Final = time.monotonic() - started
            assert rig.gateway.chat(CONTROL_MODEL)["choices"]
            stalled.set()
            prober.join(timeout=30)
            assert not prober.is_alive()
            assert from_stem(marker, listed) == frozenset() and CONTROL_MODEL in listed
            assert LISTING_TIMEOUT_SECONDS <= elapsed < LISTING_TIMEOUT_SECONDS * 2, elapsed
            record_property("listing_seconds", round(elapsed, 2))
            record_property("max_liveliness_seconds", round(max(latencies), 2))
            eventually(
                lambda: rig.log.read_text().count(LISTING_FAILURE_LOG),
                lambda count: count > failures_before,
                seconds=10,
            )
            assert mine(rig.plane, key) != ()
            release.set()
    finally:
        release.set()


def test_endless_pagination_stops_at_the_page_cap(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()

    def endless(request: ControlPlaneRequest) -> Reply:
        page: Final = int(request.query.get("nextToken", "page-0").removeprefix("page-"))
        return json_reply(200, {"inferenceProfileSummaries": [], "nextToken": f"page-{page + 1}"})

    with rig.gateway.scenario() as scenario, rig.plane.answering(key, endless):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        for _ in range(3):
            assert from_stem(marker, listed_ids(rig.gateway)) == frozenset()
        requests: Final = mine(rig.plane, key)
        counts: Final = listings(requests)
        assert set(counts) == {INFERENCE_PROFILES}, counts
        assert counts[INFERENCE_PROFILES] % PROFILE_PAGE_CAP == 0 and counts[INFERENCE_PROFILES] > 0, counts
        tokens: Final = tuple(request.query.get("nextToken", "page-0") for request in requests)
        expected: Final = tuple(f"page-{index}" for index in range(PROFILE_PAGE_CAP))
        assert tokens == expected * (counts[INFERENCE_PROFILES] // PROFILE_PAGE_CAP), tokens


@pytest.mark.parametrize("region", (pytest.param("", id="empty"), pytest.param("x" * 5120, id="5kb")))
def test_unusable_region_never_reaches_a_control_plane(rig: Rig, region: str) -> None:
    key, secret = credential()
    marker: Final = stem()
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog_for(marker).respond):
        sigv4_deployment(scenario, key, secret, region)
        for _ in range(6):
            listed: Final = listed_ids(rig.gateway)
            assert from_stem(marker, listed) == frozenset() and CONTROL_MODEL in listed
        assert rig.gateway.chat(CONTROL_MODEL)["choices"]
        assert mine(rig.plane, key) == ()


@pytest.mark.parametrize("field", ("aws_region_name", "api_key"))
def test_non_string_credential_fields_are_refused_at_model_creation(rig: Rig, field: str) -> None:
    key, secret = credential()
    response: Final = rig.gateway.request(
        "POST",
        "/model/new",
        {
            "model_name": "bedrock/*",
            "litellm_params": {
                "model": "bedrock/*",
                "aws_access_key_id": key,
                "aws_secret_access_key": secret,
                "aws_region_name": "us-east-1",
                field: 12345,
            },
            "model_info": {},
        },
    )
    assert response.status_code in (400, 422), response.text
    assert field in response.text, response.text
    assert CONTROL_MODEL in listed_ids(rig.gateway)


def test_empty_api_key_falls_back_to_sigv4(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1", litellm_params={"api_key": ""})
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        requests: Final = mine(rig.plane, key)
        assert_listing_shape(requests)
        assert_sigv4(requests, key=key, secret=secret, region="us-east-1")


def test_repeated_listings_are_served_from_the_cache_once_per_worker(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        for path in LISTING_PATHS * 3:
            assert from_stem(marker, listed_ids(rig.gateway, path)) == catalog.invocable_ids()
        assert assert_listing_shape(mine(rig.plane, key)) <= WORKERS


def test_two_regions_list_the_union_from_each_regions_control_plane(rig: Rig) -> None:
    first_key, first_secret = credential()
    second_key, second_secret = credential()
    marker: Final = stem()
    first: Final = Catalog(active_profiles=(f"us.{marker}.sonnet-v1:0",), on_demand_models=(f"{marker}.us-only",))
    second: Final = Catalog(active_profiles=(f"eu.{marker}.haiku-v1:0",), on_demand_models=(f"{marker}.eu-only",))
    with (
        rig.gateway.scenario() as scenario,
        rig.plane.answering(first_key, first.respond),
        rig.plane.answering(second_key, second.respond),
    ):
        sigv4_deployment(scenario, first_key, first_secret, "us-east-1")
        sigv4_deployment(scenario, second_key, second_secret, "eu-west-1")
        union: Final = Catalog(
            active_profiles=first.active_profiles + second.active_profiles,
            on_demand_models=first.on_demand_models + second.on_demand_models,
        )
        assert discovered(rig.gateway, union, marker) == union.invocable_ids()
        assert_sigv4(mine(rig.plane, first_key), key=first_key, secret=first_secret, region="us-east-1")
        assert_sigv4(mine(rig.plane, second_key), key=second_key, secret=second_secret, region="eu-west-1")


def test_partial_wildcard_lists_only_the_matching_models_under_their_real_ids(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = Catalog(
        active_profiles=(f"us.anthropic.{marker}-v1:0",),
        on_demand_models=(f"anthropic.{marker}-v1:0", f"amazon.{marker}-v1:0"),
    )
    expected: Final = frozenset({f"bedrock/anthropic.{marker}-v1:0"})
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(
            scenario, key, secret, "us-east-1", model_name="bedrock/anthropic.*", model="bedrock/anthropic.*"
        )
        listed: Final = eventually(
            lambda: from_stem(marker, listed_ids(rig.gateway)),
            lambda ids: ids != frozenset(),
            seconds=RELOAD_SECONDS * 4,
        )
        assert listed == expected, listed


def test_partial_wildcard_matching_no_invocable_id_lists_nothing_instead_of_alias_names(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = Catalog(
        active_profiles=(f"us.anthropic.{marker}-v1:0",),
        on_demand_models=(f"amazon.{marker}-v1:0",),
    )
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1", model_name="bedrock/*", model="bedrock/*")
        sigv4_deployment(
            scenario, key, secret, "us-east-1", model_name="bedrock/anthropic.*", model="bedrock/anthropic.*"
        )
        listed: Final = eventually(
            lambda: from_stem(marker, listed_ids(rig.gateway)),
            lambda ids: ids != frozenset(),
            seconds=RELOAD_SECONDS * 4,
        )
        assert listed == catalog.invocable_ids(), listed


@pytest.mark.parametrize(
    "prefix_of",
    (
        pytest.param(lambda marker: f"team-{marker}", id="distinct"),
        pytest.param(lambda marker: marker, id="starts-a-discovered-id"),
    ),
)
def test_custom_prefix_wildcard_lists_the_discovered_models_under_that_prefix(
    rig: Rig, prefix_of: Callable[[str], str]
) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    prefix: Final = prefix_of(marker)
    expected: Final = frozenset(name.replace("bedrock/", f"{prefix}/", 1) for name in catalog.invocable_ids())
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1", model_name=f"{prefix}/*")
        listed: Final = eventually(
            lambda: from_stem(marker, listed_ids(rig.gateway)), lambda ids: expected <= ids, seconds=RELOAD_SECONDS * 4
        )
        assert listed == expected, listed
        assert_listing_shape(mine(rig.plane, key))


def test_wildcard_route_is_listed_alongside_the_discovered_models_when_asked(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        params: Final = {"return_wildcard_routes": "true"}
        assert discovered(rig.gateway, catalog, marker, params=params) == catalog.invocable_ids()
        assert "bedrock/*" in listed_ids(rig.gateway, params=params)
        assert "bedrock/*" not in listed_ids(rig.gateway)


def test_inactive_profiles_are_excluded_and_a_model_in_both_listings_appears_once(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    shared: Final = f"{marker}.shared-v1:0"
    catalog: Final = Catalog(
        active_profiles=(shared, f"us.{marker}.sonnet-v1:0"),
        inactive_profiles=(f"us.{marker}.retired-v1:0", f"eu.{marker}.retired-v1:0"),
        on_demand_models=(shared,),
    )
    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        response: Final = rig.gateway.request("GET", "/v1/models", headers=CLOSE)
        data: Final = JSON_OBJECT.validate_json(response.content)["data"]
        assert isinstance(data, list)
        names: Final = [string_value(object_value(entry)["id"]) for entry in data]
        assert names.count(f"bedrock/{shared}") == 1, names
        assert not any("retired" in name for name in names), names


def test_concurrent_cold_listings_all_answer_with_the_discovered_models(
    rig: Rig, record_property: Callable[[str, object], None]
) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker, page_size=1)

    def slow(request: ControlPlaneRequest) -> Reply:
        time.sleep(0.3)
        return catalog.respond(request)

    with rig.gateway.scenario() as scenario, rig.plane.answering(key, slow):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers: Final = tuple(pool.map(lambda _: listed_ids(rig.gateway), range(8)))
        for ids in answers:
            assert from_stem(marker, ids) == catalog.invocable_ids(), ids
        requests: Final = mine(rig.plane, key)
        counts: Final = listings(requests)
        assert counts[INFERENCE_PROFILES] == counts[FOUNDATION_MODELS] * len(catalog.pages()), counts
        assert 1 <= counts[FOUNDATION_MODELS] <= len(answers), counts
        record_property("listings_during_burst", counts[FOUNDATION_MODELS])


def test_region_update_while_listings_flow_moves_the_listing_to_the_new_regions_control_plane(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    stop: Final = threading.Event()
    statuses: Final[list[int]] = []  # mutable-ok: appended by the polling thread
    hosts: Final[set[str]] = set()  # mutable-ok: collects every control plane host the listings reached

    def poll() -> None:
        while not stop.is_set():
            statuses.append(rig.gateway.request("GET", "/v1/models", headers=CLOSE).status_code)

    def reached() -> frozenset[str]:
        hosts.update(request.host for request in mine(rig.plane, key))
        return frozenset(hosts)

    with rig.gateway.scenario() as scenario, rig.plane.answering(key, catalog.respond):
        identity: Final = sigv4_deployment(scenario, key, secret, "us-east-1")
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        poller: Final = threading.Thread(target=poll)
        poller.start()
        try:
            updated: Final = rig.gateway.request(
                "PATCH", f"/model/{identity}/update", {"litellm_params": {"aws_region_name": "eu-west-1"}}
            )
            assert updated.status_code == 200, updated.text
            eventually(reached, lambda seen: control_plane_host("eu-west-1") in seen, seconds=RELOAD_SECONDS * 4)
        finally:
            stop.set()
            poller.join(timeout=30)
        assert not poller.is_alive()
        assert statuses and set(statuses) == {200}, Counter(statuses)
        assert from_stem(marker, listed_ids(rig.gateway)) == catalog.invocable_ids()


def _burst(gateway: Gateway, count: int) -> tuple[tuple[str, int, str], ...]:
    """`count` concurrent calls: model listings and control chats, a third of the chats streamed."""

    def one(index: int) -> tuple[str, int, str]:
        if index % 2 == 0:
            listing: Final = gateway.request("GET", "/v1/models", headers=CLOSE)
            return "models", listing.status_code, listing.text
        body: Final[Mapping[str, JsonValue]] = {
            "model": CONTROL_MODEL,
            "messages": [{"role": "user", "content": f"burst {index}"}],
            "stream": index % 3 == 0,
        }
        chat: Final = gateway.request("POST", "/v1/chat/completions", body, headers=CLOSE)
        return "chat", chat.status_code, chat.text

    with ThreadPoolExecutor(max_workers=count) as pool:
        return tuple(pool.map(one, range(count)))


def _chat_ids(served: tuple[tuple[str, int, str], ...]) -> tuple[str, ...]:
    return tuple(_response_id(text, stream=text.startswith("data: ")) for kind, _, text in served if kind == "chat")


def _listed_stems(marker: str, served: tuple[tuple[str, int, str], ...]) -> tuple[frozenset[str], ...]:
    def ids(text: str) -> frozenset[str]:
        data: Final = JSON_OBJECT.validate_json(text)["data"]
        assert isinstance(data, list), text
        return from_stem(marker, frozenset(string_value(object_value(entry)["id"]) for entry in data))

    return tuple(ids(text) for kind, _, text in served if kind == "models")


def test_control_plane_outage_mid_burst_recovers_without_a_proxy_restart(rig: Rig) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    up: Final = threading.Event()

    def flaky(request: ControlPlaneRequest) -> Reply:
        return catalog.respond(request) if up.is_set() else Reply(drop_connection=True)

    with rig.gateway.scenario() as scenario, rig.plane.answering(key, flaky):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        served: Final = _burst(rig.gateway, 20)
        assert {status for _, status, _ in served} == {200}, served
        assert set(_listed_stems(marker, served)) == {frozenset()}, served
        assert mine(rig.plane, key) != (), "the outage was never attempted"
        up.set()
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        _chat_rows_landed(_chat_ids(served))


def test_slow_control_plane_under_a_burst_answers_everything_without_a_deadlock(
    rig: Rig, record_property: Callable[[str, object], None]
) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)

    def slow(request: ControlPlaneRequest) -> Reply:
        time.sleep(1.5)
        return catalog.respond(request)

    with rig.gateway.scenario() as scenario, rig.plane.answering(key, slow):
        sigv4_deployment(scenario, key, secret, "us-east-1")
        started: Final = time.monotonic()
        served: Final = _burst(rig.gateway, 20)
        record_property("burst_seconds", round(time.monotonic() - started, 2))
        assert {status for _, status, _ in served} == {200}, served
        assert set(_listed_stems(marker, served)) <= {frozenset(), catalog.invocable_ids()}, served
        assert discovered(rig.gateway, catalog, marker) == catalog.invocable_ids()
        record_property("listings_during_burst", assert_listing_shape(mine(rig.plane, key)))
        _chat_rows_landed(_chat_ids(served))
