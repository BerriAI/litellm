from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.responses_vendor import same_response
from integration._support.wire import wire_server
from integration.authorization._responses_compact import (
    ALIASES,
    MODEL,
    OTHER_TEAM_DETAIL,
    OTHER_USER_DETAIL,
    SCRIPTED_FAILURES,
    UNMANAGED_DETAIL,
    CompactUpstream,
    Tenants,
    Upstreamed,
    assert_compact_rows,
    assert_compacted,
    assert_compacted_reply,
    assert_refused,
    async_compact,
    base_url,
    compact_body,
    compact_hits,
    compact_post,
    compact_raw,
    compacted_id,
    compacts,
    leaks,
    marker,
    mint,
    minted_id,
    sdk,
    sdk_compact,
    successful_markers,
    tenants,
    upstream_compact_body,
    upstreamed,
    vendor_id,
)
from pydantic import JsonValue

_CLIENTS: Final = ("sync", "async", "httpx")


def _compact_with(
    client: Literal["sync", "async", "httpx"],
    gateway: Gateway,
    key: str,
    prefix: str,
    model: str,
    text: str,
    previous: str,
) -> None:
    match client:
        case "sync":
            assert_compacted(sdk_compact(gateway, key, prefix, model, text, previous), model, text)
        case "async":
            assert_compacted(async_compact(gateway, key, prefix, model, text, previous), model, text)
        case "httpx":
            assert_compacted_reply(
                compact_post(gateway, key, prefix, compact_body(model, text, {"previous_response_id": previous}), text),
                model,
                text,
            )


@pytest.mark.parametrize("prefix", ALIASES)
@pytest.mark.parametrize("client", _CLIENTS)
def test_owner_compact_forwards_the_raw_previous_response_id(
    gateway: Gateway, client: Literal["sync", "async", "httpx"], prefix: str
) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        assert len(upstreamed(wire)) == 1
        text: Final = marker("compact")
        _compact_with(client, gateway, actors.owner, prefix, actors.model, text, owned)
        assert compacts(upstreamed(wire)) == (
            upstream_compact_body(text, {"previous_response_id": minted_id(created)}),
        )
        assert_compact_rows(actors.owner, frozenset({text}))


def test_admin_compact_decodes_owned_ids_and_forwards_vendor_ids(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        foreign: Final = vendor_id()
        owned_text: Final = marker("admin_owned")
        foreign_text: Final = marker("admin_vendor")
        assert_compacted(
            sdk_compact(gateway, gateway.key, "/v1", actors.model, owned_text, owned), actors.model, owned_text
        )
        assert_compacted(
            sdk_compact(gateway, gateway.key, "/v1", actors.model, foreign_text, foreign), actors.model, foreign_text
        )
        assert compacts(upstreamed(wire)) == (
            upstream_compact_body(owned_text, {"previous_response_id": minted_id(created)}),
            upstream_compact_body(foreign_text, {"previous_response_id": foreign}),
        )


def test_compacted_response_id_belongs_to_the_compacting_owner(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        first: Final = marker("compact")
        compacted: Final = sdk_compact(gateway, actors.owner, "/v1", actors.model, first, owned)
        assert_compacted(compacted, actors.model, first)
        upstreamed(wire)
        follow: Final = marker("follow")
        followed: Final = sdk(gateway, actors.owner, "/v1").responses.create(
            model=actors.model, input=follow, previous_response_id=compacted.id
        )
        assert same_response(followed.id, minted_id(follow)), followed.model_dump_json()
        assert [(request.target, request.body) for request in upstreamed(wire)] == [
            ("/v1/responses", {"model": MODEL, "input": follow, "previous_response_id": compacted_id(first)})
        ]
        second: Final = marker("compact_again")
        assert_compacted(
            sdk_compact(gateway, actors.owner, "", actors.model, second, compacted.id), actors.model, second
        )
        assert compacts(upstreamed(wire)) == (
            upstream_compact_body(second, {"previous_response_id": compacted_id(first)}),
        )
        refused_user: Final = marker("refused_user")
        refused_team: Final = marker("refused_team")
        assert_refused(
            compact_post(
                gateway,
                actors.other_user,
                "/v1",
                compact_body(actors.model, refused_user, {"previous_response_id": compacted.id}),
                refused_user,
            ),
            OTHER_USER_DETAIL,
        )
        assert_refused(
            compact_post(
                gateway,
                actors.other_team,
                "/openai/v1",
                compact_body(actors.model, refused_team, {"previous_response_id": compacted.id}),
                refused_team,
            ),
            OTHER_TEAM_DETAIL,
        )
        assert upstreamed(wire) == ()


def test_compact_without_previous_response_id_is_forwarded_unchanged(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        texts: Final = tuple(marker(f"bare{index}") for index in range(3))
        for key, prefix, text in zip((actors.owner, actors.other_user, gateway.key), ALIASES, texts, strict=True):
            assert_compacted_reply(
                compact_post(gateway, key, prefix, compact_body(actors.model, text, {}), text), actors.model, text
            )
        assert compacts(upstreamed(wire)) == tuple(upstream_compact_body(text, {}) for text in texts)
        assert_compact_rows(actors.owner, frozenset({texts[0]}))


@pytest.mark.parametrize("prefix", ALIASES)
def test_compact_refuses_other_tenants_and_unmanaged_ids_before_the_provider(gateway: Gateway, prefix: str) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        refusals: Final = (
            (actors.other_user, owned, OTHER_USER_DETAIL, marker("other_user")),
            (actors.other_team, owned, OTHER_TEAM_DETAIL, marker("other_team")),
            (actors.owner, vendor_id(), UNMANAGED_DETAIL, marker("unmanaged")),
        )
        for key, previous, detail, text in refusals:
            assert_refused(
                compact_post(
                    gateway, key, prefix, compact_body(actors.model, text, {"previous_response_id": previous}), text
                ),
                detail,
            )
        assert upstreamed(wire) == ()
        barrier: Final = marker("barrier")
        assert_compacted_reply(
            compact_post(
                gateway,
                actors.owner,
                prefix,
                compact_body(actors.model, barrier, {"previous_response_id": owned}),
                barrier,
            ),
            actors.model,
            barrier,
        )
        assert compacts(upstreamed(wire)) == (
            upstream_compact_body(barrier, {"previous_response_id": minted_id(created)}),
        )
        assert_compact_rows(actors.owner, frozenset({barrier}))
        refused_texts: Final = frozenset(text for _, _, _, text in refusals)
        assert successful_markers(actors.owner, refused_texts) == frozenset()
        assert successful_markers(actors.other_user, refused_texts) == frozenset()
        assert successful_markers(actors.other_team, refused_texts) == frozenset()


_MALFORMED: Final[Mapping[str, Callable[[str], JsonValue]]] = {
    "int": lambda owned: 0,
    "list": lambda owned: [owned],
    "object": lambda owned: {"id": owned},
    "empty": lambda owned: "",
    "null": lambda owned: None,
}


@pytest.mark.parametrize("shape", tuple(_MALFORMED))
def test_non_string_previous_response_id_never_reaches_the_provider_decoded(gateway: Gateway, shape: str) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        text: Final = marker(f"malformed_{shape}")
        value: Final = _MALFORMED[shape](owned)
        response: Final = compact_post(
            gateway, actors.other_user, "/v1", compact_body(actors.model, text, {"previous_response_id": value}), text
        )
        received: Final = upstreamed(wire)
        assert not leaks(received, minted_id(created)), received
        assert (response.status_code, compacts(received)) == _MALFORMED_OUTCOME[shape](text, value), response.text
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def _forwarded(text: str, value: JsonValue) -> tuple[int, tuple[Mapping[str, JsonValue], ...]]:
    return 200, (upstream_compact_body(text, {"previous_response_id": value}),)


def _dropped(text: str, value: JsonValue) -> tuple[int, tuple[Mapping[str, JsonValue], ...]]:
    return 200, (upstream_compact_body(text, {}),)


_MALFORMED_OUTCOME: Final[Mapping[str, Callable[[str, JsonValue], tuple[int, tuple[Mapping[str, JsonValue], ...]]]]] = {
    "int": _forwarded,
    "list": _forwarded,
    "object": _forwarded,
    "empty": _forwarded,
    "null": _dropped,
}


def _tampered(owned: str) -> str:
    middle: Final = len(owned) // 2
    swapped: Final = "B" if owned[middle] == "A" else "A"
    return f"{owned[:middle]}{swapped}{owned[middle + 1 :]}"


_DEGENERATE: Final[Mapping[str, Callable[[str], str]]] = {
    "oversized": lambda owned: "resp_" + "a" * 5120,
    "oversized_unprefixed": lambda owned: "x" * 5120,
    "bare_prefix": lambda owned: "resp_",
    "tampered": _tampered,
    "doubled_prefix": lambda owned: f"resp_{owned}",
}


@pytest.mark.parametrize("shape", tuple(_DEGENERATE))
def test_degenerate_previous_response_ids_are_refused_as_unmanaged(gateway: Gateway, shape: str) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        text: Final = marker(f"degenerate_{shape}")
        assert_refused(
            compact_post(
                gateway,
                actors.owner,
                "/v1",
                compact_body(actors.model, text, {"previous_response_id": _DEGENERATE[shape](owned)}),
                text,
            ),
            UNMANAGED_DETAIL,
        )
        assert upstreamed(wire) == ()
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def _duplicated(model: str, text: str, first: str, last: str) -> bytes:
    return (
        f'{{"model": {json.dumps(model)}, "input": [{{"role": "user", "content": [{{"type": "input_text", '
        f'"text": {json.dumps(text)}}}]}}], "previous_response_id": {json.dumps(first)}, '
        f'"previous_response_id": {json.dumps(last)}}}'
    ).encode()


def test_duplicated_previous_response_id_is_checked_on_the_value_the_provider_gets(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        victim_mint: Final = marker("victim")
        attacker_mint: Final = marker("attacker")
        victim: Final = mint(gateway, actors.owner, actors.model, victim_mint)
        attacker: Final = mint(gateway, actors.other_user, actors.model, attacker_mint)
        upstreamed(wire)
        same: Final = marker("dup_same")
        assert_compacted_reply(
            compact_raw(gateway, actors.owner, _duplicated(actors.model, same, victim, victim), same),
            actors.model,
            same,
        )
        victim_last: Final = marker("dup_victim_last")
        assert_refused(
            compact_raw(
                gateway, actors.other_user, _duplicated(actors.model, victim_last, attacker, victim), victim_last
            ),
            OTHER_USER_DETAIL,
        )
        own_last: Final = marker("dup_own_last")
        assert_compacted_reply(
            compact_raw(gateway, actors.other_user, _duplicated(actors.model, own_last, victim, attacker), own_last),
            actors.model,
            own_last,
        )
        received: Final = upstreamed(wire)
        assert compacts(received) == (
            upstream_compact_body(same, {"previous_response_id": minted_id(victim_mint)}),
            upstream_compact_body(own_last, {"previous_response_id": minted_id(attacker_mint)}),
        )


def test_client_sent_retained_response_id_cannot_address_another_owner(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        victim_mint: Final = marker("victim")
        attacker_mint: Final = marker("attacker")
        victim: Final = mint(gateway, actors.owner, actors.model, victim_mint)
        attacker: Final = mint(gateway, actors.other_user, actors.model, attacker_mint)
        upstreamed(wire)
        retained_victim: Final = marker("retained_victim")
        assert_refused(
            compact_post(
                gateway,
                actors.other_user,
                "/v1",
                compact_body(
                    actors.model,
                    retained_victim,
                    {"previous_response_id": attacker, "_litellm_addressed_response_id": victim},
                ),
                retained_victim,
            ),
            OTHER_USER_DETAIL,
        )
        retained_number: Final = marker("retained_number")
        assert_refused(
            compact_post(
                gateway,
                actors.other_user,
                "/v1",
                compact_body(
                    actors.model, retained_number, {"previous_response_id": victim, "_litellm_addressed_response_id": 7}
                ),
                retained_number,
            ),
            OTHER_USER_DETAIL,
        )
        assert upstreamed(wire) == ()
        retained_own: Final = marker("retained_own")
        assert_compacted_reply(
            compact_post(
                gateway,
                actors.other_user,
                "/v1",
                compact_body(
                    actors.model,
                    retained_own,
                    {"previous_response_id": victim, "_litellm_addressed_response_id": attacker},
                ),
                retained_own,
            ),
            actors.model,
            retained_own,
        )
        received: Final = upstreamed(wire)
        assert compacts(received) == (
            upstream_compact_body(retained_own, {"previous_response_id": minted_id(attacker_mint)}),
        )
        assert not leaks(received, minted_id(victim_mint)), received


@pytest.mark.parametrize("status", tuple(SCRIPTED_FAILURES.values()))
def test_provider_errors_on_an_owned_compact_reach_the_caller(gateway: Gateway, status: int) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        failing: Final = marker(f"fail{status}")
        response: Final = compact_post(
            gateway, actors.owner, "/v1", compact_body(actors.model, failing, {"previous_response_id": owned}), failing
        )
        assert response.status_code == status, response.text
        assert f"scripted {status} for {failing}" in response.json()["error"]["message"], response.text
        assert compact_hits(upstreamed(wire), failing) == (
            upstream_compact_body(failing, {"previous_response_id": minted_id(created)}),
        )
        after: Final = marker("after_failure")
        assert_compacted_reply(
            compact_post(
                gateway, actors.owner, "/v1", compact_body(actors.model, after, {"previous_response_id": owned}), after
            ),
            actors.model,
            after,
        )
        assert compacts(upstreamed(wire)) == (
            upstream_compact_body(after, {"previous_response_id": minted_id(created)}),
        )
        assert_compact_rows(actors.owner, frozenset({after}))


def test_response_minted_by_a_key_without_owner_is_addressable_by_any_key(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        ownerless: Final = scenario.key(models=[actors.model])
        created: Final = marker("ownerless")
        minted: Final = mint(gateway, ownerless, actors.model, created)
        upstreamed(wire)
        text: Final = marker("compact_ownerless")
        assert_compacted(sdk_compact(gateway, actors.other_user, "/v1", actors.model, text, minted), actors.model, text)
        assert compacts(upstreamed(wire)) == (
            upstream_compact_body(text, {"previous_response_id": minted_id(created)}),
        )


def test_identical_compacts_each_reach_the_provider_with_the_cache_on(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        text: Final = marker("twin")
        body: Final = compact_body(actors.model, text, {"previous_response_id": owned})
        rows: Final = (marker("twin_row"), marker("twin_row"))
        for row in rows:
            assert_compacted_reply(compact_post(gateway, actors.owner, "/v1", body, row), actors.model, text)
        assert (
            compacts(upstreamed(wire))
            == (upstream_compact_body(text, {"previous_response_id": minted_id(created)}),) * 2
        )
        assert_compact_rows(actors.owner, frozenset(rows))
        assert_refused(compact_post(gateway, actors.other_user, "/v1", body, marker("twin_refused")), OTHER_USER_DETAIL)
        assert upstreamed(wire) == ()


def test_cached_create_still_checks_the_previous_response_id_owner(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        follow: Final = marker("cached_follow")
        owner_client: Final = sdk(gateway, actors.owner, "/v1")
        sent: Final = {"model": MODEL, "input": follow, "previous_response_id": minted_id(created)}

        def create() -> tuple[str, tuple[Upstreamed, ...]]:
            reply: Final = owner_client.responses.create(model=actors.model, input=follow, previous_response_id=owned)
            return reply.id, upstreamed(wire)

        first_id, first_received = create()
        assert [(request.target, request.body) for request in first_received] == [("/v1/responses", sent)]
        hit_id, hit_received = eventually(create, lambda attempt: not attempt[1], seconds=60)
        assert same_response(first_id, minted_id(follow)) and same_response(hit_id, minted_id(follow)), hit_id
        assert hit_received == ()
        refused: Final = httpx.post(
            f"{base_url(gateway)}/v1/responses",
            json={"model": actors.model, "input": follow, "previous_response_id": owned},
            headers={"Authorization": f"Bearer {actors.other_user}"},
            timeout=120,
            trust_env=False,
        )
        assert_refused(refused, OTHER_USER_DETAIL)
        assert upstreamed(wire) == ()


@dataclass(frozen=True, slots=True)
class _BurstCall:
    kind: Literal["owner", "other_user", "other_team", "unmanaged", "admin_vendor", "bare"]
    prefix: str
    text: str


_BURST_KINDS: Final = ("owner", "other_user", "other_team", "unmanaged", "admin_vendor", "bare")
_REFUSALS: Final[Mapping[str, str]] = {
    "other_user": OTHER_USER_DETAIL,
    "other_team": OTHER_TEAM_DETAIL,
    "unmanaged": UNMANAGED_DETAIL,
}


def _burst_call(gateway: Gateway, actors: Tenants, owned: str, foreign: str, call: _BurstCall) -> httpx.Response:
    plans: Final[Mapping[str, tuple[str, Mapping[str, JsonValue]]]] = {
        "owner": (actors.owner, {"previous_response_id": owned}),
        "other_user": (actors.other_user, {"previous_response_id": owned}),
        "other_team": (actors.other_team, {"previous_response_id": owned}),
        "unmanaged": (actors.owner, {"previous_response_id": foreign}),
        "admin_vendor": (gateway.key, {"previous_response_id": foreign}),
        "bare": (actors.owner, {}),
    }
    key, extra = plans[call.kind]
    return compact_post(gateway, key, call.prefix, compact_body(actors.model, call.text, extra), call.text)


def test_concurrent_mixed_compact_burst_gives_every_caller_its_own_verdict(gateway: Gateway) -> None:
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        created: Final = marker("mint")
        owned: Final = mint(gateway, actors.owner, actors.model, created)
        upstreamed(wire)
        foreign: Final = vendor_id()
        calls: Final = tuple(
            _BurstCall(_BURST_KINDS[index % 6], ALIASES[index % 3], marker(f"burst{index}")) for index in range(36)
        )
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            responses: Final = tuple(pool.map(partial(_burst_call, gateway, actors, owned, foreign), calls))
        for call, response in zip(calls, responses, strict=True):
            if call.kind in _REFUSALS:
                assert_refused(response, _REFUSALS[call.kind])
            else:
                assert_compacted_reply(response, actors.model, call.text)
        received: Final = upstreamed(wire)
        expected: Final[Mapping[str, Mapping[str, JsonValue]]] = {
            "owner": {"previous_response_id": minted_id(created)},
            "admin_vendor": {"previous_response_id": foreign},
            "bare": {},
        }
        assert {call.text: compact_hits(received, call.text) for call in calls} == {
            call.text: (upstream_compact_body(call.text, expected[call.kind]),) if call.kind in expected else ()
            for call in calls
        }
        owner_texts: Final = frozenset(call.text for call in calls if call.kind in ("owner", "bare"))
        assert_compact_rows(actors.owner, owner_texts)
        refused_texts: Final = frozenset(call.text for call in calls if call.kind in _REFUSALS)
        assert successful_markers(actors.owner, refused_texts) == frozenset()
        assert successful_markers(actors.other_user, refused_texts) == frozenset()
        assert successful_markers(actors.other_team, refused_texts) == frozenset()


def _settings_proxy_config(tmp_path: Path, name: str, settings: str) -> Path:
    config: Final = tmp_path / f"{name}.yaml"
    config.write_text(f"general_settings:\n{settings}")
    return config


def test_allow_unmanaged_response_ids_still_scopes_proxy_issued_ids_on_compact(
    gateway: Gateway, tmp_path: Path
) -> None:
    config: Final = _settings_proxy_config(tmp_path, "allow_unmanaged", "  allow_unmanaged_response_ids: true\n")
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        with owned_proxy(gateway, tmp_path, {"INTEGRATION_PROXY_WORKERS": "1"}, config=config) as allowed:
            created: Final = marker("mint")
            owned: Final = mint(allowed, actors.owner, actors.model, created)
            upstreamed(wire)
            foreign: Final = vendor_id()
            vendor_text: Final = marker("allowed_vendor")
            assert_compacted_reply(
                compact_post(
                    allowed,
                    actors.owner,
                    "/v1",
                    compact_body(actors.model, vendor_text, {"previous_response_id": foreign}),
                    vendor_text,
                ),
                actors.model,
                vendor_text,
            )
            other_text: Final = marker("allowed_other_user")
            assert_refused(
                compact_post(
                    allowed,
                    actors.other_user,
                    "/v1",
                    compact_body(actors.model, other_text, {"previous_response_id": owned}),
                    other_text,
                ),
                OTHER_USER_DETAIL,
            )
            owner_text: Final = marker("allowed_owner")
            assert_compacted_reply(
                compact_post(
                    allowed,
                    actors.owner,
                    "/v1",
                    compact_body(actors.model, owner_text, {"previous_response_id": owned}),
                    owner_text,
                ),
                actors.model,
                owner_text,
            )
            received: Final = upstreamed(wire)
    assert compacts(received) == (
        upstream_compact_body(vendor_text, {"previous_response_id": foreign}),
        upstream_compact_body(owner_text, {"previous_response_id": minted_id(created)}),
    )


def test_disable_responses_id_security_lets_any_key_compact_and_forwards_raw_ids(
    gateway: Gateway, tmp_path: Path
) -> None:
    config: Final = _settings_proxy_config(tmp_path, "disable_security", "  disable_responses_id_security: true\n")
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        sealed_mint: Final = marker("sealed")
        sealed: Final = mint(gateway, actors.owner, actors.model, sealed_mint)
        with owned_proxy(gateway, tmp_path, {"INTEGRATION_PROXY_WORKERS": "1"}, config=config) as disabled:
            open_mint: Final = marker("open")
            opened: Final = mint(disabled, actors.owner, actors.model, open_mint)
            upstreamed(wire)
            foreign: Final = vendor_id()
            plans: Final = (
                (marker("disabled_open"), opened, minted_id(open_mint)),
                (marker("disabled_sealed"), sealed, minted_id(sealed_mint)),
                (marker("disabled_vendor"), foreign, foreign),
            )
            for text, previous, _ in plans:
                assert_compacted_reply(
                    compact_post(
                        disabled,
                        actors.other_user,
                        "/v1",
                        compact_body(actors.model, text, {"previous_response_id": previous}),
                        text,
                    ),
                    actors.model,
                    text,
                )
            received: Final = upstreamed(wire)
    assert compacts(received) == tuple(
        upstream_compact_body(text, {"previous_response_id": raw}) for text, _, raw in plans
    )


def test_null_and_empty_id_security_settings_keep_the_compact_checks(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _settings_proxy_config(
        tmp_path,
        "null_and_empty",
        "  allow_unmanaged_response_ids: null\n  disable_responses_id_security: ''\n",
    )
    upstream: Final = CompactUpstream()
    with wire_server(upstream.respond) as wire, gateway.scenario() as scenario:
        actors: Final = tenants(scenario, wire)
        with owned_proxy(gateway, tmp_path, {"INTEGRATION_PROXY_WORKERS": "1"}, config=config) as defaulted:
            created: Final = marker("mint")
            owned: Final = mint(defaulted, actors.owner, actors.model, created)
            upstreamed(wire)
            vendor_text: Final = marker("defaulted_vendor")
            assert_refused(
                compact_post(
                    defaulted,
                    actors.owner,
                    "/v1",
                    compact_body(actors.model, vendor_text, {"previous_response_id": vendor_id()}),
                    vendor_text,
                ),
                UNMANAGED_DETAIL,
            )
            other_text: Final = marker("defaulted_other_user")
            assert_refused(
                compact_post(
                    defaulted,
                    actors.other_user,
                    "/v1",
                    compact_body(actors.model, other_text, {"previous_response_id": owned}),
                    other_text,
                ),
                OTHER_USER_DETAIL,
            )
            assert upstreamed(wire) == ()
            owner_text: Final = marker("defaulted_owner")
            assert_compacted_reply(
                compact_post(
                    defaulted,
                    actors.owner,
                    "/v1",
                    compact_body(actors.model, owner_text, {"previous_response_id": owned}),
                    owner_text,
                ),
                actors.model,
                owner_text,
            )
            received: Final = upstreamed(wire)
    assert compacts(received) == (upstream_compact_body(owner_text, {"previous_response_id": minted_id(created)}),)
