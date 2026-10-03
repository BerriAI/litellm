"""
Optional team ownership on registered tags: persistence, FK and uniqueness constraints,
and the proxy-admin-only guard for assigning or releasing an owner.
"""

import uuid
from typing import Final

import httpx
import psycopg.errors
import pytest
from pydantic import JsonValue

from litellm.proxy._types import hash_token
from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows, write_rows


def _tag_rows(name: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT tag_name, team_id FROM "LiteLLM_TagTable" WHERE tag_name = %s', (name,))


def _tag_name() -> str:
    return f"integration-{uuid.uuid4().hex}"


def _create_tag(scenario: Scenario, name: str, **fields: JsonValue) -> None:
    scenario.gateway.post("/tag/new", {"name": name, "models": [], **fields})
    scenario.cleanups.callback(_delete_tag, scenario.gateway, name)


def _delete_tag(gateway: Gateway, name: str) -> None:
    gateway.post("/tag/delete", {"name": name})
    assert _tag_rows(name) == []


def _tag_info(gateway: Gateway, name: str) -> dict[str, JsonValue]:
    return object_value(gateway.post("/tag/info", {"names": [name]})[name])


def _new_team(gateway: Gateway) -> str:
    created: Final = gateway.post("/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}"})
    return string_value(created["team_id"])


def test_deleting_the_owning_team_nulls_out_tag_ownership(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = _new_team(gateway)
        name: Final = _tag_name()
        _create_tag(scenario, name, team_id=team_id)
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        gateway.post("/team/delete", {"team_ids": [team_id]})

        assert _tag_rows(name) == [{"tag_name": name, "team_id": None}]
        assert _tag_info(gateway, name)["team_id"] is None


def test_tag_team_id_foreign_key_and_index(gateway: Gateway) -> None:
    name: Final = _tag_name()
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        write_rows(
            'INSERT INTO "LiteLLM_TagTable" (tag_name, models, team_id) VALUES (%s, %s, %s)',
            (name, [], "no-such-team"),
        )

    indexes: Final = read_rows("SELECT indexdef FROM pg_indexes WHERE tablename = 'LiteLLM_TagTable'", ())
    assert any("team_id" in str(index["indexdef"]) for index in indexes)


def test_tag_name_is_globally_unique_across_team_owners(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_a: Final = scenario.team()
        team_b: Final = scenario.team()
        name: Final = _tag_name()
        _create_tag(scenario, name, team_id=team_a)

        response: Final = gateway.request("POST", "/tag/new", {"name": name, "team_id": team_b})
        assert response.status_code == 400, response.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        with pytest.raises(psycopg.errors.UniqueViolation):
            write_rows(
                'INSERT INTO "LiteLLM_TagTable" (tag_name, models) VALUES (%s, %s)',
                (name, []),
            )


def test_non_admin_key_cannot_set_tag_team_ownership(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        user_id: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user_id, allowed_routes=["/tag/new", "/tag/update"])
        name: Final = _tag_name()

        create: Final = gateway.request("POST", "/tag/new", {"name": name, "team_id": team_id}, key=key)
        assert create.status_code == 403, create.text
        assert _tag_rows(name) == []

        _create_tag(scenario, name, team_id=team_id)

        assign: Final = gateway.request("POST", "/tag/update", {"name": name, "team_id": scenario.team()}, key=key)
        assert assign.status_code == 403, assign.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        release: Final = gateway.request("POST", "/tag/update", {"name": name, "team_id": None}, key=key)
        assert release.status_code == 403, release.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]

        unrelated: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "description": "denied edit"}, key=key
        )
        assert unrelated.status_code == 403, unrelated.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_id}]
        assert _tag_info(gateway, name)["team_id"] == team_id


def test_team_admin_manages_their_own_team_tags(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        admin_id: Final = scenario.user()
        team_a: Final = scenario.team(members_with_roles=[{"role": "admin", "user_id": admin_id}])
        team_b: Final = scenario.team()
        admin_key: Final = scenario.key(user_id=admin_id, team_id=team_a)

        name: Final = _tag_name()
        create: Final = gateway.request(
            "POST", "/tag/new", {"name": name, "team_id": team_a, "models": []}, key=admin_key
        )
        assert create.status_code == 200, create.text
        scenario.cleanups.callback(lambda: gateway.request("POST", "/tag/delete", {"name": name}))
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        update: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "description": "admin edit"}, key=admin_key
        )
        assert update.status_code == 200, update.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        transfer: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "team_id": team_b}, key=admin_key
        )
        assert transfer.status_code == 403, transfer.text
        assert _tag_rows(name) == [{"tag_name": name, "team_id": team_a}]

        foreign_tag: Final = _tag_name()
        _create_tag(scenario, foreign_tag, team_id=team_b)
        foreign: Final = gateway.request("POST", "/tag/delete", {"name": foreign_tag}, key=admin_key)
        assert foreign.status_code == 403, foreign.text
        assert _tag_rows(foreign_tag) == [{"tag_name": foreign_tag, "team_id": team_b}]

        unowned_tag: Final = _tag_name()
        _create_tag(scenario, unowned_tag)
        unowned: Final = gateway.request("POST", "/tag/delete", {"name": unowned_tag}, key=admin_key)
        assert unowned.status_code == 403, unowned.text
        assert _tag_rows(unowned_tag) == [{"tag_name": unowned_tag, "team_id": None}]

        delete: Final = gateway.request("POST", "/tag/delete", {"name": name}, key=admin_key)
        assert delete.status_code == 200, delete.text
        assert _tag_rows(name) == []


def test_regular_team_member_cannot_manage_tags_and_sees_only_own_team_tags(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        member_id: Final = scenario.user()
        team_a: Final = scenario.team(members_with_roles=[{"role": "user", "user_id": member_id}])
        team_b: Final = scenario.team()
        member_key: Final = scenario.key(user_id=member_id, team_id=team_a)

        team_a_tag: Final = _tag_name()
        _create_tag(scenario, team_a_tag, team_id=team_a)
        team_b_tag: Final = _tag_name()
        _create_tag(scenario, team_b_tag, team_id=team_b)

        member_tag: Final = _tag_name()
        create: Final = gateway.request(
            "POST", "/tag/new", {"name": member_tag, "team_id": team_a, "models": []}, key=member_key
        )
        assert create.status_code == 403, create.text
        assert _tag_rows(member_tag) == []

        update: Final = gateway.request(
            "POST", "/tag/update", {"name": team_a_tag, "description": "member edit"}, key=member_key
        )
        assert update.status_code == 403, update.text

        delete: Final = gateway.request("POST", "/tag/delete", {"name": team_a_tag}, key=member_key)
        assert delete.status_code == 403, delete.text
        assert _tag_rows(team_a_tag) == [{"tag_name": team_a_tag, "team_id": team_a}]

        listing: Final = gateway.request("GET", "/tag/list", key=member_key)
        assert listing.status_code == 200, listing.text
        entries: Final = {entry["name"]: entry for entry in listing.json()}
        assert team_a_tag in entries
        assert entries[team_a_tag]["team_id"] == team_a
        assert team_b_tag not in entries


def test_deleting_two_owning_teams_nulls_all_their_tags(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_a: Final = _new_team(gateway)
        team_b: Final = _new_team(gateway)
        team_c: Final = scenario.team()
        tag_a: Final = _tag_name()
        tag_b: Final = _tag_name()
        tag_c: Final = _tag_name()
        _create_tag(scenario, tag_a, team_id=team_a)
        _create_tag(scenario, tag_b, team_id=team_b)
        _create_tag(scenario, tag_c, team_id=team_c)

        gateway.post("/team/delete", {"team_ids": [team_a, team_b]})

        assert _tag_rows(tag_a) == [{"tag_name": tag_a, "team_id": None}]
        assert _tag_rows(tag_b) == [{"tag_name": tag_b, "team_id": None}]
        assert _tag_rows(tag_c) == [{"tag_name": tag_c, "team_id": team_c}]
        assert _tag_info(gateway, tag_a)["team_id"] is None
        assert _tag_info(gateway, tag_b)["team_id"] is None
        assert _tag_info(gateway, tag_c)["team_id"] == team_c


def _chat_with_tag(
    gateway: Gateway,
    model: str,
    marker: str,
    *,
    key: str | None = None,
    metadata_tags: list[str] | None = None,
    root_tags: list[str] | None = None,
    headers: dict[str, str] | None = None,
    base: Gateway | None = None,
) -> httpx.Response:
    body: dict[str, JsonValue] = {"model": model, "messages": [{"role": "user", "content": marker}]}
    if metadata_tags is not None:
        body["metadata"] = {"tags": metadata_tags}
    if root_tags is not None:
        body["tags"] = root_tags
    return (base or gateway).request("POST", "/v1/chat/completions", body, key=key, headers=headers)


def _upstream_marker_chats(upstream: httpx.Client, marker: str) -> list[tuple[str, ...]]:
    observed: Final = upstream.get("/__observations")
    observed.raise_for_status()
    requests: Final = object_value(observed.json())["requests"]
    assert isinstance(requests, list), observed.text
    tagged: Final = [
        object_value(value)
        for value in requests
        if marker in str(object_value(value)["body"]) and object_value(value)["path"] == "/v1/chat/completions"
    ]
    return sorted(
        tuple(
            str(object_value(message)["content"]) for message in object_value(object_value(entry)["body"])["messages"]
        )
        for entry in tagged
    )


def _assert_tag_ownership_denied(response: httpx.Response, tag: str, team_id: str) -> None:
    assert response.status_code == 403, response.text
    assert "tag_ownership_denied" in response.text, response.text
    assert tag in response.text, response.text
    assert team_id in response.text, response.text


def test_owned_tag_gates_inference_requests(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        team_a: Final = scenario.team()
        team_b: Final = scenario.team()
        key_a: Final = scenario.key(team_id=team_a)
        key_b: Final = scenario.key(team_id=team_b)
        tag: Final = _tag_name()
        _create_tag(scenario, tag, team_id=team_a)
        unowned: Final = _tag_name()
        _create_tag(scenario, unowned)
        unregistered: Final = f"unregistered-{uuid.uuid4().hex}"
        marker: Final = f"lit8516 {uuid.uuid4().hex}"
        upstream.get("/__observations").raise_for_status()

        allowed: Final = _chat_with_tag(gateway, model, f"{marker}-owner", key=key_a, metadata_tags=[tag])
        assert allowed.status_code == 200, allowed.text

        denied_body: Final = _chat_with_tag(gateway, model, f"{marker}-foreign", key=key_b, metadata_tags=[tag])
        _assert_tag_ownership_denied(denied_body, tag, team_a)

        denied_header: Final = _chat_with_tag(
            gateway, model, f"{marker}-header", key=key_b, headers={"x-litellm-tags": tag}
        )
        _assert_tag_ownership_denied(denied_header, tag, team_a)

        denied_root: Final = _chat_with_tag(gateway, model, f"{marker}-root", key=key_b, root_tags=[tag])
        _assert_tag_ownership_denied(denied_root, tag, team_a)

        unowned_ok: Final = _chat_with_tag(gateway, model, f"{marker}-unowned", key=key_b, metadata_tags=[unowned])
        assert unowned_ok.status_code == 200, unowned_ok.text

        unregistered_ok: Final = _chat_with_tag(
            gateway, model, f"{marker}-unregistered", key=key_b, metadata_tags=[unregistered]
        )
        assert unregistered_ok.status_code == 200, unregistered_ok.text

        master_denied: Final = _chat_with_tag(gateway, model, f"{marker}-master", metadata_tags=[tag])
        _assert_tag_ownership_denied(master_denied, tag, team_a)

        assert _upstream_marker_chats(upstream, marker) == [
            (f"{marker}-owner",),
            (f"{marker}-unowned",),
            (f"{marker}-unregistered",),
        ]

        hashed_b: Final = hash_token(key_b)
        failure_rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND status = %s',
                (hashed_b, "failure"),
            ),
            lambda rows: len(rows) >= 3,
            seconds=30,
        )
        assert all(row["request_tags"] == [] for row in failure_rows), failure_rows

        flush_marker: Final = f"flush-{uuid.uuid4().hex}"
        flushed: Final = _chat_with_tag(
            gateway, model, f"{marker}-flush", key=key_b, metadata_tags=[flush_marker]
        )
        assert flushed.status_code == 200, flushed.text
        eventually(
            lambda: read_rows(
                'SELECT 1 AS used FROM "LiteLLM_DailyTagSpend" WHERE tag = %s AND api_key = %s LIMIT 1',
                (flush_marker, hashed_b),
            ),
            lambda rows: len(rows) == 1,
            seconds=30,
        )
        assert (
            read_rows(
                'SELECT 1 AS used FROM "LiteLLM_DailyTagSpend" WHERE tag = %s AND api_key = %s',
                (tag, hashed_b),
            )
            == []
        )


def test_owned_tag_denies_inherited_and_mixed_sources(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        team_a: Final = scenario.team()
        team_b: Final = scenario.team()
        team_c: Final = scenario.team(metadata={"tags": []})
        tag: Final = _tag_name()
        _create_tag(scenario, tag, team_id=team_a)
        unowned: Final = _tag_name()
        _create_tag(scenario, unowned)
        key_b_inherited: Final = scenario.key(team_id=team_b, metadata={"tags": [tag]})
        key_b_plain: Final = scenario.key(team_id=team_b)
        gateway.post("/team/update", {"team_id": team_c, "metadata": {"tags": [tag]}})
        key_c: Final = scenario.key(team_id=team_c)
        marker: Final = f"lit8516 {uuid.uuid4().hex}"
        upstream.get("/__observations").raise_for_status()

        inherited_key: Final = _chat_with_tag(gateway, model, f"{marker}-key-inherited", key=key_b_inherited)
        _assert_tag_ownership_denied(inherited_key, tag, team_a)

        inherited_team: Final = _chat_with_tag(gateway, model, f"{marker}-team-inherited", key=key_c)
        _assert_tag_ownership_denied(inherited_team, tag, team_a)

        mixed: Final = _chat_with_tag(
            gateway, model, f"{marker}-mixed", key=key_b_plain, metadata_tags=[unowned, tag]
        )
        _assert_tag_ownership_denied(mixed, tag, team_a)

        assert _upstream_marker_chats(upstream, marker) == []

        hashed_inherited: Final = hash_token(key_b_inherited)
        inherited_failure_rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND status = %s',
                (hashed_inherited, "failure"),
            ),
            lambda rows: len(rows) >= 1,
            seconds=30,
        )
        assert all(row["request_tags"] == [] for row in inherited_failure_rows), inherited_failure_rows

        flush_marker: Final = f"flush-{uuid.uuid4().hex}"
        flushed: Final = _chat_with_tag(
            gateway, model, f"{marker}-flush", key=key_b_plain, metadata_tags=[flush_marker]
        )
        assert flushed.status_code == 200, flushed.text
        eventually(
            lambda: read_rows(
                'SELECT 1 AS used FROM "LiteLLM_DailyTagSpend" WHERE tag = %s AND api_key = %s LIMIT 1',
                (flush_marker, hash_token(key_b_plain)),
            ),
            lambda rows: len(rows) == 1,
            seconds=30,
        )
        assert (
            read_rows(
                'SELECT 1 AS used FROM "LiteLLM_DailyTagSpend" WHERE tag = %s AND api_key = %s',
                (tag, hashed_inherited),
            )
            == []
        )


def test_tag_ownership_transitions_take_effect_with_warm_cache(gateway: Gateway, peer: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        team_a: Final = _new_team(gateway)
        team_b: Final = scenario.team()
        key_a: Final = string_value(gateway.post("/key/generate", {"team_id": team_a})["key"])
        key_b: Final = scenario.key(team_id=team_b)
        tag: Final = _tag_name()
        _create_tag(scenario, tag, team_id=team_a)
        marker: Final = f"lit8516 {uuid.uuid4().hex}"

        warm: Final = _chat_with_tag(gateway, model, f"{marker}-warm", key=key_a, metadata_tags=[tag])
        assert warm.status_code == 200, warm.text
        warm_peer: Final = _chat_with_tag(
            gateway, model, f"{marker}-warm-peer", key=key_a, metadata_tags=[tag], base=peer
        )
        assert warm_peer.status_code == 200, warm_peer.text

        gateway.post("/tag/update", {"name": tag, "team_id": team_b})
        denied_a: Final = _chat_with_tag(gateway, model, f"{marker}-transfer-a", key=key_a, metadata_tags=[tag])
        _assert_tag_ownership_denied(denied_a, tag, team_b)
        allowed_b: Final = _chat_with_tag(gateway, model, f"{marker}-transfer-b", key=key_b, metadata_tags=[tag])
        assert allowed_b.status_code == 200, allowed_b.text

        converged: Final = eventually(
            lambda: _chat_with_tag(
                gateway, model, f"{marker}-peer-converge", key=key_a, metadata_tags=[tag], base=peer
            ).status_code,
            lambda status: status == 403,
            seconds=15,
        )
        assert converged == 403

        gateway.post("/tag/update", {"name": tag, "team_id": None})
        released_a: Final = _chat_with_tag(gateway, model, f"{marker}-release-a", key=key_a, metadata_tags=[tag])
        assert released_a.status_code == 200, released_a.text
        released_b: Final = _chat_with_tag(gateway, model, f"{marker}-release-b", key=key_b, metadata_tags=[tag])
        assert released_b.status_code == 200, released_b.text

        gateway.post("/tag/update", {"name": tag, "team_id": team_a})
        gateway.post("/team/delete", {"team_ids": [team_a]})
        assert _tag_rows(tag) == [{"tag_name": tag, "team_id": None}]
        assert _tag_info(gateway, tag)["team_id"] is None
        after_delete: Final = _chat_with_tag(gateway, model, f"{marker}-after-delete", key=key_b, metadata_tags=[tag])
        assert after_delete.status_code == 200, after_delete.text

        allowed_markers: Final = {
            f"{marker}-warm",
            f"{marker}-warm-peer",
            f"{marker}-transfer-b",
            f"{marker}-release-a",
            f"{marker}-release-b",
            f"{marker}-after-delete",
        }
        seen: Final = {content[0] for content in _upstream_marker_chats(upstream, marker) if content}
        allowed_seen: Final = {content for content in seen if content in allowed_markers}
        denied_seen: Final = seen - allowed_markers
        assert allowed_seen == allowed_markers, seen
        assert denied_seen <= {f"{marker}-peer-converge"}, denied_seen


def test_failed_owner_mutation_keeps_existing_authorization(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        team_a: Final = scenario.team()
        admin_id: Final = scenario.user()
        team_b: Final = scenario.team(members_with_roles=[{"role": "admin", "user_id": admin_id}])
        admin_b_key: Final = scenario.key(user_id=admin_id, team_id=team_b)
        key_a: Final = scenario.key(team_id=team_a)
        key_b: Final = scenario.key(team_id=team_b)
        tag: Final = _tag_name()
        _create_tag(scenario, tag, team_id=team_a)
        marker: Final = f"lit8516 {uuid.uuid4().hex}"
        upstream.get("/__observations").raise_for_status()

        forbidden: Final = gateway.request(
            "POST", "/tag/update", {"name": tag, "team_id": team_b}, key=admin_b_key
        )
        assert forbidden.status_code == 403, forbidden.text

        denied_b: Final = _chat_with_tag(gateway, model, f"{marker}-b", key=key_b, metadata_tags=[tag])
        _assert_tag_ownership_denied(denied_b, tag, team_a)
        allowed_a: Final = _chat_with_tag(gateway, model, f"{marker}-a", key=key_a, metadata_tags=[tag])
        assert allowed_a.status_code == 200, allowed_a.text

        assert _upstream_marker_chats(upstream, marker) == [(f"{marker}-a",)]


def test_registering_a_tag_used_only_by_its_owner_starts_enforcing_ownership(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        team_a: Final = scenario.team()
        team_b: Final = scenario.team()
        key_a: Final = scenario.key(team_id=team_a)
        key_b: Final = scenario.key(team_id=team_b)
        tag: Final = _tag_name()
        marker: Final = f"lit8516 {uuid.uuid4().hex}"
        upstream.get("/__observations").raise_for_status()

        before: Final = _chat_with_tag(gateway, model, f"{marker}-before", key=key_a, metadata_tags=[tag])
        assert before.status_code == 200, before.text
        _wait_for_daily_tag_spend(tag)

        _create_tag(scenario, tag, team_id=team_a)

        denied_b: Final = _chat_with_tag(gateway, model, f"{marker}-after-b", key=key_b, metadata_tags=[tag])
        _assert_tag_ownership_denied(denied_b, tag, team_a)
        allowed_a: Final = _chat_with_tag(gateway, model, f"{marker}-after-a", key=key_a, metadata_tags=[tag])
        assert allowed_a.status_code == 200, allowed_a.text

        assert _upstream_marker_chats(upstream, marker) == [
            (f"{marker}-after-a",),
            (f"{marker}-before",),
        ]


def _wait_for_daily_tag_spend(tag: str) -> None:
    eventually(
        lambda: read_rows('SELECT api_key FROM "LiteLLM_DailyTagSpend" WHERE tag = %s', (tag,)),
        lambda rows: len(rows) >= 1,
        seconds=30,
    )


def test_owned_tag_create_rejects_name_used_by_another_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team()
        team_b: Final = scenario.team()
        key_b: Final = scenario.key(team_id=team_b)
        tag: Final = _tag_name()

        used: Final = _chat_with_tag(gateway, model, f"lit8516 {uuid.uuid4().hex}", key=key_b, metadata_tags=[tag])
        assert used.status_code == 200, used.text
        _wait_for_daily_tag_spend(tag)

        create: Final = gateway.request("POST", "/tag/new", {"name": tag, "team_id": team_a})
        assert create.status_code == 409, create.text
        assert tag in create.text and team_a in create.text
        assert _tag_rows(tag) == []

        still_ok: Final = _chat_with_tag(
            gateway, model, f"lit8516 {uuid.uuid4().hex}", key=key_b, metadata_tags=[tag]
        )
        assert still_ok.status_code == 200, still_ok.text


def test_owned_tag_create_allows_name_used_only_by_the_owner(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        admin_id: Final = scenario.user()
        team_a: Final = scenario.team(members_with_roles=[{"role": "admin", "user_id": admin_id}])
        admin_key: Final = scenario.key(user_id=admin_id, team_id=team_a)
        key_a: Final = scenario.key(team_id=team_a)
        tag: Final = _tag_name()

        used: Final = _chat_with_tag(gateway, model, f"lit8516 {uuid.uuid4().hex}", key=key_a, metadata_tags=[tag])
        assert used.status_code == 200, used.text
        _wait_for_daily_tag_spend(tag)

        create: Final = gateway.request("POST", "/tag/new", {"name": tag, "team_id": team_a}, key=admin_key)
        assert create.status_code == 200, create.text
        scenario.cleanups.callback(lambda: gateway.request("POST", "/tag/delete", {"name": tag}))
        assert _tag_rows(tag) == [{"tag_name": tag, "team_id": team_a}]


def test_owned_tag_create_rejects_name_used_by_the_master_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team()
        tag: Final = _tag_name()

        used: Final = _chat_with_tag(gateway, model, f"lit8516 {uuid.uuid4().hex}", metadata_tags=[tag])
        assert used.status_code == 200, used.text
        _wait_for_daily_tag_spend(tag)

        create: Final = gateway.request("POST", "/tag/new", {"name": tag, "team_id": team_a})
        assert create.status_code == 409, create.text
        assert _tag_rows(tag) == []


def test_unowned_tag_create_ignores_foreign_usage(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_b: Final = scenario.team()
        key_b: Final = scenario.key(team_id=team_b)
        tag: Final = _tag_name()

        used: Final = _chat_with_tag(gateway, model, f"lit8516 {uuid.uuid4().hex}", key=key_b, metadata_tags=[tag])
        assert used.status_code == 200, used.text
        _wait_for_daily_tag_spend(tag)

        _create_tag(scenario, tag)
        assert _tag_rows(tag) == [{"tag_name": tag, "team_id": None}]


def test_owned_tag_create_rejects_name_used_by_a_deleted_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team()
        team_b: Final = scenario.team()
        key_b: Final = string_value(gateway.post("/key/generate", {"team_id": team_b})["key"])
        tag: Final = _tag_name()

        used: Final = _chat_with_tag(gateway, model, f"lit8516 {uuid.uuid4().hex}", key=key_b, metadata_tags=[tag])
        assert used.status_code == 200, used.text
        _wait_for_daily_tag_spend(tag)
        gateway.post("/key/delete", {"keys": [key_b]})

        create: Final = gateway.request("POST", "/tag/new", {"name": tag, "team_id": team_b})
        assert create.status_code == 409, create.text
        assert _tag_rows(tag) == []
