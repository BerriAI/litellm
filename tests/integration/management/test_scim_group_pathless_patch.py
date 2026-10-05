import os
import signal
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.process import owned_proxy_process
from integration.authorization._guardrail_opt_out import upstream_hits
from pydantic import JsonValue

PATCH_OP_SCHEMA: Final = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
GROUP_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:Group"
BURST_TEAMS: Final = 10
BURST_REQUESTS_PER_TEAM: Final = 3
KILL_AFTER_RESPONSES: Final = 5


def patch_group(
    candidate: Gateway, team: str, operations: Sequence[JsonValue], *, key: str | None = None
) -> httpx.Response:
    return candidate.request(
        "PATCH",
        f"/scim/v2/Groups/{team}",
        {"schemas": [PATCH_OP_SCHEMA], "Operations": list(operations)},
        key=key,
    )


def pathless(op: str, value: JsonValue) -> dict[str, JsonValue]:
    return {"op": op, "value": value}


def pathed(op: str, path: str, value: JsonValue | None = None) -> dict[str, JsonValue]:
    return {"op": op, "path": path, **({} if value is None else {"value": value})}


def team_info(candidate: Gateway, team: str) -> dict[str, JsonValue]:
    return object_value(candidate.get("/team/info", {"team_id": team})["team_info"])


def team_metadata(candidate: Gateway, team: str) -> dict[str, JsonValue]:
    return object_value(team_info(candidate, team).get("metadata") or {})


def team_alias(candidate: Gateway, team: str) -> JsonValue:
    return team_info(candidate, team).get("team_alias")


def alias_and_metadata(candidate: Gateway, team: str) -> tuple[JsonValue, dict[str, JsonValue]]:
    info: Final = team_info(candidate, team)
    return info.get("team_alias"), object_value(info.get("metadata") or {})


def member_ids(candidate: Gateway, team: str) -> frozenset[str]:
    members: Final = team_info(candidate, team).get("members_with_roles") or []
    assert isinstance(members, list), members
    return frozenset(string_value(object_value(member)["user_id"]) for member in members)


def group_member_ids(candidate: Gateway, team: str) -> frozenset[str]:
    members: Final = candidate.get(f"/scim/v2/Groups/{team}").get("members") or []
    assert isinstance(members, list), members
    return frozenset(string_value(object_value(member)["value"]) for member in members)


def scim_data(metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(metadata["scim_data"])


def scim_group(scenario: Scenario, members: Sequence[str]) -> str:
    created: Final = scenario.gateway.request(
        "POST",
        "/scim/v2/Groups",
        {
            "schemas": [GROUP_SCHEMA],
            "displayName": f"integration-{uuid.uuid4().hex}",
            "members": [{"value": member} for member in members],
        },
    )
    assert created.status_code == 201, created.text
    team: Final = string_value(object_value(created.json())["id"])
    scenario.cleanups.callback(scenario.delete_team, team)
    return team


def model_names(candidate: Gateway) -> frozenset[str]:
    entries: Final = candidate.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    return frozenset(string_value(object_value(entry)["model_name"]) for entry in entries)


def test_pathless_replace_renames_the_team_and_keeps_the_resource_under_scim_data(gateway: Gateway) -> None:
    renamed: Final = f"okta-renamed-{uuid.uuid4().hex}"
    external: Final = f"ext-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        response: Final = patch_group(
            gateway, team, [pathless("replace", {"id": team, "displayName": renamed, "externalId": external})]
        )
        assert response.status_code == 200, response.text
        assert response.json()["displayName"] == renamed, response.text
        assert team_alias(gateway, team) == renamed
        metadata: Final = team_metadata(gateway, team)
        assert set(metadata) == {"externalId", "scim_data", "scim_managed"}, metadata
        assert metadata["externalId"] == external and metadata["scim_managed"] is True, metadata
        assert scim_data(metadata) == {"id": team, "displayName": renamed, "externalId": external}, metadata


def test_pathless_replace_with_members_is_an_absolute_roster(gateway: Gateway) -> None:
    renamed: Final = f"roster-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        first: Final = scenario.user(user_role="internal_user")
        second: Final = scenario.user(user_role="internal_user")
        team: Final = scim_group(scenario, [first])
        assert member_ids(gateway, team) == {first}
        response: Final = patch_group(
            gateway, team, [pathless("replace", {"displayName": renamed, "members": [{"value": second}]})]
        )
        assert response.status_code == 200, response.text
        assert member_ids(gateway, team) == {second}
        assert group_member_ids(gateway, team) == {second}
        assert team_alias(gateway, team) == renamed
        metadata: Final = team_metadata(gateway, team)
        assert "" not in metadata, metadata
        assert "members" not in scim_data(metadata), metadata


def test_pathless_add_applies_the_attribute(gateway: Gateway) -> None:
    external: Final = f"ext-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        alias: Final = team_alias(gateway, team)
        response: Final = patch_group(gateway, team, [pathless("add", {"externalId": external})])
        assert response.status_code == 200, response.text
        metadata: Final = team_metadata(gateway, team)
        assert "" not in metadata, metadata
        assert metadata["externalId"] == external, metadata
        assert scim_data(metadata) == {"externalId": external}, metadata
        assert team_alias(gateway, team) == alias


def test_pathed_operations_are_unchanged(gateway: Gateway) -> None:
    renamed: Final = f"pathed-{uuid.uuid4().hex}"
    external: Final = f"ext-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        first: Final = scenario.user(user_role="internal_user")
        second: Final = scenario.user(user_role="internal_user")
        team: Final = scim_group(scenario, [first])
        response: Final = patch_group(
            gateway,
            team,
            [
                pathed("replace", "displayName", renamed),
                pathed("replace", "externalId", external),
                pathed("add", "members", [{"value": second}]),
                pathed("remove", f'members[value eq "{first}"]'),
            ],
        )
        assert response.status_code == 200, response.text
        assert response.json()["displayName"] == renamed, response.text
        assert team_alias(gateway, team) == renamed
        assert member_ids(gateway, team) == {second}
        assert group_member_ids(gateway, team) == {second}
        metadata: Final = team_metadata(gateway, team)
        assert metadata["externalId"] == external, metadata
        assert "" not in metadata, metadata


def test_pathless_patch_merges_into_the_put_snapshot(gateway: Gateway) -> None:
    put_alias: Final = f"put-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scim_group(scenario, [member])
        put: Final = gateway.request(
            "PUT",
            f"/scim/v2/Groups/{team}",
            {
                "schemas": [GROUP_SCHEMA],
                "id": team,
                "displayName": put_alias,
                "externalId": "ext-v1",
                "members": [{"value": member}],
            },
        )
        assert put.status_code == 200, put.text
        assert scim_data(team_metadata(gateway, team))["externalId"] == "ext-v1"
        response: Final = patch_group(gateway, team, [pathless("add", {"externalId": "ext-v2"})])
        assert response.status_code == 200, response.text
        metadata: Final = team_metadata(gateway, team)
        snapshot: Final = scim_data(metadata)
        assert "" not in metadata, metadata
        assert metadata["externalId"] == "ext-v2", metadata
        assert snapshot["displayName"] == put_alias and snapshot["externalId"] == "ext-v2", snapshot
        assert team_alias(gateway, team) == put_alias
        assert member_ids(gateway, team) == {member}


def test_group_patch_drops_the_empty_key_left_by_an_earlier_push(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        gateway.post("/team/update", {"team_id": team, "metadata": {"": {"displayName": "stale"}, "env": "staging"}})
        assert "" in team_metadata(gateway, team)
        response: Final = patch_group(gateway, team, [pathed("replace", "externalId", "ext-after")])
        assert response.status_code == 200, response.text
        metadata: Final = team_metadata(gateway, team)
        assert "" not in metadata, metadata
        assert metadata["env"] == "staging" and metadata["externalId"] == "ext-after", metadata


def test_later_path_op_wins_over_the_pathless_value(gateway: Gateway) -> None:
    first: Final = f"first-{uuid.uuid4().hex}"
    second: Final = f"second-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        response: Final = patch_group(
            gateway, team, [pathless("replace", {"displayName": first}), pathed("replace", "displayName", second)]
        )
        assert response.status_code == 200, response.text
        assert response.json()["displayName"] == second, response.text
        assert team_alias(gateway, team) == second
        metadata: Final = team_metadata(gateway, team)
        assert "" not in metadata, metadata
        assert scim_data(metadata)["displayName"] == second, metadata


def test_read_only_attributes_do_not_become_metadata_keys(gateway: Gateway) -> None:
    renamed: Final = f"readonly-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        response: Final = patch_group(
            gateway,
            team,
            [
                pathless(
                    "replace",
                    {
                        "id": team,
                        "schemas": [GROUP_SCHEMA],
                        "meta": {"resourceType": "Group"},
                        "displayName": renamed,
                    },
                )
            ],
        )
        assert response.status_code == 200, response.text
        metadata: Final = team_metadata(gateway, team)
        assert set(metadata) == {"scim_data", "scim_managed"}, metadata
        assert team_alias(gateway, team) == renamed


def test_pathless_rename_is_visible_from_the_peer_proxy(gateway: Gateway, peer: Gateway) -> None:
    renamed: Final = f"peer-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        response: Final = patch_group(gateway, team, [pathless("replace", {"displayName": renamed})])
        assert response.status_code == 200, response.text
        group: Final = eventually(
            lambda: peer.get(f"/scim/v2/Groups/{team}"), lambda observed: observed.get("displayName") == renamed
        )
        assert group["displayName"] == renamed, group
        assert team_alias(peer, team) == renamed
        assert "" not in team_metadata(peer, team)


def test_pathless_remove_is_rejected(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        before: Final = alias_and_metadata(gateway, team)
        response: Final = patch_group(gateway, team, [pathless("remove", {"externalId": "ext-gone"})])
        assert response.status_code == 400, response.text
        assert "RFC 7644 Section 3.5.2.2" in response.text, response.text
        assert alias_and_metadata(gateway, team) == before


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "value": "new-name"},
        {"op": "replace", "value": 7},
        {"op": "replace", "value": ["new-name"]},
        {"op": "replace", "value": ""},
        {"op": "replace", "value": "x" * 5120},
        {"op": "replace", "value": None},
        {"op": "replace"},
        {"op": "add", "value": "new-name"},
    ],
    ids=["string", "int", "list", "empty-string", "5kb-string", "null", "missing", "add-string"],
)
def test_pathless_op_without_an_object_value_is_rejected(gateway: Gateway, operation: dict[str, JsonValue]) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        before: Final = alias_and_metadata(gateway, team)
        response: Final = patch_group(gateway, team, [operation])
        assert response.status_code == 400, response.text
        assert "RFC 7644 Section 3.5.2" in response.text, response.text
        assert alias_and_metadata(gateway, team) == before


def test_pathless_empty_object_changes_nothing_but_marks_the_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        alias: Final = team_alias(gateway, team)
        response: Final = patch_group(gateway, team, [pathless("replace", {})])
        assert response.status_code == 200, response.text
        assert team_alias(gateway, team) == alias
        metadata: Final = team_metadata(gateway, team)
        assert metadata == {"scim_managed": True, "scim_data": {}}, metadata


def test_duplicate_pathless_ops_are_idempotent(gateway: Gateway) -> None:
    external: Final = f"ext-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        response: Final = patch_group(gateway, team, [pathless("add", {"externalId": external})] * 2)
        assert response.status_code == 200, response.text
        metadata: Final = team_metadata(gateway, team)
        assert metadata == {"scim_managed": True, "externalId": external, "scim_data": {"externalId": external}}, (
            metadata
        )


def test_unauthenticated_patch_changes_nothing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        before: Final = alias_and_metadata(gateway, team)
        response: Final = patch_group(
            gateway, team, [pathless("replace", {"displayName": "intruder"})], key=f"sk-{uuid.uuid4().hex}"
        )
        assert response.status_code == 401, response.text
        assert alias_and_metadata(gateway, team) == before


def test_unknown_group_is_404(gateway: Gateway) -> None:
    response: Final = patch_group(
        gateway, f"missing-{uuid.uuid4().hex}", [pathless("replace", {"displayName": "ghost"})]
    )
    assert response.status_code == 404, response.text


def test_malformed_patch_body_is_rejected(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        before: Final = alias_and_metadata(gateway, team)
        response: Final = gateway.request(
            "PATCH", f"/scim/v2/Groups/{team}", {"schemas": [PATCH_OP_SCHEMA], "Operations": "nope"}
        )
        assert response.status_code in (400, 422), response.text
        assert alias_and_metadata(gateway, team) == before


def test_pathless_and_pathed_scalar_coercion_agree(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        via_pathless: Final = scenario.team()
        via_path: Final = scenario.team()
        first: Final = patch_group(gateway, via_pathless, [pathless("replace", {"displayName": 7})])
        second: Final = patch_group(gateway, via_path, [pathed("replace", "displayName", 7)])
        assert first.status_code == second.status_code == 200, (first.text, second.text)
        assert team_alias(gateway, via_pathless) == team_alias(gateway, via_path), (first.text, second.text)
        assert "" not in team_metadata(gateway, via_pathless)


def _patched_state(
    candidate: Gateway, team: str, operations: Sequence[JsonValue]
) -> tuple[JsonValue, dict[str, JsonValue]]:
    response: Final = patch_group(candidate, team, operations)
    assert response.status_code == 200, response.text
    return alias_and_metadata(candidate, team)


def test_repeated_pathless_patch_is_idempotent(gateway: Gateway) -> None:
    renamed: Final = f"repeat-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        operations: Final = [pathless("replace", {"displayName": renamed, "externalId": "ext-repeat"})]
        states: Final = tuple(_patched_state(gateway, team, operations) for _ in range(3))
        assert all(state == states[0] for state in states), states
        assert states[0][0] == renamed, states
        assert "" not in states[0][1], states


def test_concurrent_pathless_renames_converge(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        aliases: Final = tuple(f"race-{index}-{uuid.uuid4().hex}" for index in range(10))
        candidates: Final = (gateway, peer)
        with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
            responses: Final = tuple(
                pool.map(
                    lambda indexed: patch_group(
                        candidates[indexed[0] % 2], team, [pathless("replace", {"displayName": indexed[1]})]
                    ),
                    enumerate(aliases),
                )
            )
        assert all(response.status_code == 200 for response in responses), [
            response.text for response in responses if response.status_code != 200
        ]
        alias: Final = team_alias(gateway, team)
        assert alias in aliases, alias
        metadata: Final = team_metadata(gateway, team)
        assert "" not in metadata, metadata
        assert scim_data(metadata)["displayName"] == alias, metadata


def test_team_key_keeps_serving_after_the_rename(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, models=[model])
        response: Final = patch_group(
            gateway, team, [pathless("replace", {"displayName": f"serving-{uuid.uuid4().hex}"})]
        )
        assert response.status_code == 200, response.text
        eventually(lambda: model_names(peer), lambda names: model in names, seconds=60)
        for candidate, marker in ((gateway, f"primary-{uuid.uuid4().hex}"), (peer, f"peer-{uuid.uuid4().hex}")):
            completion: Final = candidate.chat(model, key=key, text=marker)
            assert completion["choices"], completion
            assert upstream_hits(gateway, marker) == 1, marker


@dataclass(frozen=True, slots=True)
class _Attempt:
    team: str
    alias: str
    response: httpx.Response | None


class _KillSwitch:
    def __init__(self, after: int, action: Callable[[], None]) -> None:
        self._after: Final = after
        self._action: Final = action
        self._lock: Final = threading.Lock()
        self._responses = 0

    def tick(self) -> None:
        with self._lock:
            self._responses += 1
            if self._responses == self._after:
                self._action()


def _burst_operations(alias: str, index: int) -> Sequence[JsonValue]:
    shapes: Final = (
        [pathless("replace", {"displayName": alias, "externalId": f"ext-{index}"})],
        [pathless("add", {"displayName": alias})],
        [pathed("replace", "displayName", alias)],
    )
    return shapes[index % len(shapes)]


def _attempt(candidate: Gateway, team: str, index: int, switch: _KillSwitch) -> _Attempt:
    alias: Final = f"burst-{index}-{uuid.uuid4().hex}"
    try:
        return _Attempt(team, alias, patch_group(candidate, team, _burst_operations(alias, index)))
    except httpx.TransportError:
        return _Attempt(team, alias, None)
    finally:
        switch.tick()


def _team_attempts(candidate: Gateway, team: str, offset: int, switch: _KillSwitch) -> tuple[_Attempt, ...]:
    return tuple(_attempt(candidate, team, offset + index, switch) for index in range(BURST_REQUESTS_PER_TEAM))


def _burst(candidate: Gateway, teams: Sequence[str], disruption: Callable[[], None]) -> tuple[_Attempt, ...]:
    switch: Final = _KillSwitch(KILL_AFTER_RESPONSES, disruption)
    with ThreadPoolExecutor(max_workers=len(teams)) as pool:
        per_team: Final = tuple(
            pool.submit(_team_attempts, candidate, team, index * BURST_REQUESTS_PER_TEAM, switch)
            for index, team in enumerate(teams)
        )
        return tuple(chain.from_iterable(future.result() for future in per_team))


def _burst_teams(scenario: Scenario) -> Mapping[str, str]:
    origins: Final = tuple(f"origin-{uuid.uuid4().hex}" for _ in range(BURST_TEAMS))
    return MappingProxyType({scenario.team(team_alias=origin): origin for origin in origins})


def _assert_team_reflected(candidate: Gateway, team: str, origin: str, sent: Sequence[_Attempt]) -> None:
    aliases: Final = frozenset(attempt.alias for attempt in sent)
    alias, metadata = alias_and_metadata(candidate, team)
    assert "" not in metadata, metadata
    if alias == origin:
        assert all(attempt.response is None for attempt in sent), (team, sent)
        assert "scim_managed" not in metadata, metadata
        return
    assert alias in aliases, (alias, aliases)
    assert metadata["scim_managed"] is True, metadata
    if sent[-1].response is not None:
        assert alias == sent[-1].alias, (alias, sent[-1].alias)
    snapshot: Final = metadata.get("scim_data")
    if snapshot is not None:
        assert object_value(snapshot).get("displayName") in aliases, snapshot


def _assert_burst_reflected(candidate: Gateway, teams: Mapping[str, str], attempts: Sequence[_Attempt]) -> None:
    answered: Final = tuple(attempt for attempt in attempts if attempt.response is not None)
    assert answered, "The whole burst failed to reach the proxy"
    for attempt in answered:
        assert attempt.response is not None and attempt.response.status_code == 200, attempt.response
        assert attempt.response.json()["displayName"] == attempt.alias, attempt.response.text
    for team, origin in teams.items():
        _assert_team_reflected(candidate, team, origin, tuple(attempt for attempt in attempts if attempt.team == team))


def _patch_status(candidate: Gateway, team: str) -> int | None:
    operations: Final = [pathless("replace", {"displayName": f"probe-{uuid.uuid4().hex}"})]
    try:
        return patch_group(candidate, team, operations).status_code
    except httpx.TransportError:
        return None


def _serving_workers(root: int) -> tuple[psutil.Process, ...]:
    return tuple(child for child in psutil.Process(root).children() if child.children())


@pytest.mark.timeout(240)
def test_worker_kill_mid_burst_keeps_serving_and_leaves_no_empty_key(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned, gateway.scenario() as scenario:
        teams: Final = _burst_teams(scenario)
        workers: Final = eventually(
            lambda: _serving_workers(owned.process.pid), lambda children: len(children) >= 2, seconds=30
        )
        attempts: Final = _burst(owned.gateway, tuple(teams), lambda: os.kill(workers[0].pid, signal.SIGKILL))
        assert not workers[0].is_running() or workers[0].status() == psutil.STATUS_ZOMBIE, workers[0]
        probe: Final = scenario.team()
        eventually(lambda: _patch_status(owned.gateway, probe), lambda status_code: status_code == 200, seconds=60)
        _assert_burst_reflected(owned.gateway, teams, attempts)


@pytest.mark.timeout(300)
def test_rolling_restart_mid_burst_drains_without_empty_keys(gateway: Gateway, tmp_path: Path) -> None:
    with gateway.scenario() as scenario, owned_proxy_process(gateway, tmp_path, {}, workers=2) as replacement:
        teams: Final = _burst_teams(scenario)
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as retiring:
            attempts: Final = _burst(retiring.gateway, tuple(teams), retiring.process.terminate)
        _assert_burst_reflected(replacement.gateway, teams, attempts)
