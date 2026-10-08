"""DELETE /scim/v2/Groups/{id} deletes the group's team the way /team/delete does.

The members' group entries, their membership rows and every key on the team go in one locked pass whatever the
roster size, the configured admin group's members are demoted, and a member a concurrent group write added while
the delete waited on the team lock, or while it was demoting the roster it had read, is demoted with the rest. The
group is 404 after, on every worker and on the peer proxy, and an unrelated key keeps serving.
"""

import os
import re
import signal
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Final

import psutil
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.scim import (
    add_to_group,
    assert_gone,
    assert_landed,
    body,
    create_group,
    created_team,
    delete_group,
    groups_of,
    held_team_lock,
    held_user_row,
    keys_of,
    member_ids,
    membership_user_ids,
    replace_group,
    seed_users,
    team_key,
    user_role,
    waiters_on_lock_held_by,
    waiters_on_rows_locked_by,
)
from pydantic import JsonValue, TypeAdapter

PROXY_ADMIN: Final = "proxy_admin"
DEMOTED_ROLE: Final = "internal_user_viewer"
MASTER_KEY_USER: Final = "default_user_id"
BURST_GROUPS: Final = 20
OWNED_PROXY_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
DELETED_AUDIT_ROWS_SQL: Final = """
SELECT table_name, action, object_id, changed_by, changed_by_api_key
FROM "LiteLLM_AuditLog"
WHERE object_id IN (%s, %s) AND action = 'deleted'
ORDER BY table_name
"""
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_WORKER_PIDS: Final = TypeAdapter(tuple[int, ...])


def _hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _admin_group_config(tmp_path: Path, admin_alias: str) -> Path:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = {**object_value(config["litellm_settings"]), "scim_admin_group": admin_alias}
    path: Final = tmp_path / "scim-admin-group.yaml"
    path.write_text(yaml.safe_dump({**config, "litellm_settings": settings}))
    return path


def _roles_of(candidate: Gateway, users: tuple[str, ...]) -> frozenset[str]:
    return frozenset(user_role(candidate, user) for user in users)


def _models_status(candidate: Gateway, key: str) -> int:
    return candidate.request("GET", "/v1/models", key=key).status_code


def _assert_revoked_everywhere(proxies: tuple[Gateway, ...], key: str) -> None:
    for candidate in proxies:
        eventually(partial(_models_status, candidate, key), lambda code: code == 401, seconds=30)


def _deleted_audit_rows(team: str, hashed_key: str) -> list[dict[str, JsonValue]]:
    return read_rows(DELETED_AUDIT_ROWS_SQL, (team, hashed_key))


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    return _WORKER_PIDS.validate_python(_STARTED_WORKER.findall(text)), text.count("Application startup complete.")


def test_deleting_one_group_leaves_the_members_other_group_intact(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        users: Final = seed_users(scenario, 3)
        doomed: Final = created_team(scenario, create_group(gateway, users))
        kept: Final = created_team(scenario, create_group(gateway, users))
        kept_key: Final = team_key(gateway, kept)

        deleted: Final = delete_group(gateway, doomed)
        assert deleted.status_code == 204, deleted.text

        assert gateway.request("GET", f"/scim/v2/Groups/{doomed}").status_code == 404
        assert_landed(gateway, kept, users)
        assert groups_of(gateway, users[0]) == frozenset({kept})
        assert keys_of(kept) == frozenset({_hashed(kept_key)})
        assert _models_status(gateway, kept_key) == 200


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
def test_deleting_the_admin_group_demotes_its_members_and_writes_an_audit_row_naming_the_scim_caller(
    gateway: Gateway, tmp_path: Path
) -> None:
    admin_alias: Final = f"integration-admins-{uuid.uuid4().hex}"
    with (
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"], "LITELLM_STORE_AUDIT_LOGS": "true"},
            config=_admin_group_config(tmp_path, admin_alias),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        candidate: Final = owned.gateway
        users: Final = seed_users(scenario, 3)
        team: Final = created_team(scenario, create_group(candidate, users, display_name=admin_alias))
        assert _roles_of(candidate, users) == frozenset({PROXY_ADMIN})
        key: Final = team_key(candidate, team)

        deleted: Final = delete_group(candidate, team)
        assert deleted.status_code == 204, deleted.text

        assert_gone(candidate, team, users, key)
        assert _roles_of(candidate, users) == frozenset({DEMOTED_ROLE})
        rows: Final = eventually(lambda: _deleted_audit_rows(team, _hashed(key)), lambda found: len(found) >= 2, 30)
        caller: Final = {"changed_by": MASTER_KEY_USER, "changed_by_api_key": _hashed(candidate.key)}
        assert rows == [
            {"table_name": "LiteLLM_TeamTable", "action": "deleted", "object_id": team, **caller},
            {"table_name": "LiteLLM_VerificationToken", "action": "deleted", "object_id": _hashed(key), **caller},
        ]


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
def test_a_member_added_while_the_admin_group_delete_waits_on_the_team_lock_is_demoted_too(
    gateway: Gateway, tmp_path: Path
) -> None:
    admin_alias: Final = f"integration-admins-{uuid.uuid4().hex}"
    with (
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"]},
            config=_admin_group_config(tmp_path, admin_alias),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
            client_timeout=60,
        ) as owned,
        owned.gateway.scenario() as scenario,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        candidate: Final = owned.gateway
        users: Final = seed_users(scenario, 5)
        (late,) = seed_users(scenario, 1)
        team: Final = created_team(scenario, create_group(candidate, users, display_name=admin_alias))
        assert _roles_of(candidate, users) == frozenset({PROXY_ADMIN})
        key: Final = team_key(candidate, team)

        with held_team_lock(team) as holder:
            put: Final = pool.submit(replace_group, candidate, team, (*users, late), display_name=admin_alias)
            eventually(lambda: waiters_on_lock_held_by(holder), lambda waiting: waiting == 1, seconds=30)
            delete: Final = pool.submit(delete_group, candidate, team)
            eventually(lambda: waiters_on_lock_held_by(holder), lambda waiting: waiting == 2, seconds=30)
        replaced: Final = put.result()
        deleted: Final = delete.result()

        assert replaced.status_code == 200, replaced.text
        assert member_ids(body(replaced)) == frozenset((*users, late))
        assert deleted.status_code == 204, deleted.text
        assert_gone(candidate, team, (*users, late), key)
        assert _roles_of(candidate, (*users, late)) == frozenset({DEMOTED_ROLE})


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
def test_a_member_added_while_the_admin_group_delete_demotes_the_roster_it_read_is_demoted_too(
    gateway: Gateway, tmp_path: Path
) -> None:
    admin_alias: Final = f"integration-admins-{uuid.uuid4().hex}"
    with (
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"]},
            config=_admin_group_config(tmp_path, admin_alias),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
            client_timeout=60,
        ) as owned,
        owned.gateway.scenario() as scenario,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        candidate: Final = owned.gateway
        users: Final = seed_users(scenario, 5)
        (late,) = seed_users(scenario, 1)
        team: Final = created_team(scenario, create_group(candidate, users, display_name=admin_alias))
        assert _roles_of(candidate, users) == frozenset({PROXY_ADMIN})
        key: Final = team_key(candidate, team)

        with held_user_row(users[0]) as holder:
            delete: Final = pool.submit(delete_group, candidate, team)
            eventually(lambda: waiters_on_rows_locked_by(holder), lambda waiting: waiting == 1, seconds=30)
            added: Final = add_to_group(candidate, team, (late,))
            assert added.status_code == 200, added.text
            assert late in member_ids(body(added))
            assert user_role(candidate, late) == PROXY_ADMIN
        deleted: Final = delete.result()

        assert deleted.status_code == 204, deleted.text
        assert_gone(candidate, team, (*users, late), key)
        assert _roles_of(candidate, (*users, late)) == frozenset({DEMOTED_ROLE})


def test_deleting_an_unknown_group_is_404(gateway: Gateway) -> None:
    missing: Final = f"missing-{uuid.uuid4().hex}"
    response: Final = delete_group(gateway, missing)
    assert response.status_code == 404, response.text
    assert "error" in body(response), response.text
    assert missing in response.text


def test_deleting_a_group_twice_is_404_the_second_time(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        users: Final = seed_users(scenario, 2)
        team: Final = created_team(scenario, create_group(gateway, users))
        first: Final = delete_group(gateway, team)
        assert first.status_code == 204, first.text
        second: Final = delete_group(gateway, team)
        assert second.status_code == 404, second.text
        assert "error" in body(second), second.text
        assert gateway.request("GET", f"/scim/v2/Groups/{team}").status_code == 404
        assert membership_user_ids(team) == frozenset()


def test_deleting_a_group_with_a_non_admin_key_is_refused(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        users: Final = seed_users(scenario, 2)
        team: Final = created_team(scenario, create_group(gateway, users))
        outsider: Final = scenario.key()
        response: Final = delete_group(gateway, team, key=outsider)
        assert response.status_code in (401, 403), response.text
        assert "error" in body(response), response.text
        assert_landed(gateway, team, users)


@pytest.mark.parametrize("group_id", ("123", "x" * 5000, "a%2Fb"), ids=("int_like", "five_kb", "encoded_slash"))
def test_deleting_a_group_with_a_hostile_id_is_404(gateway: Gateway, group_id: str) -> None:
    response: Final = delete_group(gateway, group_id)
    assert response.status_code == 404, response.text
    assert isinstance(response.json(), dict), response.text
    assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_deleting_an_empty_group_is_204_then_404(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = created_team(scenario, create_group(gateway, ()))
        assert member_ids(gateway.get(f"/scim/v2/Groups/{team}")) == frozenset()
        deleted: Final = delete_group(gateway, team)
        assert deleted.status_code == 204, deleted.text
        assert gateway.request("GET", f"/scim/v2/Groups/{team}").status_code == 404
        assert membership_user_ids(team) == frozenset()


def test_group_delete_on_one_proxy_revokes_the_team_key_and_the_group_on_the_peer(
    gateway: Gateway, peer: Gateway
) -> None:
    with gateway.scenario() as scenario:
        users: Final = seed_users(scenario, 3)
        team: Final = created_team(scenario, create_group(gateway, users))
        key: Final = team_key(gateway, team)
        assert _models_status(peer, key) == 200
        assert member_ids(peer.get(f"/scim/v2/Groups/{team}")) == frozenset(users)

        deleted: Final = delete_group(gateway, team)
        assert deleted.status_code == 204, deleted.text

        assert peer.request("GET", f"/scim/v2/Groups/{team}").status_code == 404
        _assert_revoked_everywhere((peer, gateway), key)
        assert_gone(peer, team, users, key)


def test_twenty_group_deletes_split_across_two_proxies_all_land(gateway: Gateway, peer: Gateway) -> None:
    proxies: Final = (gateway, peer)
    with gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=10) as pool:
        bystander: Final = scenario.key()
        rosters: Final = tuple(seed_users(scenario, 3) for _ in range(BURST_GROUPS))
        teams: Final = tuple(created_team(scenario, create_group(gateway, users)) for users in rosters)
        keys: Final = tuple(team_key(gateway, team) for team in teams)

        assigned: Final = tuple(proxies[index % 2] for index in range(BURST_GROUPS))
        responses: Final = tuple(pool.map(delete_group, assigned, teams))

        for response in responses:
            assert response.status_code == 204, response.text
        for team, users, key in zip(teams, rosters, keys):
            _assert_revoked_everywhere(proxies, key)
            assert_gone(gateway, team, users, key)
            assert peer.request("GET", f"/scim/v2/Groups/{team}").status_code == 404
        assert _models_status(gateway, bystander) == 200
        assert _models_status(peer, bystander) == 200


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
def test_group_delete_after_a_worker_was_killed_still_completes(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy_process(
            gateway, tmp_path, {}, remove_environment=("DATABASE_URL_READ_REPLICA",), workers=2
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        candidate: Final = owned.gateway
        workers, _ = eventually(
            lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 2 and found[1] == 2, seconds=120
        )
        first_users: Final = seed_users(scenario, 3)
        second_users: Final = seed_users(scenario, 3)
        first: Final = created_team(scenario, create_group(candidate, first_users))
        second: Final = created_team(scenario, create_group(candidate, second_users))
        first_key: Final = team_key(candidate, first)
        second_key: Final = team_key(candidate, second)

        victim: Final = psutil.Process(workers[0])
        victim.suspend()
        victim.send_signal(signal.SIGKILL)
        deleted_first: Final = delete_group(candidate, first)
        assert deleted_first.status_code == 204, deleted_first.text
        assert_gone(candidate, first, first_users, first_key)

        eventually(lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 3 and found[1] == 3, 180)
        deleted_second: Final = delete_group(candidate, second)
        assert deleted_second.status_code == 204, deleted_second.text
        assert_gone(candidate, second, second_users, second_key)
