import os
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Final

import httpx
import psycopg
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, write_rows
from integration._support.scim import (
    PATCH_OP_SCHEMA,
    TAKE_TEAM_LOCK_SQL,
    TAKE_USER_ROW_LOCK_SQL,
    create_group,
    created_team,
    seed_users,
    users_referencing,
    waiters_on_lock_held_by,
    waiters_on_rows_locked_by,
)

pytestmark: Final = pytest.mark.shares_worker_state

SCIM_CORE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:User"
SET_EMAIL_SQL: Final = 'UPDATE "LiteLLM_UserTable" SET user_email = %s WHERE user_id = %s'
PROFILE_NAME: Final = "Race"
USER_ALIAS_SQL: Final = 'SELECT user_alias FROM "LiteLLM_UserTable" WHERE user_id = %s'
DROP_TEAM_SQL: Final = 'UPDATE "LiteLLM_UserTable" SET teams = array_remove(teams, %s) WHERE user_id = %s'
APPEND_TEAM_SQL: Final = 'UPDATE "LiteLLM_UserTable" SET teams = array_append(teams, %s) WHERE user_id = %s'
DROP_FROM_ROSTER_SQL: Final = """
UPDATE "LiteLLM_TeamTable"
SET members_with_roles = COALESCE(
    (SELECT jsonb_agg(member) FROM jsonb_array_elements(members_with_roles::jsonb) AS member
     WHERE member->>'user_id' IS DISTINCT FROM %(user)s),
    '[]'::jsonb
)
WHERE team_id = %(team)s
"""
READD_TEAM_SQL: Final = 'UPDATE "LiteLLM_UserTable" SET teams = array_append(teams, %(team)s) WHERE user_id = %(user)s'
TEAM_ENTRIES_SQL: Final = (
    'SELECT cardinality(array_positions(teams, %s))::text AS entries FROM "LiteLLM_UserTable" WHERE user_id = %s'
)


def _put_profile(candidate: Gateway, user: str, email: str) -> httpx.Response:
    return candidate.request(
        "PUT",
        f"/scim/v2/Users/{user}",
        {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": user,
            "name": {"givenName": PROFILE_NAME, "familyName": "Profile"},
            "emails": [{"value": email, "primary": True}],
            "active": True,
        },
    )


def _patch_display_name(candidate: Gateway, user: str, email: str) -> httpx.Response:
    return candidate.request(
        "PATCH",
        f"/scim/v2/Users/{user}",
        {"schemas": [PATCH_OP_SCHEMA], "Operations": [{"op": "replace", "path": "displayName", "value": PROFILE_NAME}]},
    )


def _post_existing_email(candidate: Gateway, user: str, email: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/scim/v2/Users",
        {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": f"integration-scim-{uuid.uuid4().hex}",
            "name": {"givenName": PROFILE_NAME, "familyName": "Profile"},
            "emails": [{"value": email, "primary": True}],
            "active": True,
        },
    )


@contextmanager
def _held_user_row(user: str) -> Generator[psycopg.Connection]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as holder:
        holder.execute(TAKE_USER_ROW_LOCK_SQL, (user,))
        yield holder


@pytest.mark.parametrize(
    ("write", "status"),
    [(_put_profile, 200), (_patch_display_name, 200), (_post_existing_email, 201)],
    ids=["put", "patch", "post_existing_email"],
)
def test_a_scim_user_write_keeps_a_team_a_group_push_added_after_the_write_read_the_user(
    gateway: Gateway, write: Callable[[Gateway, str, str], httpx.Response], status: int
) -> None:
    with gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=1) as pool:
        (user,) = seed_users(scenario, 1)
        email: Final = f"{user}@example.com"
        write_rows(SET_EMAIL_SQL, (email, user))
        team: Final = created_team(scenario, create_group(gateway, ()))

        with _held_user_row(user) as holder:
            pending: Final = pool.submit(write, gateway, user, email)
            eventually(
                lambda: waiters_on_rows_locked_by(holder.info.backend_pid), lambda waiting: waiting >= 1, seconds=30
            )
            holder.execute(APPEND_TEAM_SQL, (team, user))
        response: Final = pending.result()

        assert response.status_code == status, response.text
        assert users_referencing(team) == frozenset({user})
        assert read_rows(USER_ALIAS_SQL, (user,)) == [{"user_alias": PROFILE_NAME}]


def _put_with_group(candidate: Gateway, user: str, team: str) -> httpx.Response:
    return candidate.request(
        "PUT",
        f"/scim/v2/Users/{user}",
        {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": user,
            "emails": [{"value": f"{user}@example.com", "primary": True}],
            "groups": [{"value": team}],
            "active": True,
        },
    )


def _patch_add_group(candidate: Gateway, user: str, team: str) -> httpx.Response:
    return candidate.request(
        "PATCH",
        f"/scim/v2/Users/{user}",
        {"schemas": [PATCH_OP_SCHEMA], "Operations": [{"op": "add", "path": "groups", "value": [{"value": team}]}]},
    )


@pytest.mark.parametrize("write", [_put_with_group, _patch_add_group], ids=["put", "patch"])
def test_a_scim_user_write_naming_a_team_repairs_a_member_whose_teams_lost_it(
    gateway: Gateway, write: Callable[[Gateway, str, str], httpx.Response]
) -> None:
    with gateway.scenario() as scenario:
        (user,) = seed_users(scenario, 1)
        team: Final = created_team(scenario, create_group(gateway, (user,)))
        write_rows(DROP_TEAM_SQL, (team, user))
        assert users_referencing(team) == frozenset()

        response: Final = write(gateway, user, team)

        assert response.status_code == 200, response.text
        assert users_referencing(team) == frozenset({user})


@contextmanager
def _held_team_lock(team: str) -> Generator[psycopg.Connection]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as holder:
        holder.execute(TAKE_TEAM_LOCK_SQL, (team,))
        yield holder


@pytest.mark.parametrize(
    ("concurrent_write", "expected_entries"),
    [(DROP_FROM_ROSTER_SQL, "0"), (READD_TEAM_SQL, "1")],
    ids=["removed_from_roster", "team_readded"],
)
def test_a_member_repair_rereads_the_roster_and_teams_under_the_team_lock(
    gateway: Gateway, concurrent_write: str, expected_entries: str
) -> None:
    with gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=1) as pool:
        (user,) = seed_users(scenario, 1)
        team: Final = created_team(scenario, create_group(gateway, (user,)))
        write_rows(DROP_TEAM_SQL, (team, user))

        with _held_team_lock(team) as holder:
            pending: Final = pool.submit(_put_with_group, gateway, user, team)
            eventually(
                lambda: waiters_on_lock_held_by(holder.info.backend_pid), lambda waiting: waiting >= 1, seconds=30
            )
            holder.execute(concurrent_write, {"team": team, "user": user})
        response: Final = pending.result()

        assert response.status_code == 200, response.text
        assert read_rows(TEAM_ENTRIES_SQL, (team, user)) == [{"entries": expected_entries}]
