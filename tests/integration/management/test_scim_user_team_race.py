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
from integration._support.database import write_rows
from integration._support.scim import (
    PATCH_OP_SCHEMA,
    TAKE_USER_ROW_LOCK_SQL,
    create_group,
    created_team,
    seed_users,
    users_referencing,
    waiters_on_rows_locked_by,
)

pytestmark: Final = pytest.mark.shares_worker_state

SCIM_CORE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:User"
SET_EMAIL_SQL: Final = 'UPDATE "LiteLLM_UserTable" SET user_email = %s WHERE user_id = %s'
APPEND_TEAM_SQL: Final = 'UPDATE "LiteLLM_UserTable" SET teams = array_append(teams, %s) WHERE user_id = %s'


def _put_profile(candidate: Gateway, user: str, email: str) -> httpx.Response:
    return candidate.request(
        "PUT",
        f"/scim/v2/Users/{user}",
        {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": user,
            "name": {"givenName": "Race", "familyName": "Profile"},
            "emails": [{"value": email, "primary": True}],
            "active": True,
        },
    )


def _patch_display_name(candidate: Gateway, user: str, email: str) -> httpx.Response:
    return candidate.request(
        "PATCH",
        f"/scim/v2/Users/{user}",
        {"schemas": [PATCH_OP_SCHEMA], "Operations": [{"op": "replace", "path": "displayName", "value": email}]},
    )


def _post_existing_email(candidate: Gateway, user: str, email: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/scim/v2/Users",
        {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": f"integration-scim-{uuid.uuid4().hex}",
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
                lambda: waiters_on_rows_locked_by(holder.info.backend_pid), lambda waiting: waiting == 1, seconds=30
            )
            holder.execute(APPEND_TEAM_SQL, (team, user))
        response: Final = pending.result()

        assert response.status_code == status, response.text
        assert users_referencing(team) == frozenset({user})
