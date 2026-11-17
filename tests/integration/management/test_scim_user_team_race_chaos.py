import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, write_rows
from integration._support.process import owned_proxy_process
from integration._support.scim import PATCH_OP_SCHEMA, add_to_group, create_group, created_team, seed_users

pytestmark: Final = pytest.mark.shares_worker_state

SCIM_CORE_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:User"
BURST_USERS: Final = 24
TEAMS: Final = 4
SET_EMAILS_SQL: Final = (
    """UPDATE "LiteLLM_UserTable" SET user_email = user_id || '@example.com' WHERE user_id = ANY(%s)"""
)
TEAM_ENTRIES_SQL: Final = """
SELECT user_id, cardinality(array_positions(teams, %s))::text AS entries
FROM "LiteLLM_UserTable" WHERE user_id = ANY(%s) ORDER BY user_id
"""
ROSTER_SQL: Final = """
SELECT member->>'user_id' AS user_id
FROM "LiteLLM_TeamTable", jsonb_array_elements(members_with_roles::jsonb) AS member
WHERE team_id = %s ORDER BY 1
"""


def _put_profile(candidate: Gateway, user: str) -> httpx.Response:
    return candidate.request(
        "PUT",
        f"/scim/v2/Users/{user}",
        {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": user,
            "name": {"givenName": "Burst", "familyName": "Profile"},
            "emails": [{"value": f"{user}@example.com", "primary": True}],
            "active": True,
        },
    )


def _patch_display_name(candidate: Gateway, user: str) -> httpx.Response:
    return candidate.request(
        "PATCH",
        f"/scim/v2/Users/{user}",
        {"schemas": [PATCH_OP_SCHEMA], "Operations": [{"op": "replace", "path": "displayName", "value": "Burst"}]},
    )


def _post_existing_email(candidate: Gateway, user: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/scim/v2/Users",
        {
            "schemas": [SCIM_CORE_USER_SCHEMA],
            "userName": f"integration-scim-{uuid.uuid4().hex}",
            "name": {"givenName": "Burst", "familyName": "Profile"},
            "emails": [{"value": f"{user}@example.com", "primary": True}],
            "active": True,
        },
    )


USER_WRITES: Final = (_put_profile, _patch_display_name, _post_existing_email)


def _status(call: Callable[[], httpx.Response]) -> int | None:
    try:
        return call().status_code
    except httpx.TransportError:
        return None


def _entries(team: str, users: tuple[str, ...]) -> dict[str, str]:
    return {str(row["user_id"]): str(row["entries"]) for row in read_rows(TEAM_ENTRIES_SQL, (team, list(users)))}


def test_group_pushes_racing_user_writes_land_once_on_every_user_while_a_worker_is_killed(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned, owned.gateway.scenario() as scenario:
        candidate: Final = owned.gateway
        users: Final = seed_users(scenario, BURST_USERS)
        write_rows(SET_EMAILS_SQL, (list(users),))
        teams: Final = tuple(created_team(scenario, create_group(candidate, ())) for _ in range(TEAMS))
        team_of: Final = {user: teams[index % TEAMS] for index, user in enumerate(users)}
        workers: Final = eventually(owned.worker_pids, lambda pids: len(pids) >= 2, seconds=30)

        with ThreadPoolExecutor(max_workers=2 * BURST_USERS) as pool:
            pushes: Final = {
                user: pool.submit(_status, lambda user=user: add_to_group(candidate, team_of[user], (user,)))
                for user in users
            }
            writes: Final = tuple(
                pool.submit(_status, lambda user=user, index=index: USER_WRITES[index % 3](candidate, user))
                for index, user in enumerate(users)
            )
            owned.kill_worker(min(workers))
        pushed: Final = {user: future.result() for user, future in pushes.items()}
        written: Final = tuple(future.result() for future in writes)

        assert set(pushed.values()) | set(written) <= {200, 201, None}, (pushed, written)
        landed: Final = {user: _entries(team_of[user], (user,))[user] for user in users}
        assert {user: landed[user] for user in users if pushed[user] == 200} == {
            user: "1" for user in users if pushed[user] == 200
        }
        assert set(landed.values()) <= {"0", "1"}, landed

        serving: Final = eventually(
            lambda: _status(lambda: candidate.request("GET", f"/scim/v2/Users/{users[0]}")),
            lambda status: status == 200,
            seconds=30,
        )
        assert serving == 200
        retried: Final = tuple(add_to_group(candidate, team_of[user], (user,)) for user in users)
        assert [response.status_code for response in retried] == [200] * BURST_USERS, [r.text for r in retried]
        for team in teams:
            expected: Final = sorted(user for user in users if team_of[user] == team)
            assert _entries(team, users) == {user: "1" if team_of[user] == team else "0" for user in users}
            assert [row["user_id"] for row in read_rows(ROSTER_SQL, (team,))] == expected
