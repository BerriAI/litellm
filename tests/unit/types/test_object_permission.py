from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import BaseModel

from litellm.types.object_permission import (
    response_without_empty_auth_profile_grants,
    row_without_empty_auth_profile_grants,
)


class _PermissionRow(BaseModel):
    object_permission_id: str
    mcp_servers: list[str]
    mcp_auth_profiles: list[str]


class _TeamRow(BaseModel):
    team_id: str
    updated_at: datetime
    object_permission: _PermissionRow | None


_UPDATED_AT: Final = datetime(2026, 11, 18, tzinfo=timezone.utc)


def test_a_row_whose_permission_grants_nothing_answers_without_the_grant_field() -> None:
    row: Final = _TeamRow(
        team_id="payments",
        updated_at=_UPDATED_AT,
        object_permission=_PermissionRow(object_permission_id="op", mcp_servers=["analytics"], mcp_auth_profiles=[]),
    )

    assert response_without_empty_auth_profile_grants(row) == {
        "team_id": "payments",
        "updated_at": "2026-11-18T00:00:00Z",
        "object_permission": {"object_permission_id": "op", "mcp_servers": ["analytics"]},
    }


def test_a_granting_row_or_a_row_without_a_permission_is_answered_as_it_is() -> None:
    granting: Final = _TeamRow(
        team_id="bots",
        updated_at=_UPDATED_AT,
        object_permission=_PermissionRow(object_permission_id="op", mcp_servers=[], mcp_auth_profiles=["ana-bots"]),
    )
    bare: Final = _TeamRow(team_id="plain", updated_at=_UPDATED_AT, object_permission=None)

    assert response_without_empty_auth_profile_grants(granting) is granting
    assert response_without_empty_auth_profile_grants(bare) is bare


@pytest.mark.parametrize(
    ("stored_grants", "expected_permission"),
    [
        ([], {"object_permission_id": "perm-1", "mcp_servers": ["analytics"]}),
        (
            ["ana-bots"],
            {"object_permission_id": "perm-1", "mcp_servers": ["analytics"], "mcp_auth_profiles": ["ana-bots"]},
        ),
    ],
)
def test_a_dumped_key_row_shows_grants_only_when_it_has_them(
    stored_grants: list[str], expected_permission: dict[str, object]
) -> None:
    updated_row: Final = {
        "token": "hashed",
        "object_permission": {
            "object_permission_id": "perm-1",
            "mcp_servers": ["analytics"],
            "mcp_auth_profiles": stored_grants,
        },
    }

    assert row_without_empty_auth_profile_grants(updated_row) == {
        "token": "hashed",
        "object_permission": expected_permission,
    }
