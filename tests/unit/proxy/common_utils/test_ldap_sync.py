from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.common_utils.ldap_directory import DirectoryError, DirectoryGroup
from litellm.proxy.common_utils.ldap_sync import resolve_user_ids
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.ldap_sync import LDAPSyncSettings

SETTINGS: Final = LDAPSyncSettings(
    enabled=True,
    server_url="ldaps://directory.example.org",
    group_search_base="ou=groups,dc=example,dc=org",
    group_filter="(cn=project-*)",
    user_search_base="ou=people,dc=example,dc=org",
)


class Users:
    def __init__(self, rows: Sequence[LiteLLM_UserTable]) -> None:
        self.rows = rows
        self.db = self
        self.writer_db = self
        self.litellm_usertable = self

    async def find_many(
        self,
        *,
        where: Mapping[str, object] | None = None,
        take: int | None = None,
        skip: int | None = None,
        order: Mapping[str, str] | None = None,
    ) -> Sequence[LiteLLM_UserTable]:
        return self.rows


@pytest.mark.asyncio
async def test_existing_users_match_case_insensitively_without_creating_unknown_users():
    users = Users([LiteLLM_UserTable(user_id="alice", user_email="Alice@example.org")])
    result = await resolve_user_ids(
        UserRepository(users),
        DirectoryGroup("project-a", "project-a", ("alice@example.org", "unknown@example.org")),
        SETTINGS,
    )
    assert result == frozenset({"alice"})
    assert len(users.rows) == 1


@pytest.mark.asyncio
async def test_duplicate_email_accounts_abort_group_matching():
    users = Users(
        [
            LiteLLM_UserTable(user_id="alice", user_email="Alice@example.org"),
            LiteLLM_UserTable(user_id="other", user_email="alice@example.org"),
        ]
    )
    result = await resolve_user_ids(
        UserRepository(users), DirectoryGroup("project-a", "project-a", ("alice@example.org",)), SETTINGS
    )
    assert isinstance(result, DirectoryError)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["user_id", "sso_user_id"])
async def test_configured_identity_field_matches_existing_user(field: str):
    users = Users([LiteLLM_UserTable(user_id="alice", sso_user_id="z1234")])
    identity = "alice" if field == "user_id" else "z1234"
    result = await resolve_user_ids(
        UserRepository(users),
        DirectoryGroup("project-a", "project-a", (identity,)),
        SETTINGS.model_copy(update={"litellm_match_field": field}),
    )
    assert result == frozenset({"alice"})
