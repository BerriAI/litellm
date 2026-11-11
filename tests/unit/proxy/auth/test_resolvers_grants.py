from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import AsyncMock

import pytest

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import UserNotFoundError, forget_missing_user
from litellm.proxy.auth.resolvers.grants import GrantResolver, UserLookup, user_models
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

USER_ID = "user-1"
TEAM_ID = "team-1"

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@pytest.mark.parametrize("user_exists", (True, False), ids=("created-on-another-worker", "still-missing"))
async def test_session_grants_recheck_a_recently_missing_user(
    user_exists: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)
    user_id: Final = "session-user-after-miss"
    row: Final = LiteLLM_UserTable(user_id=user_id, user_role="internal_user")
    read: Final = AsyncMock(return_value=None)
    database: Final = cast(
        "PrismaClient",
        SimpleNamespace(
            db=SimpleNamespace(
                litellm_usertable=SimpleNamespace(find_unique=read, find_first=AsyncMock(return_value=None))
            )
        ),
    )
    resolver: Final = GrantResolver(database, UserApiKeyCache())
    forget_missing_user(user_id)
    try:
        with pytest.raises(UserNotFoundError):
            await resolver.resolve_identity(UserLookup(user_id=user_id), team_id=None)
        read.return_value = row if user_exists else None

        if user_exists:
            assert await resolver.resolve_identity(UserLookup(user_id=user_id), team_id=None) == (row, None, user_id)
        else:
            with pytest.raises(UserNotFoundError):
                await resolver.resolve_identity(UserLookup(user_id=user_id), team_id=None)
    finally:
        forget_missing_user(user_id)


class _Loaders:
    """Fake row readers standing in for the ``auth_checks`` loaders, recording every call they receive."""

    def __init__(self, *, user=None, membership=None, user_error=None):
        self._user = user
        self._membership = membership
        self._user_error = user_error
        self.user_calls = []
        self.membership_calls = []

    async def load_user(self, **kwargs):
        self.user_calls.append(kwargs)
        if self._user_error is not None:
            raise self._user_error
        return self._user

    async def load_membership(self, **kwargs):
        self.membership_calls.append(kwargs)
        return self._membership

    def resolver(self) -> GrantResolver:
        return GrantResolver(
            object(),
            object(),
            load_user=self.load_user,
            load_membership=self.load_membership,
        )


def _user(teams=(TEAM_ID,), user_id=USER_ID) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(user_id=user_id, user_role="internal_user", teams=list(teams), models=["gpt-5.5"])


async def test_resolve_identity_reads_membership_under_the_matched_rows_id():
    legacy_uuid = "bb8ab11f-09aa-47ae-b063-6e80506ac3bc"
    loaders = _Loaders(user=_user(user_id=legacy_uuid))

    user_object, _membership, effective_user_id = await loaders.resolver().resolve_identity(
        UserLookup(user_id="matt@example.com", user_email="matt@example.com", sso_user_id="matt@example.com"),
        team_id=TEAM_ID,
    )

    assert user_object is not None and user_object.user_id == legacy_uuid
    assert effective_user_id == legacy_uuid
    assert loaders.membership_calls[0]["user_id"] == legacy_uuid
    assert loaders.user_calls[0]["user_email"] == "matt@example.com"


async def test_resolve_identity_without_a_user_id_reads_nothing():
    loaders = _Loaders(user=_user())

    outcome = await loaders.resolver().resolve_identity(UserLookup(user_id=None), team_id=TEAM_ID)

    assert outcome == (None, None, None)
    assert loaders.user_calls == []
    assert loaders.membership_calls == []


async def test_resolve_identity_lets_loader_errors_surface():
    loaders = _Loaders(user_error=UserNotFoundError(user_id=USER_ID))

    with pytest.raises(UserNotFoundError):
        await loaders.resolver().resolve_identity(UserLookup(user_id=USER_ID), team_id=None)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(["gpt-5.5", "claude-opus-5"], ("gpt-5.5", "claude-opus-5")), ([], ()), ([{"not": "a model"}], ())],
    ids=["models", "empty", "unusable-column"],
)
def test_user_models_reads_the_column_as_a_tuple_of_names(stored, expected):
    assert user_models(LiteLLM_UserTable(user_id=USER_ID, models=stored)) == expected
