from fastapi import HTTPException
import pytest

from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    ProxyException,
)
from litellm.proxy.auth.auth_checks import TeamNotFoundError, UserNotFoundError
from litellm.proxy.auth.resolvers.grants import (
    GrantResolver,
    LookupDegraded,
    NotAMember,
    ResolvedGrants,
    TeamGone,
    UserGone,
    UserLookup,
    raise_public,
    user_models,
)

USER_ID = "user-1"
TEAM_ID = "team-1"


class _Loaders:
    """Fake row readers standing in for the ``auth_checks`` loaders, recording every call they receive."""

    def __init__(self, *, user=None, team=None, membership=None, user_error=None, team_error=None):
        self._user = user
        self._team = team
        self._membership = membership
        self._user_error = user_error
        self._team_error = team_error
        self.user_calls = []
        self.team_calls = []
        self.membership_calls = []

    async def load_user(self, **kwargs):
        self.user_calls.append(kwargs)
        if self._user_error is not None:
            raise self._user_error
        return self._user

    async def load_team(self, **kwargs):
        self.team_calls.append(kwargs)
        if self._team_error is not None:
            raise self._team_error
        return self._team

    async def load_membership(self, **kwargs):
        self.membership_calls.append(kwargs)
        return self._membership

    def resolver(self) -> GrantResolver:
        return GrantResolver(
            object(),
            object(),
            load_user=self.load_user,
            load_team=self.load_team,
            load_membership=self.load_membership,
        )


def _user(teams=(TEAM_ID,), user_id=USER_ID) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(user_id=user_id, user_role="internal_user", teams=list(teams), models=["gpt-5.5"])


def _team(models=("gpt-5.5",)) -> LiteLLM_TeamTableCachedObj:
    return LiteLLM_TeamTableCachedObj(team_id=TEAM_ID, team_alias="alias", models=list(models))


async def test_resolve_returns_live_rows_for_a_member():
    membership = LiteLLM_TeamMembership(user_id=USER_ID, team_id=TEAM_ID, spend=1.5)
    loaders = _Loaders(user=_user(), team=_team(models=("new-a", "new-b")), membership=membership)

    outcome = await loaders.resolver().resolve(UserLookup(user_id=USER_ID), team_id=TEAM_ID)

    assert outcome == ResolvedGrants(
        user_object=_user(),
        team_object=_team(models=("new-a", "new-b")),
        team_membership=membership,
        effective_user_id=USER_ID,
    )
    assert loaders.team_calls[0]["team_id"] == TEAM_ID
    assert loaders.membership_calls[0]["user_id"] == USER_ID
    assert loaders.membership_calls[0]["team_id"] == TEAM_ID


async def test_resolve_denies_a_user_removed_from_the_team_without_reading_the_team():
    loaders = _Loaders(user=_user(teams=("other-team",)), team=_team())

    outcome = await loaders.resolver().resolve(UserLookup(user_id=USER_ID), team_id=TEAM_ID)

    assert outcome == NotAMember(user_id=USER_ID, team_id=TEAM_ID)
    assert loaders.team_calls == []


async def test_resolve_reports_a_deleted_user():
    loaders = _Loaders(user_error=UserNotFoundError(user_id=USER_ID), team=_team())

    outcome = await loaders.resolver().resolve(UserLookup(user_id=USER_ID), team_id=TEAM_ID)

    assert outcome == UserGone(user_id=USER_ID)
    assert loaders.team_calls == []


async def test_resolve_reports_a_deleted_team():
    loaders = _Loaders(user=_user(), team_error=TeamNotFoundError(team_id=TEAM_ID))

    outcome = await loaders.resolver().resolve(UserLookup(user_id=USER_ID), team_id=TEAM_ID)

    assert outcome == TeamGone(team_id=TEAM_ID)


@pytest.mark.parametrize(
    "loaders",
    [
        _Loaders(user_error=Exception("No db connected")),
        _Loaders(user=_user(), team_error=HTTPException(status_code=500, detail="db timeout")),
    ],
    ids=["user-read-failed", "team-read-failed"],
)
async def test_resolve_marks_an_unreadable_row_as_degraded_not_denied(loaders):
    outcome = await loaders.resolver().resolve(UserLookup(user_id=USER_ID), team_id=TEAM_ID)

    assert isinstance(outcome, LookupDegraded)


async def test_resolve_without_a_team_skips_team_and_membership_reads():
    loaders = _Loaders(user=_user(teams=()))

    outcome = await loaders.resolver().resolve(UserLookup(user_id=USER_ID), team_id=None)

    assert outcome == ResolvedGrants(
        user_object=_user(teams=()), team_object=None, team_membership=None, effective_user_id=USER_ID
    )
    assert loaders.team_calls == []
    assert loaders.membership_calls == []


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


def test_raise_public_maps_a_deleted_user_to_401():
    with pytest.raises(ProxyException) as exc_info:
        raise_public(UserGone(user_id=USER_ID))
    assert exc_info.value.code == "401"
    assert USER_ID in exc_info.value.message


def test_raise_public_maps_a_removed_member_to_403():
    with pytest.raises(HTTPException) as exc_info:
        raise_public(NotAMember(user_id=USER_ID, team_id=TEAM_ID))
    assert exc_info.value.status_code == 403
    assert TEAM_ID in str(exc_info.value.detail)


def test_raise_public_maps_a_deleted_team_to_404():
    with pytest.raises(TeamNotFoundError) as exc_info:
        raise_public(TeamGone(team_id=TEAM_ID))
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(["gpt-5.5", "claude-opus-5"], ("gpt-5.5", "claude-opus-5")), ([], ()), ([{"not": "a model"}], ())],
    ids=["models", "empty", "unusable-column"],
)
def test_user_models_reads_the_column_as_a_tuple_of_names(stored, expected):
    assert user_models(LiteLLM_UserTable(user_id=USER_ID, models=stored)) == expected
