from collections.abc import Awaitable, Callable
from typing import Final

import pytest

import litellm
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamTable,
)
from litellm.proxy.auth import identity as auth_identity
from litellm.proxy.auth.auth_exception_handler import DB_UNAVAILABLE_FALLBACK_USER_ID
from litellm.proxy.auth.host.facts import EndUserRow
from litellm.proxy.auth.host.python import PythonAuthHost
from litellm.rust_bridge._native import RustDenial, RustIdentity, RustKeyContext, RustTeamGrant
from litellm.types.proxy.agent_identity import ManagedAgentContext
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles


class _Outcome:
    def __init__(self, identity: RustIdentity | None = None, denial: RustDenial | None = None) -> None:
        self.identity = identity
        self.denial = denial


@pytest.fixture
def project_outcome(
    auth_host: Callable[..., PythonAuthHost],
) -> Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]]:
    async def run(identity: RustIdentity) -> UserAPIKeyAuth:
        return await auth_identity.project(_Outcome(identity=identity), auth_host("jwt"))  # pyright: ignore[reportArgumentType]  # outcome double

    return run


NO_END_USER: Final = EndUserRow(end_user_id=None, end_user=None, params={})


def _context(
    key: UserAPIKeyAuth,
    *,
    end_user: EndUserRow = NO_END_USER,
    owner: LiteLLM_UserTable | None = None,
    team: RustTeamGrant | None = None,
    project: LiteLLM_ProjectTableCachedObj | None = None,
) -> RustKeyContext:
    return RustKeyContext(key, end_user, owner=owner, team=team or RustTeamGrant.NoTeam(), project=project)


@pytest.mark.asyncio
async def test_no_master_key_reports_the_key_after_the_pass_through_override(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    token: Final = await project_outcome(RustIdentity.NoMasterKey(token="from-header"))

    assert (token.api_key, token.user_role) == ("from-header", LitellmUserRoles.INTERNAL_USER)


@pytest.mark.asyncio
async def test_virtual_key_is_marked_and_named_by_its_hash(
    auth_host: Callable[..., PythonAuthHost],
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    host: Final = auth_host()
    context: Final = _context(UserAPIKeyAuth(token="hashed", user_id="u", team_id="t"))

    token: Final = await auth_identity.project(_Outcome(identity=RustIdentity.VirtualKey(context=context)), host)  # pyright: ignore[reportArgumentType]  # outcome double

    assert (token.api_key, token.user_id, token.team_id) == ("hashed", "u", "t")
    assert token.via_virtual_key is True
    assert host.call().request.state.requires_key_model_check is True


@pytest.mark.asyncio
async def test_keyless_team_permission_never_reaches_a_key_without_a_team(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    stale: Final = LiteLLM_ObjectPermissionTable(object_permission_id="op-1")
    context: Final = _context(UserAPIKeyAuth(token="hashed", team_object_permission=stale))

    token: Final = await project_outcome(RustIdentity.VirtualKey(context=context))

    assert token.team_object_permission is None


@pytest.mark.asyncio
async def test_loaded_team_permission_and_metadata_replace_the_keys_copy(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    stale: Final = LiteLLM_ObjectPermissionTable(object_permission_id="op-stale")
    fresh: Final = LiteLLM_ObjectPermissionTable(object_permission_id="op-fresh")
    key: Final = UserAPIKeyAuth(token="hashed", team_id="t", team_object_permission=stale, team_metadata={"v": 1})
    team: Final = LiteLLM_TeamTableCachedObj(team_id="t", object_permission=fresh, metadata={"v": 2})

    token: Final = await project_outcome(
        RustIdentity.VirtualKey(context=_context(key, team=RustTeamGrant.Loaded(team=team)))
    )

    assert (token.team_object_permission, token.team_metadata) == (fresh, {"v": 2})


@pytest.mark.asyncio
async def test_unloadable_team_keeps_the_keys_metadata_with_the_permission_it_names(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    permission: Final = LiteLLM_ObjectPermissionTable(object_permission_id="op-1")
    key: Final = UserAPIKeyAuth(token="hashed", team_id="t", team_metadata={"guardrails": ["g"]})

    token: Final = await project_outcome(
        RustIdentity.VirtualKey(context=_context(key, team=RustTeamGrant.KeyCopy(object_permission=permission))),
    )

    assert (token.team_object_permission, token.team_metadata) == (permission, {"guardrails": ["g"]})


@pytest.mark.asyncio
async def test_owner_budget_and_project_values_reach_the_virtual_key(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u", model_max_budget={"gpt-a": 1.0})
    project_row: Final = LiteLLM_ProjectTableCachedObj(project_id="p", project_alias="alias", metadata={"m": 1})
    context: Final = _context(UserAPIKeyAuth(token="hashed", user_id="u"), owner=owner, project=project_row)

    token: Final = await project_outcome(RustIdentity.VirtualKey(context=context))

    assert (token.user_model_max_budget, token.project_alias, token.project_metadata) == (
        {"gpt-a": 1.0},
        "alias",
        {"m": 1},
    )


@pytest.mark.asyncio
async def test_unreachable_database_gets_the_restricted_fallback_identity(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    token: Final = await project_outcome(RustIdentity.DbUnavailable(message="db down"))

    assert (token.user_id, token.user_role, token.request_route) == (
        DB_UNAVAILABLE_FALLBACK_USER_ID,
        LitellmUserRoles.INTERNAL_USER,
        "/chat/completions",
    )


@pytest.mark.asyncio
async def test_known_end_user_limits_and_permission_reach_the_virtual_key(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    permission: Final = LiteLLM_ObjectPermissionTable(object_permission_id="op-eu")
    end_user: Final = EndUserRow(
        end_user_id="eu",
        end_user=LiteLLM_EndUserTable(user_id="eu", blocked=False, object_permission=permission),
        params={"end_user_id": "eu", "end_user_tpm_limit": 7},
    )
    context: Final = _context(UserAPIKeyAuth(token="hashed"), end_user=end_user)

    token: Final = await project_outcome(RustIdentity.VirtualKey(context=context))

    assert (token.end_user_id, token.end_user_tpm_limit) == ("eu", 7)
    assert token.end_user_object_permission == permission


@pytest.mark.asyncio
async def test_unknown_end_user_limits_are_dropped_without_a_default_budget(
    monkeypatch: pytest.MonkeyPatch, project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]]
) -> None:
    monkeypatch.setattr(litellm, "max_end_user_budget_id", None)
    end_user: Final = EndUserRow(end_user_id="eu", end_user=None, params={"end_user_id": "eu", "end_user_tpm_limit": 7})
    context: Final = _context(UserAPIKeyAuth(token="hashed"), end_user=end_user)

    token: Final = await project_outcome(RustIdentity.VirtualKey(context=context))

    assert token.end_user_tpm_limit is None


@pytest.mark.asyncio
async def test_mapped_jwt_keeps_its_claims_and_fills_a_missing_email(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    token: Final = await project_outcome(
        RustIdentity.MappedJwtKey(
            claims={"sub": "u"},
            end_user_id="eu",
            user_email="u@example.com",
            auto_registered=False,
            minted=False,
            org_id=None,
            context=_context(UserAPIKeyAuth(token="hashed", user_id="u")),
        ),
    )

    assert (token.jwt_claims, token.user_email, token.end_user_id) == ({"sub": "u"}, "u@example.com", "eu")
    assert token.via_virtual_key is True


@pytest.mark.asyncio
async def test_master_key_is_named_by_its_alias_never_the_key(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    end_user: Final = EndUserRow(end_user_id="eu", end_user=None, params={"end_user_id": "eu"})

    token: Final = await project_outcome(RustIdentity.MasterKey(hash="master-hash", end_user=end_user))

    assert token.api_key != "sk-master"
    assert (token.user_role, token.user_id, token.end_user_id) == (
        LitellmUserRoles.PROXY_ADMIN,
        "default_user_id",
        "eu",
    )
    assert token.via_virtual_key is True


def _jwt(**overrides: object) -> RustIdentity.Jwt:
    fields: Final[dict[str, object]] = {
        "claims": {"sub": "u"},
        "is_proxy_admin": False,
        "user_id": "u",
        "user_email": "u@example.com",
        "team_id": None,
        "org_id": None,
        "end_user_id": None,
        "agent_id": None,
        "managed_agent": None,
        "user": None,
        "team": None,
        "membership": None,
        "org": None,
        "end_user": None,
        **overrides,
    }
    return RustIdentity.Jwt(**fields)  # pyright: ignore[reportArgumentType]  # test double fields


@pytest.mark.asyncio
async def test_jwt_proxy_admin_is_projected_as_an_admin_with_its_claims(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    user: Final = LiteLLM_UserTable(user_id="u", user_role="internal_user", tpm_limit=5)

    token: Final = await project_outcome(_jwt(is_proxy_admin=True, user=user))

    assert (token.user_role, token.user_id, token.jwt_claims, token.user_tpm_limit) == (
        LitellmUserRoles.PROXY_ADMIN,
        "u",
        {"sub": "u"},
        None,
    )


@pytest.mark.asyncio
async def test_a_finished_jwt_is_projected_as_the_token_the_host_finished(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    finished: Final = UserAPIKeyAuth(user_id="u", project_alias="p")

    assert await project_outcome(RustIdentity.FinishedJwt(token=finished)) is finished


def test_jwt_user_takes_its_role_and_limits_from_its_user_row() -> None:
    user: Final = LiteLLM_UserTable(user_id="u", user_role="internal_user", tpm_limit=5)

    token: Final = auth_identity.auth_from_result(auth_identity.jwt_result(_jwt(user=user), "jwt"))

    assert (token.user_role, token.user_tpm_limit, token.user_email) == (
        LitellmUserRoles.INTERNAL_USER,
        5,
        "u@example.com",
    )


def test_jwt_result_carries_every_row_the_identity_names() -> None:
    team: Final = LiteLLM_TeamTable(team_id="t2", models=["gpt-b"], tpm_limit=9)
    membership: Final = LiteLLM_TeamMembership(user_id="u", team_id="t2")
    org: Final = LiteLLM_OrganizationTable(
        organization_id="o1", budget_id="b", models=[], created_by="x", updated_by="x"
    )
    end_user: Final = LiteLLM_EndUserTable(user_id="eu", blocked=False)

    result: Final = auth_identity.jwt_result(
        _jwt(
            team_id="t2",
            org_id="o1",
            end_user_id="eu",
            team=team,
            membership=membership,
            org=org,
            end_user=EndUserRow(end_user_id="eu", end_user=end_user, params={}),
        ),
        "jwt",
    )

    assert (
        result["token"],
        result["team_object"],
        result.get("team_membership"),
        result["org_object"],
        result["end_user_object"],
    ) == ("jwt", team, membership, org, end_user)
    assert auth_identity.auth_from_result(result).team_tpm_limit == 9


def test_delegated_managed_agent_carries_its_one_shot_proof() -> None:
    managed: Final = ManagedAgentContext(agent_id="a", mode="delegated", user_id="u")

    token: Final = auth_identity.auth_from_result(
        auth_identity.jwt_result(_jwt(agent_id="a", managed_agent=managed), "jwt")
    )

    assert token.managed_agent_context == managed
    assert token._managed_delegation_verified is True  # pyright: ignore[reportPrivateUsage]  # the proof is private
    assert token.user_role == LitellmUserRoles.INTERNAL_USER


@pytest.mark.asyncio
async def test_a_mapped_key_without_an_owner_takes_the_model_budget_of_the_user_the_policy_loaded(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    token: Final = await project_outcome(
        RustIdentity.MappedJwtKey(
            claims={"sub": "u"},
            end_user_id=None,
            user_email=None,
            auto_registered=True,
            minted=False,
            org_id=None,
            user=LiteLLM_UserTable(user_id="u", model_max_budget={"gpt-a": 2.0}),
            context=_context(UserAPIKeyAuth(token="hashed")),
        )
    )

    assert token.user_model_max_budget == {"gpt-a": 2.0}


@pytest.mark.asyncio
async def test_auto_registered_minted_key_takes_the_jwt_org(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    owner: Final = LiteLLM_UserTable(user_id="u", model_max_budget={"gpt-a": 1.0})

    token: Final = await project_outcome(
        RustIdentity.MappedJwtKey(
            claims={"sub": "u"},
            end_user_id=None,
            user_email="u@example.com",
            auto_registered=True,
            minted=True,
            org_id="jwt-org",
            user=None,
            context=_context(UserAPIKeyAuth(token="hashed", user_id="u", org_id="key-org"), owner=owner),
        ),
    )

    assert (token.org_id, token.api_key, token.jwt_claims, token.user_model_max_budget) == (
        "jwt-org",
        "hashed",
        {"sub": "u"},
        {"gpt-a": 1.0},
    )


@pytest.mark.asyncio
async def test_an_existing_key_linked_to_a_jwt_keeps_its_own_org(
    project_outcome: Callable[[RustIdentity], Awaitable[UserAPIKeyAuth]],
) -> None:
    token: Final = await project_outcome(
        RustIdentity.MappedJwtKey(
            claims={"sub": "u"},
            end_user_id=None,
            user_email=None,
            auto_registered=True,
            minted=False,
            org_id="jwt-org",
            user=None,
            context=_context(UserAPIKeyAuth(token="hashed", user_id="u", org_id="key-org")),
        ),
    )

    assert token.org_id == "key-org"
