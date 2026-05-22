"""Read-world seed for the authz matrix tests: 2 orgs, 3 teams, 9 actors."""

import enum
import uuid
from dataclasses import dataclass
from typing import Any, Dict

from prisma import Json

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.utils import PrismaClient, hash_token


class Actor(str, enum.Enum):
    PROXY_ADMIN = "proxy_admin"
    ORG_ADMIN = "org_admin"
    TEAM_ADMIN = "team_admin"
    INTERNAL_USER = "internal_user"
    OWNER = "owner"
    UNRELATED_SAME_ORG = "unrelated_same_org"
    CROSS_ORG_USER = "cross_org_user"
    SERVICE_ACCOUNT = "service_account"
    ORG_B_ADMIN = "org_b_admin"


PREFIX = "behavior-pin-"
ORG_A = PREFIX + "org-a"
ORG_B = PREFIX + "org-b"
TEAM_ALPHA = PREFIX + "team-alpha"
TEAM_BETA = PREFIX + "team-beta"
TEAM_GAMMA = PREFIX + "team-gamma"
BUDGET_ID = PREFIX + "budget"


@dataclass(frozen=True)
class SeededKey:
    user_id: str
    cleartext: str
    hashed: str


@dataclass(frozen=True)
class World:
    org_a_id: str
    org_b_id: str
    team_alpha_id: str
    team_beta_id: str
    team_gamma_id: str
    keys: Dict[Actor, SeededKey]


def _new_clear_key() -> str:
    return "sk-" + uuid.uuid4().hex


def _actor_profile() -> Dict[Actor, Dict[str, Any]]:
    return {
        Actor.PROXY_ADMIN: {
            "user_role": LitellmUserRoles.PROXY_ADMIN.value,
            "team_id": None,
            "organization_id": None,
        },
        Actor.ORG_ADMIN: {
            "user_role": LitellmUserRoles.ORG_ADMIN.value,
            "team_id": None,
            "organization_id": ORG_A,
        },
        Actor.TEAM_ADMIN: {
            "user_role": LitellmUserRoles.INTERNAL_USER.value,
            "team_id": TEAM_ALPHA,
            "organization_id": ORG_A,
        },
        Actor.INTERNAL_USER: {
            "user_role": LitellmUserRoles.INTERNAL_USER.value,
            "team_id": TEAM_ALPHA,
            "organization_id": ORG_A,
        },
        Actor.OWNER: {
            "user_role": LitellmUserRoles.INTERNAL_USER.value,
            "team_id": TEAM_ALPHA,
            "organization_id": ORG_A,
        },
        Actor.UNRELATED_SAME_ORG: {
            "user_role": LitellmUserRoles.INTERNAL_USER.value,
            "team_id": TEAM_ALPHA,
            "organization_id": ORG_A,
        },
        Actor.CROSS_ORG_USER: {
            "user_role": LitellmUserRoles.INTERNAL_USER.value,
            "team_id": TEAM_BETA,
            "organization_id": ORG_B,
        },
        Actor.SERVICE_ACCOUNT: {
            "user_role": LitellmUserRoles.INTERNAL_USER.value,
            "team_id": TEAM_ALPHA,
            "organization_id": ORG_A,
        },
        Actor.ORG_B_ADMIN: {
            "user_role": LitellmUserRoles.ORG_ADMIN.value,
            "team_id": None,
            "organization_id": ORG_B,
        },
    }


async def _wipe_world(prisma: PrismaClient) -> None:
    await prisma.db.litellm_verificationtoken.delete_many(
        where={"user_id": {"startswith": PREFIX}}
    )
    await prisma.db.litellm_organizationmembership.delete_many(
        where={"user_id": {"startswith": PREFIX}}
    )
    await prisma.db.litellm_teammembership.delete_many(
        where={"user_id": {"startswith": PREFIX}}
    )
    await prisma.db.litellm_usertable.delete_many(
        where={"user_id": {"startswith": PREFIX}}
    )
    await prisma.db.litellm_teamtable.delete_many(
        where={"team_id": {"startswith": PREFIX}}
    )
    await prisma.db.litellm_organizationtable.delete_many(
        where={"organization_id": {"startswith": PREFIX}}
    )
    await prisma.db.litellm_budgettable.delete_many(where={"budget_id": BUDGET_ID})


async def seed_world(prisma: PrismaClient) -> World:
    await _wipe_world(prisma)

    await prisma.db.litellm_budgettable.create(
        data={
            "budget_id": BUDGET_ID,
            "created_by": "behavior-pin-seeder",
            "updated_by": "behavior-pin-seeder",
        }
    )

    for org_id, alias in [(ORG_A, "alpha"), (ORG_B, "beta")]:
        await prisma.db.litellm_organizationtable.create(
            data={
                "organization_id": org_id,
                "organization_alias": alias,
                "budget_id": BUDGET_ID,
                "created_by": "behavior-pin-seeder",
                "updated_by": "behavior-pin-seeder",
            }
        )

    profiles = _actor_profile()
    user_ids: Dict[Actor, str] = {actor: PREFIX + actor.value for actor in Actor}

    for actor, profile in profiles.items():
        teams_list = [profile["team_id"]] if profile["team_id"] else []
        await prisma.db.litellm_usertable.create(
            data={
                "user_id": user_ids[actor],
                "user_role": profile["user_role"],
                "team_id": profile["team_id"],
                "organization_id": profile["organization_id"],
                "teams": teams_list,
            }
        )

    # _get_user_in_team in key_management_endpoints.py walks members_with_roles
    # (a JSON list of {user_id, role}), not the String[] members column —
    # populate both to match what /team/new produces.
    await prisma.db.litellm_teamtable.create(
        data={
            "team_id": TEAM_ALPHA,
            "team_alias": "alpha-1",
            "organization_id": ORG_A,
            "admins": [user_ids[Actor.TEAM_ADMIN]],
            "members": [
                user_ids[Actor.TEAM_ADMIN],
                user_ids[Actor.INTERNAL_USER],
                user_ids[Actor.OWNER],
                user_ids[Actor.UNRELATED_SAME_ORG],
                user_ids[Actor.SERVICE_ACCOUNT],
            ],
            "members_with_roles": Json(
                [
                    {"user_id": user_ids[Actor.TEAM_ADMIN], "role": "admin"},
                    {"user_id": user_ids[Actor.INTERNAL_USER], "role": "user"},
                    {"user_id": user_ids[Actor.OWNER], "role": "user"},
                    {"user_id": user_ids[Actor.UNRELATED_SAME_ORG], "role": "user"},
                    {"user_id": user_ids[Actor.SERVICE_ACCOUNT], "role": "user"},
                ]
            ),
        }
    )
    await prisma.db.litellm_teamtable.create(
        data={
            "team_id": TEAM_BETA,
            "team_alias": "beta-1",
            "organization_id": ORG_B,
            "admins": [],
            "members": [user_ids[Actor.CROSS_ORG_USER]],
            "members_with_roles": Json(
                [
                    {"user_id": user_ids[Actor.CROSS_ORG_USER], "role": "user"},
                ]
            ),
        }
    )
    # TEAM_GAMMA: ORG_A team with no actor members — the "same-org,
    # not-my-team" read target.
    await prisma.db.litellm_teamtable.create(
        data={
            "team_id": TEAM_GAMMA,
            "team_alias": "gamma-1",
            "organization_id": ORG_A,
            "admins": [],
            "members": [],
            "members_with_roles": Json([]),
        }
    )

    for actor, org_id, role in [
        (Actor.ORG_ADMIN, ORG_A, "org_admin"),
        (Actor.TEAM_ADMIN, ORG_A, "internal_user"),
        (Actor.INTERNAL_USER, ORG_A, "internal_user"),
        (Actor.OWNER, ORG_A, "internal_user"),
        (Actor.UNRELATED_SAME_ORG, ORG_A, "internal_user"),
        (Actor.SERVICE_ACCOUNT, ORG_A, "internal_user"),
        (Actor.CROSS_ORG_USER, ORG_B, "internal_user"),
        (Actor.ORG_B_ADMIN, ORG_B, "org_admin"),
    ]:
        await prisma.db.litellm_organizationmembership.create(
            data={
                "user_id": user_ids[actor],
                "organization_id": org_id,
                "user_role": role,
            }
        )

    for actor, team_id in [
        (Actor.TEAM_ADMIN, TEAM_ALPHA),
        (Actor.INTERNAL_USER, TEAM_ALPHA),
        (Actor.OWNER, TEAM_ALPHA),
        (Actor.UNRELATED_SAME_ORG, TEAM_ALPHA),
        (Actor.SERVICE_ACCOUNT, TEAM_ALPHA),
        (Actor.CROSS_ORG_USER, TEAM_BETA),
    ]:
        await prisma.db.litellm_teammembership.create(
            data={"user_id": user_ids[actor], "team_id": team_id}
        )

    keys: Dict[Actor, SeededKey] = {}
    for actor, profile in profiles.items():
        cleartext = _new_clear_key()
        hashed = hash_token(cleartext)
        token_data: Dict[str, Any] = {
            "token": hashed,
            "key_name": PREFIX + actor.value + "-key",
            "user_id": user_ids[actor],
            # LiteLLM_VerificationTokenView's models field rejects NULL even
            # though the column is nullable in Postgres.
            "models": [],
        }
        if profile["team_id"]:
            token_data["team_id"] = profile["team_id"]
        if profile["organization_id"]:
            token_data["organization_id"] = profile["organization_id"]
        if actor == Actor.SERVICE_ACCOUNT:
            token_data["metadata"] = Json({"service_account_id": user_ids[actor]})
        await prisma.db.litellm_verificationtoken.create(data=token_data)
        keys[actor] = SeededKey(
            user_id=user_ids[actor], cleartext=cleartext, hashed=hashed
        )

    return World(
        org_a_id=ORG_A,
        org_b_id=ORG_B,
        team_alpha_id=TEAM_ALPHA,
        team_beta_id=TEAM_BETA,
        team_gamma_id=TEAM_GAMMA,
        keys=keys,
    )
