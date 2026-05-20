"""Immutable read-world seed for behavior-pinning tests.

Seeds 8 actor profiles across 2 orgs / 2 teams so authz matrix tests can drive
``(actor, target, expected)`` tuples against the real proxy. Each actor has
exactly one virtual key (cleartext + hashed) so tests pass the cleartext as a
``Bearer`` token and the real auth stack accepts it.

All seeded rows are namespaced under the ``behavior-pin-`` prefix on their
primary key, so the seed is identifiable in psql and the wipe step is targeted.
"""

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


# Stable IDs so re-seed is idempotent and the world is identifiable in psql.
PREFIX = "behavior-pin-"
ORG_A = PREFIX + "org-a"
ORG_B = PREFIX + "org-b"
TEAM_ALPHA = PREFIX + "team-alpha"
TEAM_BETA = PREFIX + "team-beta"
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
    keys: Dict[Actor, SeededKey]


def _new_clear_key() -> str:
    return "sk-" + uuid.uuid4().hex


def _actor_profile() -> Dict[Actor, Dict[str, Any]]:
    """Per-actor (role, scoping) profile used for both the user row and its key.

    Scoping rules:
    - PROXY_ADMIN: global, no team/org scope on its key.
    - ORG_ADMIN: org_a, no team scope.
    - TEAM_ADMIN / INTERNAL_USER / OWNER / UNRELATED_SAME_ORG / SERVICE_ACCOUNT:
      team_alpha within org_a.
    - CROSS_ORG_USER: team_beta within org_b.

    The auth layer reads ``user_role`` off the user row pointed at by the key's
    ``user_id``, so setting it once on the user is enough — keys do not need a
    separate role field.
    """
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
    }


async def _wipe_world(prisma: PrismaClient) -> None:
    """Delete prior seed rows so re-seed is idempotent across sessions."""
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

    # Budget — orgs require a non-null budget_id.
    await prisma.db.litellm_budgettable.create(
        data={
            "budget_id": BUDGET_ID,
            "created_by": "behavior-pin-seeder",
            "updated_by": "behavior-pin-seeder",
        }
    )

    # Orgs.
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

    # Users.
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

    # Teams.
    # NOTE: _get_user_in_team in key_management_endpoints.py checks
    # ``members_with_roles`` (a JSON array of {user_id, role}), NOT the plain
    # ``members`` String[] column — so the JSON list is what the team-key authz
    # gate inspects. Populate both to match what the real /team/new handler
    # would produce.
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

    # Org memberships (UI lookups + ORG_ADMIN scoping use these).
    for actor, org_id, role in [
        (Actor.ORG_ADMIN, ORG_A, "org_admin"),
        (Actor.TEAM_ADMIN, ORG_A, "internal_user"),
        (Actor.INTERNAL_USER, ORG_A, "internal_user"),
        (Actor.OWNER, ORG_A, "internal_user"),
        (Actor.UNRELATED_SAME_ORG, ORG_A, "internal_user"),
        (Actor.SERVICE_ACCOUNT, ORG_A, "internal_user"),
        (Actor.CROSS_ORG_USER, ORG_B, "internal_user"),
    ]:
        await prisma.db.litellm_organizationmembership.create(
            data={
                "user_id": user_ids[actor],
                "organization_id": org_id,
                "user_role": role,
            }
        )

    # Team memberships.
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

    # Verification tokens — one cleartext per actor, hashed via the real
    # credential boundary so user_api_key_auth accepts it.
    keys: Dict[Actor, SeededKey] = {}
    for actor, profile in profiles.items():
        cleartext = _new_clear_key()
        hashed = hash_token(cleartext)
        token_data: Dict[str, Any] = {
            "token": hashed,
            "key_name": PREFIX + actor.value + "-key",
            "user_id": user_ids[actor],
            # LiteLLM_VerificationTokenView is non-Optional on models — Postgres lets the
            # column be NULL, but the pydantic view used by user_api_key_auth rejects None.
            "models": [],
        }
        if profile["team_id"]:
            token_data["team_id"] = profile["team_id"]
        if profile["organization_id"]:
            token_data["organization_id"] = profile["organization_id"]
        if actor == Actor.SERVICE_ACCOUNT:
            # LiteLLM convention: service-account keys carry an explicit metadata flag.
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
        keys=keys,
    )
