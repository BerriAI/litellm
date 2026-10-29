import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import yaml
from prisma import Prisma
from pydantic import JsonValue

from litellm.proxy.utils import PrismaClient, hash_token
from tests.integration._support.client import Gateway, Scenario, string_value
from tests.integration._support.prisma_clients import connected_proxy_prisma
from tests.integration._support.process import OwnedState, owned_proxy

TEAM_ALPHA: Final = "behavior-pin-team-alpha"
TEAM_BETA: Final = "behavior-pin-team-beta"
TEAM_GAMMA: Final = "behavior-pin-team-gamma"
ORG_A: Final = "behavior-pin-org-a"
ORG_B: Final = "behavior-pin-org-b"


class Actor(str, Enum):
    PROXY_ADMIN = "proxy_admin"
    ORG_ADMIN = "org_admin"
    TEAM_ADMIN = "team_admin"
    INTERNAL_USER = "internal_user"
    OWNER = "owner"
    UNRELATED_SAME_ORG = "unrelated_same_org"
    CROSS_ORG_USER = "cross_org_user"
    SERVICE_ACCOUNT = "service_account"
    ORG_B_ADMIN = "org_b_admin"


@dataclass(frozen=True, slots=True)
class SeededKey:
    user_id: str
    cleartext: str
    hashed: str


@dataclass(frozen=True, slots=True)
class ActorSet:
    org_a_id: str
    org_b_id: str
    team_alpha_id: str
    team_beta_id: str
    team_gamma_id: str
    keys: Mapping[Actor, SeededKey]


@dataclass(frozen=True, slots=True)
class ScratchResources:
    prefix: str

    def tag(self, suffix: str = "") -> str:
        return f"{self.prefix}-{suffix}" if suffix else self.prefix


@dataclass(frozen=True, slots=True)
class GatewayClient:
    gateway: Gateway
    replacements: Mapping[str, str]

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, JsonValue] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        resolved_path: Final = _replace_text(path, self.replacements)
        resolved_body: Final = _replace_body(json, self.replacements) if json is not None else None
        return self.gateway.request(method, resolved_path, resolved_body, headers=headers, params=params)

    async def post(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, JsonValue] | None = None,
    ) -> httpx.Response:
        return await self.request("POST", path, headers=headers, json=json)

    async def get(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return await self.request("GET", path, headers=headers, params=params)

    async def put(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, JsonValue] | None = None,
    ) -> httpx.Response:
        return await self.request("PUT", path, headers=headers, json=json)

    async def delete(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, JsonValue] | None = None,
    ) -> httpx.Response:
        return await self.request("DELETE", path, headers=headers, json=json)


@contextmanager
def owned_settings_client(
    client: GatewayClient, directory: Path, litellm_settings: Mapping[str, JsonValue]
) -> Iterator[GatewayClient]:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {**base, "litellm_settings": {**base["litellm_settings"], **litellm_settings}}
    path: Final = directory / "owned-settings.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(client.gateway, directory, {}, config=path, state=OwnedState.SHARED_WITH_WORKER) as candidate:
        yield GatewayClient(candidate, client.replacements)


@dataclass(frozen=True, slots=True)
class ManagementDatabase:
    prisma: PrismaClient
    gateway: Gateway

    @property
    def db(self) -> Prisma:
        return self.prisma.db


def _replace_text(value: str, replacements: Mapping[str, str]) -> str:
    return value.replace(TEAM_ALPHA, replacements[TEAM_ALPHA]).replace(
        TEAM_BETA, replacements[TEAM_BETA]
    ).replace(TEAM_GAMMA, replacements[TEAM_GAMMA]).replace(ORG_A, replacements[ORG_A]).replace(
        ORG_B, replacements[ORG_B]
    )


def _replace_body(
    value: JsonValue, replacements: Mapping[str, str]
) -> JsonValue:
    if isinstance(value, str):
        return _replace_text(value, replacements)
    if isinstance(value, list):
        return [_replace_body(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_body(item, replacements) for key, item in value.items()}
    return value


def _seeded_key(
    scenario: Scenario,
    actor: Actor,
    user_id: str,
    *,
    team_id: str | None = None,
    organization_id: str | None = None,
) -> SeededKey:
    token: Final = scenario.key(
        key_alias=f"integration-{actor.value}-{uuid.uuid4().hex}",
        user_id=user_id,
        team_id=team_id,
        organization_id=organization_id,
    )
    return SeededKey(user_id=user_id, cleartext=token, hashed=hash_token(token))


async def create_actor_set(
    gateway: Gateway, scenario: Scenario, db: ManagementDatabase
) -> ActorSet:
    run_id: Final = uuid.uuid4().hex
    org_a_id: Final = scenario.organization(organization_alias=f"alpha-{run_id}")
    org_b_id: Final = scenario.organization(organization_alias=f"beta-{run_id}")
    proxy_admin_id: Final = scenario.user(user_role="proxy_admin")
    org_admin_a_id: Final = scenario.org_member(org_a_id, "org_admin")
    org_admin_b_id: Final = scenario.org_member(org_b_id, "org_admin")
    await db.db.litellm_usertable.update(
        where={"user_id": org_admin_a_id}, data={"user_role": "org_admin"}
    )
    await db.db.litellm_usertable.update(
        where={"user_id": org_admin_b_id}, data={"user_role": "org_admin"}
    )
    team_alpha_id: Final = scenario.team(
        team_id=f"integration-{run_id}-alpha",
        team_alias=f"alpha-{run_id}",
        organization_id=org_a_id,
        team_member_permissions=["/key/health"],
    )
    team_beta_id: Final = scenario.team(
        team_id=f"integration-{run_id}-beta",
        team_alias=f"beta-{run_id}",
        organization_id=org_b_id,
        team_member_permissions=["/key/health"],
    )
    team_gamma_id: Final = scenario.team(
        team_id=f"integration-{run_id}-gamma",
        team_alias=f"gamma-{run_id}",
        organization_id=org_a_id,
        team_member_permissions=["/key/health"],
    )
    team_admin_id: Final = scenario.member(team_alpha_id, role="admin")
    internal_user_id: Final = scenario.member(team_alpha_id)
    owner_id: Final = scenario.member(team_alpha_id)
    unrelated_id: Final = scenario.member(team_alpha_id)
    service_account_id: Final = scenario.member(team_alpha_id)
    cross_org_id: Final = scenario.member(team_beta_id)
    keys: Final[Mapping[Actor, SeededKey]] = MappingProxyType(
        {
            Actor.PROXY_ADMIN: _seeded_key(scenario, Actor.PROXY_ADMIN, proxy_admin_id),
            Actor.ORG_ADMIN: _seeded_key(
                scenario, Actor.ORG_ADMIN, org_admin_a_id, organization_id=org_a_id
            ),
            Actor.TEAM_ADMIN: _seeded_key(
                scenario, Actor.TEAM_ADMIN, team_admin_id, team_id=team_alpha_id, organization_id=org_a_id
            ),
            Actor.INTERNAL_USER: _seeded_key(
                scenario, Actor.INTERNAL_USER, internal_user_id, team_id=team_alpha_id, organization_id=org_a_id
            ),
            Actor.OWNER: _seeded_key(
                scenario, Actor.OWNER, owner_id, team_id=team_alpha_id, organization_id=org_a_id
            ),
            Actor.UNRELATED_SAME_ORG: _seeded_key(
                scenario, Actor.UNRELATED_SAME_ORG, unrelated_id, team_id=team_alpha_id, organization_id=org_a_id
            ),
            Actor.CROSS_ORG_USER: _seeded_key(
                scenario, Actor.CROSS_ORG_USER, cross_org_id, team_id=team_beta_id, organization_id=org_b_id
            ),
            Actor.SERVICE_ACCOUNT: _seeded_key(
                scenario, Actor.SERVICE_ACCOUNT, service_account_id, team_id=team_alpha_id, organization_id=org_a_id
            ),
            Actor.ORG_B_ADMIN: _seeded_key(
                scenario, Actor.ORG_B_ADMIN, org_admin_b_id, organization_id=org_b_id
            ),
        }
    )
    return ActorSet(org_a_id, org_b_id, team_alpha_id, team_beta_id, team_gamma_id, keys)


def create_gateway_client(gateway: Gateway, actors: ActorSet) -> GatewayClient:
    replacements: Final[Mapping[str, str]] = MappingProxyType(
        {
            TEAM_ALPHA: actors.team_alpha_id,
            TEAM_BETA: actors.team_beta_id,
            TEAM_GAMMA: actors.team_gamma_id,
            ORG_A: actors.org_a_id,
            ORG_B: actors.org_b_id,
        }
    )
    return GatewayClient(gateway, replacements)


async def create_scratch_key(
    gateway_client: GatewayClient,
    seeder_cleartext: str,
    scratch_prefix: str,
    *,
    user_id: str,
    team_id: str | None = None,
    organization_id: str | None = None,
    key_alias: str | None = None,
) -> str:
    body: Final[dict[str, JsonValue]] = {
        "key_alias": key_alias or scratch_prefix,
        "user_id": user_id,
        **({"team_id": team_id} if team_id is not None else {}),
        **({"organization_id": organization_id} if organization_id is not None else {}),
    }
    created: Final = gateway_client.gateway.post(
        "/key/generate", body, key=seeder_cleartext
    )
    return string_value(created["key"])


async def create_scratch_team(
    db: ManagementDatabase,
    team_id: str,
    *,
    organization_id: str | None = None,
    admin_user_ids: Sequence[str] = (),
    member_user_ids: Sequence[str] = (),
    team_member_permissions: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    max_budget: float | None = None,
    tpm_limit: int | None = None,
    rpm_limit: int | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> str:
    admins: Final = list(admin_user_ids)
    members: Final = list(member_user_ids)
    members_with_roles: Final = [
        {"user_id": identity, "role": "admin"} for identity in admins
    ] + [{"user_id": identity, "role": "user"} for identity in members]
    fields: Final[dict[str, JsonValue]] = {
        "team_id": team_id,
        "team_alias": team_id,
        "admins": admins,
        "members": admins + members,
        "members_with_roles": members_with_roles,
        **({"organization_id": organization_id} if organization_id is not None else {}),
        **(
            {"team_member_permissions": list(team_member_permissions)}
            if team_member_permissions is not None
            else {}
        ),
        **({"models": list(models)} if models is not None else {}),
        **({"max_budget": max_budget} if max_budget is not None else {}),
        **({"tpm_limit": tpm_limit} if tpm_limit is not None else {}),
        **({"rpm_limit": rpm_limit} if rpm_limit is not None else {}),
        **({"metadata": dict(metadata)} if metadata is not None else {}),
    }
    created: Final = db.gateway.post("/team/new", fields)
    return string_value(created["team_id"])


async def create_scratch_user(
    db: ManagementDatabase,
    scratch_prefix: str,
    *,
    suffix: str,
    user_email: str | None = None,
) -> str:
    user_id: Final = f"{scratch_prefix}-{suffix}"
    fields: Final[dict[str, JsonValue]] = {
        "user_id": user_id,
        "user_role": "internal_user",
        "auto_create_key": False,
        **({"user_email": user_email} if user_email is not None else {}),
    }
    created: Final = db.gateway.post("/user/new", fields)
    return string_value(created["user_id"])


async def create_scratch_org(
    db: ManagementDatabase,
    scratch_prefix: str,
    *,
    max_budget: float | None = None,
    tpm_limit: int | None = None,
    rpm_limit: int | None = None,
    models: Sequence[str] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
    suffix: str = "org",
) -> str:
    organization_id: Final = f"{scratch_prefix}-{suffix}"
    budget_fields: Final[dict[str, JsonValue]] = {
        "budget_id": f"{organization_id}-budget",
        **({"max_budget": max_budget} if max_budget is not None else {}),
        **({"tpm_limit": tpm_limit} if tpm_limit is not None else {}),
        **({"rpm_limit": rpm_limit} if rpm_limit is not None else {}),
    }
    budget: Final = db.gateway.post("/budget/new", budget_fields)
    budget_id: Final = string_value(budget["budget_id"])
    fields: Final[dict[str, JsonValue]] = {
        "organization_id": organization_id,
        "organization_alias": organization_id,
        "budget_id": budget_id,
        **({"models": list(models)} if models is not None else {}),
        **({"metadata": dict(metadata)} if metadata is not None else {}),
    }
    created: Final = db.gateway.post("/organization/new", fields)
    return string_value(created["organization_id"])


async def create_scratch_actor(
    db: ManagementDatabase,
    scratch_prefix: str,
    *,
    user_role: str,
    org_admin_of: Sequence[str] = (),
    organization_id: str | None = None,
    suffix: str = "actor",
) -> SeededKey:
    api_user_role: Final = (
        user_role
        if user_role in {"proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer"}
        else "internal_user"
    )
    created_user: Final = db.gateway.post(
        "/user/new",
        {
            "user_id": f"{scratch_prefix}-{suffix}",
            "user_role": api_user_role,
            "auto_create_key": False,
            **({"organization_id": organization_id} if organization_id is not None else {}),
        },
    )
    user_id: Final = string_value(created_user["user_id"])
    token: Final = string_value(
        db.gateway.post(
            "/key/generate",
            {"user_id": user_id, "key_alias": f"{scratch_prefix}-{suffix}-key"},
        )["key"]
    )
    tuple(
        db.gateway.post(
            "/organization/member_add",
            {"organization_id": org_id, "member": {"user_id": user_id, "role": "org_admin"}},
        )
        for org_id in org_admin_of
    )
    return SeededKey(user_id, token, hash_token(token))


@asynccontextmanager
async def management_database(gateway: Gateway) -> AsyncIterator[ManagementDatabase]:
    async with connected_proxy_prisma() as client:
        yield ManagementDatabase(client, gateway)


@asynccontextmanager
async def scratch_resources(db: ManagementDatabase) -> AsyncIterator[ScratchResources]:
    resource: Final = ScratchResources(f"integration-{uuid.uuid4().hex}")
    yield resource
    prefix: Final = resource.prefix
    await db.db.litellm_verificationtoken.delete_many(
        where={"OR": [{"key_alias": {"startswith": prefix}}, {"key_name": {"startswith": prefix}}]}
    )
    await db.db.litellm_teammembership.delete_many(where={"team_id": {"startswith": prefix}})
    await db.db.litellm_organizationmembership.delete_many(where={"user_id": {"startswith": prefix}})
    await db.db.litellm_teamtable.delete_many(where={"team_id": {"startswith": prefix}})
    await db.db.litellm_usertable.delete_many(
        where={"OR": [{"user_id": {"startswith": prefix}}, {"user_email": {"startswith": prefix}}]}
    )
    await db.db.litellm_organizationtable.delete_many(where={"organization_id": {"startswith": prefix}})
    await db.db.litellm_budgettable.delete_many(where={"budget_id": {"startswith": prefix}})
    users: Final = await db.db.litellm_usertable.find_many(where={"teams": {"isEmpty": False}})
    for user in users:
        teams: Final = [team_id for team_id in user.teams if not team_id.startswith(prefix)]
        if teams != list(user.teams):
            await db.db.litellm_usertable.update(
                where={"user_id": user.user_id},
                data={"teams": {"set": teams}},
            )
