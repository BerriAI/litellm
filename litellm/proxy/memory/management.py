from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from litellm.proxy._types import UI_TEAM_ID, LitellmUserRoles, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.memory.memory_endpoints import is_memory_team_admin, require_memory_prisma
from litellm.proxy.memory.policy import (
    MemoryAccess,
    MemoryIdentity,
    invalidate_memory_configuration,
    memory_digest,
    memory_primary_client,
    resolve_memory_access,
)
from litellm.proxy.memory.store import MemoryStore
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.project_repository import ProjectRepository
from litellm.repositories.table_repositories import (
    MemoryPolicyRepository,
    MemoryPreferenceRepository,
    OrganizationMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.types.memory_v2 import (
    MemoryCapture,
    MemoryEntry,
    MemoryPolicy,
    MemoryPolicyInput,
    MemoryPreference,
    MemorySearch,
    MemoryStatus,
    MemoryTarget,
)

_AUTH: Final = Depends(user_api_key_auth)

router: Final = APIRouter(
    prefix="/v2/memory",
    tags=[  # mutable-ok: Prisma serializes these as native JSON containers.
        "memory management"
    ],
)


async def require_policy_admin(
    auth: UserAPIKeyAuth, target_type: MemoryTarget, target_id: str, *, write: bool = True
) -> None:
    prisma: Final = memory_primary_client(require_memory_prisma())
    if write and auth.user_role in (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY):
        raise HTTPException(status_code=403, detail="Memory policies require administrator write access")
    proxy_admin: Final = (
        auth.user_role == LitellmUserRoles.PROXY_ADMIN or not write and user_api_key_has_admin_view(auth)
    )
    if target_type == "gateway":
        if not proxy_admin or target_id != "*":
            raise HTTPException(status_code=403, detail="Gateway memory policies require a proxy administrator")
        return
    if target_type == "organization":
        organization: Final = await OrganizationRepository(prisma).find_by_id(target_id)
        membership: Final = await OrganizationMembershipRepository(prisma).table.find_first(
            where={  # mutable-ok: Prisma serializes these as native JSON containers.
                "organization_id": target_id,
                "user_id": auth.user_id or "",
                "user_role": "org_admin",
            }
        )
        if organization and (proxy_admin or membership):
            return
    if target_type == "team":
        team: Final = await TeamRepository(prisma).find_by_id(target_id)
        if team and (proxy_admin or await is_memory_team_admin(prisma, auth, target_id)):
            return
    if target_type == "project":
        project: Final = await ProjectRepository(prisma).find_by_id(target_id)
        if project and (proxy_admin or project.team_id and await is_memory_team_admin(prisma, auth, project.team_id)):
            return
    if target_type == "key":
        key: Final = await VerificationTokenRepository(prisma).find_by_id(target_id, id_field="token")
        if key and (proxy_admin or key.team_id and await is_memory_team_admin(prisma, auth, key.team_id)):
            return
    if target_type == "user" and proxy_admin and await UserRepository(prisma).find_by_id(target_id):
        return
    raise HTTPException(status_code=403, detail="You cannot administer memory for this target")


@router.get("/policies", response_model=list[MemoryPolicy])
async def list_policies(
    target_type: MemoryTarget | None = None,
    target_id: str | None = None,
    offset: int = Query(0, ge=0),
    auth: UserAPIKeyAuth = _AUTH,
) -> list[MemoryPolicy]:
    if target_type is not None and target_id is not None:
        await require_policy_admin(auth, target_type, target_id, write=False)
    elif not user_api_key_has_admin_view(auth):
        raise HTTPException(status_code=403, detail="Select a target you administer")
    elif target_type is not None or target_id is not None:
        raise HTTPException(status_code=400, detail="Provide both target_type and target_id")
    rows: Final = await MemoryPolicyRepository(memory_primary_client(require_memory_prisma())).table.find_many(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "target_type": target_type,
            "target_id": target_id,
        }
        if target_type and target_id
        else None,
        take=100,
        skip=offset,
        order={  # mutable-ok: Prisma serializes these as native JSON containers.
            "policy_id": "asc"
        },
    )
    return [  # mutable-ok: Prisma serializes these as native JSON containers.
        MemoryPolicy.model_validate(row, from_attributes=True) for row in rows
    ]


@router.put("/policies", response_model=MemoryPolicy)
async def set_policy(policy: MemoryPolicyInput, auth: UserAPIKeyAuth = _AUTH) -> MemoryPolicy:
    await require_policy_admin(auth, policy.target_type, policy.target_id)
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN and policy.scope in ("user", "organization"):
        if policy.target_type != "organization" or policy.scope != "organization":
            raise HTTPException(status_code=403, detail="This shared scope requires a proxy administrator")
    policy_id: Final = memory_digest(policy.target_type, policy.target_id)
    fields: Final = {  # mutable-ok: Prisma serializes these as native JSON containers.
        **policy.model_dump(),
        "updated_by": auth.user_id or "proxy-admin",
    }
    row: Final = await MemoryPolicyRepository(memory_primary_client(require_memory_prisma())).table.upsert(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "policy_id": policy_id
        },
        data={  # mutable-ok: Prisma serializes these as native JSON containers.
            "create": {  # mutable-ok: Prisma serializes these as native JSON containers.
                **fields,
                "policy_id": policy_id,
            },
            "update": fields,
        },
    )
    await invalidate_memory_configuration()
    return MemoryPolicy.model_validate(row, from_attributes=True)


@router.delete("/policies/{policy_id}", status_code=204)
async def delete_policy(policy_id: str, auth: UserAPIKeyAuth = _AUTH) -> Response:
    table: Final = MemoryPolicyRepository(memory_primary_client(require_memory_prisma())).table
    row: Final = await table.find_unique(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "policy_id": policy_id
        }
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Memory policy not found")
    policy: Final = MemoryPolicy.model_validate(row, from_attributes=True)
    await require_policy_admin(auth, policy.target_type, policy.target_id)
    await table.delete(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "policy_id": policy_id
        }
    )
    await invalidate_memory_configuration()
    return Response(status_code=204)


@router.get("/preference", response_model=MemoryPreference)
async def get_preference(auth: UserAPIKeyAuth = _AUTH) -> MemoryPreference:
    subject: Final = MemoryIdentity.from_auth(auth).preference_subject
    row: Final = await MemoryPreferenceRepository(memory_primary_client(require_memory_prisma())).table.find_unique(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "subject": subject
        }
    )
    return MemoryPreference(enabled=row.enabled if row else False)


@router.put("/preference", response_model=MemoryPreference)
async def set_preference(preference: MemoryPreference, auth: UserAPIKeyAuth = _AUTH) -> MemoryPreference:
    identity: Final = MemoryIdentity.from_auth(auth)
    if identity.read_only:
        raise HTTPException(status_code=403, detail="Read-only users cannot change memory preferences")
    subject: Final = identity.preference_subject
    if not preference.enabled:
        await MemoryPreferenceRepository(memory_primary_client(require_memory_prisma())).table.delete_many(
            where={  # mutable-ok: Prisma serializes these as native JSON containers.
                "subject": subject
            }
        )
        return preference
    await MemoryPreferenceRepository(memory_primary_client(require_memory_prisma())).table.upsert(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "subject": subject
        },
        data={  # mutable-ok: Prisma serializes these as native JSON containers.
            "create": {  # mutable-ok: Prisma serializes these as native JSON containers.
                "subject": subject,
                "enabled": preference.enabled,
            },
            "update": {  # mutable-ok: Prisma serializes these as native JSON containers.
                "enabled": preference.enabled
            },
        },
    )
    return preference


@router.get("/status", response_model=MemoryStatus)
async def get_status(
    key_id: str | None = Query(None, pattern=r"^[a-f0-9]{64}$"),
    auth: UserAPIKeyAuth = _AUTH,
) -> MemoryStatus:
    return (await access_for_key(auth, key_id)).status


async def access_for_key(auth: UserAPIKeyAuth, key_id: str | None) -> MemoryAccess:
    prisma: Final = memory_primary_client(require_memory_prisma())
    if key_id is None:
        return await resolve_memory_access(prisma, MemoryIdentity.from_auth(auth))
    key: Final = await VerificationTokenRepository(prisma).find_by_id(key_id, id_field="token")
    if key is None or not (
        user_api_key_has_admin_view(auth)
        or key_id == MemoryIdentity.from_auth(auth).key_id
        or (auth.is_session_token or auth.team_id == UI_TEAM_ID)
        and auth.user_id
        and key.user_id == auth.user_id
    ):
        raise HTTPException(status_code=403, detail="You cannot access memory for this key")
    team: Final = await TeamRepository(prisma).find_by_id(key.team_id) if key.team_id else None
    identity: Final = MemoryIdentity(
        key_id=key_id,
        user_id=key.user_id,
        team_id=key.team_id,
        project_id=key.project_id,
        organization_id=key.org_id or (team.organization_id if team else None),
        read_only=MemoryIdentity.from_auth(auth).read_only,
    )
    return await resolve_memory_access(prisma, identity)


@router.get("/entries", response_model=list[MemoryEntry])
async def list_entries(
    query: str = Query("", max_length=500),
    limit: int = Query(20, ge=1, le=20),
    offset: int = Query(0, ge=0),
    key_id: str | None = Query(None, pattern=r"^[a-f0-9]{64}$"),
    auth: UserAPIKeyAuth = _AUTH,
) -> list[MemoryEntry]:
    prisma: Final = memory_primary_client(require_memory_prisma())
    access: Final = await access_for_key(auth, key_id)
    return await MemoryStore(prisma, access).search(
        MemorySearch(query=query, limit=limit, offset=offset), require_active=False
    )


@router.post("/entries", response_model=MemoryEntry)
async def capture_entry(
    capture: MemoryCapture,
    key_id: str | None = Query(None, pattern=r"^[a-f0-9]{64}$"),
    auth: UserAPIKeyAuth = _AUTH,
) -> MemoryEntry:
    prisma: Final = memory_primary_client(require_memory_prisma())
    access: Final = await access_for_key(auth, key_id)
    return await MemoryStore(prisma, access).capture(capture)


@router.delete("/entries/{memory_id}", status_code=204)
async def delete_entry(
    memory_id: str,
    key_id: str | None = Query(None, pattern=r"^[a-f0-9]{64}$"),
    auth: UserAPIKeyAuth = _AUTH,
) -> Response:
    prisma: Final = memory_primary_client(require_memory_prisma())
    access: Final = await access_for_key(auth, key_id)
    if not await MemoryStore(prisma, access).delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)
