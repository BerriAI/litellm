import hashlib
import json
from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException

from litellm.proxy._types import UI_TEAM_ID, LitellmUserRoles, UserAPIKeyAuth
from litellm.repositories.table_repositories import MemoryPolicyRepository, MemoryPreferenceRepository
from litellm.types.memory_v2 import MemoryPolicy, MemoryScope, MemoryStatus


def memory_digest(*parts: str | None) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MemoryIdentity:
    key_id: str | None
    user_id: str | None
    team_id: str | None
    project_id: str | None
    organization_id: str | None
    read_only: bool

    @classmethod
    def from_auth(cls, auth: UserAPIKeyAuth) -> "MemoryIdentity":
        token: Final = auth.token or auth.api_key
        key_id: Final = (
            token
            if token
            and len(token) == 64
            and all(c in "0123456789abcdef" for c in token)
            and not auth.is_session_token
            and auth.team_id != UI_TEAM_ID
            else None
        )
        return cls(
            key_id=key_id,
            user_id=auth.user_id,
            team_id=auth.team_id,
            project_id=auth.project_id,
            organization_id=auth.org_id,
            read_only=auth.user_role
            in (
                LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
                LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            ),
        )

    @property
    def preference_subject(self) -> str:
        if self.user_id:
            return memory_digest("user", self.user_id)
        if self.key_id:
            return memory_digest("key", self.key_id)
        raise HTTPException(status_code=403, detail="Memory requires an authenticated user or virtual key")

    @property
    def policy_targets(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (kind, value)
            for kind, value in (
                ("key", self.key_id),
                ("user", self.user_id),
                ("project", self.project_id),
                ("team", self.team_id),
                ("organization", self.organization_id),
                ("gateway", "*"),
            )
            if value
        )

    def namespace(self, scope: MemoryScope) -> str | None:
        if scope == "key":
            return (
                memory_digest(scope, self.organization_id, self.team_id, self.project_id, self.key_id)
                if self.key_id
                else None
            )
        if scope == "user":
            return memory_digest(scope, self.organization_id, self.user_id) if self.user_id else None
        if scope == "team":
            return memory_digest(scope, self.organization_id, self.team_id) if self.team_id else None
        if scope == "project":
            return (
                memory_digest(scope, self.organization_id, self.team_id, self.project_id) if self.project_id else None
            )
        return memory_digest(scope, self.organization_id) if self.organization_id else None


@dataclass(frozen=True)
class MemoryAccess:
    identity: MemoryIdentity
    policy: MemoryPolicy | None
    opted_in: bool

    @property
    def namespace(self) -> str | None:
        return self.identity.namespace(self.policy.scope) if self.policy else None

    @property
    def active(self) -> bool:
        return bool(
            self.namespace
            and self.policy
            and (self.policy.activation == "automatic" or self.policy.activation == "opt_in" and self.opted_in)
        )

    @property
    def status(self) -> MemoryStatus:
        return MemoryStatus(
            active=self.active,
            activation=self.policy.activation if self.policy else "disabled",
            scope=self.policy.scope if self.policy else None,
            opted_in=self.opted_in,
            policy_id=self.policy.policy_id if self.policy else None,
        )


async def resolve_memory_access(prisma_client: object, identity: MemoryIdentity) -> MemoryAccess:
    rows: Final = await MemoryPolicyRepository(prisma_client).table.find_many(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "OR": [  # mutable-ok: Prisma serializes these as native JSON containers.
                {  # mutable-ok: Prisma serializes these as native JSON containers.
                    "target_type": kind,
                    "target_id": target,
                }
                for kind, target in identity.policy_targets
            ]
        },
        take=len(identity.policy_targets),
    )
    policies: Final = {  # mutable-ok: Prisma serializes these as native JSON containers.
        (row.target_type, row.target_id): MemoryPolicy.model_validate(row, from_attributes=True) for row in rows
    }
    policy: Final = next((policies[target] for target in identity.policy_targets if target in policies), None)
    if not identity.user_id and not identity.key_id:
        return MemoryAccess(identity=identity, policy=None, opted_in=False)
    preference: Final = await MemoryPreferenceRepository(prisma_client).table.find_unique(
        where={  # mutable-ok: Prisma serializes these as native JSON containers.
            "subject": identity.preference_subject
        }
    )
    return MemoryAccess(identity=identity, policy=policy, opted_in=preference.enabled if preference else False)
