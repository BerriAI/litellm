from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Final
from uuid import uuid4

from fastapi import HTTPException
from prisma import Json
from prisma.models import LiteLLM_SCIMResource, LiteLLM_SCIMSource
from prisma.types import LiteLLM_SCIMResourceCreateInput
from pydantic import TypeAdapter

from litellm.proxy._types import LiteLLM_UserTable as UserPolicy
from litellm.proxy.utils import PrismaClient
from litellm.repositories.base_repository import is_unique_violation
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMPatchOp, SCIMPatchOperation, SCIMUser


def human_email(user: SCIMUser) -> str:
    preferred: Final = next((email.value for email in user.emails or () if email.primary), None)
    first: Final = user.emails[0].value if user.emails else None
    email: Final = preferred or first or user.userName
    if not email:
        raise HTTPException(400, "userName or email is required")
    return email.casefold()


def changes_readonly_attribute(operation: SCIMPatchOperation) -> bool:
    path: Final = (operation.path or "").lower()
    value: Final = operation.value
    fields: Final = TypeAdapter(dict[str, object]).validate_python(value) if isinstance(value, dict) else {}
    attributes: Final = tuple(key.lower() for key in fields) if not path else (path,)
    return any(attribute.startswith(("groups", "externalid")) for attribute in attributes)


def validate_human_patch(patch: SCIMPatchOp) -> None:
    if any(changes_readonly_attribute(operation) for operation in patch.Operations):
        raise HTTPException(400, "externalId is immutable; update group membership through this source's Groups")


def patched_username(current: str | None, operation: SCIMPatchOperation) -> str | None:
    direct: Final = bool(operation.path and operation.path.casefold() == "username")
    fields: Final = (
        TypeAdapter(Mapping[str, object]).validate_python(operation.value)
        if operation.path is None and isinstance(operation.value, dict)
        else None
    )
    if not direct and (fields is None or "userName" not in fields):
        return current
    candidate: Final = (
        None
        if operation.op == "remove"
        else operation.value
        if direct
        else fields["userName"]
        if fields is not None
        else None
    )
    if not isinstance(candidate, str) or not candidate:
        raise HTTPException(400, "userName is required")
    return candidate


@dataclass(frozen=True, slots=True)
class SourceHumanProvisioner:
    client: PrismaClient
    source: LiteLLM_SCIMSource

    async def create(self, user: SCIMUser) -> SCIMUser:
        from litellm.proxy.management_endpoints.scim.agent_provisioning import user_document

        if not user.externalId or not user.userName:
            raise HTTPException(400, "externalId and userName are required")
        try:
            row: Final = await self.reserve(user)
        except Exception as exc:
            if is_unique_violation(exc):
                raise HTTPException(409, "This human identity belongs to another provisioning record") from exc
            raise
        if row.deleted:
            raise HTTPException(409, "This subject was deleted; automatic recreation is not permitted")
        if user_document(row).agent_user is not None:
            raise HTTPException(409, "A provisioned agent-user cannot become a human")
        return await self.update(row, user)

    async def reserve(self, user: SCIMUser) -> LiteLLM_SCIMResource:
        if user.externalId is None or user.userName is None:
            raise HTTPException(400, "externalId and userName are required")
        async with self.client.tx() as tx:
            existing: Final = await tx.litellm_scimresource.find_unique(
                where={
                    "source_id_kind_external_id": {
                        "source_id": self.source.source_id,
                        "kind": "Users",
                        "external_id": user.externalId,
                    }
                }
            )
            if existing is not None:
                return existing
            email: Final = human_email(user)
            matches: Final = await tx.litellm_usertable.find_many(
                where={
                    "OR": [
                        {"user_id": user.userName},
                        {"user_email": {"equals": email, "mode": "insensitive"}},
                    ]
                }
            )
            if matches:
                raise HTTPException(
                    409, "This local user already exists; automatic directory adoption is not permitted"
                )
            local_id: Final = user.userName
            scim_id: Final = str(uuid4())
            document: Final = user.model_copy(update={"id": scim_id})
            data: Final = LiteLLM_SCIMResourceCreateInput(
                id=scim_id,
                source_id=self.source.source_id,
                kind="Users",
                external_id=user.externalId,
                user_name=user.userName,
                display_name=user.displayName or user.userName,
                document=Json(document.model_dump(by_alias=True, mode="json", exclude_none=True)),
                active=user.active,
                local_id=local_id,
                human_email=email,
                human_subject_key=f"{self.source.tenant_id}:{user.externalId}",
            )
            return await tx.litellm_scimresource.create(data=data)

    async def update(self, row: LiteLLM_SCIMResource, change: SCIMUser | SCIMPatchOp) -> SCIMUser:
        from litellm.proxy.management_endpoints.scim import scim_v2
        from litellm.proxy.management_endpoints.scim.agent_provisioning import user_document

        if row.local_id is None:
            raise HTTPException(409, "The human provisioned record is incomplete")
        username: Final = (
            change.userName
            if isinstance(change, SCIMUser)
            else reduce(patched_username, change.Operations, row.user_name)
        )
        if isinstance(change, SCIMUser):
            await self.claim_email(row, human_email(change))
        else:
            validate_human_patch(change)
        async with self.client.tx() as tx:
            existing: Final = await tx.litellm_usertable.find_unique(where={"user_id": row.local_id})
        if existing is None:
            desired: Final = change if isinstance(change, SCIMUser) else user_document(row)
            created: Final = await scim_v2.create_user(
                user=desired.model_copy(update={"userName": row.local_id, "groups": None})
            )
            if created.id != row.local_id:
                raise HTTPException(409, "The human local identity changed during provisioning")
        if isinstance(change, SCIMPatchOp):
            async with self.client.tx() as tx:
                current: Final = await tx.litellm_usertable.find_unique(where={"user_id": row.local_id})
            if current is None:
                raise HTTPException(409, "The human local identity was removed during provisioning")
            preview, _ = scim_v2.apply_scim_user_patch(UserPolicy.model_validate(current.model_dump()), change)
            email: Final = TypeAdapter[str | None](str | None).validate_python(preview.get("user_email"))
            if email:
                await self.claim_email(row, email.casefold())
        result: Final = (
            await scim_v2.patch_user(user_id=row.local_id, patch_ops=change)
            if isinstance(change, SCIMPatchOp)
            else await scim_v2.update_user(user_id=row.local_id, user=change.model_copy(update={"groups": None}))
        )
        document: Final = result.model_copy(update={"id": row.id, "externalId": row.external_id, "userName": username})
        async with self.client.tx() as tx:
            await tx.litellm_scimresource.update(
                where={"id": row.id},
                data={
                    "document": Json(document.model_dump(by_alias=True, mode="json", exclude_none=True)),
                    "active": document.active,
                    "user_name": document.userName,
                },
            )
        return document

    async def claim_email(self, row: LiteLLM_SCIMResource, email: str) -> None:
        try:
            async with self.client.tx() as tx:
                await tx.litellm_scimresource.update(where={"id": row.id}, data={"human_email": email})
        except Exception as exc:
            if is_unique_violation(exc):
                raise HTTPException(409, "This email belongs to another provisioning record") from exc
            raise
