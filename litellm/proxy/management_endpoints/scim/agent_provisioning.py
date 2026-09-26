import re
from collections import deque
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Final, Literal

from fastapi import HTTPException
from prisma import Prisma
from prisma.models import LiteLLM_SCIMResource, LiteLLM_SCIMSource
from prisma.types import (
    LiteLLM_SCIMResourceUpdateInput,
    LiteLLM_SCIMResourceWhereUniqueInput,
)
from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import SCIMSourceRepository
from litellm.types.proxy.management_endpoints.scim_agent_provisioning import (
    SCIM_AGENT_USER_SCHEMA,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMGroup,
    SCIMMember,
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMUser,
    SCIMUserName,
)


@dataclass(frozen=True, slots=True)
class SCIMProvisioningFailure:
    status: int
    message: str


def reject(failure: SCIMProvisioningFailure) -> None:
    raise HTTPException(failure.status, failure.message)


async def source_for_auth(auth: object, client: PrismaClient) -> LiteLLM_SCIMSource | None:
    if not isinstance(auth, UserAPIKeyAuth) or not auth.token:
        return None
    source: Final = await SCIMSourceRepository(client, use_writer=True).table.find_unique(
        where={"key_hash": auth.token}
    )
    if source is not None and not source.enabled:
        reject(SCIMProvisioningFailure(403, "This provisioning source is disabled"))
    return source


def user_document(row: LiteLLM_SCIMResource) -> SCIMUser:
    return SCIMUser.model_validate(
        {**TypeAdapter(dict[str, object]).validate_python(row.document), "id": row.id, "active": row.active}
    )


def group_document(row: LiteLLM_SCIMResource) -> SCIMGroup:
    return SCIMGroup.model_validate(
        {
            **TypeAdapter(dict[str, object]).validate_python(row.document),
            "id": row.id,
            "members": [{"value": value} for value in row.member_ids],
        }
    )


async def remove_group_member(tx: Prisma, group: LiteLLM_SCIMResource, member_id: str) -> None:
    where: Final[LiteLLM_SCIMResourceWhereUniqueInput] = {"id": group.id}
    data: Final[LiteLLM_SCIMResourceUpdateInput] = {
        "member_ids": [member for member in group.member_ids if member != member_id]
    }
    await tx.litellm_scimresource.update(where=where, data=data)


def _user_changes(item: SCIMPatchOperation, current: SCIMUser) -> dict[str, object] | SCIMProvisioningFailure:
    allowed: Final = {
        "active": "active",
        "displayname": "displayName",
        "username": "userName",
        "name": "name",
        "emails": "emails",
    }
    if item.path is None:
        value: Final = item.value
        if item.op == "remove" or not isinstance(value, dict):
            return SCIMProvisioningFailure(400, "An object value or attribute path is required")
        changes: Final = TypeAdapter(dict[str, object]).validate_python(value)
        if any(key not in allowed.values() for key in changes):
            return SCIMProvisioningFailure(400, "Agent subject and parent identity are immutable")
        return changes
    name_fields: Final = {"name." + name.lower(): name for name in SCIMUserName.model_fields}
    name_field: Final = name_fields.get(item.path.lower())
    if name_field is not None:
        return {
            "name": {
                **(current.name.model_dump() if current.name else {}),
                name_field: None if item.op == "remove" else item.value,
            }
        }
    email_type: Final = re.fullmatch(r'emails\[type eq "([^"\r\n]+)"\]\.value', item.path, re.IGNORECASE)
    if email_type is not None:
        others: Final = tuple(email.model_dump() for email in current.emails or () if email.type != email_type[1])
        selected: Final = next(
            (email.model_dump() for email in current.emails or () if email.type == email_type[1]),
            {"type": email_type[1]},
        )
        return {"emails": others if item.op == "remove" else ({**selected, "value": item.value}, *others)}
    key: Final = allowed.get(item.path.lower())
    if key is None:
        return SCIMProvisioningFailure(400, "This attribute is immutable or unsupported for an agent-user")
    return {key: None if item.op == "remove" else item.value}


def _patch_user_operation(
    current: SCIMUser | SCIMProvisioningFailure, item: SCIMPatchOperation
) -> SCIMUser | SCIMProvisioningFailure:
    if isinstance(current, SCIMProvisioningFailure):
        return current
    changes: Final = _user_changes(item, current)
    if isinstance(changes, SCIMProvisioningFailure):
        return changes
    try:
        updated: Final = SCIMUser.model_validate({**current.model_dump(by_alias=True), **changes})
    except ValidationError:
        return SCIMProvisioningFailure(400, "Invalid agent-user attribute value")
    if not updated.userName:
        return SCIMProvisioningFailure(400, "userName is required")
    return updated


def apply_user_patch(user: SCIMUser, patch: SCIMPatchOp) -> SCIMUser | SCIMProvisioningFailure:
    return reduce(_patch_user_operation, patch.Operations, user)


def _patched_members(current: SCIMGroup, item: SCIMPatchOperation) -> tuple[SCIMMember, ...] | SCIMProvisioningFailure:
    path: Final = item.path or ""
    selected: Final = re.fullmatch(r'members\[value eq "([^"\r\n]+)"\]', path, re.IGNORECASE)
    if selected is not None and item.op == "remove":
        return tuple(member for member in current.members or () if member.value != selected[1])
    if path.lower() != "members":
        return SCIMProvisioningFailure(400, "Unsupported group PATCH attribute")
    try:
        incoming: Final = TypeAdapter(tuple[SCIMMember, ...]).validate_python(() if item.value is None else item.value)
    except ValidationError:
        return SCIMProvisioningFailure(400, "Invalid group members")
    if item.op == "replace":
        return incoming
    if item.op == "remove":
        removed: Final = frozenset(member.value for member in incoming)
        return tuple(member for member in current.members or () if member.value not in removed) if incoming else ()
    return tuple({member.value: member for member in (*tuple(current.members or ()), *incoming)}.values())


def _patch_group_operation(
    current: SCIMGroup | SCIMProvisioningFailure, item: SCIMPatchOperation
) -> SCIMGroup | SCIMProvisioningFailure:
    if isinstance(current, SCIMProvisioningFailure):
        return current
    if (item.path or "").lower() == "displayname" and item.op != "remove" and isinstance(item.value, str):
        return current.model_copy(update={"displayName": item.value})
    members: Final = _patched_members(current, item)
    if isinstance(members, SCIMProvisioningFailure):
        return members
    return current.model_copy(update={"members": list(members)})


def group_members_after_patch(group: SCIMGroup, patch: SCIMPatchOp) -> SCIMGroup | SCIMProvisioningFailure:
    return reduce(_patch_group_operation, patch.Operations, group)


def _identity_patch_children(value: object) -> tuple[object, ...] | Literal[True]:
    if isinstance(value, dict):
        fields: Final = TypeAdapter(dict[str, object]).validate_python(value)
        if any(
            key.lower().startswith(SCIM_AGENT_USER_SCHEMA.lower()) or key.lower() in ("agent_user", "identityparentid")
            for key in fields
        ):
            return True
        return tuple(fields.values())
    if isinstance(value, list):
        return TypeAdapter(tuple[object, ...]).validate_python(value)
    if isinstance(value, str) and value.lower().startswith(SCIM_AGENT_USER_SCHEMA.lower()):
        return True
    return ()


def patch_changes_identity(patch: SCIMPatchOp) -> bool:
    pending: Final = deque(  # mutable-ok: work queue avoids recursion on arbitrarily nested untrusted PATCH values
        chain.from_iterable((item.path, item.value) for item in patch.Operations)
    )
    while pending:
        match _identity_patch_children(pending.popleft()):
            case True:
                return True
            case children:
                pending.extend(children)
    return False
