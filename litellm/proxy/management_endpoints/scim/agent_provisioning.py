import re
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce, wraps
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Concatenate, Final, Literal, ParamSpec, TypeVar
from uuid import UUID, uuid4

from fastapi import HTTPException
from prisma import Json, Prisma
from prisma.models import LiteLLM_SCIMResource, LiteLLM_SCIMSource
from prisma.types import (
    LiteLLM_AgentsTableCreateInput,
    LiteLLM_SCIMResourceCreateInput,
    LiteLLM_SCIMResourceUpdateInput,
    LiteLLM_SCIMResourceWhereInput,
    LiteLLM_SCIMResourceWhereUniqueInput,
    LiteLLM_VerifiedSubjectCreateInput,
)
from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import PrismaClient
from litellm.repositories.base_repository import is_unique_violation
from litellm.repositories.chunked_in import count_in, find_many_in
from litellm.repositories.table_repositories import SCIMResourceRepository, SCIMSourceRepository
from litellm.types.proxy.management_endpoints.scim_agent_provisioning import (
    SCIM_AGENT_USER_SCHEMA,
    canonical_directory_id,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMGroup,
    SCIMListResponse,
    SCIMMember,
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMUser,
    SCIMUserName,
)

if TYPE_CHECKING:
    from litellm.proxy.management_endpoints.scim.scim_v2 import ProvisionedGroupWrite


@dataclass(frozen=True, slots=True)
class _GroupDatabase:
    db: Prisma


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
            "members": tuple(SCIMMember(value=value) for value in row.member_ids),
        }
    )


async def remove_group_member(tx: Prisma, group: LiteLLM_SCIMResource, member_id: str) -> None:
    where: Final[LiteLLM_SCIMResourceWhereUniqueInput] = {"id": group.id}
    data: Final[LiteLLM_SCIMResourceUpdateInput] = {
        "member_ids": [member for member in group.member_ids if member != member_id]
    }
    await tx.litellm_scimresource.update(where=where, data=data)


def _user_changes(item: SCIMPatchOperation, current: SCIMUser) -> dict[str, object] | SCIMProvisioningFailure:
    allowed: Final = MappingProxyType(
        {
            "active": "active",
            "displayname": "displayName",
            "username": "userName",
            "name": "name",
            "emails": "emails",
        }
    )
    if item.path is None:
        value: Final = item.value
        if item.op == "remove" or not isinstance(value, dict):
            return SCIMProvisioningFailure(400, "An object value or attribute path is required")
        changes: Final = TypeAdapter(dict[str, object]).validate_python(value)
        if any(key not in allowed.values() for key in changes):
            return SCIMProvisioningFailure(400, "Agent subject and parent identity are immutable")
        return changes
    name_fields: Final = MappingProxyType({"name." + name.lower(): name for name in SCIMUserName.model_fields})
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
    merged: Final = MappingProxyType({member.value: member for member in (*(current.members or ()), *incoming)})
    return tuple(merged.values())


def _patch_group_operation(
    current: SCIMGroup | SCIMProvisioningFailure, item: SCIMPatchOperation
) -> SCIMGroup | SCIMProvisioningFailure:
    if isinstance(current, SCIMProvisioningFailure):
        return current
    if item.path is None:
        return _replace_group_attributes(current, item)
    if item.path.lower() == "displayname" and item.op != "remove" and isinstance(item.value, str):
        return current.model_copy(update={"displayName": item.value})
    members: Final = _patched_members(current, item)
    if isinstance(members, SCIMProvisioningFailure):
        return members
    return current.model_copy(update={"members": list(members)})


def _replace_group_attributes(current: SCIMGroup, item: SCIMPatchOperation) -> SCIMGroup | SCIMProvisioningFailure:
    raw: Final[object] = item.value
    if item.op == "remove" or not isinstance(raw, dict):
        return SCIMProvisioningFailure(400, "An object value or attribute path is required")
    fields: Final = TypeAdapter(Mapping[str, object]).validate_python(raw)
    if any(key.lower() not in ("displayname", "members") for key in fields):
        return SCIMProvisioningFailure(400, "Unsupported group PATCH attribute")
    display_name: Final = next((value for key, value in fields.items() if key.lower() == "displayname"), None)
    if display_name is not None and (not isinstance(display_name, str) or not display_name):
        return SCIMProvisioningFailure(400, "Invalid group displayName")
    renamed: Final = current if display_name is None else current.model_copy(update={"displayName": display_name})
    member_values: Final = next((value for key, value in fields.items() if key.lower() == "members"), None)
    if member_values is None:
        return renamed
    return _patch_group_operation(renamed, SCIMPatchOperation(op=item.op, path="members", value=member_values))


def group_members_after_patch(group: SCIMGroup, patch: SCIMPatchOp) -> SCIMGroup | SCIMProvisioningFailure:
    return reduce(_patch_group_operation, patch.Operations, group)


Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")


def serialized_source(
    operation: Callable[Concatenate["AgentProvisioningService", Parameters], Awaitable[Result]],
) -> Callable[Concatenate["AgentProvisioningService", Parameters], Awaitable[Result]]:
    @wraps(operation)
    async def execute(
        service: "AgentProvisioningService", *args: Parameters.args, **kwargs: Parameters.kwargs
    ) -> Result:
        async with service.source_transaction():
            return await operation(service, *args, **kwargs)

    return execute


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


class AgentProvisioningService:
    def __init__(self, client: PrismaClient, source: LiteLLM_SCIMSource) -> None:
        self.client = client
        self.source = source

    @asynccontextmanager
    async def source_transaction(self) -> AsyncGenerator[Prisma]:
        async with self.client.tx(timeout=timedelta(seconds=30)) as tx:
            await tx.execute_raw(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "scim-source:" + self.source.source_id
            )
            current: Final = await tx.litellm_scimsource.find_unique(where={"source_id": self.source.source_id})
            if current is None or not current.enabled or current.key_hash != self.source.key_hash:
                raise HTTPException(403, "This provisioning source is disabled or its token has changed")
            yield tx

    async def list(
        self, kind: Literal["Users", "Groups"], start: int, count: int, filter_value: str | None
    ) -> SCIMListResponse:
        from litellm.proxy.management_endpoints.scim.scim_v2 import parse_scim_eq_filter

        parsed: Final = parse_scim_eq_filter(filter_value) if filter_value else None
        fields: Final = MappingProxyType(
            {
                "username": "user_name",
                "externalid": "external_id",
                "displayname": "display_name",
                "id": "id",
            }
        )
        if filter_value and (parsed is None or parsed[0] not in fields):
            reject(SCIMProvisioningFailure(400, "Unsupported SCIM filter"))
        filter_clause: Final[LiteLLM_SCIMResourceWhereInput] = (
            {"user_name": parsed[1]}
            if parsed and parsed[0] == "username"
            else {"external_id": canonical_directory_id(parsed[1])}
            if parsed and parsed[0] == "externalid"
            else {"display_name": parsed[1]}
            if parsed and parsed[0] == "displayname"
            else {"id": parsed[1]}
            if parsed
            else {}
        )
        where: Final[LiteLLM_SCIMResourceWhereInput] = {
            "source_id": self.source.source_id,
            "kind": kind,
            "deleted": False,
            **filter_clause,
        }
        async with self.client.tx() as tx:
            rows: Final = await tx.litellm_scimresource.find_many(
                where=where, skip=start - 1, take=min(count, 100), order={"id": "asc"}
            )
            total: Final = await tx.litellm_scimresource.count(where=where)
        return SCIMListResponse(
            totalResults=total,
            startIndex=start,
            itemsPerPage=len(rows),
            Resources=[user_document(row) for row in rows]
            if kind == "Users"
            else [group_document(row) for row in rows],
        )

    async def get(self, kind: Literal["Users", "Groups"], resource_id: str) -> SCIMUser | SCIMGroup:
        async with self.client.tx() as tx:
            row: Final = await self._resource(tx, kind, resource_id)
        return user_document(row) if kind == "Users" else group_document(row)

    async def _resource(self, tx: Prisma, kind: Literal["Users", "Groups"], resource_id: str) -> LiteLLM_SCIMResource:
        where: Final[LiteLLM_SCIMResourceWhereUniqueInput] = {"id": resource_id}
        row: Final = await tx.litellm_scimresource.find_unique(where=where)
        if row is None or row.source_id != self.source.source_id or row.kind != kind or row.deleted:
            raise HTTPException(404, "SCIM resource not found in this provisioning source")
        return row

    async def create_user(self, user: SCIMUser) -> SCIMUser:
        from litellm.proxy.management_endpoints.scim.human_provisioning import SourceHumanProvisioner

        async with self.source_transaction() as tx:
            if not user.externalId or not user.userName:
                raise HTTPException(400, "externalId and userName are required")
            if user.agent_user is not None:
                return await self._create_native(
                    tx,
                    user,
                    external_id=user.externalId,
                    user_name=user.userName,
                    parent=str(user.agent_user.identityParentId),
                )
            result: Final = await SourceHumanProvisioner(self.client, self.source).create_in_transaction(tx, user)
        await SourceHumanProvisioner.finish_update(result)
        return result.document

    async def _create_native(
        self, tx: Prisma, user: SCIMUser, *, external_id: str, user_name: str, parent: str
    ) -> SCIMUser:
        try:
            oid: Final = str(UUID(external_id))
        except ValueError:
            raise HTTPException(400, "Agent externalId must be the Entra object ID")
        tenant: Final = self.source.tenant_id
        issuer: Final = f"https://login.microsoftonline.com/{tenant}/v2.0"
        try:
            previous: Final = await tx.litellm_scimresource.find_unique(
                where={
                    "source_id_kind_external_id": {
                        "source_id": self.source.source_id,
                        "kind": "Users",
                        "external_id": oid,
                    }
                }
            )
            if previous is not None:
                if previous.deleted:
                    raise HTTPException(409, "This subject was deleted; automatic recreation is not permitted")
                return await self._update_native(tx, previous, user)
            registered: Final = await tx.litellm_agentidentity.find_unique(
                where={
                    "provider_tenant_id_client_id": {
                        "provider": "microsoft_entra",
                        "tenant_id": tenant,
                        "client_id": parent,
                    }
                }
            )
            if registered is not None:
                raise HTTPException(
                    409, "This parent identity is already registered; automatic adoption is not permitted"
                )
            agent_id: Final = str(uuid4())
            scim_id: Final = str(uuid4())
            document: Final = user.model_copy(update={"id": scim_id})
            resource_data: Final[LiteLLM_SCIMResourceCreateInput] = LiteLLM_SCIMResourceCreateInput(
                id=scim_id,
                source_id=self.source.source_id,
                kind="Users",
                external_id=oid,
                user_name=user_name,
                display_name=user.displayName or user_name,
                document=Json(document.model_dump(by_alias=True, mode="json", exclude_none=True)),
                active=user.active,
                local_id=agent_id,
            )
            await tx.litellm_scimresource.create(data=resource_data)
            agent_data: Final[LiteLLM_AgentsTableCreateInput] = LiteLLM_AgentsTableCreateInput(
                agent_id=agent_id,
                agent_name=user.displayName or user_name,
                agent_card_params=Json({}),
                identity_managed=True,
                enabled=False,
                execution_mode="autonomous",
                created_by="scim:" + self.source.source_id,
                updated_by="scim:" + self.source.source_id,
                identity={
                    "create": {
                        "provider": "microsoft_entra",
                        "tenant_id": tenant,
                        "issuer": issuer,
                        "client_id": parent,
                        "provisioning_source_id": self.source.source_id,
                        "revision": str(uuid4()),
                    }
                },
                retired_identities={
                    "create": {
                        "provider": "microsoft_entra",
                        "tenant_id": tenant,
                        "issuer": issuer,
                        "client_id": parent,
                    }
                },
            )
            await tx.litellm_agentstable.create(data=agent_data)
            subject_data: Final[LiteLLM_VerifiedSubjectCreateInput] = LiteLLM_VerifiedSubjectCreateInput(
                issuer=issuer,
                tenant_id=tenant,
                oid=oid,
                kind="agent_user",
                agent_id=agent_id,
                parent_client_id=parent,
                scim_resource_id=scim_id,
                verified_via="scim",
            )
            await tx.litellm_verifiedsubject.create(data=subject_data)
            return document
        except Exception as exc:
            if is_unique_violation(exc):
                raise HTTPException(409, "The subject, parent identity or agent name is already registered") from exc
            raise

    async def _update_native(self, tx: Prisma, row: LiteLLM_SCIMResource, user: SCIMUser) -> SCIMUser:
        old: Final = user_document(row)
        try:
            subject_id: Final = str(UUID(user.externalId or ""))
        except ValueError:
            raise HTTPException(409, "Agent subject and parent identity are immutable") from None
        if user.agent_user != old.agent_user or subject_id != row.external_id:
            raise HTTPException(409, "Agent subject and parent identity are immutable")
        if row.local_id is None or await tx.litellm_agentstable.find_unique(where={"agent_id": row.local_id}) is None:
            raise HTTPException(409, "The registered agent was deleted; automatic recreation is not permitted")
        if not user.userName:
            raise HTTPException(400, "userName is required")
        document: Final = user.model_copy(
            update={"id": row.id, "active": user.active if "active" in user.model_fields_set else row.active}
        )
        updated: Final = await tx.litellm_scimresource.update_many(
            where={"id": row.id, "updated_at": row.updated_at},
            data={
                "user_name": user.userName,
                "display_name": user.displayName or user.userName or row.display_name,
                "document": Json(document.model_dump(by_alias=True, mode="json", exclude_none=True)),
                "active": document.active,
            },
        )
        if updated != 1:
            raise HTTPException(409, "The provisioned subject changed concurrently; retry")
        return document

    async def update_user(self, resource_id: str, user: SCIMUser | SCIMPatchOp) -> SCIMUser:
        from litellm.proxy.management_endpoints.scim.human_provisioning import SourceHumanProvisioner

        async with self.source_transaction() as tx:
            row: Final = await self._resource(tx, "Users", resource_id)
            current: Final = user_document(row)
            if current.agent_user is not None:
                updated: Final = apply_user_patch(current, user) if isinstance(user, SCIMPatchOp) else user
                if isinstance(updated, SCIMProvisioningFailure):
                    raise HTTPException(updated.status, updated.message)
                return await self._update_native(tx, row, updated)
            if isinstance(user, SCIMUser) and (
                user.agent_user is not None
                or canonical_directory_id(user.externalId or "") != canonical_directory_id(row.external_id)
            ):
                raise HTTPException(409, "A human subject cannot be rebound or converted into an agent-user")
            if isinstance(user, SCIMPatchOp) and patch_changes_identity(user):
                raise HTTPException(409, "A human cannot be converted into an agent-user")
            result: Final = await SourceHumanProvisioner(self.client, self.source).update_in_transaction(tx, row, user)
        await SourceHumanProvisioner.finish_update(result)
        return result.document

    @serialized_source
    async def delete(self, kind: Literal["Users", "Groups"], resource_id: str) -> None:
        from litellm.proxy.management_endpoints.scim import scim_v2

        where: Final[LiteLLM_SCIMResourceWhereUniqueInput] = {"id": resource_id}
        async with self.client.tx() as tx:
            row: Final = await tx.litellm_scimresource.find_unique(where=where)
            if row is None or row.source_id != self.source.source_id or row.kind != kind:
                raise HTTPException(404, "SCIM resource not found in this provisioning source")
            if row.deleted:
                return
        if row.local_id is not None:
            try:
                if kind == "Groups":
                    await scim_v2.delete_group(group_id=row.local_id)
                elif user_document(row).agent_user is None:
                    await scim_v2.delete_user(user_id=row.local_id)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
        async with self.client.tx() as tx:
            if kind == "Users":
                memberships: Final[LiteLLM_SCIMResourceWhereInput] = {
                    "source_id": self.source.source_id,
                    "kind": "Groups",
                    "member_ids": {"has": row.id},
                }
                groups: Final = await tx.litellm_scimresource.find_many(where=memberships)
                for group in groups:
                    await remove_group_member(tx, group, row.id)
            retired: Final[LiteLLM_SCIMResourceUpdateInput] = {"active": False, "deleted": True, "member_ids": []}
            await tx.litellm_scimresource.update(where=where, data=retired)

    async def create_group(self, group: SCIMGroup) -> SCIMGroup:
        from litellm.proxy.management_endpoints.scim import scim_v2

        admin_group: Final = await scim_v2.provisioning_group_admin_role()
        async with self.source_transaction() as tx:
            document, result = await self._create_group(tx, group, admin_group)
        await scim_v2.finish_provisioned_group(result)
        return document

    async def _create_group(
        self, tx: Prisma, group: SCIMGroup, admin_group: str | None
    ) -> tuple[SCIMGroup, "ProvisionedGroupWrite | None"]:
        if not group.externalId:
            raise HTTPException(400, "externalId is required for a directory group")
        external_id: Final = canonical_directory_id(group.externalId)
        old: Final = await tx.litellm_scimresource.find_unique(
            where={
                "source_id_kind_external_id": {
                    "source_id": self.source.source_id,
                    "kind": "Groups",
                    "external_id": external_id,
                }
            }
        )
        if old is not None and old.deleted:
            raise HTTPException(409, "This directory group was deleted")
        if old is not None:
            return await self._update_group(tx, old.id, group, admin_group)
        members: Final = tuple(dict.fromkeys(member.value for member in group.members or ()))
        scim_id: Final = str(uuid4())
        document: Final = group.model_copy(update={"id": scim_id, "externalId": external_id})
        await self._validate_members(members, tx=tx)
        resource_data: Final[LiteLLM_SCIMResourceCreateInput] = LiteLLM_SCIMResourceCreateInput(
            id=scim_id,
            source_id=self.source.source_id,
            kind="Groups",
            external_id=external_id,
            display_name=group.displayName,
            document=Json(document.model_dump(by_alias=True, mode="json", exclude_none=True)),
            member_ids=list(members),
        )
        row: Final = await tx.litellm_scimresource.create(data=resource_data)
        result: Final = await self._sync_human_members(tx, row, admin_group)
        return group_document(row), result

    async def _validate_members(self, members: tuple[str, ...], *, tx: Prisma | None = None) -> None:
        if not members:
            return
        count: Final = await count_in(
            SCIMResourceRepository(self.client, use_writer=True).table
            if tx is None
            else SCIMResourceRepository(_GroupDatabase(tx)).table,
            "id",
            members,
            where=LiteLLM_SCIMResourceWhereInput(source_id=self.source.source_id, kind="Users", deleted=False),
        )
        if count != len(frozenset(members)):
            raise HTTPException(400, "Group members must exist in this provisioning source")

    async def update_group(self, resource_id: str, change: SCIMGroup | SCIMPatchOp) -> SCIMGroup:
        from litellm.proxy.management_endpoints.scim import scim_v2

        admin_group: Final = await scim_v2.provisioning_group_admin_role()
        async with self.source_transaction() as tx:
            document, result = await self._update_group(tx, resource_id, change, admin_group)
        await scim_v2.finish_provisioned_group(result)
        return document

    async def _update_group(
        self, tx: Prisma, resource_id: str, change: SCIMGroup | SCIMPatchOp, admin_group: str | None
    ) -> tuple[SCIMGroup, "ProvisionedGroupWrite | None"]:
        old: Final = await self._resource(tx, "Groups", resource_id)
        updated: Final = (
            group_members_after_patch(group_document(old), change) if isinstance(change, SCIMPatchOp) else change
        )
        if isinstance(updated, SCIMProvisioningFailure):
            raise HTTPException(updated.status, updated.message)
        if canonical_directory_id(updated.externalId or "") != canonical_directory_id(old.external_id):
            raise HTTPException(409, "Directory group externalId is immutable")
        members: Final = tuple(dict.fromkeys(member.value for member in updated.members or ()))
        await self._validate_members(members, tx=tx)
        data: Final = LiteLLM_SCIMResourceUpdateInput(
            display_name=updated.displayName,
            document=Json(updated.model_dump(by_alias=True, mode="json", exclude_none=True)),
            member_ids=list(members),
        )
        count: Final = await tx.litellm_scimresource.update_many(
            where=LiteLLM_SCIMResourceWhereInput(id=old.id, updated_at=old.updated_at), data=data
        )
        if count != 1:
            raise HTTPException(409, "The group changed concurrently; retry")
        row: Final = await self._resource(tx, "Groups", resource_id)
        result: Final = await self._sync_human_members(tx, row, admin_group)
        return group_document(row), result

    async def _sync_human_members(
        self, tx: Prisma, group: LiteLLM_SCIMResource, admin_group: str | None
    ) -> "ProvisionedGroupWrite | None":
        from litellm.proxy.management_endpoints.scim import scim_v2

        users: Final = await find_many_in(
            SCIMResourceRepository(_GroupDatabase(tx)).table,
            "id",
            group.member_ids,
            where={
                "source_id": self.source.source_id,
                "kind": "Users",
                "deleted": False,
            },
        )
        if any(row.local_id is None and user_document(row).agent_user is None for row in users):
            raise HTTPException(409, "The human provisioned record is incomplete")
        humans: Final = [
            SCIMMember(value=row.local_id)
            for row in users
            if row.local_id is not None and user_document(row).agent_user is None
        ]
        if not humans and group.local_id is None:
            return None
        document: Final = SCIMGroup(
            schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
            id=group.local_id or group.id,
            externalId=group.external_id,
            displayName=group.display_name,
            members=humans,
        )
        result: Final = await scim_v2.write_provisioned_group(tx, self.client, document, admin_group)
        if group.local_id is None:
            where: Final[LiteLLM_SCIMResourceWhereUniqueInput] = {"id": group.id}
            linked: Final[LiteLLM_SCIMResourceUpdateInput] = {"local_id": result.team_id}
            await tx.litellm_scimresource.update(where=where, data=linked)
        return result
