import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce, wraps
from typing import Concatenate, Final, Literal, ParamSpec, TypeVar
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
from litellm.repositories.table_repositories import SCIMSourceRepository
from litellm.types.proxy.management_endpoints.scim_agent_provisioning import SCIM_AGENT_USER_SCHEMA
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMGroup,
    SCIMListResponse,
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


Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")


def serialized_source(
    operation: Callable[Concatenate["AgentProvisioningService", Parameters], Awaitable[Result]],
) -> Callable[Concatenate["AgentProvisioningService", Parameters], Awaitable[Result]]:
    @wraps(operation)
    async def execute(
        service: "AgentProvisioningService", *args: Parameters.args, **kwargs: Parameters.kwargs
    ) -> Result:
        async with service.client.tx(timeout=timedelta(seconds=30)) as tx:
            await tx.execute_raw(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", "scim-source:" + service.source.source_id
            )
            current: Final = await tx.litellm_scimsource.find_unique(where={"source_id": service.source.source_id})
            if current is None or not current.enabled or current.key_hash != service.source.key_hash:
                raise HTTPException(403, "This provisioning source is disabled or its token has changed")
            return await operation(service, *args, **kwargs)

    return execute


def patch_changes_identity(patch: SCIMPatchOp) -> bool:
    def marked(value: object) -> bool:
        if isinstance(value, dict):
            fields: Final = TypeAdapter(dict[str, object]).validate_python(value)
            return any(
                key.lower().startswith(SCIM_AGENT_USER_SCHEMA.lower())
                or key.lower() in ("agent_user", "identityparentid")
                or marked(item)
                for key, item in fields.items()
            )
        if isinstance(value, list):
            return any(marked(item) for item in TypeAdapter(list[object]).validate_python(value))
        return isinstance(value, str) and value.lower().startswith(SCIM_AGENT_USER_SCHEMA.lower())

    return any(marked(item.path) or marked(item.value) for item in patch.Operations)


class AgentProvisioningService:
    def __init__(self, client: PrismaClient, source: LiteLLM_SCIMSource):
        self.client = client
        self.source = source

    async def list(
        self, kind: Literal["Users", "Groups"], start: int, count: int, filter_value: str | None
    ) -> SCIMListResponse:
        from litellm.proxy.management_endpoints.scim.scim_v2 import parse_scim_eq_filter

        parsed: Final = parse_scim_eq_filter(filter_value) if filter_value else None
        fields: Final = {
            "username": "user_name",
            "externalid": "external_id",
            "displayname": "display_name",
            "id": "id",
        }
        if filter_value and (parsed is None or parsed[0] not in fields):
            reject(SCIMProvisioningFailure(400, "Unsupported SCIM filter"))
        filter_clause: Final[LiteLLM_SCIMResourceWhereInput] = (
            {"user_name": parsed[1]}
            if parsed and parsed[0] == "username"
            else {"external_id": parsed[1]}
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
        row: Final = await tx.litellm_scimresource.find_unique(where={"id": resource_id})
        if row is None or row.source_id != self.source.source_id or row.kind != kind or row.deleted:
            raise HTTPException(404, "SCIM resource not found in this provisioning source")
        return row

    @serialized_source
    async def create_user(self, user: SCIMUser) -> SCIMUser:
        if not user.externalId or not user.userName:
            raise HTTPException(400, "externalId and userName are required")
        if user.agent_user is None:
            return await self._create_human(user)
        try:
            oid: Final = str(UUID(user.externalId))
        except ValueError:
            raise HTTPException(400, "Agent externalId must be the Entra object ID")
        parent: Final = str(user.agent_user.identityParentId)
        tenant: Final = self.source.tenant_id
        issuer: Final = f"https://login.microsoftonline.com/{tenant}/v2.0"
        try:
            async with self.client.tx() as tx:
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
                    user_name=user.userName,
                    display_name=user.displayName or user.userName,
                    document=Json(document.model_dump(by_alias=True, mode="json", exclude_none=True)),
                    active=user.active,
                    local_id=agent_id,
                )
                await tx.litellm_scimresource.create(data=resource_data)
                agent_data: Final[LiteLLM_AgentsTableCreateInput] = LiteLLM_AgentsTableCreateInput(
                    agent_id=agent_id,
                    agent_name=user.displayName or user.userName,
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

    async def _create_human(self, user: SCIMUser) -> SCIMUser:
        from litellm.proxy.management_endpoints.scim.human_provisioning import SourceHumanProvisioner

        return await SourceHumanProvisioner(self.client, self.source).create(user)

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

    @serialized_source
    async def update_user(self, resource_id: str, user: SCIMUser | SCIMPatchOp) -> SCIMUser:
        return await self._update_user(resource_id, user)

    async def _update_user(self, resource_id: str, user: SCIMUser | SCIMPatchOp) -> SCIMUser:
        async with self.client.tx() as tx:
            row: Final = await self._resource(tx, "Users", resource_id)
            current: Final = user_document(row)
            if current.agent_user is not None:
                updated: Final = apply_user_patch(current, user) if isinstance(user, SCIMPatchOp) else user
                if isinstance(updated, SCIMProvisioningFailure):
                    raise HTTPException(updated.status, updated.message)
                return await self._update_native(tx, row, updated)
            if isinstance(user, SCIMUser) and (user.agent_user is not None or user.externalId != row.external_id):
                raise HTTPException(409, "A human subject cannot be rebound or converted into an agent-user")
            if isinstance(user, SCIMPatchOp) and patch_changes_identity(user):
                raise HTTPException(409, "A human cannot be converted into an agent-user")
        from litellm.proxy.management_endpoints.scim.human_provisioning import SourceHumanProvisioner

        return await SourceHumanProvisioner(self.client, self.source).update(row, user)

    @serialized_source
    async def delete(self, kind: Literal["Users", "Groups"], resource_id: str) -> None:
        from litellm.proxy.management_endpoints.scim import scim_v2

        async with self.client.tx() as tx:
            row: Final = await tx.litellm_scimresource.find_unique(where={"id": resource_id})
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
            await tx.litellm_scimresource.update(
                where={"id": row.id}, data={"active": False, "deleted": True, "member_ids": []}
            )

    @serialized_source
    async def create_group(self, group: SCIMGroup) -> SCIMGroup:
        if not group.externalId:
            raise HTTPException(400, "externalId is required for a directory group")
        async with self.client.tx() as tx:
            old: Final = await tx.litellm_scimresource.find_unique(
                where={
                    "source_id_kind_external_id": {
                        "source_id": self.source.source_id,
                        "kind": "Groups",
                        "external_id": group.externalId,
                    }
                }
            )
            if old is not None and old.deleted:
                raise HTTPException(409, "This directory group was deleted")
        if old is not None:
            return await self._update_group(old.id, group)
        members: Final = tuple(dict.fromkeys(member.value for member in group.members or ()))
        scim_id: Final = str(uuid4())
        document: Final = group.model_copy(update={"id": scim_id})
        async with self.client.tx() as tx:
            await self._validate_members(tx, members)
            resource_data: Final[LiteLLM_SCIMResourceCreateInput] = LiteLLM_SCIMResourceCreateInput(
                id=scim_id,
                source_id=self.source.source_id,
                kind="Groups",
                external_id=group.externalId,
                display_name=group.displayName,
                document=Json(document.model_dump(by_alias=True, mode="json", exclude_none=True)),
                member_ids=list(members),
            )
            row: Final = await tx.litellm_scimresource.create(data=resource_data)
        await self._sync_human_members(row)
        return group_document(row)

    async def _validate_members(self, tx: Prisma, members: tuple[str, ...]) -> None:
        if not members:
            return
        rows: Final = await tx.litellm_scimresource.find_many(
            where={"id": {"in": list(members)}, "source_id": self.source.source_id, "kind": "Users", "deleted": False}
        )
        if frozenset(row.id for row in rows) != frozenset(members):
            raise HTTPException(400, "Group members must exist in this provisioning source")

    @serialized_source
    async def update_group(self, resource_id: str, change: SCIMGroup | SCIMPatchOp) -> SCIMGroup:
        return await self._update_group(resource_id, change)

    async def _update_group(self, resource_id: str, change: SCIMGroup | SCIMPatchOp) -> SCIMGroup:
        async with self.client.tx() as tx:
            old: Final = await self._resource(tx, "Groups", resource_id)
            updated: Final = (
                group_members_after_patch(group_document(old), change) if isinstance(change, SCIMPatchOp) else change
            )
            if isinstance(updated, SCIMProvisioningFailure):
                raise HTTPException(updated.status, updated.message)
            if updated.externalId != old.external_id:
                raise HTTPException(409, "Directory group externalId is immutable")
            members: Final = tuple(dict.fromkeys(member.value for member in updated.members or ()))
            await self._validate_members(tx, members)
            count: Final = await tx.litellm_scimresource.update_many(
                where={"id": old.id, "updated_at": old.updated_at},
                data={
                    "display_name": updated.displayName,
                    "document": Json(updated.model_dump(by_alias=True, mode="json", exclude_none=True)),
                    "member_ids": list(members),
                },
            )
            if count != 1:
                raise HTTPException(409, "The group changed concurrently; retry")
            row: Final = await self._resource(tx, "Groups", resource_id)
        await self._sync_human_members(row)
        return group_document(row)

    async def _sync_human_members(self, group: LiteLLM_SCIMResource) -> None:
        from litellm.proxy.management_endpoints.scim import scim_v2

        async with self.client.tx() as tx:
            users: Final = await tx.litellm_scimresource.find_many(
                where={
                    "id": {"in": group.member_ids},
                    "source_id": self.source.source_id,
                    "kind": "Users",
                    "deleted": False,
                }
            )
        humans: Final = [
            SCIMMember(value=row.local_id)
            for row in users
            if row.local_id is not None and user_document(row).agent_user is None
        ]
        if not humans and group.local_id is None:
            return
        document: Final = SCIMGroup(
            schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
            id=group.local_id or group.id,
            externalId=group.external_id,
            displayName=group.display_name,
            members=humans,
        )
        async with self.client.tx() as tx:
            existing_team: Final = await tx.litellm_teamtable.find_unique(where={"team_id": group.local_id or group.id})
        result: Final = (
            await scim_v2.create_group(group=document)
            if existing_team is None
            else await scim_v2.update_group(group_id=existing_team.team_id, group=document)
        )
        if group.local_id is None:
            async with self.client.tx() as tx:
                await tx.litellm_scimresource.update(where={"id": group.id}, data={"local_id": result.id})
