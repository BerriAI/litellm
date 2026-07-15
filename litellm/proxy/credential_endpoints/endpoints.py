"""
CRUD endpoints for storing reusable credentials.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import _get_masked_values
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from pydantic import ValidationError

from litellm.models.credentials import CredentialInfo
from litellm.proxy.credential_endpoints.access_decision import (
    OPAQUE_DENY_REASON,
    Allow,
    Deny,
    decide_credential_patch,
)
from litellm.proxy.management_endpoints.logging_exporter_access import (
    is_destination_visible,
    parse_credential_info,
)
from litellm.proxy.management_endpoints.logging_exporter_validation import (
    is_admin_gated_credential_info,
    validate_credential_access,
)
from litellm.proxy.utils import handle_exception_on_proxy, jsonify_object
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.utils import (
    CreateCredentialItem,
    CredentialItem,
    UpdateCredentialItem,
)

router = APIRouter()


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only the proxy admin can manage logging credentials"},
        )


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN


def _summarize_validation_error(ve: ValidationError) -> str:
    parts = (".".join(str(loc) for loc in err["loc"]) + ": " + err["msg"] for err in ve.errors())
    return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class CallerAdminScope:
    """The teams and orgs a caller administers, for destination visibility.

    ``team_ids`` are the teams the caller admins directly (role=admin in
    ``members_with_roles``) unioned with every team in an org the caller is
    org-admin of, since an org admin manages their org's teams even ones they
    aren't a direct member of. ``org_ids`` are the orgs the caller is org-admin
    of. The list endpoint matches a destination's ``access.teams`` against
    ``team_ids`` and ``access.orgs`` against ``org_ids``; the PATCH decider uses
    ``team_ids`` alone, since a team admin may only grant team ids.
    """

    team_ids: frozenset[str]
    org_ids: frozenset[str]


async def _caller_admin_scope(
    user_api_key_dict: UserAPIKeyAuth, prisma_client: "Optional[PrismaClient]"
) -> CallerAdminScope:
    """The teams and orgs the caller administers.

    Empty when the caller has no user_id, no DB connection, or admins nothing.
    Uses cached ``get_user_object`` / ``get_team_object`` plus one bounded query
    for org teams; the role match is done in Python because ``members_with_roles``
    is a JSON column. Best-effort: a lookup miss returns the empty scope, which
    denies visibility and reduces the PATCH decider to no-ops (the safe fallback).
    """
    if user_api_key_dict.user_id is None or prisma_client is None:
        return CallerAdminScope(frozenset(), frozenset())
    from litellm.proxy.auth.auth_checks import get_team_object, get_user_object
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    try:
        user_obj = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
        if user_obj is None:
            return CallerAdminScope(frozenset(), frozenset())

        # Direct team-admin grants: fetch the caller's teams concurrently, keep the
        # ones where they hold the admin role.
        own_team_ids = tuple(tid for tid in (getattr(user_obj, "teams", None) or []) if isinstance(tid, str))
        team_objs = await asyncio.gather(
            *(
                get_team_object(
                    team_id=tid,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
                for tid in own_team_ids
            )
        )
        team_admin_of = frozenset(
            tid
            for tid, team_obj in zip(own_team_ids, team_objs)
            if any(
                member.user_id == user_api_key_dict.user_id and member.role == "admin"
                for member in (team_obj.members_with_roles or [])
            )
        )

        # Org-admin grants: every team in any org the caller admins, even if the
        # caller isn't a direct member of that team.
        org_admin_of = frozenset(
            m.organization_id
            for m in (user_obj.organization_memberships or [])
            if m.organization_id and m.user_role == LitellmUserRoles.ORG_ADMIN.value
        )
        org_teams = (
            await prisma_client.db.litellm_teamtable.find_many(where={"organization_id": {"in": list(org_admin_of)}})
            if org_admin_of
            else []
        )
        org_grantable = frozenset(t.team_id for t in org_teams if t.team_id)

        return CallerAdminScope(team_admin_of | org_grantable, org_admin_of)
    except Exception:  # noqa: BLE001  # fail closed to an empty admin scope so a lookup error never widens access
        verbose_proxy_logger.exception("caller admin-scope lookup failed")
        return CallerAdminScope(frozenset(), frozenset())


async def _caller_grantable_team_ids(
    user_api_key_dict: UserAPIKeyAuth, prisma_client: "Optional[PrismaClient]"
) -> frozenset[str]:
    """Team ids the caller may add to / remove from a destination's ``access.teams``
    (the PATCH decider's grant scope)."""
    return (await _caller_admin_scope(user_api_key_dict, prisma_client)).team_ids


def _credential_in_memory(credential_name: str) -> Optional[CredentialItem]:
    return next(
        (cred for cred in litellm.credential_list if cred.credential_name == credential_name),
        None,
    )


async def _credential_for_admin_gate(credential_name: str, prisma_client: object) -> Optional[CredentialItem]:
    """Authoritative credential lookup for the admin gate on update/delete.

    The in-process ``litellm.credential_list`` can be stale: a credential created
    via the API on another horizontally-scaled instance, or before a restart,
    exists only in the DB. Gating on the in-memory copy alone would let a logging
    credential that isn't resident be updated/deleted without the proxy-admin
    check. Prefer the in-memory copy, fall back to the DB so the gate sees the
    real ``credential_info``.
    """
    existing = _credential_in_memory(credential_name)
    if existing is not None:
        return existing
    if prisma_client is None:
        return None
    try:
        return await CredentialsRepository(prisma_client).find_by_name(credential_name)
    except Exception:  # noqa: BLE001  # treat any lookup failure as credential-not-found
        return None


class CredentialHelperUtils:
    @staticmethod
    def encrypt_credential_values(
        credential: CredentialItem, new_encryption_key: Optional[str] = None
    ) -> CredentialItem:
        """Encrypt values in credential.credential_values and add to DB"""
        encrypted_credential_values = {}
        for key, value in (credential.credential_values or {}).items():
            encrypted_credential_values[key] = encrypt_value_helper(value, new_encryption_key)

        # Return a new object to avoid mutating the caller's credential, which
        # is kept in memory and should remain unencrypted.
        return CredentialItem(
            credential_name=credential.credential_name,
            credential_values=encrypted_credential_values,
            credential_info=credential.credential_info or {},
        )


@router.post(
    "/credentials",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def create_credential(
    request: Request,
    fastapi_response: Response,
    credential: CreateCredentialItem,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    Stores credential in DB.
    Reloads credentials in memory.
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client

    # POST stays proxy-admin only across the board: route gate was widened so
    # team-admins can PATCH access on existing logging destinations, but
    # creation of any credential (logging or provider) remains admin-only.
    _require_proxy_admin(user_api_key_dict)
    validate_credential_access(credential.credential_info)

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        if credential.model_id:
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail="LLM router not found. Please ensure you have a valid router instance.",
                )
            # get model from router
            model = llm_router.get_deployment(credential.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail="Model not found")
            credential_values = llm_router.get_deployment_credentials(credential.model_id)
            if credential_values is None:
                raise HTTPException(status_code=404, detail="Model not found")
            credential.credential_values = credential_values

        if credential.credential_values is None:
            raise HTTPException(
                status_code=400,
                detail="Credential values are required. Unable to infer credential values from model ID.",
            )
        processed_credential = CredentialItem(
            credential_name=credential.credential_name,
            credential_values=credential.credential_values,
            credential_info=credential.credential_info,
        )
        encrypted_credential = CredentialHelperUtils.encrypt_credential_values(processed_credential)
        credentials_dict = encrypted_credential.model_dump()
        credentials_dict_jsonified = jsonify_object(credentials_dict)
        await CredentialsRepository(prisma_client).create(
            data={
                **credentials_dict_jsonified,
                "created_by": user_api_key_dict.user_id,
                "updated_by": user_api_key_dict.user_id,
            }
        )

        ## ADD TO LITELLM ##
        CredentialAccessor.upsert_credentials([processed_credential])

        return {"success": True, "message": "Credential created successfully"}
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.get(
    "/credentials",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def get_credentials(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.

    Proxy admins see every credential (values masked). A non-proxy-admin sees
    only the logging destinations actually visible to a scope they administer:
    the same ``is_destination_visible`` predicate the assignment validator and
    the request-time resolver use, so the list can never show a destination a
    caller could neither assign nor route to. Provider credentials, and logging
    destinations scoped to other tenants, stay invisible. A caller who
    administers nothing gets 403 (Veria F2).
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if _is_proxy_admin(user_api_key_dict):
            visible = list(litellm.credential_list)
        else:
            scope = await _caller_admin_scope(user_api_key_dict, prisma_client)
            if not scope.team_ids and not scope.org_ids:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": (
                            "Listing logging destinations requires team-admin or "
                            "org-admin status. Ask your proxy admin to add you to a "
                            "team or org."
                        )
                    },
                )
            visible = [
                credential
                for credential in litellm.credential_list
                if (info := parse_credential_info(credential.credential_info)) is not None
                and info.credential_type == "logging"
                and is_destination_visible(info, scope.team_ids, scope.org_ids)
            ]
        masked_credentials = [
            {
                "credential_name": credential.credential_name,
                "credential_values": _get_masked_values(credential.credential_values),
                "credential_info": credential.credential_info,
            }
            for credential in visible
        ]
        return {"success": True, "credentials": masked_credentials}
    except HTTPException:
        raise
    except Exception as e:
        return handle_exception_on_proxy(e)


@router.get(
    "/credentials/by_name/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
async def get_credential_by_name(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    try:
        for credential in litellm.credential_list:
            if credential.credential_name == credential_name:
                masked_credential = CredentialItem(
                    credential_name=credential.credential_name,
                    credential_values=_get_masked_values(
                        credential.credential_values,
                        unmasked_length=4,
                        number_of_asterisks=4,
                    ),
                    credential_info=credential.credential_info,
                )
                return masked_credential
        raise HTTPException(
            status_code=404,
            detail="Credential not found. Got credential name: " + credential_name,
        )
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.get(
    "/credentials/by_model/{model_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
async def get_credential_by_model(
    request: Request,
    fastapi_response: Response,
    model_id: str = Path(..., description="The model ID to look up credentials for"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import llm_router

    try:
        if llm_router is None:
            raise HTTPException(status_code=500, detail="LLM router not found")
        model = llm_router.get_deployment(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        credential_values = llm_router.get_deployment_credentials(model_id)
        if credential_values is None:
            raise HTTPException(status_code=404, detail="Model not found")
        masked_credential_values = _get_masked_values(
            credential_values,
            unmasked_length=4,
            number_of_asterisks=4,
        )
        credential = CredentialItem(
            credential_name="{}-credential-{}".format(model.model_name, model_id),
            credential_values=masked_credential_values,
            credential_info={},
        )
        return credential
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.delete(
    "/credentials/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def delete_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    # DELETE stays proxy-admin only. The route gate lets team-admins reach
    # /credentials/{name} for PATCH; reject any DELETE that isn't proxy-admin.
    _require_proxy_admin(user_api_key_dict)

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        await CredentialsRepository(prisma_client).delete_by_name(credential_name)

        ## DELETE FROM LITELLM ##
        litellm.credential_list = [cred for cred in litellm.credential_list if cred.credential_name != credential_name]
        return {"success": True, "message": "Credential deleted successfully"}
    except Exception as e:
        return handle_exception_on_proxy(e)


def update_db_credential(
    db_credential: CredentialItem,
    updated_patch: CredentialItem,
    new_encryption_key: Optional[str] = None,
) -> CredentialItem:
    """
    Update a credential in the DB.
    """
    merged_credential = CredentialItem(
        credential_name=db_credential.credential_name,
        credential_info=db_credential.credential_info,
        credential_values=db_credential.credential_values,
    )

    encrypted_credential = CredentialHelperUtils.encrypt_credential_values(
        updated_patch,
        new_encryption_key,
    )
    # update model name
    if encrypted_credential.credential_name:
        merged_credential.credential_name = encrypted_credential.credential_name

    # update litellm params
    if encrypted_credential.credential_values:
        # Encrypt any sensitive values
        encrypted_params = {k: v for k, v in encrypted_credential.credential_values.items()}

        merged_credential.credential_values.update(encrypted_params)

    # Merge the patch into the existing credential_info so a partial update (e.g. only
    # access.teams) preserves credential_type/description/host AND the untouched
    # access subfields (global/orgs/other teams in access). See
    # _merge_credential_info for the surgical-access reasoning.
    if encrypted_credential.credential_info:
        if merged_credential.credential_info is None:
            merged_credential.credential_info = {}
        _merge_credential_info(merged_credential.credential_info, encrypted_credential.credential_info)

    return merged_credential


def _merge_credential_info(into: dict, patch: dict) -> None:
    """Merge ``patch`` into ``into`` in place, with surgical access subfields.

    A prior top-level dict.update let a patch like ``{access: {teams: [...]}}``
    replace the entire stored ``access`` object, wiping ``access.global=true``
    and ``access.orgs`` entries that the decider intentionally protected by
    refusing to allow them in the patch (Veria F1: scope tampering). Now
    ``access`` is merged subfield-by-subfield, so a non-admin patch carrying
    only ``access.teams`` keeps existing ``access.global`` / ``access.orgs``
    intact. The DB write and the in-memory cache sync both call this so the
    two stores can't drift.
    """
    patch_copy = dict(patch)
    patch_access = patch_copy.pop("access", None)
    into.update(patch_copy)
    if patch_access is None:
        return
    existing_access = into.get("access")
    if isinstance(existing_access, dict) and isinstance(patch_access, dict):
        existing_access.update(patch_access)
    else:
        into["access"] = patch_access


async def _authorize_credential_patch(
    *,
    credential_name: str,
    patch: UpdateCredentialItem,
    existing: Optional[CredentialItem],
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: "Optional[PrismaClient]",
) -> None:
    """Raise 403 unless the caller is allowed to apply ``patch`` to ``existing``.

    The decider widening only applies when the STORED credential is a logging
    destination -- a patch body alone can't promote a provider credential into
    the decider's allowed paths (Cursor BugBot bypass: ``is_admin_gated_credential_info``
    returned True for any patch carrying ``access``, so a team-admin could PATCH
    ``access.teams`` onto a provider credential and reach the decider).
    """
    existing_is_logging_gated = existing is not None and is_admin_gated_credential_info(existing.credential_info)
    if not existing_is_logging_gated:
        _require_proxy_admin(user_api_key_dict)
        return

    is_admin = _is_proxy_admin(user_api_key_dict)
    team_admin_ids = frozenset() if is_admin else await _caller_grantable_team_ids(user_api_key_dict, prisma_client)
    try:
        patch_info_typed = (
            CredentialInfo.model_validate(patch.credential_info) if patch.credential_info is not None else None
        )
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail={"error": _summarize_validation_error(ve)})
    assert existing is not None  # narrowed by existing_is_logging_gated
    try:
        existing_info_typed = CredentialInfo.model_validate(existing.credential_info)
    except ValidationError:
        # Stored info the strict access model can't parse (e.g. a legacy access key)
        # can't be run through the field-level decider. Fail closed: only the proxy
        # admin may patch such a row, instead of 500-ing every caller.
        if not is_admin:
            raise HTTPException(status_code=403, detail={"error": OPAQUE_DENY_REASON})
        return
    decision = decide_credential_patch(
        is_proxy_admin=is_admin,
        caller_team_admin_ids=team_admin_ids,
        existing_info=existing_info_typed,
        patch_info=patch_info_typed,
        patch_values=patch.credential_values,
        patch_name_changed=(patch.credential_name is not None and patch.credential_name != credential_name),
    )
    if isinstance(decision, Deny):
        reason = decision.reason if decision.from_user_input else OPAQUE_DENY_REASON
        raise HTTPException(status_code=403, detail={"error": reason})
    assert isinstance(decision, Allow)


def _patch_to_credential_item(patch: UpdateCredentialItem, credential_name: str) -> CredentialItem:
    """Translate the partial PATCH body into the legacy CredentialItem shape
    the downstream merge expects (non-None dicts)."""
    return CredentialItem(
        credential_name=patch.credential_name or credential_name,
        credential_values=patch.credential_values or {},
        credential_info=patch.credential_info or {},
    )


@router.patch(
    "/credentials/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def update_credential(
    request: Request,
    fastapi_response: Response,
    credential: UpdateCredentialItem,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.

    Both ``credential_values`` and ``credential_info`` are optional; a team-admin
    typically patches only ``credential_info.access`` to grant or revoke their
    own team. A proxy admin may patch either or both. See
    ``decide_credential_patch`` for the exact contract.
    """
    from litellm.proxy.proxy_server import prisma_client

    existing = await _credential_for_admin_gate(credential_name, prisma_client)
    await _authorize_credential_patch(
        credential_name=credential_name,
        patch=credential,
        existing=existing,
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )
    validate_credential_access(credential.credential_info)

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        credentials_repository = CredentialsRepository(prisma_client)
        db_credential = await credentials_repository.find_by_name(credential_name)
        if db_credential is None:
            raise HTTPException(status_code=404, detail="Credential not found in DB.")
        merged_credential = update_db_credential(db_credential, _patch_to_credential_item(credential, credential_name))
        credential_object_jsonified = jsonify_object(merged_credential.model_dump())
        await credentials_repository.update_by_name(
            credential_name,
            data={
                **credential_object_jsonified,
                "updated_by": user_api_key_dict.user_id,
            },
        )
        _sync_in_memory_credential(
            old_name=credential_name,
            merged=merged_credential,
            patch=credential,
        )
        return {"success": True, "message": "Credential updated successfully"}
    except Exception as e:
        return handle_exception_on_proxy(e)


def _sync_in_memory_credential(
    *,
    old_name: str,
    merged: CredentialItem,
    patch: UpdateCredentialItem,
) -> None:
    """Mirror the DB write into ``litellm.credential_list``.

    Skips when the credential isn't resident in memory (e.g. created on
    another scaled instance, restored from DB on the next reload). The
    in-memory ``credential_info`` is merged subfield-by-subfield via
    ``_merge_credential_info`` so a partial patch can't clobber stored
    ``access`` subfields it didn't touch.
    """
    existing_in_memory: Optional[CredentialItem] = None
    for cred in litellm.credential_list:
        if cred.credential_name == old_name:
            existing_in_memory = cred
            break
    if existing_in_memory is None:
        return

    in_memory_values = dict(existing_in_memory.credential_values or {})
    if patch.credential_values:
        in_memory_values.update(patch.credential_values)
    in_memory_info = dict(existing_in_memory.credential_info or {})
    if patch.credential_info:
        _merge_credential_info(in_memory_info, patch.credential_info)
    updated_in_memory = CredentialItem(
        credential_name=merged.credential_name,
        credential_values=in_memory_values,
        credential_info=in_memory_info,
    )
    if merged.credential_name != old_name:
        litellm.credential_list = [c for c in litellm.credential_list if c.credential_name != old_name]
    CredentialAccessor.upsert_credentials([updated_in_memory])
