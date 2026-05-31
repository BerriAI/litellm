"""
Allow proxy admin to add/update/delete models in the db

Currently most endpoints are in `proxy_server.py`, but those should  be moved here over time.

Endpoints here:

model/{model_id}/update - PATCH endpoint for model update.
"""

#### MODEL MANAGEMENT ####

import asyncio
import datetime
import json
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_ProxyModelTable,
    LiteLLM_TeamTable,
    LitellmTableNames,
    LitellmUserRoles,
    ModelInfoDelete,
    PrismaCompatibleUpdateDBModel,
    ProxyErrorTypes,
    ProxyException,
    TeamModelAddRequest,
    TeamModelDeleteRequest,
    UserAPIKeyAuth,
)
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.management_endpoints.common_utils import _is_user_team_admin
from litellm.proxy.management_endpoints.team_endpoints import (
    team_model_add,
    team_model_delete,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    update_team as _legacy_update_team,
)
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    UpdateUsefulLinksRequest,
)
from litellm.types.router import (
    SPECIAL_MODEL_INFO_PARAMS,
    Deployment,
    DeploymentTypedDict,
    LiteLLMParamsTypedDict,
    updateDeployment,
)
from litellm.utils import get_utc_datetime

router = APIRouter()


# Credential fields that must never be silently re-pointed at a new endpoint,
# and the routing fields that change where they are sent.
_CREDENTIAL_LITELLM_PARAMS = (
    "api_key",
    "litellm_credential_name",
    "aws_secret_access_key",
    "aws_session_token",
    "vertex_credentials",
)
# Caller-controlled endpoint URLs that must be SSRF-validated on write.
_URL_LITELLM_PARAMS = (
    "api_base",
    "base_url",
    "aws_bedrock_runtime_endpoint",
    "aws_sts_endpoint",
)
# Destination fields whose change must drop an inherited credential (URLs above
# plus the provider selector, which isn't a URL so isn't SSRF-validated).
_DESTINATION_LITELLM_PARAMS = _URL_LITELLM_PARAMS + ("custom_llm_provider",)


def _field_explicitly_set(model: Optional[BaseModel], field: str) -> bool:
    """True if `field` was explicitly provided (to a value or null) on a model."""
    return model is not None and field in model.model_fields_set


def _validate_model_url_params(litellm_params: dict) -> None:
    """SSRF-guard api_base/base_url on stored model configs, mirroring the
    request-time guard in auth_utils.is_request_body_safe so the proxy applies
    one consistent policy. Gated on litellm.user_url_validation (default True)
    with user_url_allowed_hosts as the escape hatch for internal endpoints."""
    if not getattr(litellm, "user_url_validation", False):
        return
    for url_field in _URL_LITELLM_PARAMS:
        url_value = litellm_params.get(url_field)
        if not url_value or not isinstance(url_value, str):
            continue
        try:
            validate_url(url_value)
        except SSRFError as e:
            raise ProxyException(
                message=(
                    f"{url_field}={url_value!r} is rejected by the SSRF guard "
                    f"({e}). Add the host to general_settings.user_url_allowed_hosts "
                    "to allow it."
                ),
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param=url_field,
            )


def _decrypted_param(value: object) -> object:
    """Decrypt a stored litellm_param for comparison; non-string or
    non-encrypted values pass through unchanged."""
    return (
        decrypt_value_helper(value, "litellm_param", return_original_value=True)
        if isinstance(value, str)
        else value
    )


def _strip_credentials_on_destination_change(
    merged_litellm_params: dict,
    patch_plaintext: dict,
    db_plaintext: "Callable[[str], object]",
) -> None:
    """If the patch changes a destination field (api_base/base_url/custom_llm_provider)
    without supplying a fresh credential, drop the inherited secret(s) from the
    merged params so a stored credential is never silently re-pointed at a new
    (possibly attacker-controlled) endpoint."""
    destination_changed = any(
        field in patch_plaintext and patch_plaintext[field] != db_plaintext(field)
        for field in _DESTINATION_LITELLM_PARAMS
    )
    supplied_new_credential = any(
        field in patch_plaintext for field in _CREDENTIAL_LITELLM_PARAMS
    )
    if destination_changed and not supplied_new_credential:
        for field in _CREDENTIAL_LITELLM_PARAMS:
            merged_litellm_params.pop(field, None)


def _is_pricing_field(field: str) -> bool:
    """Custom per-token / per-second / per-pixel / per-request (and tiered) cost
    overrides all follow this naming, so match by convention rather than an
    enumerated subset that drifts out of date."""
    return "cost_per" in field or field.endswith("_cost")


def _contains_env_reference(value: object) -> bool:
    """True if any (possibly nested) string is an `os.environ/` reference, which
    resolves to a server-side environment secret at call time. Iterative (stack)
    rather than recursive, matching _reject_os_environ_references."""
    stack: List[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if item.startswith("os.environ/"):
                return True
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _assert_privileged_model_fields_authorized(
    litellm_params: Optional[BaseModel],
    model_info: Optional[BaseModel],
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """Restrict privileged model fields on non-admin (team-admin) writes.

    - Any custom pricing/cost field gates spend/budget enforcement, so a
      non-admin must not set or clear one.
    - `litellm_credential_name` resolves a globally-stored credential by name
      with no ownership model, so only a proxy admin may bind one.
    - `os.environ/` references resolve to the server's environment secrets
      (e.g. the operator's global provider key), so a non-admin must supply
      literal values, not references (mirrors /health/test_connection).
    """
    if is_proxy_admin(user_api_key_dict):
        return
    for model in (litellm_params, model_info):
        if model is None:
            continue
        for field, value in model.model_dump(exclude_unset=True).items():
            if _is_pricing_field(field):
                raise ProxyException(
                    message=f"Only proxy admins can set model pricing fields (e.g. {field}).",
                    type=ProxyErrorTypes.auth_error.value,
                    code=status.HTTP_403_FORBIDDEN,
                    param=field,
                )
            if _contains_env_reference(value):
                raise ProxyException(
                    message=(
                        f"os.environ/ references are not permitted in non-admin "
                        f"model parameters (field: {field}); supply a literal value."
                    ),
                    type=ProxyErrorTypes.auth_error.value,
                    code=status.HTTP_403_FORBIDDEN,
                    param=field,
                )
    if (
        _field_explicitly_set(litellm_params, "litellm_credential_name")
        and litellm_params.litellm_credential_name is not None  # type: ignore
    ):
        raise ProxyException(
            message="Only proxy admins can bind a stored credential (litellm_credential_name) to a model.",
            type=ProxyErrorTypes.auth_error.value,
            code=status.HTTP_403_FORBIDDEN,
            param="litellm_credential_name",
        )


def _assert_team_model_has_own_credential(
    model_params: Deployment, user_api_key_dict: UserAPIKeyAuth
) -> None:
    """A team-scoped model created by a non-admin must carry its own credential,
    so it cannot silently inherit the proxy's global provider keys at call time."""
    if is_proxy_admin(user_api_key_dict):
        return
    if model_params.model_info is None or model_params.model_info.team_id is None:
        return
    litellm_params = model_params.litellm_params
    has_credential = any(
        getattr(litellm_params, field, None)
        for field in ("api_key", "aws_secret_access_key", "vertex_credentials")
    )
    if not has_credential:
        raise ProxyException(
            message=(
                "Team-scoped models must provide their own credential (e.g. "
                "litellm_params.api_key) and cannot inherit the proxy's global keys."
            ),
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="litellm_params.api_key",
        )


async def update_team(*args, **kwargs):
    """
    Backward-compatible shim for tests/legacy call sites that patch this symbol.
    Team model management now uses team_model_add/team_model_delete directly.
    """
    return await _legacy_update_team(*args, **kwargs)


class UpdatePublicModelGroupsRequest(BaseModel):
    """Request model for updating public model groups"""

    model_groups: List[str] = Field(
        description="List of model group names to make public"
    )

    model_config = ConfigDict(extra="forbid")


async def get_db_model(
    model_id: str, prisma_client: PrismaClient
) -> Optional[Deployment]:
    db_model = cast(
        Optional[BaseModel],
        await prisma_client.db.litellm_proxymodeltable.find_unique(
            where={"model_id": model_id}
        ),
    )

    if not db_model:
        return None

    deployment_pydantic_obj = Deployment(**db_model.model_dump(exclude_none=True))
    return deployment_pydantic_obj


def update_db_model(
    db_model: Deployment, updated_patch: updateDeployment
) -> PrismaCompatibleUpdateDBModel:
    merged_deployment_dict = DeploymentTypedDict(
        model_name=db_model.model_name,
        litellm_params=LiteLLMParamsTypedDict(
            **db_model.litellm_params.model_dump(exclude_none=True)  # type: ignore
        ),
        model_info=db_model.model_info.model_dump(exclude_none=True),
    )
    # update model name
    if updated_patch.model_name:
        merged_deployment_dict["model_name"] = updated_patch.model_name

    # update litellm params
    patch_litellm_plaintext: dict = {}
    if updated_patch.litellm_params:
        patch_litellm_plaintext = updated_patch.litellm_params.model_dump(
            exclude_none=True
        )
        # SSRF-guard the patched destination (api_base/base_url).
        _validate_model_url_params(patch_litellm_plaintext)
        # Encrypt any sensitive values
        encrypted_params = {
            k: encrypt_value_helper(v) for k, v in patch_litellm_plaintext.items()
        }

        merged_deployment_dict["litellm_params"].update(encrypted_params)  # type: ignore

    # update model info
    if updated_patch.model_info:
        if "model_info" not in merged_deployment_dict:
            merged_deployment_dict["model_info"] = {}
        merged_deployment_dict["model_info"].update(
            updated_patch.model_info.model_dump(exclude_none=True)
        )

    # The deployment id is the immutable primary key; never let a patched (or
    # freshly-constructed, auto-id'd) model_info blob change it.
    if db_model.model_info is not None and db_model.model_info.id is not None:
        merged_deployment_dict.setdefault("model_info", {})["id"] = (  # type: ignore
            db_model.model_info.id
        )

    # Honor explicit-null clears LAST, after both merges, so a model_info blob the UI
    # passes through (which today re-sends the OLD pricing on every save) cannot
    # silently undo a litellm_params clear via .update().
    #
    # Restricted to SPECIAL_MODEL_INFO_PARAMS (input/output cost per token/character
    # and cache read/write costs) so this path cannot be used to null out privileged
    # model_info fields like team_id or access groups. SPECIAL_MODEL_INFO_PARAMS are
    # mirrored between litellm_params and model_info by Deployment.__init__, so the
    # clear propagates to both blobs.
    if updated_patch.litellm_params:
        for field in updated_patch.litellm_params.model_fields_set:
            if (
                field in SPECIAL_MODEL_INFO_PARAMS
                and getattr(updated_patch.litellm_params, field) is None
            ):
                merged_deployment_dict["litellm_params"].pop(field, None)  # type: ignore
                merged_deployment_dict.get("model_info", {}).pop(field, None)
    if updated_patch.model_info:
        for field in updated_patch.model_info.model_fields_set:
            if (
                field in SPECIAL_MODEL_INFO_PARAMS
                and getattr(updated_patch.model_info, field) is None
            ):
                merged_deployment_dict["model_info"].pop(field, None)  # type: ignore
                merged_deployment_dict.get("litellm_params", {}).pop(field, None)  # type: ignore

    # Refuse to silently re-point a stored credential at a new endpoint: if the
    # destination changed without a fresh credential, drop the inherited secret
    # so it must be re-entered rather than forwarded to the new destination.
    if updated_patch.litellm_params:
        _strip_credentials_on_destination_change(
            merged_deployment_dict["litellm_params"],  # type: ignore
            patch_litellm_plaintext,
            lambda field: _decrypted_param(
                getattr(db_model.litellm_params, field, None)
            ),
        )

    # convert to prisma compatible format

    prisma_compatible_model_dict = PrismaCompatibleUpdateDBModel()
    if "model_name" in merged_deployment_dict:
        prisma_compatible_model_dict["model_name"] = merged_deployment_dict[
            "model_name"
        ]

    if "litellm_params" in merged_deployment_dict:
        prisma_compatible_model_dict["litellm_params"] = json.dumps(
            merged_deployment_dict["litellm_params"]
        )

    if "model_info" in merged_deployment_dict:
        model_info = merged_deployment_dict["model_info"]
        for key, value in model_info.items():
            if isinstance(value, datetime.datetime):
                model_info[key] = value.isoformat()
        prisma_compatible_model_dict["model_info"] = json.dumps(model_info)

    if updated_patch.blocked is not None:
        prisma_compatible_model_dict["blocked"] = updated_patch.blocked

    return prisma_compatible_model_dict


@router.patch(
    "/model/{model_id}/update",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def patch_model(
    model_id: str,  # Get model_id from path parameter
    patch_data: updateDeployment,  # Create a specific schema for PATCH operations
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    PATCH Endpoint for partial model updates.

    Only updates the fields specified in the request while preserving other existing values.
    Follows proper PATCH semantics by only modifying provided fields.

    Args:
        model_id: The ID of the model to update
        patch_data: The fields to update and their new values
        user_api_key_dict: User authentication information

    Returns:
        Updated model information

    Raises:
        ProxyException: For various error conditions including authentication and database errors
    """
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        premium_user,
        prisma_client,
        store_model_in_db,
    )

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Verify model exists and is stored in DB
        if not store_model_in_db:
            raise ProxyException(
                message="Model updates only supported for DB-stored models",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param=None,
            )

        # Fetch existing model
        db_model = await get_db_model(model_id=model_id, prisma_client=prisma_client)

        if db_model is None:
            # Check if model exists in config but not DB
            if llm_router and llm_router.get_deployment(model_id=model_id) is not None:
                raise ProxyException(
                    message="Cannot edit config-based model. Store model in DB via /model/new first.",
                    type=ProxyErrorTypes.validation_error.value,
                    code=status.HTTP_400_BAD_REQUEST,
                    param=None,
                )
            raise ProxyException(
                message=f"Model {model_id} not found on proxy.",
                type=ProxyErrorTypes.not_found_error,
                code=status.HTTP_404_NOT_FOUND,
                param=None,
            )

        await ModelManagementAuthChecks.can_user_make_model_call(
            model_params=db_model,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            premium_user=premium_user,
        )

        # Pause/resume (`blocked`) is a proxy-admin-only privilege. Team admins
        # passed the auth check above for team-scoped models, but they must not
        # be able to unblock (or block) a model their proxy admin has paused.
        if (
            patch_data.blocked is not None
            and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        ):
            raise ProxyException(
                message="Only proxy admins can change a model's blocked flag.",
                type=ProxyErrorTypes.auth_error.value,
                code=status.HTTP_403_FORBIDDEN,
                param="blocked",
            )

        # Pricing overrides and credential binding are proxy-admin-only.
        _assert_privileged_model_fields_authorized(
            litellm_params=patch_data.litellm_params,
            model_info=patch_data.model_info,
            user_api_key_dict=user_api_key_dict,
        )

        # Handle team model updates with proper alias management
        update_data = await _update_team_model_in_db(
            db_model=db_model,
            patch_data=patch_data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )

        # Add metadata about update
        update_data["updated_by"] = (
            user_api_key_dict.user_id or litellm_proxy_admin_name
        )
        update_data["updated_at"] = cast(str, get_utc_datetime())

        # Perform partial update
        updated_model = await prisma_client.db.litellm_proxymodeltable.update(
            where={"model_id": model_id},
            data=update_data,
        )

        # Clear cache and reload models (uses config setting or defaults to preserving config models for DB updates)
        await clear_cache()

        ## CREATE AUDIT LOG ##
        asyncio.create_task(
            create_object_audit_log(
                object_id=model_id,
                action="updated",
                user_api_key_dict=user_api_key_dict,
                table_name=LitellmTableNames.PROXY_MODEL_TABLE_NAME,
                before_value=db_model.model_dump_json(exclude_none=True),
                after_value=updated_model.model_dump_json(exclude_none=True),
                litellm_changed_by=user_api_key_dict.user_id,
                litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
            )
        )

        return updated_model

    except Exception as e:
        verbose_proxy_logger.exception(f"Error in patch_model: {str(e)}")

        if isinstance(e, (HTTPException, ProxyException)):
            raise e

        raise ProxyException(
            message=f"Error updating model: {str(e)}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )


################################# Helper Functions #################################
####################################################################################
####################################################################################
####################################################################################


async def _add_model_to_db(
    model_params: Deployment,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    new_encryption_key: Optional[str] = None,
    should_create_model_in_db: bool = True,
) -> Optional[LiteLLM_ProxyModelTable]:
    # encrypt litellm params #
    _litellm_params_dict = model_params.litellm_params.dict(exclude_none=True)
    _original_litellm_model_name = model_params.litellm_params.model
    for k, v in _litellm_params_dict.items():
        encrypted_value = encrypt_value_helper(
            value=v, new_encryption_key=new_encryption_key
        )
        model_params.litellm_params[k] = encrypted_value
    _data: dict = {
        "model_id": model_params.model_info.id,
        "model_name": model_params.model_name,
        "litellm_params": model_params.litellm_params.model_dump_json(exclude_none=True),  # type: ignore
        "model_info": model_params.model_info.model_dump_json(  # type: ignore
            exclude_none=True
        ),
        "created_by": user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
        "updated_by": user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
    }
    if model_params.model_info.id is not None:
        _data["model_id"] = model_params.model_info.id
    if should_create_model_in_db:
        model_response = await prisma_client.db.litellm_proxymodeltable.create(
            data=_data  # type: ignore
        )
    else:
        model_response = LiteLLM_ProxyModelTable(**_data)
    return model_response


async def _add_team_model_to_db(
    model_params: Deployment,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> Optional[LiteLLM_ProxyModelTable]:
    """
    If 'team_id' is provided,

    - generate a unique 'model_name' for the model (e.g. 'model_name_{team_id}_{uuid})
    - store the model in the db with the unique 'model_name'
    - add the public model name to the team's allowed models list
    """
    _team_id = model_params.model_info.team_id
    if _team_id is None:
        return None

    # Capture the original public name FIRST, before any mutations
    original_model_name = model_params.model_name

    # Set team_public_model_name in model_info using the captured original_model_name
    # This must happen BEFORE mutating model_params.model_name so _add_model_to_db
    # serializes the correct team_public_model_name (not the internal UUID name)
    if original_model_name:
        model_params.model_info.team_public_model_name = original_model_name

    # Generate and assign unique internal model_name LAST
    # (after team_public_model_name is safely stored)
    unique_model_name = f"model_name_{_team_id}_{uuid.uuid4()}"
    model_params.model_name = unique_model_name

    ## CREATE MODEL IN DB ##
    model_response = await _add_model_to_db(
        model_params=model_params,
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )

    if original_model_name:
        await team_model_add(
            data=TeamModelAddRequest(
                team_id=_team_id,
                models=[original_model_name],
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=user_api_key_dict,
        )

    return model_response


async def _update_team_model_in_db(
    db_model: Deployment,
    patch_data: updateDeployment,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> PrismaCompatibleUpdateDBModel:
    """
    Handle team model updates with proper alias management.

    If patch_data contains a team_id:
    - Creates unique internal model_name and team alias
    - Adds model to team object
    - Preserves team_public_model_name for external reference
    """
    # Validate team_id if present in patch_data
    from litellm.proxy.proxy_server import premium_user

    await ModelManagementAuthChecks.allow_team_model_action(
        model_params=patch_data,
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
        premium_user=premium_user,
    )

    db_team_id = db_model.model_info.team_id if db_model.model_info else None
    explicit_patch_team_id = (
        patch_data.model_info.team_id if patch_data.model_info else None
    )

    # A team admin must not move a model to a different team.
    if (
        explicit_patch_team_id is not None
        and db_team_id is not None
        and explicit_patch_team_id != db_team_id
    ):
        raise ProxyException(
            message="Cannot reassign a model to a different team.",
            type=ProxyErrorTypes.auth_error.value,
            code=status.HTTP_403_FORBIDDEN,
            param="model_info.team_id",
        )

    # Inherit the persisted team_id when the patch omits it, so a team-scoped
    # model keeps its team scoping (and its internal model_name) instead of being
    # renamed into the global pool via a model_info-less PATCH.
    patch_team_id = (
        explicit_patch_team_id if explicit_patch_team_id is not None else db_team_id
    )

    # Genuinely non-team model: standard update.
    if patch_team_id is None:
        return update_db_model(db_model=db_model, updated_patch=patch_data)

    # Determine public model name
    public_model_name = _get_public_model_name(
        patch_data=patch_data,
        db_model=db_model,
    )

    # Ensure model_info exists and set team_public_model_name
    if patch_data.model_info is None:
        from litellm.types.router import ModelInfo

        patch_data.model_info = ModelInfo()
    patch_data.model_info.team_public_model_name = public_model_name

    # Check if team assignment is new or changed
    is_new_team_assignment = db_team_id != patch_team_id

    if is_new_team_assignment:
        await _setup_new_team_model_assignment(
            team_id=patch_team_id,
            public_model_name=public_model_name,
            patch_data=patch_data,
            user_api_key_dict=user_api_key_dict,
        )
    else:
        await _update_existing_team_model_assignment(
            team_id=patch_team_id,
            public_model_name=public_model_name,
            db_model=db_model,
            patch_data=patch_data,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )

    return update_db_model(db_model=db_model, updated_patch=patch_data)


def _get_public_model_name(
    patch_data: updateDeployment,
    db_model: Deployment,
) -> str:
    """Determine the public model name from patch or existing model."""
    if patch_data.model_name:
        return patch_data.model_name

    if db_model.model_info and db_model.model_info.team_public_model_name:
        return db_model.model_info.team_public_model_name

    return db_model.model_name


async def _setup_new_team_model_assignment(
    team_id: str,
    public_model_name: str,
    patch_data: updateDeployment,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """Set up a new team model with unique name and team membership."""
    unique_model_name = f"model_name_{team_id}_{uuid.uuid4()}"
    patch_data.model_name = unique_model_name

    await team_model_add(
        data=TeamModelAddRequest(
            team_id=team_id,
            models=[public_model_name],
        ),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=user_api_key_dict,
    )


async def _get_team_deployments(
    team_id: str, prisma_client: PrismaClient
) -> List[LiteLLM_ProxyModelTable]:
    """
    Fetch all deployments for a given team_id from the database.

    Centralizes team deployment queries to ensure consistent filtering and error handling.
    This is the established helper pattern for team deployment DB access in this module.

    Note: prisma-client-py 0.11.0 does not support JSON path filtering, so we filter
    by the model_name prefix (team models use "model_name_{team_id}_*") and confirm
    team_id in model_info with Python-side filtering.
    """
    prefix = f"model_name_{team_id}_"
    response = await prisma_client.db.litellm_proxymodeltable.find_many(
        where={
            "model_name": {"startswith": prefix},
        }
    )
    if not response:
        return []

    # Confirm team_id in model_info (defensive check)
    result = []
    for row in response:
        model_info = row.model_info
        if isinstance(model_info, str):
            try:
                model_info = json.loads(model_info)
            except (TypeError, ValueError):
                continue
        if isinstance(model_info, dict) and model_info.get("team_id") == team_id:
            result.append(row)
    return result


async def _update_existing_team_model_assignment(
    team_id: str,
    public_model_name: str,
    db_model: Deployment,
    patch_data: updateDeployment,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Optional[PrismaClient],
) -> None:
    """Update an existing team model if the public name changed.

    Note on DB scan: Prisma's JSON filtering does not support compound AND conditions
    across multiple JSON paths, so we fetch all deployments for the team and filter
    team_public_model_name in Python. For teams with many deployments this scan grows
    linearly; if team deployment counts become large this should be revisited.
    """

    def _get_team_public_model_name(
        model_info: Optional[Union[dict, str]],
    ) -> Optional[str]:
        if isinstance(model_info, dict):
            value = model_info.get("team_public_model_name")
            return value if isinstance(value, str) else None
        if isinstance(model_info, str):
            try:
                parsed = json.loads(model_info)
            except (TypeError, ValueError):
                return None
            if isinstance(parsed, dict):
                value = parsed.get("team_public_model_name")
                return value if isinstance(value, str) else None
        return None

    old_public_name = (
        db_model.model_info.team_public_model_name if db_model.model_info else None
    )

    if old_public_name and public_model_name != old_public_name:
        # Clear user-supplied public name from patch before any early return so the
        # caller does not overwrite the internal UUID-based model_name in the DB.
        patch_data.model_name = None
        if prisma_client is None:
            verbose_proxy_logger.warning(
                "prisma_client not initialized; skipping public name update entirely to avoid orphaned entries"
            )
            return

        # Query DB for all team deployments to check for sibling deployments
        team_deployments = await _get_team_deployments(team_id, prisma_client)
        other_deployments_with_old_name = [
            d
            for d in team_deployments
            if d.model_name != db_model.model_name
            and _get_team_public_model_name(d.model_info) == old_public_name
        ]

        # Add new name first, then delete old name to prevent access loss on partial failure
        await team_model_add(
            data=TeamModelAddRequest(
                team_id=team_id,
                models=[public_model_name],
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=user_api_key_dict,
        )

        if not other_deployments_with_old_name:
            await team_model_delete(
                data=TeamModelDeleteRequest(
                    team_id=team_id,
                    models=[old_public_name],
                ),
                http_request=Request(scope={"type": "http"}),
                user_api_key_dict=user_api_key_dict,
            )
    elif not old_public_name and public_model_name:
        # First-time assignment of public name on an existing team deployment:
        # ensure the team's models list is updated so team routing can resolve it.
        await team_model_add(
            data=TeamModelAddRequest(
                team_id=team_id,
                models=[public_model_name],
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=user_api_key_dict,
        )
    # else: old_public_name == public_model_name (no rename needed)
    # No team_model_add/delete calls required; public name is already registered

    # Always clear patch_data.model_name to prevent caller from overwriting
    # the internal UUID-based model_name in the DB with the user-supplied public name
    patch_data.model_name = None


class ModelManagementAuthChecks:
    """
    Common auth checks for model management endpoints
    """

    @staticmethod
    def can_user_make_team_model_call(
        team_id: str,
        user_api_key_dict: UserAPIKeyAuth,
        team_obj: Optional[LiteLLM_TeamTable] = None,
        premium_user: bool = False,
    ) -> Literal[True]:
        if premium_user is False:
            raise HTTPException(
                status_code=403,
                detail={"error": CommonProxyErrors.not_premium_user.value},
            )
        if (
            user_api_key_dict.user_role
            and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return True
        elif team_obj is None or not _is_user_team_admin(
            user_api_key_dict=user_api_key_dict, team_obj=team_obj
        ):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Team ID={} does not match the API key's team ID={}, OR you are not the admin for this team. Check `/user/info` to verify your team admin status.".format(
                        team_id, user_api_key_dict.team_id
                    )
                },
            )
        return True

    @staticmethod
    async def allow_team_model_action(
        model_params: Union[Deployment, updateDeployment],
        user_api_key_dict: UserAPIKeyAuth,
        prisma_client: PrismaClient,
        premium_user: bool,
    ) -> Literal[True]:
        if model_params.model_info is None or model_params.model_info.team_id is None:
            return True
        if model_params.model_info.team_id is not None and premium_user is not True:
            raise HTTPException(
                status_code=403,
                detail={"error": CommonProxyErrors.not_premium_user.value},
            )

        _existing_team_row = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": model_params.model_info.team_id}
        )

        if _existing_team_row is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Team id={} does not exist in db".format(
                        model_params.model_info.team_id
                    )
                },
            )
        existing_team_row = LiteLLM_TeamTable(**_existing_team_row.model_dump())

        ModelManagementAuthChecks.can_user_make_team_model_call(
            team_id=model_params.model_info.team_id,
            user_api_key_dict=user_api_key_dict,
            team_obj=existing_team_row,
            premium_user=premium_user,
        )
        return True

    @staticmethod
    async def can_user_make_model_call(
        model_params: Deployment,
        user_api_key_dict: UserAPIKeyAuth,
        prisma_client: PrismaClient,
        premium_user: bool,
    ) -> Literal[True]:
        ## Check team model auth
        if (
            model_params.model_info is not None
            and model_params.model_info.team_id is not None
        ):
            team_obj_row = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": model_params.model_info.team_id}
            )
            if team_obj_row is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Team id={} does not exist in db".format(
                            model_params.model_info.team_id
                        )
                    },
                )
            team_obj = LiteLLM_TeamTable(**team_obj_row.model_dump())

            return ModelManagementAuthChecks.can_user_make_team_model_call(
                team_id=model_params.model_info.team_id,
                user_api_key_dict=user_api_key_dict,
                team_obj=team_obj,
                premium_user=premium_user,
            )
        ## Check non-team model auth
        elif user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "User does not have permission to make this model call. Your role={}. You can only make model calls if you are a PROXY_ADMIN or if you are a team admin, by specifying a team_id in the model_info.".format(
                        user_api_key_dict.user_role
                    )
                },
            )
        else:
            return True

        return True


#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964
@router.post(
    "/model/delete",
    description="Allows deleting models in the model list in the config.yaml",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_model(
    model_info: ModelInfoDelete,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import llm_router

    try:
        """
        [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964

        - Check if id in db
        - Delete
        """

        from litellm.proxy.proxy_server import (
            llm_router,
            premium_user,
            prisma_client,
            store_model_in_db,
        )

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )

        model_in_db = await prisma_client.db.litellm_proxymodeltable.find_unique(
            where={"model_id": model_info.id}
        )
        if model_in_db is None:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Model with id={model_info.id} not found in db"},
            )

        model_params = Deployment(**model_in_db.model_dump())
        await ModelManagementAuthChecks.can_user_make_model_call(
            model_params=model_params,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            premium_user=premium_user,
        )

        # delete team model alias
        if model_params.model_info.team_id is not None:
            removed_model_aliases = await delete_team_model_alias(
                # The team alias is keyed on the PUBLIC model name; the internal
                # model_name is the unique UUID, which never matches an alias, so
                # using it here leaves the alias (and team.models entry) orphaned.
                public_model_name=(
                    model_params.model_info.team_public_model_name
                    or model_params.model_name
                ),
                prisma_client=prisma_client,
            )

            valid_team_model_aliases = [
                model
                for team_id, model in removed_model_aliases
                if team_id == model_params.model_info.team_id
            ]

            ## UPDATE TEAM TO NOT LIST MODEL ##
            existing_team_row = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": model_params.model_info.team_id}
            )
            if existing_team_row is not None:
                existing_team_row.models = [
                    model
                    for model in existing_team_row.models
                    if model not in valid_team_model_aliases
                ]

                await prisma_client.db.litellm_teamtable.update(
                    where={"team_id": model_params.model_info.team_id},
                    data={"models": existing_team_row.models},
                )

        # update DB
        if store_model_in_db is True:
            """
            - store model_list in db
            - store keys separately
            """
            # encrypt litellm params #
            result = await prisma_client.db.litellm_proxymodeltable.delete(
                where={"model_id": model_info.id}
            )

            if result is None:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"Model with id={model_info.id} not found in db"},
                )

            ## DELETE FROM ROUTER ##
            # Only evict a router deployment that originated from the DB. A
            # config (static) deployment must never be removed by deleting a DB
            # row, even if a row shares its id.
            if llm_router is not None:
                router_deployment = llm_router.get_deployment(model_id=model_info.id)
                if (
                    router_deployment is not None
                    and router_deployment.model_info.db_model
                ):
                    llm_router.delete_deployment(id=model_info.id)

            ## CREATE AUDIT LOG ##
            asyncio.create_task(
                create_object_audit_log(
                    object_id=model_info.id,
                    action="deleted",
                    user_api_key_dict=user_api_key_dict,
                    table_name=LitellmTableNames.PROXY_MODEL_TABLE_NAME,
                    before_value=result.model_dump_json(exclude_none=True),
                    after_value=None,
                    litellm_changed_by=user_api_key_dict.user_id,
                    litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
                )
            )
            return {"message": f"Model: {result.model_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

    except Exception as e:
        verbose_proxy_logger.exception(
            f"Failed to delete model. Due to error - {str(e)}"
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


async def delete_team_model_alias(
    public_model_name: str,
    prisma_client: PrismaClient,
) -> List[Tuple[str, str]]:
    """
    Delete a team model alias

    Iterate through all team model aliases and delete the one that matches the model_id

    Returns:
    - List of team id + model alias pairs that were removed
    """
    team_model_aliases = await prisma_client.db.litellm_modeltable.find_many(
        include={"team": True}
    )
    tasks = []
    removed_model_aliases = []
    for team_model_alias in team_model_aliases:
        model_aliases = team_model_alias.model_aliases  # {"alias": "public model name"}
        id = team_model_alias.id

        if public_model_name in model_aliases.values():
            key = list(model_aliases.keys())[
                list(model_aliases.values()).index(public_model_name)
            ]
            if team_model_alias.team is not None:
                removed_model_aliases.append((team_model_alias.team.team_id, key))
            del model_aliases[key]
            tasks.append(
                prisma_client.db.litellm_modeltable.update(
                    where={"id": id},
                    data={"model_aliases": json.dumps(model_aliases)},
                )
            )
    await asyncio.gather(*tasks)

    return removed_model_aliases


#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964
@router.post(
    "/model/new",
    description="Allows adding new models to the model list in the config.yaml",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def add_new_model(
    model_params: Deployment,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Add a new model to the proxy.

    Parameters:
    - model_name: str - The name users will use to call this model (required)
    - litellm_params: dict - LiteLLM-specific parameters (required)
        - model: str - The actual model identifier, e.g., "azure/my-deployment-name" (required - this is the only required field in litellm_params)
        - api_key: str - API key for the provider (optional)
        - api_base: str - API base URL (optional)
        - Other optional params: api_version, timeout, max_retries, etc.
    - model_info: dict - Additional model metadata returned in /v1/model/info (optional)

    Example curl:

    ```bash
    curl -L -X POST 'http://0.0.0.0:4000/model/new' \
    -H 'Authorization: Bearer LITELLM_VIRTUAL_KEY' \
    -H 'Content-Type: application/json' \
    -d '{
      "model_name": "my-azure-model",
      "litellm_params": {
        "model": "azure/my-deployment-name",
        "api_key": "my-azure-api-key",
        "api_base": "https://my-endpoint.openai.azure.com"
      },
      "model_info": {
        "my_custom_key": "my_custom_value"
      }
    }'
    ```

    Returns:
    - The created model entry with model_id
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        premium_user,
        prisma_client,
        proxy_config,
        proxy_logging_obj,
        store_model_in_db,
    )

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )

        ## Auth check
        await ModelManagementAuthChecks.can_user_make_model_call(
            model_params=model_params,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            premium_user=premium_user,
        )

        # Pricing overrides and credential binding are proxy-admin-only.
        _assert_privileged_model_fields_authorized(
            litellm_params=model_params.litellm_params,
            model_info=model_params.model_info,
            user_api_key_dict=user_api_key_dict,
        )
        # A non-admin team model must carry its own credential (no global-key fallback).
        _assert_team_model_has_own_credential(model_params, user_api_key_dict)
        # SSRF-guard the destination on create.
        _validate_model_url_params(
            model_params.litellm_params.model_dump(exclude_none=True)
        )
        # Reject a model_info.id that collides with a live deployment, so a DB
        # row cannot hijack (and on delete evict) a config/router deployment's id.
        if (
            model_params.model_info.id is not None
            and llm_router is not None
            and llm_router.has_model_id(model_params.model_info.id)
        ):
            raise ProxyException(
                message=(
                    f"model_info.id={model_params.model_info.id} already belongs to an "
                    "existing deployment; omit model_info.id to auto-generate one."
                ),
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="model_info.id",
            )

        model_response: Optional[LiteLLM_ProxyModelTable] = None
        # update DB
        if store_model_in_db is True:
            """
            - store model_list in db
            - store keys separately
            """

            try:
                _original_litellm_model_name = model_params.model_name
                if model_params.model_info.team_id is None:
                    model_response = await _add_model_to_db(
                        model_params=model_params,
                        user_api_key_dict=user_api_key_dict,
                        prisma_client=prisma_client,
                    )
                else:
                    model_response = await _add_team_model_to_db(
                        model_params=model_params,
                        user_api_key_dict=user_api_key_dict,
                        prisma_client=prisma_client,
                    )
                await proxy_config.add_deployment(
                    prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
                )
                # don't let failed slack alert block the /model/new response
                _alerting = general_settings.get("alerting", []) or []
                if "slack" in _alerting:
                    # send notification - new model added
                    await proxy_logging_obj.slack_alerting_instance.model_added_alert(
                        model_name=model_params.model_name,
                        litellm_model_name=_original_litellm_model_name,
                        passed_model_info=model_params.model_info,
                    )
            except Exception as e:
                verbose_proxy_logger.exception(f"Exception in add_new_model: {e}")

        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

        if model_response is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to add model to db. Check your server logs for more details."
                },
            )

        ## CREATE AUDIT LOG ##
        asyncio.create_task(
            create_object_audit_log(
                object_id=model_response.model_id,
                action="created",
                user_api_key_dict=user_api_key_dict,
                table_name=LitellmTableNames.PROXY_MODEL_TABLE_NAME,
                before_value=None,
                after_value=(
                    model_response.model_dump_json(exclude_none=True)
                    if isinstance(model_response, BaseModel)
                    else None
                ),
                litellm_changed_by=user_api_key_dict.user_id,
                litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
            )
        )

        return model_response

    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.add_new_model(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


#### MODEL MANAGEMENT ####
@router.post(
    "/model/update",
    description="Edit existing model params",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_model(
    model_params: updateDeployment,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Old endpoint for model update. Makes a PUT request.

    Use `/model/{model_id}/update` to PATCH the stored model in db.
    """
    from litellm.proxy.proxy_server import (
        LITELLM_PROXY_ADMIN_NAME,
        llm_router,
        premium_user,
        prisma_client,
        store_model_in_db,
    )

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )

        _model_id = None
        _model_info = getattr(model_params, "model_info", None)
        if _model_info is None:
            raise Exception("model_info not provided")

        _model_id = _model_info.id
        if _model_id is None:
            raise Exception("model_info.id not provided")

        _existing_litellm_params = (
            await prisma_client.db.litellm_proxymodeltable.find_unique(
                where={"model_id": _model_id}
            )
        )

        if _existing_litellm_params is None:
            if (
                llm_router is not None
                and llm_router.get_deployment(model_id=_model_id) is not None
            ):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Can't edit model. Model in config. Store model in db via `/model/new`. to edit."
                    },
                )
            else:
                raise Exception("model not found")
        deployment = Deployment(**_existing_litellm_params.model_dump())

        await ModelManagementAuthChecks.can_user_make_model_call(
            model_params=deployment,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            premium_user=premium_user,
        )

        # Pricing/credential field authz, parity with patch_model / add_new_model.
        _assert_privileged_model_fields_authorized(
            litellm_params=model_params.litellm_params,
            model_info=model_params.model_info,
            user_api_key_dict=user_api_key_dict,
        )

        # update DB
        if store_model_in_db is True:
            _existing_litellm_params_dict = dict(
                _existing_litellm_params.litellm_params
            )

            if model_params.litellm_params is None:
                raise Exception("litellm_params not provided")

            _new_litellm_params_dict = model_params.litellm_params.dict(
                exclude_none=True
            )
            # SSRF-guard the patched destination on this legacy update path too.
            _validate_model_url_params(_new_litellm_params_dict)

            ### ENCRYPT PARAMS ###
            for k, v in _new_litellm_params_dict.items():
                encrypted_value = encrypt_value_helper(value=v)
                model_params.litellm_params[k] = encrypted_value

            ### MERGE WITH EXISTING DATA ###
            merged_dictionary = {}
            _mp = model_params.litellm_params.dict()

            for key, value in _mp.items():
                if value is not None:
                    merged_dictionary[key] = value
                elif (
                    key in _existing_litellm_params_dict
                    and _existing_litellm_params_dict[key] is not None
                ):
                    merged_dictionary[key] = _existing_litellm_params_dict[key]
                else:
                    pass

            # Don't silently re-point an inherited credential at a new endpoint.
            _strip_credentials_on_destination_change(
                merged_dictionary,
                _new_litellm_params_dict,
                lambda field: _decrypted_param(
                    _existing_litellm_params_dict.get(field)
                ),
            )

            _data: dict = {
                "litellm_params": json.dumps(merged_dictionary),  # type: ignore
                "updated_by": user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
            }
            model_response = await prisma_client.db.litellm_proxymodeltable.update(
                where={"model_id": _model_id},
                data=_data,  # type: ignore
            )

            # Clear cache and reload models (uses config setting or defaults to preserving config models for DB updates)
            await clear_cache()

            ## CREATE AUDIT LOG ##
            asyncio.create_task(
                create_object_audit_log(
                    object_id=_model_id,
                    action="updated",
                    user_api_key_dict=user_api_key_dict,
                    table_name=LitellmTableNames.PROXY_MODEL_TABLE_NAME,
                    before_value=(
                        _existing_litellm_params.model_dump_json(exclude_none=True)
                        if isinstance(_existing_litellm_params, BaseModel)
                        else None
                    ),
                    after_value=(
                        model_response.model_dump_json(exclude_none=True)
                        if isinstance(model_response, BaseModel)
                        else None
                    ),
                    litellm_changed_by=user_api_key_dict.user_id,
                    litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
                )
            )

            return model_response
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.update_model(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/model_group/make_public",
    description="Update which model groups are public",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_public_model_groups(
    request: UpdatePublicModelGroupsRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update which model groups are public.

    This endpoint allows admins to specify which model groups should be publicly accessible.
    Public model groups are visible via the /public/model_hub endpoint.

    Args:
        request: Request containing list of model group names to make public
        user_api_key_dict: User authentication information

    Returns:
        Success message with updated public model groups

    Raises:
        ProxyException: For various error conditions including authentication errors
    """
    try:
        # Update the public model groups
        import litellm
        from litellm.proxy.proxy_server import proxy_config, store_model_in_db

        # Check if user has admin permissions
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only proxy admins can update public model groups. Your role={}".format(
                        user_api_key_dict.user_role
                    )
                },
            )

        # Check if STORE_MODEL_IN_DB is enabled
        if store_model_in_db is not True:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

        # Load existing config first (this may overwrite in-memory litellm settings
        # from DB values via _update_config_from_db), so set the in-memory value AFTER
        config = await proxy_config.get_config()

        # Update config with new settings
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}

        config["litellm_settings"]["public_model_groups"] = request.model_groups

        # Save the updated config
        await proxy_config.save_config(new_config=config)

        # Set in-memory value AFTER get_config() and save_config() to avoid
        # get_config() overwriting with stale DB value
        litellm.public_model_groups = request.model_groups

        verbose_proxy_logger.debug(
            f"Updated public model groups to: {request.model_groups} by user: {user_api_key_dict.user_id}"
        )

        return {
            "message": "Successfully updated public model groups",
            "public_model_groups": request.model_groups,
            "updated_by": user_api_key_dict.user_id,
        }

    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating public model groups: {str(e)}")

        if isinstance(e, HTTPException):
            raise e

        raise ProxyException(
            message=f"Error updating public model groups: {str(e)}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )


@router.post(
    "/model_hub/update_useful_links",
    description="Update useful links",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_useful_links(
    request: UpdateUsefulLinksRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update useful links.
    """
    try:
        # Update the public model groups
        import litellm
        from litellm.proxy.proxy_server import proxy_config

        # Check if user has admin permissions
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only proxy admins can update public model groups. Your role={}".format(
                        user_api_key_dict.user_role
                    )
                },
            )

        # Load existing config first (this may overwrite in-memory litellm settings
        # from DB values via _update_config_from_db), so set the in-memory value AFTER
        config = await proxy_config.get_config()

        # Update config with new settings
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}

        config["litellm_settings"]["public_model_groups_links"] = request.useful_links

        # Save the updated config
        await proxy_config.save_config(new_config=config)

        # Set in-memory value AFTER get_config() and save_config() to avoid
        # get_config() overwriting with stale DB value
        litellm.public_model_groups_links = request.useful_links

        verbose_proxy_logger.debug(
            f"Updated useful links to: {request.useful_links} by user: {user_api_key_dict.user_id}"
        )

        return {
            "message": "Successfully updated useful links",
            "useful_links": request.useful_links,
            "updated_by": user_api_key_dict.user_id,
        }

    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating public model groups: {str(e)}")

        if isinstance(e, HTTPException):
            raise e

        raise ProxyException(
            message=f"Error updating public model groups: {str(e)}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )


def _deduplicate_litellm_router_models(models: List[Dict]) -> List[Dict]:
    """
    Deduplicate models based on their model_info.id field.
    Returns a list of unique models keeping only the first occurrence of each model ID.

    Args:
        models: List of model dictionaries containing model_info

    Returns:
        List of deduplicated model dictionaries
    """
    seen_ids = set()
    unique_models = []
    for model in models:
        model_id = model.get("model_info", {}).get("id", None)
        if model_id is not None and model_id not in seen_ids:
            unique_models.append(model)
            seen_ids.add(model_id)
    return unique_models


async def clear_cache():
    """
    Clear router caches and reload models.
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        prisma_client,
        proxy_config,
        proxy_logging_obj,
        verbose_proxy_logger,
    )

    if llm_router is None or prisma_client is None:
        verbose_proxy_logger.debug(
            "llm_router or prisma_client is None, skipping cache clear"
        )
        return

    try:
        # Only clear DB models, preserve config models
        verbose_proxy_logger.debug("Clearing only DB models, preserving config models")

        # Get current models and filter out DB models
        current_models = llm_router.model_list.copy()
        config_models = []
        db_model_ids = []

        for model in current_models:
            model_info = model.get("model_info", {})
            if model_info.get("db_model", False):
                # This is a DB model, mark for deletion
                db_model_ids.append(model_info.get("id"))
            else:
                # This is a config model, preserve it
                config_models.append(model)

        # Clear only DB models
        for model_id in db_model_ids:
            llm_router.delete_deployment(id=model_id)

        # Clear auto routers
        llm_router.auto_routers.clear()

        # Reload only DB models
        await proxy_config.add_deployment(
            prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
        )

        verbose_proxy_logger.debug(
            f"Cleared {len(db_model_ids)} DB models, preserved {len(config_models)} config models"
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Failed to clear cache and reload models. Due to error - {str(e)}"
        )
