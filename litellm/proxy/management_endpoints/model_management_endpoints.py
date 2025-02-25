"""
Allow proxy admin to add/update/delete models in the db

Currently most endpoints are in `proxy_server.py`, but those should  be moved here over time.

Endpoints here: 

model/{model_id}/update - PATCH endpoint for model update.
"""

#### MODEL MANAGEMENT ####

import json
import uuid
from typing import Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmUserRoles,
    PrismaCompatibleUpdateDBModel,
    ProxyErrorTypes,
    ProxyException,
    TeamModelAddRequest,
    UpdateTeamRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.management_endpoints.team_endpoints import (
    team_model_add,
    update_team,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.router import (
    Deployment,
    DeploymentTypedDict,
    LiteLLMParamsTypedDict,
    updateDeployment,
)
from litellm.utils import get_utc_datetime

router = APIRouter()


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
    )
    # update model name
    if updated_patch.model_name:
        merged_deployment_dict["model_name"] = updated_patch.model_name

    # update litellm params
    if updated_patch.litellm_params:
        # Encrypt any sensitive values
        encrypted_params = {
            k: encrypt_value_helper(v)
            for k, v in updated_patch.litellm_params.model_dump(
                exclude_none=True
            ).items()
        }

        merged_deployment_dict["litellm_params"].update(encrypted_params)  # type: ignore

    # update model info
    if updated_patch.model_info:
        if "model_info" not in merged_deployment_dict:
            merged_deployment_dict["model_info"] = {}
        merged_deployment_dict["model_info"].update(
            updated_patch.model_info.model_dump(exclude_none=True)
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
        prisma_compatible_model_dict["model_info"] = json.dumps(
            merged_deployment_dict["model_info"]
        )
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

        # Create update dictionary only for provided fields
        update_data = update_db_model(db_model=db_model, updated_patch=patch_data)

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
):
    # encrypt litellm params #
    _litellm_params_dict = model_params.litellm_params.dict(exclude_none=True)
    _orignal_litellm_model_name = model_params.litellm_params.model
    for k, v in _litellm_params_dict.items():
        encrypted_value = encrypt_value_helper(value=v)
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
    model_response = await prisma_client.db.litellm_proxymodeltable.create(
        data=_data  # type: ignore
    )
    return model_response


async def _add_team_model_to_db(
    model_params: Deployment,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
):
    """
    If 'team_id' is provided,

    - generate a unique 'model_name' for the model (e.g. 'model_name_{team_id}_{uuid})
    - store the model in the db with the unique 'model_name'
    - store a team model alias mapping {"model_name": "model_name_{team_id}_{uuid}"}
    """
    _team_id = model_params.model_info.team_id
    if _team_id is None:
        return None
    original_model_name = model_params.model_name
    if original_model_name:
        model_params.model_info.team_public_model_name = original_model_name

    unique_model_name = f"model_name_{_team_id}_{uuid.uuid4()}"

    model_params.model_name = unique_model_name

    ## CREATE MODEL IN DB ##
    model_response = await _add_model_to_db(
        model_params=model_params,
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )

    ## CREATE MODEL ALIAS IN DB ##
    await update_team(
        data=UpdateTeamRequest(
            team_id=_team_id,
            model_aliases={original_model_name: unique_model_name},
        ),
        user_api_key_dict=user_api_key_dict,
        http_request=Request(scope={"type": "http"}),
    )

    # add model to team object
    await team_model_add(
        data=TeamModelAddRequest(
            team_id=_team_id,
            models=[original_model_name],
        ),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=user_api_key_dict,
    )

    return model_response


def check_if_team_id_matches_key(
    team_id: Optional[str], user_api_key_dict: UserAPIKeyAuth
) -> bool:
    can_make_call = True
    if (
        user_api_key_dict.user_role
        and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
    ):
        return True
    if team_id is None:
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            can_make_call = False
    else:
        if user_api_key_dict.team_id != team_id:
            can_make_call = False
    return can_make_call
