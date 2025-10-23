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
from litellm._uuid import uuid
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from litellm._logging import verbose_proxy_logger
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
    UpdateTeamRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.management_endpoints.common_utils import _is_user_team_admin
from litellm.proxy.management_endpoints.team_endpoints import (
    team_model_add,
    update_team,
)
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    UpdateUsefulLinksRequest,
)
from litellm.types.router import (
    Deployment,
    DeploymentTypedDict,
    LiteLLMParamsTypedDict,
    updateDeployment,
)
from litellm.utils import get_utc_datetime

router = APIRouter()


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
        model_info = merged_deployment_dict["model_info"]
        for key, value in model_info.items():
            if isinstance(value, datetime.datetime):
                model_info[key] = value.isoformat()
        prisma_compatible_model_dict["model_info"] = json.dumps(model_info)

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
    _orignal_litellm_model_name = model_params.litellm_params.model
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
                public_model_name=model_params.model_name,
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
            if llm_router is not None:
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
    from litellm.proxy.proxy_server import (
        general_settings,
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

            _data: dict = {
                "litellm_params": json.dumps(merged_dictionary),  # type: ignore
                "updated_by": user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
            }
            model_response = await prisma_client.db.litellm_proxymodeltable.update(
                where={"model_id": _model_id},
                data=_data,  # type: ignore
            )

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

        litellm.public_model_groups = request.model_groups

        # Load existing config
        config = await proxy_config.get_config()

        # Update config with new settings
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}

        config["litellm_settings"]["public_model_groups"] = request.model_groups

        # Save the updated config
        await proxy_config.save_config(new_config=config)

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

        litellm.public_model_groups_links = request.useful_links

        # Load existing config
        config = await proxy_config.get_config()

        # Update config with new settings
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}

        config["litellm_settings"]["public_model_groups_links"] = request.useful_links

        # Save the updated config
        await proxy_config.save_config(new_config=config)

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
        
        verbose_proxy_logger.debug(f"Cleared {len(db_model_ids)} DB models, preserved {len(config_models)} config models")
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Failed to clear cache and reload models. Due to error - {str(e)}"
        )
