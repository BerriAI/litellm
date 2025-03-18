"""
Allow proxy admin to add/update/delete models in the db

Currently most endpoints are in `proxy_server.py`, but those should  be moved here over time.

Endpoints here: 

model/{model_id}/update - PATCH endpoint for model update.
"""

#### MODEL MANAGEMENT ####

import asyncio
import json
import uuid
from typing import Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_ProxyModelTable,
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
from litellm.proxy.management_endpoints.team_endpoints import (
    team_model_add,
    update_team,
)
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
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

        if model_params.model_info.team_id is not None and premium_user is not True:
            raise HTTPException(
                status_code=403,
                detail={"error": CommonProxyErrors.not_premium_user.value},
            )

        if not check_if_team_id_matches_key(
            team_id=model_params.model_info.team_id, user_api_key_dict=user_api_key_dict
        ):
            raise HTTPException(
                status_code=403,
                detail={"error": "Team ID does not match the API key's team ID"},
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
        # update DB
        if store_model_in_db is True:
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
                raise Exception("model not found")
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
