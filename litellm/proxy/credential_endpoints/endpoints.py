"""
CRUD endpoints for storing reusable credentials.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import _get_masked_values
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.utils import handle_exception_on_proxy, jsonify_object
from litellm.types.utils import CreateCredentialItem, CredentialItem

router = APIRouter()


class CredentialHelperUtils:
    @staticmethod
    def encrypt_credential_values(credential: CredentialItem) -> CredentialItem:
        """Encrypt values in credential.credential_values and add to DB"""
        encrypted_credential_values = {}
        for key, value in credential.credential_values.items():
            encrypted_credential_values[key] = encrypt_value_helper(value)
        credential.credential_values = encrypted_credential_values
        return credential


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
            credential_values = llm_router.get_deployment_credentials(
                credential.model_id
            )
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
        encrypted_credential = CredentialHelperUtils.encrypt_credential_values(
            processed_credential
        )
        credentials_dict = encrypted_credential.model_dump()
        credentials_dict_jsonified = jsonify_object(credentials_dict)
        await prisma_client.db.litellm_credentialstable.create(
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
    """
    try:
        masked_credentials = [
            {
                "credential_name": credential.credential_name,
                "credential_values": _get_masked_values(credential.credential_values),
                "credential_info": credential.credential_info,
            }
            for credential in litellm.credential_list
        ]
        return {"success": True, "credentials": masked_credentials}
    except Exception as e:
        return handle_exception_on_proxy(e)


@router.get(
    "/credentials/by_name/{credential_name}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
@router.get(
    "/credentials/by_model/{model_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialItem,
)
async def get_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: Optional[str] = None,
    model_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import llm_router

    try:
        if model_id:
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
            # return credential object
            return credential
        elif credential_name:
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
        else:
            raise HTTPException(
                status_code=404, detail="Credential name or model ID required"
            )
    except Exception as e:
        verbose_proxy_logger.exception(e)
        raise handle_exception_on_proxy(e)


@router.delete(
    "/credentials/{credential_name}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def delete_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        await prisma_client.db.litellm_credentialstable.delete(
            where={"credential_name": credential_name}
        )

        ## DELETE FROM LITELLM ##
        litellm.credential_list = [
            cred
            for cred in litellm.credential_list
            if cred.credential_name != credential_name
        ]
        return {"success": True, "message": "Credential deleted successfully"}
    except Exception as e:
        return handle_exception_on_proxy(e)


def update_db_credential(
    db_credential: CredentialItem, updated_patch: CredentialItem
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
        updated_patch
    )
    # update model name
    if encrypted_credential.credential_name:
        merged_credential.credential_name = encrypted_credential.credential_name

    # update litellm params
    if encrypted_credential.credential_values:
        # Encrypt any sensitive values
        encrypted_params = {
            k: v for k, v in encrypted_credential.credential_values.items()
        }

        merged_credential.credential_values.update(encrypted_params)

    # update model info
    if encrypted_credential.credential_info:
        """Update credential info"""
        if "credential_info" not in merged_credential.credential_info:
            merged_credential.credential_info = {}
        merged_credential.credential_info.update(encrypted_credential.credential_info)

    return merged_credential


@router.patch(
    "/credentials/{credential_name}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def update_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str,
    credential: CredentialItem,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        db_credential = await prisma_client.db.litellm_credentialstable.find_unique(
            where={"credential_name": credential_name},
        )
        if db_credential is None:
            raise HTTPException(status_code=404, detail="Credential not found in DB.")
        merged_credential = update_db_credential(db_credential, credential)
        credential_object_jsonified = jsonify_object(merged_credential.model_dump())
        await prisma_client.db.litellm_credentialstable.update(
            where={"credential_name": credential_name},
            data={
                **credential_object_jsonified,
                "updated_by": user_api_key_dict.user_id,
            },
        )
        return {"success": True, "message": "Credential updated successfully"}
    except Exception as e:
        return handle_exception_on_proxy(e)
