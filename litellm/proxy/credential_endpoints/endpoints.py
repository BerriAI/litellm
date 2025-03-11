"""
CRUD endpoints for storing reusable credentials.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.utils import handle_exception_on_proxy, jsonify_object
from litellm.types.utils import CredentialItem

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
    credential: CredentialItem,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    Stores credential in DB.
    Reloads credentials in memory.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        credential = CredentialHelperUtils.encrypt_credential_values(credential)
        credentials_dict = credential.model_dump()
        credentials_dict_jsonified = jsonify_object(credentials_dict)
        await prisma_client.db.litellm_credentialstable.create(
            data={
                **credentials_dict_jsonified,
                "created_by": user_api_key_dict.user_id,
                "updated_by": user_api_key_dict.user_id,
            }
        )

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
                "credential_values": credential.credential_values,
            }
            for credential in litellm.credential_list
        ]
        return {"success": True, "credentials": masked_credentials}
    except Exception as e:
        return handle_exception_on_proxy(e)


@router.get(
    "/credentials/{credential_name}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def get_credential(
    request: Request,
    fastapi_response: Response,
    credential_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    try:
        for credential in litellm.credential_list:
            if credential.credential_name == credential_name:
                masked_credential = {
                    "credential_name": credential.credential_name,
                    "credential_values": credential.credential_values,
                }
                return {"success": True, "credential": masked_credential}
        return {"success": False, "message": "Credential not found"}
    except Exception as e:
        return handle_exception_on_proxy(e)


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
        return {"success": True, "message": "Credential deleted successfully"}
    except Exception as e:
        return handle_exception_on_proxy(e)


@router.put(
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
        credential_object_jsonified = jsonify_object(credential.model_dump())
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
