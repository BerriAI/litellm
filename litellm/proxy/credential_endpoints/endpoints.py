"""
CRUD endpoints for storing reusable credentials.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Path

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


def validate_aws_credential(credential_name: str, test_role_assumption: bool = False) -> dict:
    """
    Validate that an AWS credential exists and optionally test role assumption.

    Args:
        credential_name: Name of the credential to validate
        test_role_assumption: If True and the credential contains aws_role_name,
                             attempt to assume the role to verify it works

    Returns:
        dict with validation results:
        - valid: bool indicating if credential is valid
        - message: str with details about validation
        - credential_params: list of AWS params found in the credential

    Raises:
        HTTPException if credential not found or validation fails
    """
    credential_values = CredentialAccessor.get_credential_values(credential_name)
    if not credential_values:
        raise HTTPException(
            status_code=404,
            detail=f"AWS credential '{credential_name}' not found"
        )

    # Check which AWS params are present
    aws_params = [
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_region_name",
        "aws_session_name",
        "aws_profile_name",
        "aws_role_name",
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_bedrock_runtime_endpoint",
        "aws_external_id",
    ]
    found_params = [p for p in aws_params if p in credential_values]

    if not found_params:
        raise HTTPException(
            status_code=400,
            detail=f"Credential '{credential_name}' does not contain any AWS authentication parameters"
        )

    result = {
        "valid": True,
        "message": f"Credential '{credential_name}' contains valid AWS parameters",
        "credential_params": found_params,
    }

    # Optionally test role assumption
    if test_role_assumption and "aws_role_name" in credential_values:
        try:
            from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

            aws_llm = BaseAWSLLM()
            # Attempt to get credentials - this will try to assume the role
            aws_llm.get_credentials(
                aws_access_key_id=credential_values.get("aws_access_key_id"),
                aws_secret_access_key=credential_values.get("aws_secret_access_key"),
                aws_session_token=credential_values.get("aws_session_token"),
                aws_region_name=credential_values.get("aws_region_name", "us-east-1"),
                aws_session_name=credential_values.get("aws_session_name", "litellm-validation"),
                aws_profile_name=credential_values.get("aws_profile_name"),
                aws_role_name=credential_values.get("aws_role_name"),
                aws_web_identity_token=credential_values.get("aws_web_identity_token"),
                aws_sts_endpoint=credential_values.get("aws_sts_endpoint"),
                aws_external_id=credential_values.get("aws_external_id"),
            )
            result["role_assumption_tested"] = True
            result["message"] = f"Credential '{credential_name}' successfully assumed role {credential_values['aws_role_name']}"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to assume role with credential '{credential_name}': {str(e)}"
            )

    return result


class CredentialHelperUtils:
    @staticmethod
    def encrypt_credential_values(credential: CredentialItem, new_encryption_key: Optional[str] = None) -> CredentialItem:
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
    "/credentials/by_name/{credential_name:path}",
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
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
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
    db_credential: CredentialItem, updated_patch: CredentialItem, new_encryption_key: Optional[str] = None
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
    "/credentials/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def update_credential(
    request: Request,
    fastapi_response: Response,
    credential: CredentialItem,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
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


@router.post(
    "/credentials/{credential_name:path}/validate_aws",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
)
async def validate_aws_credential_endpoint(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name to validate"),
    test_role_assumption: bool = False,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Validate that an AWS credential exists and optionally test role assumption.

    This endpoint can be used to verify that:
    1. The credential exists in the credential store
    2. The credential contains valid AWS authentication parameters
    3. (Optional) The role can be successfully assumed if aws_role_name is present

    Args:
        credential_name: Name of the credential to validate
        test_role_assumption: If True, attempt to assume the role to verify it works

    Returns:
        Validation results including found AWS parameters and role assumption status
    """
    try:
        result = validate_aws_credential(
            credential_name=credential_name,
            test_role_assumption=test_role_assumption,
        )
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception_on_proxy(e)
