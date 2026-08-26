"""
CRUD endpoints for storing reusable credentials.
"""

from typing import (
    Final,
    cast,  # noqa: TID251  # jsonify_object in proxy/utils.py is annotated with a bare dict
)

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import _get_masked_values
from litellm.llms.anthropic.wif import (
    _IDENTITY_SOURCE_PARAM,  # pyright: ignore[reportPrivateUsage]  # one canonical param name, shared with the litellm_params identity-source resolver
    _INTERNAL_ISSUER_FIELD_MAP,  # pyright: ignore[reportPrivateUsage]  # one canonical field map, shared with the litellm_params identity-source resolver
    _build_variant,  # pyright: ignore[reportPrivateUsage]  # one canonical builder, shared with the litellm_params identity-source resolver
)
from litellm.llms.base_llm.auth.identity_source import (
    AnthropicIdentitySourceKind,
    InternalIssuerSource,
)
from litellm.llms.base_llm.auth.internal_issuer import internal_issuer_jwks_document
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.credential_hydration import (
    hydrate_named_credential,
    hydrate_named_credential_authoritative,
    named_credential_wif_fields,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.utils import handle_exception_on_proxy, jsonify_object
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.router import anthropic_wif_fields_named
from litellm.types.utils import CreateCredentialItem, CredentialItem

router: Final = APIRouter()


def _reject_non_admin_wif_fields(
    wif_fields: tuple[str, ...],
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """A credential referenced by ``litellm_credential_name`` feeds its values into the same
    workload identity federation resolution as a deployment's own ``litellm_params``. Only proxy
    admins may touch a server-owned WIF field, whether they write it, drop it, or edit a stored
    credential that already carries one.
    """
    if not wif_fields or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return
    raise HTTPException(
        status_code=403,
        detail={  # mutable-ok: starlette json.dumps()s HTTPException.detail raw, needs a real dict
            "error": (
                f"Only proxy admins can change {wif_fields[0]!r}, a server-owned workload identity federation "
                "parameter."
            )
        },
    )


def _incoming_wif_fields(credential: CredentialItem) -> tuple[str, ...]:
    """WIF fields the request payload itself touches: the ones it sets (to any value, ``None``
    included, since the key alone is what the federation resolver reacts to), plus the ones it
    names in ``credential_values_to_delete``, since dropping a federation field off the stored
    credential breaks every deployment referencing it just as installing one would redirect them.
    """
    return anthropic_wif_fields_named(credential.credential_values) + anthropic_wif_fields_named(
        credential.credential_values_to_delete or ()
    )


def _stored_wif_fields(stored_credential: CredentialItem) -> tuple[str, ...]:
    return anthropic_wif_fields_named(stored_credential.credential_values)


def _reject_overlapping_credential_values(credential: CredentialItem) -> None:
    overlap: Final = frozenset(credential.credential_values) & frozenset(credential.credential_values_to_delete or ())
    if overlap:
        raise HTTPException(
            status_code=400,
            detail=f"credential_values_to_delete overlaps credential_values for key(s): {sorted(overlap)}",
        )


def _sync_in_memory_credential(credential: CredentialItem, credential_name: str, new_name: str) -> None:
    """Mirror a DB credential update into the in-memory ``credential_list`` used by request-time
    resolution; a no-op if the credential isn't loaded in memory (e.g. proxy restarted since boot).
    """
    existing_in_memory: CredentialItem | None = None
    for cred in litellm.credential_list:
        if cred.credential_name == credential_name:
            existing_in_memory = cred
            break

    if existing_in_memory is None:
        return

    in_memory_values: Final = dict(existing_in_memory.credential_values or {})
    if credential.credential_values:
        in_memory_values.update(credential.credential_values)
    for key in credential.credential_values_to_delete or ():
        in_memory_values.pop(key, None)
    in_memory_info: Final = dict(existing_in_memory.credential_info or {})
    if credential.credential_info:
        in_memory_info.update(credential.credential_info)
    updated_in_memory: Final = CredentialItem(
        credential_name=new_name,
        credential_values=in_memory_values,
        credential_info=in_memory_info,
    )
    # Remove old entry if renamed, then use upsert_credentials to handle duplicates
    if new_name != credential_name:
        litellm.credential_list = [c for c in litellm.credential_list if c.credential_name != credential_name]
    CredentialAccessor.upsert_credentials([updated_in_memory])


class CredentialHelperUtils:
    @staticmethod
    def encrypt_credential_values(credential: CredentialItem, new_encryption_key: str | None = None) -> CredentialItem:
        """Encrypt values in credential.credential_values and add to DB"""
        encrypted_credential_values: Final = {}
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
            model: Final = llm_router.get_deployment(credential.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail="Model not found")
            credential_values: Final = llm_router.get_deployment_credentials(credential.model_id)
            if credential_values is None:
                raise HTTPException(status_code=404, detail="Model not found")
            credential.credential_values = credential_values

        if credential.credential_values is None:
            raise HTTPException(
                status_code=400,
                detail="Credential values are required. Unable to infer credential values from model ID.",
            )
        _reject_non_admin_wif_fields(anthropic_wif_fields_named(credential.credential_values), user_api_key_dict)
        _reject_non_admin_wif_fields(
            await named_credential_wif_fields(credential.credential_name, prisma_client), user_api_key_dict
        )
        processed_credential: Final = CredentialItem(
            credential_name=credential.credential_name,
            credential_values=credential.credential_values,
            credential_info=credential.credential_info,
        )
        encrypted_credential: Final = CredentialHelperUtils.encrypt_credential_values(processed_credential)
        # exclude_none: wif.py rejects foreign-variant fields by presence, so persisting a null
        # for every unset variant field would fail the next request against this credential
        credentials_dict: Final = encrypted_credential.model_dump(exclude_none=True)
        credentials_dict_jsonified: Final = cast(  # cast-ok: deep-copies a model_dump, so keys are str
            "dict[str, object]", jsonify_object(credentials_dict)
        )
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
    """
    try:
        masked_credentials: Final = [
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
    "/credentials/{credential_name:path}/jwks",
    dependencies=(Depends(user_api_key_auth),),
    tags=["credential management"],  # mutable-ok: FastAPI's include_router does self.tags.copy(), needs a real list
)
async def get_credential_internal_issuer_jwks(
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves the dependency from the default
):
    """
    Export the public JWKS for an anthropic ``internal_issuer`` credential, so the operator can
    register it on the Anthropic federation issuer from the UI. Never touches the private signing
    key: only its derived public JWKS leaves this process. 404s for any other credential shape.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={  # mutable-ok: starlette json.dumps()s HTTPException.detail raw, needs a real dict
                "error": "Only proxy admins can export a credential's JWKS."
            },
        )

    try:
        credential: Final = await hydrate_named_credential_authoritative(credential_name, prisma_client)
        if credential is None or credential.credential_info.get("custom_llm_provider") != "anthropic":
            raise HTTPException(
                status_code=404,
                detail={  # mutable-ok: starlette json.dumps()s HTTPException.detail raw, needs a real dict
                    "error": f"No anthropic credential named {credential_name!r}."
                },
            )
        configured_source: Final = credential.credential_values.get(_IDENTITY_SOURCE_PARAM)
        if configured_source != AnthropicIdentitySourceKind.internal_issuer.value:
            raise HTTPException(
                status_code=404,
                detail={  # mutable-ok: starlette json.dumps()s HTTPException.detail raw, needs a real dict
                    "error": (
                        f"Credential {credential_name!r} is not configured with "
                        f"{_IDENTITY_SOURCE_PARAM}={AnthropicIdentitySourceKind.internal_issuer.value!r}."
                    )
                },
            )
        try:
            issuer_source: Final = _build_variant(
                InternalIssuerSource, credential.credential_values, _INTERNAL_ISSUER_FIELD_MAP
            )
            jwks_document: Final = internal_issuer_jwks_document(issuer_source)
        except (litellm.AuthenticationError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail={  # mutable-ok: starlette json.dumps()s HTTPException.detail raw, needs a real dict
                    "error": str(e)
                },
            ) from e
        return Response(content=jwks_document, media_type="application/json")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy's error contract
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
        model: Final = llm_router.get_deployment(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        credential_values: Final = llm_router.get_deployment_credentials(model_id)
        if credential_values is None:
            raise HTTPException(status_code=404, detail="Model not found")
        masked_credential_values: Final = _get_masked_values(
            credential_values,
            unmasked_length=4,
            number_of_asterisks=4,
        )
        credential: Final = CredentialItem(
            credential_name=f"{model.model_name}-credential-{model_id}",
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

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        _reject_non_admin_wif_fields(
            await named_credential_wif_fields(credential_name, prisma_client), user_api_key_dict
        )
        await CredentialsRepository(prisma_client).delete_by_name(credential_name)

        ## DELETE FROM LITELLM ##
        litellm.credential_list = [cred for cred in litellm.credential_list if cred.credential_name != credential_name]
        return {"success": True, "message": "Credential deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        return handle_exception_on_proxy(e)


def update_db_credential(
    db_credential: CredentialItem,
    updated_patch: CredentialItem,
    new_encryption_key: str | None = None,
) -> CredentialItem:
    """
    Update a credential in the DB.
    """
    merged_credential: Final = CredentialItem(
        credential_name=db_credential.credential_name,
        credential_info=db_credential.credential_info,
        credential_values=db_credential.credential_values,
    )

    encrypted_credential: Final = CredentialHelperUtils.encrypt_credential_values(
        updated_patch,
        new_encryption_key,
    )
    # update model name
    if encrypted_credential.credential_name:
        merged_credential.credential_name = encrypted_credential.credential_name

    # update litellm params
    if encrypted_credential.credential_values:
        # Encrypt any sensitive values
        encrypted_params: Final = {k: v for k, v in encrypted_credential.credential_values.items()}

        merged_credential.credential_values.update(encrypted_params)

    for key in updated_patch.credential_values_to_delete or ():
        merged_credential.credential_values.pop(key, None)

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
        _reject_overlapping_credential_values(credential)
        _reject_non_admin_wif_fields(_incoming_wif_fields(credential), user_api_key_dict)
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        credentials_repository: Final = CredentialsRepository(prisma_client)
        db_credential: Final = await credentials_repository.find_by_name(credential_name)
        if db_credential is None:
            raise HTTPException(status_code=404, detail="Credential not found in DB.")
        _reject_non_admin_wif_fields(_stored_wif_fields(db_credential), user_api_key_dict)
        if credential.credential_name != credential_name:
            shadowed_credential: Final = await hydrate_named_credential(credential.credential_name, prisma_client)
            if shadowed_credential is not None:
                _reject_non_admin_wif_fields(_stored_wif_fields(shadowed_credential), user_api_key_dict)
        merged_credential: Final = update_db_credential(db_credential, credential)
        credential_object_jsonified: Final = cast(  # cast-ok: deep-copies a model_dump, so keys are str
            "dict[str, object]", jsonify_object(merged_credential.model_dump(exclude_none=True))
        )
        await credentials_repository.update_by_name(
            credential_name,
            data={
                **credential_object_jsonified,
                "updated_by": user_api_key_dict.user_id,
            },
        )

        # Sync in-memory credential_list (skip if not in memory - e.g., proxy restarted)
        _sync_in_memory_credential(credential, credential_name, merged_credential.credential_name)

        return {"success": True, "message": "Credential updated successfully"}
    except Exception as e:
        raise handle_exception_on_proxy(e)
