"""
CRUD endpoints for storing reusable credentials.
"""

import time
from collections.abc import Mapping
from datetime import datetime
from typing import (
    Annotated,
    Final,
    cast,  # noqa: TID251  # jsonify_object in proxy/utils.py is annotated with a bare dict
)

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    GITHUB_COPILOT_AUTH_TYPE_KEY,
    GITHUB_COPILOT_PER_USER_AUTH_TYPE,
)
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import get_masked_values
from litellm.llms.anthropic.wif import (
    ExportedJwks,
    NotAnInternalIssuerCredential,
    UnbuildableIdentitySource,
    anthropic_internal_issuer_jwks,
)
from litellm.models.credentials import (
    CredentialView,
    UpdateCredentialItem,
    UserConnectionDeleteResponse,
    UserConnectionFlowHandle,
    UserConnectionPollRequest,
    UserConnectionPollResponse,
    UserConnectionStartResponse,
    UserProviderConnection,
    UserProviderConnectionsResponse,
)
from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.credential_hydration import (
    hydrate_named_credential_authoritative,
    named_credential_wif_fields,
    stored_credential_provider,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.utils import handle_exception_on_proxy, jsonify_object
from litellm.repositories.base_repository import is_unique_violation
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.router import server_owned_wif_fields_named
from litellm.types.utils import CreateCredentialItem, CredentialItem

from .user_provider_credentials import (
    GithubCopilotUserConnectionPayload,
    decode_user_provider_credential,
    delete_user_provider_credential,
    delete_user_provider_credentials_for_credential,
    invalidate_user_provider_credential_cache,
    list_user_provider_credentials,
    list_user_provider_credentials_for_credential,
    set_user_provider_credential_cache,
    upsert_user_provider_credential,
)

router: Final = APIRouter()
_CREDENTIAL_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_GITHUB_COPILOT_PROVIDER: Final = "github_copilot"
_DISPLAY_NAME_MAX_LENGTH: Final = 255


def _reject_non_admin_wif_fields(
    wif_fields: tuple[str, ...],
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """A credential referenced by ``litellm_credential_name`` feeds its values into the same
    server-owned federation or OAuth token-exchange configuration as a deployment's own
    ``litellm_params``. Only proxy admins may touch one, whether they write it, drop it, or edit a
    stored credential that already carries it.
    """
    if not wif_fields or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return
    raise ProxyException(
        message=(
            f"Only proxy admins can change {wif_fields[0]!r}, a server-owned workload identity federation "
            "or OAuth token exchange parameter."
        ),
        type=ProxyErrorTypes.auth_error.value,
        code=status.HTTP_403_FORBIDDEN,
        param=wif_fields[0],
    )


def _incoming_wif_fields(incoming_values: Mapping[str, object], credential: UpdateCredentialItem) -> tuple[str, ...]:
    """WIF fields the request touches: the ones its values set (to any value, ``None`` included,
    since the key alone is what the federation resolver reacts to), whether the caller sent them
    or named a deployment through ``model_id`` for the proxy to copy them from, plus the ones it
    names in ``credential_values_to_delete``, since dropping a federation field off the stored
    credential breaks every deployment referencing it just as installing one would redirect them.
    """
    return server_owned_wif_fields_named(incoming_values) + server_owned_wif_fields_named(
        credential.credential_values_to_delete or ()
    )


def _stored_wif_fields(stored_credential: CredentialItem) -> tuple[str, ...]:
    return server_owned_wif_fields_named(stored_credential.credential_values)


def _reject_overlapping_credential_values(credential: UpdateCredentialItem) -> None:
    overlap: Final = frozenset(credential.credential_values or ()) & frozenset(
        credential.credential_values_to_delete or ()
    )
    if overlap:
        raise HTTPException(
            status_code=400,
            detail=f"credential_values_to_delete overlaps credential_values for key(s): {sorted(overlap)}",
        )


def _without_null_values(credential_values: Mapping[str, object]) -> dict[str, object]:
    """A null carries no credential, and the federation resolver refuses a foreign variant's field by
    KEY, so a stored ``{"anthropic_issuer_url": null}`` wedges every deployment that names this
    credential. ``model_dump(exclude_none=True)`` cannot do this: it drops the model's own null
    fields, and ``credential_values`` is a mapping inside one of them.
    """
    return {key: value for key, value in credential_values.items() if value is not None}


def _sync_in_memory_credential(credential: CredentialItem, credential_name: str) -> None:
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
        in_memory_values.update(_without_null_values(credential.credential_values))
    for key in credential.credential_values_to_delete or ():
        in_memory_values.pop(key, None)
    in_memory_info: Final = dict(existing_in_memory.credential_info or {})
    if credential.credential_info:
        in_memory_info.update(credential.credential_info)
    updated_in_memory: Final = CredentialItem(
        credential_name=credential_name,
        display_name=credential.display_name,
        credential_values=in_memory_values,
        credential_info=in_memory_info,
    )
    CredentialAccessor.upsert_credentials([updated_in_memory])


class CredentialHelperUtils:
    @staticmethod
    def encrypt_credential_values(credential: CredentialItem, new_encryption_key: str | None = None) -> CredentialItem:
        """Encrypt values in credential.credential_values and add to DB"""
        encrypted_credential_values: Final[dict[str, object]] = {}  # mutable-ok: built one entry per credential value
        for key, value in (credential.credential_values or {}).items():
            encrypted_credential_values[key] = encrypt_value_helper(
                cast("str", value),  # cast-ok: credential values are str at the encryption boundary
                new_encryption_key,
            )

        # Return a new object to avoid mutating the caller's credential, which
        # is kept in memory and should remain unencrypted.
        return CredentialItem(
            credential_name=credential.credential_name,
            display_name=credential.display_name,
            credential_values=encrypted_credential_values,
            credential_info=credential.credential_info or {},
        )


def _normalized_display_name(display_name: str | None) -> str | None:
    if display_name is None:
        return None
    trimmed: Final = display_name.strip()
    if not trimmed:
        raise ProxyException(
            message="display_name cannot be blank. Send null to clear it or omit the field to leave it unchanged.",
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="display_name",
        )
    if len(trimmed) > _DISPLAY_NAME_MAX_LENGTH:
        raise ProxyException(
            message=f"display_name cannot be longer than {_DISPLAY_NAME_MAX_LENGTH} characters.",
            type=ProxyErrorTypes.validation_error.value,
            code=status.HTTP_400_BAD_REQUEST,
            param="display_name",
        )
    return trimmed


def _not_found_unless_config_defined(credential_name: str, not_found_detail: str) -> HTTPException | ProxyException:
    in_memory: Final = CredentialAccessor.find_credential(credential_name)
    if in_memory is None or in_memory.source != "config":
        return HTTPException(status_code=404, detail=not_found_detail)
    return ProxyException(
        message=f"Credential '{credential_name}' is defined in config and cannot be edited from the API or UI.",
        type=ProxyErrorTypes.validation_error.value,
        code=status.HTTP_405_METHOD_NOT_ALLOWED,
        param="credential_name",
        headers={"Allow": "GET"},
    )


def _credential_view(credential: CredentialItem, credential_values: Mapping[str, object]) -> CredentialView:
    return CredentialView(
        credential_name=credential.credential_name,
        display_name=credential.display_name,
        credential_values=credential_values,
        credential_info=credential.credential_info,
        source=credential.source,
    )


def _credential_exists_detail(credential_name: str) -> str:
    return (
        f"Credential '{credential_name}' already exists. "
        f"Update it with PATCH /credentials/{credential_name}, or delete it first."
    )


def get_llm_router() -> litellm.Router | None:
    from litellm.proxy.proxy_server import llm_router

    return llm_router


def _resolve_deployment_credentials(llm_router: litellm.Router | None, model_id: str) -> Mapping[str, object]:
    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail="LLM router not found. Please ensure you have a valid router instance.",
        )
    if llm_router.get_deployment(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    credential_values: Final = llm_router.get_deployment_credentials(model_id)
    if credential_values is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return _CREDENTIAL_DICT_ADAPTER.validate_python(credential_values)


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
    llm_router: Annotated[litellm.Router | None, Depends(get_llm_router)] = None,
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
        credential_values: Final = (
            _resolve_deployment_credentials(llm_router, credential.model_id)
            if credential.model_id
            else credential.credential_values
        )
        if credential_values is None:
            raise HTTPException(
                status_code=400,
                detail="Credential values are required. Unable to infer credential values from model ID.",
            )
        _reject_non_admin_wif_fields(server_owned_wif_fields_named(credential_values), user_api_key_dict)
        _reject_non_admin_wif_fields(
            await named_credential_wif_fields(credential.credential_name, prisma_client), user_api_key_dict
        )
        processed_credential: Final = CredentialItem(
            credential_name=credential.credential_name,
            display_name=_normalized_display_name(credential.display_name),
            credential_values=_without_null_values(_CREDENTIAL_DICT_ADAPTER.validate_python(credential_values)),
            credential_info=credential.credential_info,
        )
        encrypted_credential: Final = CredentialHelperUtils.encrypt_credential_values(processed_credential)
        credentials_dict: Final = encrypted_credential.model_dump(exclude_none=True)
        credentials_dict_jsonified: Final = cast(  # cast-ok: deep-copies a model_dump, so keys are str
            "dict[str, object]", jsonify_object(credentials_dict)
        )
        try:
            await CredentialsRepository(prisma_client).create(
                data={
                    **credentials_dict_jsonified,
                    "created_by": user_api_key_dict.user_id,
                    "updated_by": user_api_key_dict.user_id,
                }
            )
        except Exception as e:
            if not is_unique_violation(e):
                raise
            raise HTTPException(status_code=409, detail=_credential_exists_detail(credential.credential_name))

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
            _credential_view(credential, get_masked_values(credential.credential_values))
            for credential in litellm.credential_list
        ]
        return {"success": True, "credentials": masked_credentials}
    except Exception as e:
        raise handle_exception_on_proxy(e)


@router.get(
    "/credentials/by_name/{credential_name:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=CredentialView,
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
                return _credential_view(
                    credential,
                    get_masked_values(credential.credential_values, unmasked_length=4, number_of_asterisks=4),
                )
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
    tags=["credential management"],
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
            detail={"error": "Only proxy admins can export a credential's JWKS."},
        )

    try:
        credential: Final = await hydrate_named_credential_authoritative(credential_name, prisma_client)
        credential_provider: Final = (
            None
            if credential is None
            else stored_credential_provider(credential.credential_info.get("custom_llm_provider"))
        )
        if credential is None or credential_provider != "anthropic":
            raise HTTPException(
                status_code=404,
                detail={"error": f"No anthropic credential named {credential_name!r}."},
            )
        match anthropic_internal_issuer_jwks(credential.credential_values):
            case ExportedJwks(document):
                return Response(content=document, media_type="application/json")
            case NotAnInternalIssuerCredential(required_param, required_value):
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": (
                            f"Credential {credential_name!r} is not configured with "
                            f"{required_param}={required_value!r}."
                        )
                    },
                )
            case UnbuildableIdentitySource(message):
                raise HTTPException(
                    status_code=400,
                    detail={"error": message},
                )
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
        masked_credential_values: Final = get_masked_values(
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


def _authenticated_user_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    user_id: Final = user_api_key_dict.user_id
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="An authenticated user is required")
    return user_id


def _per_user_credential_or_404(credential_name: str) -> CredentialItem:
    credential: Final = CredentialAccessor.find_credential(credential_name)
    values: Final = credential.credential_values if credential is not None else None
    if (
        credential is None
        or not isinstance(values, Mapping)
        or values.get(GITHUB_COPILOT_AUTH_TYPE_KEY) != GITHUB_COPILOT_PER_USER_AUTH_TYPE
    ):
        raise HTTPException(status_code=404, detail="Credential not found")
    return credential


def _per_user_credential_names() -> tuple[str, ...]:
    return tuple(
        credential.credential_name
        for credential in litellm.credential_list
        if cast(  # cast-ok: credential_values is a plain dict at runtime
            Mapping[object, object], credential.credential_values
        ).get(GITHUB_COPILOT_AUTH_TYPE_KEY)
        == GITHUB_COPILOT_PER_USER_AUTH_TYPE
    )


@router.get(
    "/credentials/user_connections",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=UserProviderConnectionsResponse,
)
async def list_user_connections(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves the dependency from the default
) -> UserProviderConnectionsResponse:
    """List the calling user's per-user provider connections."""
    from litellm.proxy.proxy_server import prisma_client

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        rows: Final = await list_user_provider_credentials(prisma_client, user_id)
        github_logins: Final[dict[str, str]] = {}  # mutable-ok: accumulates one entry per credential row
        connected_at: Final[dict[str, str]] = {}  # mutable-ok: accumulates one entry per credential row
        for row in rows:
            if row.provider != _GITHUB_COPILOT_PROVIDER:
                continue
            payload = decode_user_provider_credential(row.credential_b64)
            if payload is not None:
                github_logins[row.credential_name] = payload.github_login
            updated = getattr(row, "updated_at", None)
            if isinstance(updated, datetime):
                connected_at[row.credential_name] = updated.isoformat()
        return UserProviderConnectionsResponse(
            connections=[
                UserProviderConnection(
                    credential_name=name,
                    provider=_GITHUB_COPILOT_PROVIDER,
                    connected=name in github_logins,
                    github_login=github_logins.get(name),
                    connected_at=connected_at.get(name),
                )
                for name in _per_user_credential_names()
            ]
        )
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy error contract
        raise handle_exception_on_proxy(e)


def _decode_flow_handle(flow_handle: str) -> UserConnectionFlowHandle | None:
    decrypted: Final = decrypt_value_helper(
        value=flow_handle,
        key="device_flow_handle",
        exception_type="debug",
        return_original_value=False,
    )
    if decrypted is None:
        return None
    try:
        return UserConnectionFlowHandle.model_validate_json(decrypted)
    except ValidationError:
        return None


@router.post(
    "/credentials/{credential_name:path}/user_connection/start",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=UserConnectionStartResponse,
)
@with_service_target("user_provider_connections")
async def start_user_connection(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves the dependency from the default
) -> UserConnectionStartResponse:
    """Begin a GitHub device flow for the calling user's connection to a per-user credential."""
    from litellm.llms.github_copilot.per_user_auth import astart_device_flow
    from litellm.proxy.proxy_server import prisma_client

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        _per_user_credential_or_404(credential_name)
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        flow: Final = await astart_device_flow()
        handle: Final = encrypt_value_helper(
            UserConnectionFlowHandle(
                user_id=user_id,
                credential_name=credential_name,
                device_code=flow.device_code,
                interval=flow.interval,
                expires_at=time.time() + flow.expires_in,
            ).model_dump_json()
        )
        return UserConnectionStartResponse(
            user_code=flow.user_code,
            verification_uri=flow.verification_uri,
            expires_in=flow.expires_in,
            interval=flow.interval,
            flow_handle=handle,
        )
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy error contract
        raise handle_exception_on_proxy(e)


@router.post(
    "/credentials/{credential_name:path}/user_connection/poll",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=UserConnectionPollResponse,
)
@with_service_target("user_provider_connections")
async def poll_user_connection(
    request: Request,
    fastapi_response: Response,
    body: UserConnectionPollRequest,
    credential_name: str = Path(..., description="The credential name, percent-decoded"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves the dependency from the default
) -> UserConnectionPollResponse:
    """Poll the device flow once and persist the connection on completion."""
    from litellm.llms.github_copilot.per_user_auth import (
        acheck_copilot_seat,
        afetch_github_login,
        apoll_device_flow,
    )
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        _per_user_credential_or_404(credential_name)
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        handle: Final = _decode_flow_handle(body.flow_handle)
        if (
            handle is None
            or handle.user_id != user_id
            or handle.credential_name != credential_name
            or handle.expires_at <= time.time()
        ):
            raise HTTPException(status_code=400, detail="invalid or expired flow_handle")

        poll: Final = await apoll_device_flow(handle.device_code)
        if poll.status in ("expired", "denied", "pending", "slow_down"):
            return UserConnectionPollResponse(status=poll.status, interval=poll.interval)

        github_token: Final = poll.access_token
        if not github_token:
            raise HTTPException(status_code=502, detail="GitHub device flow completed without a token")
        try:
            await acheck_copilot_seat(user_id=user_id, github_token=github_token, credential_name=credential_name)
        except litellm.CallerCredentialAuthenticationError:
            return UserConnectionPollResponse(status="no_copilot_seat")
        github_login: Final = await afetch_github_login(github_token)
        connection_payload: Final = GithubCopilotUserConnectionPayload(
            access_token=github_token, github_login=github_login
        )
        await upsert_user_provider_credential(
            prisma_client,
            user_id,
            credential_name,
            _GITHUB_COPILOT_PROVIDER,
            connection_payload,
        )
        if not await set_user_provider_credential_cache(
            user_api_key_cache, user_id, credential_name, connection_payload
        ):
            raise HTTPException(
                status_code=503,
                detail="GitHub connection saved, but the cache could not be refreshed; "
                "requests may be rejected for up to 60 seconds",
            )
        return UserConnectionPollResponse(status="connected", github_login=github_login)
    except litellm.CallerCredentialRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:  # noqa: BLE001  # every remaining failure maps to the proxy error shape
        raise handle_exception_on_proxy(e)


@router.delete(
    "/credentials/{credential_name:path}/user_connection",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=UserConnectionDeleteResponse,
)
@with_service_target("user_provider_connections")
async def delete_user_connection(
    request: Request,
    fastapi_response: Response,
    credential_name: str = Path(..., description="The credential name, percent-decoded"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves the dependency from the default
) -> UserConnectionDeleteResponse:
    """Disconnect the calling user's stored GitHub token for a per-user credential. Idempotent."""
    from litellm.llms.github_copilot.per_user_auth import evict_copilot_user_session
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        _per_user_credential_or_404(credential_name)
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        # Tombstone first: while a token may still be cached, the DB row stays.
        # If the tombstone write fails and the row were deleted, the stale token
        # would keep working until its TTL with no connection left to re-check.
        tombstoned: Final = await invalidate_user_provider_credential_cache(
            user_api_key_cache, user_id, credential_name
        )
        if not tombstoned:
            raise HTTPException(
                status_code=503,
                detail={"error": "could not revoke cached connection, retry"},
            )
        prior: Final = await delete_user_provider_credential(prisma_client, user_id, credential_name)
        if prior is not None:
            evict_copilot_user_session(user_id, prior.access_token)
        return UserConnectionDeleteResponse(status="disconnected")
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy error contract
        raise handle_exception_on_proxy(e)


@with_service_target("user_provider_connections")
async def _purge_user_connections_for_credential(credential_name: str) -> None:
    """Drop every user connection under a per-user credential and clear caches.

    Shared cleanup for DELETE (credential removed) and PATCH (renamed or no
    longer per-user)."""
    from litellm.llms.github_copilot.per_user_auth import evict_copilot_user_session
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return
    try:
        prior_rows: Final = await list_user_provider_credentials_for_credential(prisma_client, credential_name)
        user_ids: Final = await delete_user_provider_credentials_for_credential(prisma_client, credential_name)
    except Exception:  # noqa: BLE001  # best-effort purge must not break credential deletion
        verbose_proxy_logger.exception(
            "_purge_user_connections_for_credential: failed to drop user connections for %s", credential_name
        )
        return
    access_tokens: Final[dict[str, str]] = {}  # mutable-ok: accumulates one entry per deleted row
    for row in prior_rows:
        decoded = decode_user_provider_credential(row.credential_b64)
        if decoded is not None:
            access_tokens[row.user_id] = decoded.access_token
    for user_id in user_ids:
        await invalidate_user_provider_credential_cache(user_api_key_cache, user_id, credential_name)
        token = access_tokens.get(user_id)
        if token:
            evict_copilot_user_session(user_id, token)


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
        deleted: Final = await CredentialsRepository(prisma_client).delete_by_name(credential_name)
        if deleted is None:
            raise _not_found_unless_config_defined(
                credential_name, "Credential not found. Got credential name: " + credential_name
            )

        ## DELETE FROM LITELLM ##
        litellm.credential_list = [cred for cred in litellm.credential_list if cred.credential_name != credential_name]
        await _purge_user_connections_for_credential(credential_name)
        return {"success": True, "message": "Credential deleted successfully"}
    except Exception as e:
        raise handle_exception_on_proxy(e)


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
        display_name=updated_patch.display_name,
        credential_info=db_credential.credential_info,
        credential_values=db_credential.credential_values,
    )

    encrypted_credential: Final = CredentialHelperUtils.encrypt_credential_values(
        updated_patch,
        new_encryption_key,
    )
    # update litellm params
    if encrypted_credential.credential_values:
        # Encrypt any sensitive values
        merged_credential.credential_values.update(_without_null_values(encrypted_credential.credential_values))

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
    credential: UpdateCredentialItem,
    credential_name: str = Path(..., description="The credential name, percent-decoded; may contain slashes"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    llm_router: Annotated[litellm.Router | None, Depends(get_llm_router)] = None,
):
    """
    [BETA] endpoint. This might change unexpectedly.
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if credential.credential_name and credential.credential_name != credential_name:
            raise ProxyException(
                message="credential_name is immutable. Set display_name to change how the credential is labeled.",
                type=ProxyErrorTypes.validation_error.value,
                code=status.HTTP_400_BAD_REQUEST,
                param="credential_name",
            )
        requested_display_name: Final = _normalized_display_name(credential.display_name)
        _reject_overlapping_credential_values(credential)
        incoming_values: Final = _CREDENTIAL_DICT_ADAPTER.validate_python(
            _resolve_deployment_credentials(llm_router, credential.model_id)
            if credential.model_id
            else credential.credential_values or {}
        )
        _reject_non_admin_wif_fields(_incoming_wif_fields(incoming_values, credential), user_api_key_dict)
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )
        credentials_repository: Final = CredentialsRepository(prisma_client)
        db_credential: Final = await credentials_repository.find_by_name(credential_name)
        if db_credential is None:
            raise _not_found_unless_config_defined(credential_name, "Credential not found in DB.")
        _reject_non_admin_wif_fields(_stored_wif_fields(db_credential), user_api_key_dict)
        patch: Final = CredentialItem(
            credential_name=credential_name,
            display_name=(
                requested_display_name if "display_name" in credential.model_fields_set else db_credential.display_name
            ),
            credential_info=_CREDENTIAL_DICT_ADAPTER.validate_python(credential.credential_info),
            credential_values=incoming_values,
            credential_values_to_delete=credential.credential_values_to_delete,
        )
        merged_credential: Final = update_db_credential(db_credential, patch)
        credential_object_jsonified: Final = cast(  # cast-ok: deep-copies a model_dump, so keys are str
            "dict[str, object]", jsonify_object(merged_credential.model_dump(exclude_none=True))
        )
        await credentials_repository.update_by_name(
            credential_name,
            data={
                **credential_object_jsonified,
                "display_name": merged_credential.display_name,
                "updated_by": user_api_key_dict.user_id,
            },
        )

        # Sync in-memory credential_list (skip if not in memory - e.g., proxy restarted)
        _sync_in_memory_credential(patch, credential_name)

        merged_values: Final = cast(  # cast-ok: credential_values is a plain dict at runtime
            Mapping[object, object], merged_credential.credential_values
        )
        still_per_user: Final = merged_values.get(GITHUB_COPILOT_AUTH_TYPE_KEY) == GITHUB_COPILOT_PER_USER_AUTH_TYPE
        if not still_per_user:
            await _purge_user_connections_for_credential(credential_name)

        return {"success": True, "message": "Credential updated successfully"}
    except Exception as e:
        raise handle_exception_on_proxy(e)
