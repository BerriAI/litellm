#### Realtime WebRTC Endpoints #####

import json
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi import status as http_status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.realtime import (
    RealtimeClientSecretRequest,
    RealtimeClientSecretResponse,
)

router = APIRouter()

_REALTIME_TOKEN_VERSION = "realtime_v1"


def _encode_realtime_token_payload(
    ephemeral_key: str,
    model_id: str,
    user_id: Optional[str],
    team_id: Optional[str],
    expires_at: Optional[int],
) -> str:
    """
    Encode metadata with the upstream ephemeral key so /realtime/calls can
    route without requiring model as a query param.
    """
    payload: Dict[str, Any] = {
        "v": _REALTIME_TOKEN_VERSION,
        "ephemeral_key": ephemeral_key,
        "model_id": model_id,
        "user_id": user_id or "",
        "team_id": team_id or "",
        "expires_at": expires_at,
    }
    return json.dumps(payload, separators=(",", ":"))


def _decode_realtime_token_payload(
    decrypted_value: str,
) -> Optional[Dict[str, Any]]:
    """
    Decode realtime token payload; returns None for legacy/raw ephemeral tokens.
    """
    try:
        decoded = json.loads(decrypted_value)
    except Exception:
        return None

    if not isinstance(decoded, dict):
        return None
    if decoded.get("v") != _REALTIME_TOKEN_VERSION:
        return None
    if not isinstance(decoded.get("ephemeral_key"), str):
        return None
    if not isinstance(decoded.get("model_id"), str):
        return None
    return decoded


@router.post(
    "/v1/realtime/client_secrets",
    dependencies=[Depends(user_api_key_auth)],
    tags=["realtime"],
)
@router.post(
    "/realtime/client_secrets",
    dependencies=[Depends(user_api_key_auth)],
    tags=["realtime"],
)
@router.post(
    "/openai/v1/realtime/client_secrets",
    dependencies=[Depends(user_api_key_auth)],
    tags=["realtime"],
)
async def create_realtime_client_secret(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RealtimeClientSecretResponse:
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        route_request,
        user_model,
        version,
    )

    data: dict = {}
    try:
        body = await _read_request_body(request=request)
        req = RealtimeClientSecretRequest(**body)

        model: str = (
            (req.session.model if req.session else None)
            or req.model
            or "gpt-4o-realtime-preview"
        )

        data = {"model": model}
        
        # If session is provided, use it; otherwise create one from model
        if req.session:
            data["session"] = req.session.model_dump(exclude_none=True)
        elif req.model:
            # User provided model at root level, convert to session format
            data["session"] = {"type": "realtime", "model": model}
        
        if req.expires_after:
            data["expires_after"] = req.expires_after.model_dump(exclude_none=True)

        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            data=data,
            call_type="acreate_realtime_client_secret",
        )

        verbose_proxy_logger.debug(
            "WebRTC: /v1/realtime/client_secrets (model=%s)", model
        )

        llm_call = await route_request(
            data=data,
            route_type="acreate_realtime_client_secret",
            llm_router=llm_router,
            user_model=user_model,
        )
        upstream_resp: httpx.Response = await llm_call  # type: ignore

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=data,
        )
        verbose_proxy_logger.error(
            "litellm.proxy.realtime_endpoints.webrtc.create_realtime_client_secret(): Exception - %s",
            str(e),
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", http_status.HTTP_400_BAD_REQUEST),
            )
        raise ProxyException(
            message=getattr(e, "message", str(e)),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )

    if upstream_resp.status_code != 200:
        verbose_proxy_logger.error(
            "WebRTC client_secrets upstream error %s: %s",
            upstream_resp.status_code,
            upstream_resp.text,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )

    upstream_json: dict = upstream_resp.json()

    # Encrypt upstream ephemeral key with routing metadata so /realtime/calls
    # can recover model without requiring query params.
    raw_value: str = upstream_json.get("value", "")
    expires_at = upstream_json.get("expires_at")
    token_payload = _encode_realtime_token_payload(
        ephemeral_key=raw_value,
        model_id=model,
        user_id=getattr(user_api_key_dict, "user_id", None),
        team_id=getattr(user_api_key_dict, "team_id", None),
        expires_at=expires_at if isinstance(expires_at, int) else None,
    )
    encrypted_token: str = encrypt_value_helper(token_payload)
    upstream_json["value"] = encrypted_token

    session_obj: Optional[dict] = upstream_json.get("session")
    if isinstance(session_obj, dict):
        cs = session_obj.get("client_secret")
        if isinstance(cs, dict) and "value" in cs:
            cs["value"] = encrypted_token
        upstream_json["session"] = session_obj

    return RealtimeClientSecretResponse(**upstream_json)


@router.post(
    "/v1/realtime/calls",
    tags=["realtime"],
)
@router.post(
    "/realtime/calls",
    tags=["realtime"],
)
@router.post(
    "/openai/v1/realtime/calls",
    tags=["realtime"],
)
async def proxy_realtime_calls(
    request: Request,
    fastapi_response: Response,
) -> Response:
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        route_request,
        user_model,
        version,
    )

    # Auth: the Bearer token is the encrypted ephemeral key issued by
    # /realtime/client_secrets, not a standard proxy API key.
    auth_header: Optional[str] = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return Response(
            content=json.dumps({"error": "Missing or invalid Authorization header"}),
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            media_type="application/json",
        )

    encrypted_token = auth_header.removeprefix("Bearer ").strip()
    decrypted_token_value = decrypt_value_helper(
        value=encrypted_token,
        key="realtime_calls_auth",
    )
    if not decrypted_token_value:
        return Response(
            content=json.dumps({"error": "Invalid or expired token"}),
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            media_type="application/json",
        )

    sdp_body: bytes = await request.body()
    decoded_payload = _decode_realtime_token_payload(decrypted_token_value)
    if decoded_payload is not None:
        # Check token expiry
        expires_at = decoded_payload.get("expires_at")
        if expires_at is not None and isinstance(expires_at, int):
            if time.time() > expires_at:
                return Response(
                    content=json.dumps({"error": "Token has expired"}),
                    status_code=http_status.HTTP_401_UNAUTHORIZED,
                    media_type="application/json",
                )
        
        openai_ephemeral_key = decoded_payload.get("ephemeral_key", "")
        model = (
            decoded_payload.get("model_id")
            or request.query_params.get("model")
            or "gpt-4o-realtime-preview"
        )
        user_id = decoded_payload.get("user_id") or None
        team_id = decoded_payload.get("team_id") or None
    else:
        # Backward compatibility: older tokens contained only encrypted upstream key.
        openai_ephemeral_key = decrypted_token_value
        model = request.query_params.get("model", "gpt-4o-realtime-preview")
        user_id = None
        team_id = None

    # Build a minimal UserAPIKeyAuth with user/team IDs from the token
    # so spend tracking and budget enforcement work correctly.
    minimal_auth = UserAPIKeyAuth(
        user_id=user_id,
        team_id=team_id,
    )

    data: dict = {}
    try:
        # Build session config for the multipart form data
        session_config = {
            "type": "realtime",
            "model": model,
        }

        data = {
            "model": model,
            "openai_ephemeral_key": openai_ephemeral_key,
            "sdp_body": sdp_body,
            "session": session_config,
        }

        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=minimal_auth,
            version=version,
            proxy_config=proxy_config,
        )

        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=minimal_auth,
            data=data,
            call_type="arealtime_calls",
        )

        verbose_proxy_logger.debug(
            "WebRTC: /v1/realtime/calls (model=%s)", model
        )

        llm_call = await route_request(
            data=data,
            route_type="arealtime_calls",
            llm_router=llm_router,
            user_model=user_model,
        )
        upstream_resp: httpx.Response = await llm_call  # type: ignore

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=minimal_auth,
            original_exception=e,
            request_data=data,
        )
        verbose_proxy_logger.error(
            "litellm.proxy.realtime_endpoints.webrtc.proxy_realtime_calls(): Exception - %s",
            str(e),
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", http_status.HTTP_400_BAD_REQUEST),
            )
        raise ProxyException(
            message=getattr(e, "message", str(e)),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "application/sdp"),
    )
