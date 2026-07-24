from __future__ import annotations

from typing import cast  # noqa: TID251, RUF100  # Callback metadata requires runtime boundary narrowing.

from litellm._logging import verbose_proxy_logger
from litellm.proxy.public_relay.config import PublicRelaySettings
from litellm.proxy.public_relay.content_crypto import encrypt_content
from litellm.proxy.public_relay.repository import store_request_content


async def protect_public_content(kwargs: dict[str, object]) -> dict[str, object]:
    reservation = _reservation(kwargs)
    if reservation is None:
        return kwargs
    value = PublicRelaySettings.from_env()
    content = _content(kwargs)
    if _content_logging_enabled(kwargs) and value.content_encryption_key is not None:
        request_id = _required_string(reservation, "request_id")
        encrypted = encrypt_content(value.content_encryption_key, request_id, content)
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is not None:
            try:
                await store_request_content(
                    prisma_client,
                    request_id,
                    _required_string(reservation, "account_id"),
                    value.content_encryption_key_version,
                    encrypted.nonce_b64,
                    encrypted.ciphertext_b64,
                    value.content_retention_days,
                )
            except Exception:  # noqa: BLE001  # Logging failure must not fail the upstream model response.
                verbose_proxy_logger.exception("Failed to store encrypted public relay content")
    return _sanitized_kwargs(kwargs)


def _reservation(kwargs: dict[str, object]) -> dict[str, object] | None:
    params = _object_dict(kwargs.get("litellm_params"))
    metadata = _object_dict(params.get("metadata")) if params is not None else None
    return _object_dict(metadata.get("public_relay_reservation")) if metadata is not None else None


def _content_logging_enabled(kwargs: dict[str, object]) -> bool:
    params = _object_dict(kwargs.get("litellm_params"))
    metadata = _object_dict(params.get("metadata")) if params is not None else None
    if metadata is None:
        return False
    key_metadata = _object_dict(metadata.get("user_api_key_auth_metadata"))
    return key_metadata is not None and key_metadata.get("public_relay_log_content") is not False


def _content(kwargs: dict[str, object]) -> dict[str, object]:
    params = _object_dict(kwargs.get("litellm_params"))
    proxy_request = _object_dict(params.get("proxy_server_request")) if params is not None else None
    request_body: object = proxy_request.get("body") if proxy_request is not None else None
    standard = _object_dict(kwargs.get("standard_logging_object"))
    response = standard.get("response") if standard is not None else None
    return {"request": request_body, "response": response}


def _sanitized_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    sanitized = dict(kwargs)
    standard = _object_dict(kwargs.get("standard_logging_object"))
    if standard is not None:
        sanitized_standard = dict(standard)
        sanitized_standard["messages"] = None
        sanitized_standard["response"] = None
        sanitized["standard_logging_object"] = sanitized_standard
    params = _object_dict(kwargs.get("litellm_params"))
    if params is not None:
        sanitized_params = dict(params)
        proxy_request = _object_dict(params.get("proxy_server_request"))
        if proxy_request is not None:
            sanitized_request = dict(proxy_request)
            sanitized_request["body"] = {}
            sanitized_params["proxy_server_request"] = sanitized_request
        sanitized["litellm_params"] = sanitized_params
    for key in ("messages", "input", "prompt"):
        sanitized.pop(key, None)
    return sanitized


def _required_string(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise TypeError(f"{key} is missing")
    return result


def _object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)  # cast-ok: isinstance validates the callback mapping boundary.
