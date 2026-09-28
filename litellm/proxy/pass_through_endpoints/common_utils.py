from collections.abc import Mapping
from typing import Final

from fastapi import Request
from pydantic import JsonValue

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.callback_utils import CALLBACK_VAR_ENCRYPTED_PREFIX
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper

# Holds the name of the request header that carries the caller's LiteLLM key, which
# user_api_key_auth reads straight from general_settings, so it stays plaintext.
_CALLER_KEY_HEADER_NAME: Final = "litellm_user_api_key"


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key: Final = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")


def _encrypt_header_value(name: str, value: JsonValue, new_encryption_key: str | None) -> JsonValue:
    if not isinstance(value, str) or not value or name == _CALLER_KEY_HEADER_NAME:
        return value
    if value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX):
        return value
    try:
        return CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper(value, new_encryption_key=new_encryption_key)
    except Exception:  # noqa: BLE001  # no salt or master key configured: store as before rather than fail the write
        return value


def _decrypted(name: str, value: str) -> str | None:
    return decrypt_value_helper(
        value=value.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX),
        key=name,
        exception_type="debug",
        return_original_value=False,
    )


def _decrypt_header_value(name: str, value: object) -> object:
    if not isinstance(value, str) or not value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX):
        return value
    decrypted: Final = _decrypted(name, value)
    if decrypted is None:
        verbose_proxy_logger.warning(
            "Could not decrypt pass-through header %s; check LITELLM_SALT_KEY / master key", name
        )
        return value
    return decrypted


def undecryptable_pass_through_header_names(headers: object) -> frozenset[str]:
    """Names of `litellm_enc::` header values that do not decrypt under the current key."""
    if not isinstance(headers, Mapping):
        return frozenset()
    return frozenset(
        str(name)
        for name, value in headers.items()
        if isinstance(value, str)
        and value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX)
        and _decrypted(str(name), value) is None
    )


def decrypt_pass_through_headers(headers: Mapping[str, object] | None) -> dict[str, object] | None:
    """Decrypt `litellm_enc::` header values; plaintext and os.environ/ values pass through."""
    if headers is None:
        return None
    return {name: _decrypt_header_value(name, value) for name, value in headers.items()}


def _with_encrypted_headers(endpoint: JsonValue, new_encryption_key: str | None, reencrypt: bool) -> JsonValue:
    if not isinstance(endpoint, dict) or not isinstance(headers := endpoint.get("headers"), dict):
        return endpoint
    return {
        **endpoint,
        "headers": {
            name: _encrypt_header_value(
                name, _decrypt_header_value(name, value) if reencrypt else value, new_encryption_key
            )
            for name, value in headers.items()
        },
    }


def encrypt_pass_through_endpoints(endpoints: JsonValue) -> JsonValue:
    """Encrypt every endpoint's header values for storage; values already carrying the marker are kept."""
    if not isinstance(endpoints, list):
        return endpoints
    return [_with_encrypted_headers(endpoint, None, reencrypt=False) for endpoint in endpoints]


def reencrypt_general_settings_pass_through(
    general_settings: object, new_encryption_key: str
) -> dict[str, JsonValue] | None:
    """Return general_settings with pass-through header values re-encrypted under new_encryption_key.

    None when the row holds no pass-through endpoint list.
    """
    if not isinstance(general_settings, dict) or not isinstance(
        endpoints := general_settings.get("pass_through_endpoints"), list
    ):
        return None
    return {
        **general_settings,
        "pass_through_endpoints": [
            _with_encrypted_headers(endpoint, new_encryption_key, reencrypt=True) for endpoint in endpoints
        ],
    }
