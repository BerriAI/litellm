from typing import Final

from fastapi import Request
from pydantic import TypeAdapter, ValidationError

from litellm.types.proxy.auth.special_headers import SpecialHeaders

_AUTH_REQUIRED_SETTING: Final[TypeAdapter[bool | None]] = TypeAdapter(bool | None)


def marketplace_key_header_present(request: Request) -> bool:
    """Recognize header credentials before an optional-auth archive chooses anonymity."""
    from litellm.proxy.proxy_server import general_settings_view

    configured_header: Final = general_settings_view().get("litellm_key_header_name")
    return (
        SpecialHeaders.openai_authorization.value in request.headers
        or SpecialHeaders.custom_litellm_api_key.value in request.headers
        or (isinstance(configured_header, str) and configured_header in request.headers)
    )


def marketplace_auth_required() -> bool:
    from litellm.proxy.proxy_server import general_settings_view

    try:
        return (
            _AUTH_REQUIRED_SETTING.validate_python(general_settings_view().get("claude_code_marketplace_auth_required"))
            is True
        )
    except ValidationError:
        return True
