from typing import Final

from pydantic import TypeAdapter, ValidationError

_AUTH_REQUIRED_SETTING: Final[TypeAdapter[bool | None]] = TypeAdapter(bool | None)


def marketplace_auth_required() -> bool:
    from litellm.proxy.proxy_server import general_settings_view

    try:
        return (
            _AUTH_REQUIRED_SETTING.validate_python(general_settings_view().get("claude_code_marketplace_auth_required"))
            is True
        )
    except ValidationError:
        return True
