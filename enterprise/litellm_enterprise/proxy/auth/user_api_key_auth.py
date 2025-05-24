from typing import Any, Optional

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth


async def enterprise_custom_auth(
    request: Request, api_key: str, user_custom_auth: Optional[Any]
) -> Optional[UserAPIKeyAuth]:
    from litellm_enterprise.proxy.proxy_server import custom_auth_settings

    if user_custom_auth is None:
        return None

    if custom_auth_settings is None:
        return await user_custom_auth(request, api_key)

    if custom_auth_settings["mode"] == "on":
        return await user_custom_auth(request, api_key)
    elif custom_auth_settings["mode"] == "off":
        return None
    elif custom_auth_settings["mode"] == "auto":
        try:
            return await user_custom_auth(request, api_key)
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error in custom auth, checking litellm auth: {e}"
            )
            return None
    else:
        raise ValueError(f"Invalid mode: {custom_auth_settings['mode']}")
