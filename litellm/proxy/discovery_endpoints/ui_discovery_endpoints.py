#### Analytics Endpoints #####
import os
from fastapi import APIRouter

from litellm.types.proxy.discovery_endpoints.ui_discovery_endpoints import (
    UiDiscoveryEndpoints,
)

router = APIRouter()


@router.get("/.well-known/litellm-ui-config", response_model=UiDiscoveryEndpoints)
@router.get(
    "/litellm/.well-known/litellm-ui-config", response_model=UiDiscoveryEndpoints
)  # if mounted at root path
async def get_ui_config():
    from litellm.proxy.utils import get_proxy_base_url, get_server_root_path
    from litellm.proxy.auth.auth_utils import _has_user_setup_sso

    auto_redirect_ui_login_to_sso = os.getenv("AUTO_REDIRECT_UI_LOGIN_TO_SSO", "true").lower() == "true"

    return UiDiscoveryEndpoints(
        server_root_path=get_server_root_path(),
        proxy_base_url=get_proxy_base_url(),
        auto_redirect_to_sso=_has_user_setup_sso() and auto_redirect_ui_login_to_sso,
    )
