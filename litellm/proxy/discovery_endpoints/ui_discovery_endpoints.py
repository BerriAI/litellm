#### Analytics Endpoints #####
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

    return UiDiscoveryEndpoints(
        server_root_path=get_server_root_path(),
        proxy_base_url=get_proxy_base_url(),
    )
