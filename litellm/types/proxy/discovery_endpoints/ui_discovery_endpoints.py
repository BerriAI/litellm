from typing import List, Optional

from pydantic import BaseModel

from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry


class UiDiscoveryEndpoints(BaseModel):
    server_root_path: str
    proxy_base_url: Optional[str]
    auto_redirect_to_sso: bool
    admin_ui_disabled: bool
    sso_configured: bool
    is_control_plane: bool = False
    workers: List[WorkerRegistryEntry] = []
