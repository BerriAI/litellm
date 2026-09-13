from pydantic import BaseModel

from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry


class UiDiscoveryEndpoints(BaseModel):
    server_root_path: str
    proxy_base_url: str | None
    auto_redirect_to_sso: bool
    admin_ui_disabled: bool
    sso_configured: bool
    hide_default_credentials_hint: bool = False
    is_control_plane: bool = False
    workers: list[WorkerRegistryEntry] = []
