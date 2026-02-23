from typing import Optional

from pydantic import BaseModel


class UiDiscoveryEndpoints(BaseModel):
    server_root_path: str
    proxy_base_url: Optional[str]
    auto_redirect_to_sso: bool
    admin_ui_disabled: bool
    sso_configured: bool
