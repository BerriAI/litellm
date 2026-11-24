from typing import Final, Literal

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry

UI_CONFIG_PATH: Final = "/.well-known/litellm-ui-config"
LITELLM_UI_CONFIG_PATH: Final = f"/litellm{UI_CONFIG_PATH}"


class UiDiscoveryEndpoints(LiteLLMBaseModel):
    server_root_path: str
    proxy_base_url: str | None
    gateway_base_url: str | None = None
    auto_redirect_to_sso: bool
    admin_ui_disabled: bool
    sso_configured: bool
    hide_default_credentials_hint: bool = False
    is_control_plane: bool = False
    mcp_stdio_enabled: bool = False
    use_si_terminology: bool = False
    claude_code_marketplace_auth_required: bool = False
    workers: list[WorkerRegistryEntry] = Field(default=[])


class ControlPlaneUiDiscovery(UiDiscoveryEndpoints):
    model_config = ConfigDict(frozen=True)

    mode: Literal["control_plane_api"] = "control_plane_api"
    control_plane_api_prefix: str
    litellm_version: str | None = None
