#### Analytics Endpoints #####
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Final

from fastapi import APIRouter
from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.discovery_endpoints.ui_discovery_endpoints import (
    NativeOIDCConfig,
    UiDiscoveryEndpoints,
)

router: Final = APIRouter()

NATIVE_OIDC_SETTING_KEYS = (
    "native_oidc_issuer",
    "native_oidc_client_id",
    "native_oidc_scopes",
)

# Constant on purpose: this is reachable from an unauthenticated public route,
# so it must never echo the configured issuer, client id, scopes, or the raw
# Pydantic error (which would embed the rejected input values).
NATIVE_OIDC_INVALID_MESSAGE = (
    "native OIDC metadata was invalid or incomplete and was omitted from "
    "/.well-known/litellm-ui-config. Check general_settings.litellm_jwtauth "
    "native_oidc_issuer / native_oidc_client_id / native_oidc_scopes."
)


@lru_cache(maxsize=1)
def _warn_native_oidc_invalid_once() -> None:
    """Emit the sanitized diagnostic at most once per process.

    Deduplicated so that a public discovery request cannot be used to flood the
    proxy logs. The cache is the dedupe: the body runs only on the first call.
    """
    verbose_proxy_logger.warning(NATIVE_OIDC_INVALID_MESSAGE)


def _build_native_oidc_config(general_settings: Mapping[str, object]) -> NativeOIDCConfig | None:
    """Build the public native OIDC object, or return None and fail closed.

    Published only when JWT auth is exactly enabled and every required field
    validates. Absent configuration is silent; present-but-broken configuration
    warns once.
    """
    if general_settings.get("enable_jwt_auth") is not True:
        return None

    jwt_auth_settings = general_settings.get("litellm_jwtauth")
    if not isinstance(jwt_auth_settings, dict):
        return None

    if not any(jwt_auth_settings.get(key) is not None for key in NATIVE_OIDC_SETTING_KEYS):
        # Not configured at all -- nothing to warn about.
        return None

    try:
        # Validated rather than constructed: the settings are untyped YAML, and
        # the model is the single place that decides what a publishable issuer,
        # client_id and scope list look like.
        return NativeOIDCConfig.model_validate(
            {  # mutable-ok: literal handed straight to model_validate
                "issuer": jwt_auth_settings.get("native_oidc_issuer"),
                "client_id": jwt_auth_settings.get("native_oidc_client_id"),
                "scopes": jwt_auth_settings.get("native_oidc_scopes"),
            }
        )
    except ValidationError:
        _warn_native_oidc_invalid_once()
        return None


@router.get("/.well-known/litellm-ui-config", response_model=UiDiscoveryEndpoints)
@router.get("/litellm/.well-known/litellm-ui-config", response_model=UiDiscoveryEndpoints)  # if mounted at root path
async def get_ui_config():
    from litellm.proxy.auth.auth_utils import _has_user_setup_sso
    from litellm.proxy.proxy_server import general_settings
    from litellm.proxy.utils import get_proxy_base_url, get_server_root_path

    native_oidc: Final = _build_native_oidc_config(general_settings)

    auto_redirect_ui_login_to_sso: Final = (
        os.getenv("AUTO_REDIRECT_UI_LOGIN_TO_SSO", "false").lower() == "true"
        or general_settings.get("auto_redirect_ui_login_to_sso", False) is True
    )
    admin_ui_disabled: Final = os.getenv("DISABLE_ADMIN_UI", "false").lower() == "true"
    hide_default_credentials_hint: Final = bool(
        os.getenv("LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT", "false").lower() == "true"
        or general_settings.get("hide_default_credentials_hint", False) is True
    )

    sso_configured: Final = _has_user_setup_sso()

    from litellm.proxy.proxy_server import proxy_config

    is_control_plane: Final = len(proxy_config.worker_registry) > 0

    return UiDiscoveryEndpoints(
        server_root_path=get_server_root_path(),
        proxy_base_url=get_proxy_base_url(),
        auto_redirect_to_sso=sso_configured and auto_redirect_ui_login_to_sso,
        admin_ui_disabled=admin_ui_disabled,
        sso_configured=sso_configured,
        hide_default_credentials_hint=hide_default_credentials_hint,
        is_control_plane=is_control_plane,
        workers=proxy_config.worker_registry if is_control_plane else (),
        native_oidc=native_oidc,
    )
