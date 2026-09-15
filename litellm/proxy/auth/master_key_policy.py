from collections.abc import Mapping, Sequence
from typing import Final, Literal

from fastapi import Request
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm.proxy._types import LiteLLMRoutes, ProxyErrorTypes, ProxyException
from litellm.proxy.auth.route_checks import RouteChecks

INSECURE_MASTER_KEYS: Final = frozenset({"sk-1234"})

InsecureMasterKeyReason = Literal["example_key", "missing"]

_ALTERNATIVE_AUTH_SETTINGS: Final = ("enable_jwt_auth", "enable_oauth2_auth", "enable_oauth2_proxy_auth", "custom_auth")

LockoutAction = Literal["store_credentials", "access_credentials", "use_credentials", "manage_virtual_keys"]

MASTER_KEY_LOCKOUT_MESSAGE: Final = (
    "This functionality is unavailable until the master key has been set. "
    "Set LITELLM_MASTER_KEY (or general_settings.master_key) to a strong random key and restart the proxy."
)

_STORE_CREDENTIAL_ROUTES: Final = (
    "/credentials",
    "/credentials/{credential_name:path}",
    "/model/new",
    "/model/update",
    "/model/{model_id}/update",
    "/config/update",
)


class _ModelInfoMarker(TypedDict, total=False):
    db_model: ReadOnly[bool]


class _DeploymentMarker(TypedDict, total=False):
    model_info: ReadOnly[_ModelInfoMarker]


_DEPLOYMENT_MARKERS: Final = TypeAdapter(list[_DeploymentMarker])

_ACCESS_CREDENTIAL_ROUTES: Final = (
    "/credentials",
    "/credentials/by_name/{credential_name:path}",
    "/credentials/by_model/{model_id}",
    "/model/info",
    "/v1/model/info",
    "/v2/model/info",
    "/get/config/callbacks",
    "/config/list",
    "/config/field/info",
)


def _route_matches_any(route: str, patterns: Sequence[str]) -> bool:
    return any(
        route == pattern or RouteChecks.route_matches_pattern(route=route, pattern=pattern) for pattern in patterns
    )


def alternative_auth_enabled(general_settings: Mapping[str, object]) -> bool:
    return any(general_settings.get(k, False) for k in _ALTERNATIVE_AUTH_SETTINGS)


def insecure_master_key_reason(
    master_key: str | None, alternative_auth_enabled: bool
) -> InsecureMasterKeyReason | None:
    if master_key in INSECURE_MASTER_KEYS:
        return "example_key"
    if (master_key is None or master_key == "") and not alternative_auth_enabled:
        return "missing"
    return None


def insecure_master_key_warning(master_key: str | None, alternative_auth_enabled: bool) -> str | None:
    reason: Final = insecure_master_key_reason(master_key, alternative_auth_enabled)
    match reason:
        case "example_key":
            return (
                "LITELLM_MASTER_KEY is set to the example key 'sk-1234' from the docs. "
                "Anyone who has read the docs can administer this gateway, and publicly reachable "
                "gateways using this key have been compromised. Set a strong random master key "
                "(e.g. `python -c \"import secrets; print('sk-' + secrets.token_urlsafe(32))\"`). "
                "Storing and using upstream credentials and managing virtual keys are disabled "
                "until a strong master key is set."
            )
        case "missing":
            return (
                "No master key is set (LITELLM_MASTER_KEY or general_settings.master_key). "
                "Every request to this proxy is accepted without authentication, including "
                'admin routes. Set a strong random master key (e.g. `python -c "import secrets; '
                "print('sk-' + secrets.token_urlsafe(32))\"`) before exposing it to a network. "
                "Storing and using upstream credentials and managing virtual keys are disabled "
                "until a strong master key is set."
            )
        case None:
            return None
    assert_never(reason)


def stored_credentials_present() -> bool:
    import litellm

    if litellm.credential_list:
        return True
    from litellm import Router
    from litellm.proxy.proxy_server import llm_router

    if not isinstance(llm_router, Router):
        return False
    deployments: Final = _DEPLOYMENT_MARKERS.validate_python(
        llm_router.model_list or []  # pyright: ignore[reportUnknownMemberType]  # Router.model_list is declared bare `list`; elements are validated by the TypeAdapter
    )
    return any((d.get("model_info") or {}).get("db_model") is True for d in deployments)


def request_http_method(request: Request) -> str:
    try:
        method: Final = request.method
    except (KeyError, AttributeError):
        return ""
    return method if isinstance(method, str) else ""


def master_key_lockout_action(
    route: str,
    method: str,
    reason: InsecureMasterKeyReason | None,
    stored_credentials_present: bool,
) -> LockoutAction | None:
    if reason is None:
        return None
    if method.upper() != "GET" and _route_matches_any(route, _STORE_CREDENTIAL_ROUTES):
        return "store_credentials"
    if not stored_credentials_present:
        return None
    if method.upper() == "GET" and _route_matches_any(route, _ACCESS_CREDENTIAL_ROUTES):
        return "access_credentials"
    if method.upper() != "GET" and RouteChecks.is_llm_api_route(route=route):
        return "use_credentials"
    if method.upper() != "GET" and _route_matches_any(route, tuple(LiteLLMRoutes.key_management_routes.value)):
        return "manage_virtual_keys"
    return None


def master_key_lockout_exception() -> ProxyException:
    return ProxyException(
        message=MASTER_KEY_LOCKOUT_MESSAGE,
        type=ProxyErrorTypes.auth_error,
        param="master_key",
        code=403,
    )
