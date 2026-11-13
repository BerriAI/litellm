"""
Authorize router fallback targets against the caller's key, team and project model access.

`_enforce_key_and_fallback_model_access` only sees fallbacks the client sends in the request body.
Fallbacks configured on the router (`router_settings.fallbacks` and friends) are chosen after auth,
inside the router, so this predicate is injected into the router to re-run the same model access
checks for each fallback target before it is attempted. Enabled by default. Opt out via
`general_settings.enforce_fallback_model_access: false`. Fallbacks of a search request always run the
key, team and user search tool grants instead, the same check /search runs on the requested tool.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UI_TEAM_ID, ModelAccessDeniedProxyException, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    can_key_call_resolved_model,
    can_token_call_search_tool,
    check_customer_model_access,
    dashboard_model_billing_auth,
)
from litellm.proxy.litellm_pre_call_utils import global_then_key_alias_target
from litellm.router_api import RegistryReader, RoutingContext
from litellm.search import asearch
from litellm.types.llms.base import LiteLLMBaseModel


class _RequestMetadata(LiteLLMBaseModel):
    user_api_key_auth: UserAPIKeyAuth | None = None


class _FallbackAccessSettings(LiteLLMBaseModel):
    enforce_fallback_model_access: bool = True


async def is_model_authorized_for_token(*, model: str, valid_token: UserAPIKeyAuth, llm_router: RegistryReader) -> bool:
    try:
        await can_key_call_resolved_model(
            model=model,
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=llm_router,
        )
    except ProxyException:
        return False
    except Exception as e:  # noqa: BLE001  # fail closed: a lookup failure must neither run the fallback nor replace the provider error
        verbose_proxy_logger.warning("Skipping fallback to model=%s: authorization lookup failed: %s", model, e)
        return False
    return True


async def is_search_tool_authorized_for_token(*, search_tool_name: str, valid_token: UserAPIKeyAuth) -> bool:
    try:
        await can_token_call_search_tool(search_tool_name=search_tool_name, valid_token=valid_token)
    except ProxyException:
        return False
    except Exception as e:  # noqa: BLE001  # fail closed: a lookup failure must neither run the fallback nor replace the provider error
        verbose_proxy_logger.warning(
            "Skipping fallback to search tool=%s: authorization lookup failed: %s", search_tool_name, e
        )
        return False
    return True


def _is_search_request(request_kwargs: Mapping[str, object]) -> bool:
    return request_kwargs.get("original_generic_function") is asearch


def _token_in_metadata(metadata: object) -> UserAPIKeyAuth | None:
    try:
        return _RequestMetadata.model_validate(metadata).user_api_key_auth
    except ValidationError:
        return None


def _user_api_key_auth_from_request(request_kwargs: Mapping[str, object]) -> UserAPIKeyAuth | None:
    return next(
        (
            token
            for field in ("metadata", "litellm_metadata")
            if (token := _token_in_metadata(request_kwargs.get(field))) is not None
        ),
        None,
    )


def _enforced_by_general_settings() -> bool:
    from litellm.proxy.proxy_server import general_settings

    return _FallbackAccessSettings.model_validate(general_settings).enforce_fallback_model_access


def _dashboard_model_group(model: str, auth: UserAPIKeyAuth, llm_router: RegistryReader) -> str:
    aliased: Final = global_then_key_alias_target(model, auth)
    return llm_router.registry().resolve(aliased, kind="alias") or aliased


async def dashboard_fallback_billing_allowed(model: str, auth: UserAPIKeyAuth, llm_router: RegistryReader) -> bool:
    if auth.team_id != UI_TEAM_ID and not isinstance(auth.dashboard_session_origin, UserAPIKeyAuth):
        return True
    original: Final = auth.dashboard_session_model
    if original is not None and _dashboard_model_group(original, auth, llm_router) == _dashboard_model_group(
        model, auth, llm_router
    ):
        return True
    try:
        billed: Final = await dashboard_model_billing_auth(auth, model, llm_router)
    except ModelAccessDeniedProxyException:
        return False
    return billed.team_id == auth.team_id


@dataclass(frozen=True, slots=True)
class RouterFallbackAccessCheck:
    """
    `FallbackAccessCheck` for the proxy's router: a fallback of a search request, and while `is_enforced()`
    is true any other fallback, is attempted only when the key behind the request could have requested it
    directly. Requests that carry no key (for example internal health checks) are not restricted.
    """

    is_enforced: Callable[[], bool]

    async def __call__(self, *, model: str, request_kwargs: Mapping[str, object], ctx: RoutingContext) -> bool:
        valid_token: Final = _user_api_key_auth_from_request(request_kwargs)
        if valid_token is None:
            return True
        if _is_search_request(request_kwargs):
            return await is_search_tool_authorized_for_token(search_tool_name=model, valid_token=valid_token)
        enforced: Final = self.is_enforced()
        if enforced and not await is_model_authorized_for_token(model=model, valid_token=valid_token, llm_router=ctx):
            return False
        if not enforced:
            try:
                check_customer_model_access(model=model, valid_token=valid_token, llm_router=ctx)
            except ProxyException:
                return False
        return await dashboard_fallback_billing_allowed(model, valid_token, ctx)


router_fallback_access_check: Final = RouterFallbackAccessCheck(is_enforced=_enforced_by_general_settings)
