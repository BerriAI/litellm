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
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    can_key_call_resolved_model,
    can_token_call_search_tool,
    dashboard_model_billing_auth,
)
from litellm.router_api import RegistryReader, RoutingContext
from litellm.search import asearch
from litellm.types.llms.base import LiteLLMBaseModel


class _RequestMetadata(LiteLLMBaseModel):
    user_api_key_auth: UserAPIKeyAuth | None = None


class _FallbackAccessSettings(LiteLLMBaseModel):
    enforce_fallback_model_access: bool = True


@runtime_checkable
class _BillingMetadata(Protocol):
    def update(self, values: Mapping[str, object], /) -> None: ...


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


async def prepare_dashboard_fallback_billing(
    model: str, request_kwargs: Mapping[str, object], auth: UserAPIKeyAuth, ctx: RoutingContext
) -> bool:
    context: Final = auth.dashboard_session_context
    if context is None:
        return True
    billed: Final = (await dashboard_model_billing_auth(auth, model, ctx)).model_copy(
        update={"budget_reservation": None}
    )
    if billed.team_id == auth.team_id:
        return True
    request: Final = context.request
    if request is None:
        return False
    metadata: Final = _dashboard_billing_metadata(request_kwargs)
    if metadata is None:
        return False
    from litellm.proxy.auth.user_api_key_auth import run_centralized_common_checks
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.proxy.spend_tracking.budget_reservation import release_unbound_budget_reservation

    if auth.budget_reservation is not None:
        await release_unbound_budget_reservation(auth.budget_reservation)
    await run_centralized_common_checks(
        user_api_key_auth_obj=billed,
        request=request,
        request_data={**request_kwargs, "model": model},
        route=billed.request_route or request.url.path,
    )
    request.state.budget_reservation = billed.budget_reservation
    context.billing_auth.adopt_dashboard_billing(billed)
    auth.adopt_dashboard_billing(billed)
    metadata.update(
        MappingProxyType(
            {
                **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(billed),
                "user_api_key_auth": LiteLLMProxyRequestSetup.auth_for_request_metadata(billed),
                "user_api_key_budget_reservation": billed.budget_reservation,
            }
        )
    )
    return True


def _dashboard_billing_metadata(request_kwargs: Mapping[str, object]) -> _BillingMetadata | None:
    return next(
        (
            value
            for field in ("metadata", "litellm_metadata")
            if isinstance(value := request_kwargs.get(field), _BillingMetadata)
            and _token_in_metadata(value) is not None
        ),
        None,
    )


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
        if self.is_enforced() and not await is_model_authorized_for_token(
            model=model, valid_token=valid_token, llm_router=ctx
        ):
            return False
        try:
            return await prepare_dashboard_fallback_billing(model, request_kwargs, valid_token, ctx)
        except Exception as e:  # noqa: BLE001  # a fallback without its granting team's billing controls cannot run
            verbose_proxy_logger.warning("Skipping fallback to model=%s: billing admission failed: %s", model, e)
            return False


router_fallback_access_check: Final = RouterFallbackAccessCheck(is_enforced=_enforced_by_general_settings)
