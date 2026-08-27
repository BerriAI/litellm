"""
Authorize router fallback targets against the caller's key, team and project model access.

`_enforce_key_and_fallback_model_access` only sees fallbacks the client sends in the request body.
Fallbacks configured on the router (`router_settings.fallbacks` and friends) are chosen after auth,
inside the router, so this predicate is injected into the router to re-run the same model access
checks for each fallback target before it is attempted.
"""

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ValidationError

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.router import Router


class _RequestMetadata(BaseModel):
    user_api_key_auth: UserAPIKeyAuth | None = None


async def is_model_authorized_for_token(*, model: str, valid_token: UserAPIKeyAuth, llm_router: Router) -> bool:
    try:
        await can_key_call_resolved_model(
            model=model,
            llm_model_list=None,
            valid_token=valid_token,
            llm_router=llm_router,
        )
    except ProxyException:
        return False
    return True


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


async def router_fallback_access_check(*, model: str, request_kwargs: Mapping[str, object], llm_router: Router) -> bool:
    """
    `FallbackAccessCheck` for the proxy's router: a fallback target is attempted only when the
    key behind the request could have requested it directly. Requests that carry no key (for
    example internal health checks) are not restricted.
    """
    valid_token: Final = _user_api_key_auth_from_request(request_kwargs)
    if valid_token is None:
        return True
    return await is_model_authorized_for_token(model=model, valid_token=valid_token, llm_router=llm_router)
