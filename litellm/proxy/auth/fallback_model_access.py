"""
Authorize router fallback targets against LiteLLM and external model-access policies.

`_enforce_key_and_fallback_model_access` only sees fallbacks the client sends in the request body.
Fallbacks configured on the router (`router_settings.fallbacks` and friends) are chosen after auth,
inside the router, so this predicate is injected into the router to re-run the same model access
checks for each fallback target before it is attempted. LiteLLM's built-in fallback check is
opt-in; configured Oso authorization applies automatically.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.oso_authorization import OsoAuthorizer, enforce_oso_model_authorization
from litellm.router import Router


class _RequestMetadata(BaseModel):
    user_api_key_auth: UserAPIKeyAuth | None = None


class _FallbackAccessSettings(BaseModel):
    enforce_fallback_model_access: bool = False


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
    except Exception as e:  # noqa: BLE001  # fail closed: a lookup failure must neither run the fallback nor replace the provider error
        verbose_proxy_logger.warning("Skipping fallback to model=%s: authorization lookup failed: %s", model, e)
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


def _general_settings() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import general_settings

    return general_settings


def _enforced_by_general_settings() -> bool:
    return _FallbackAccessSettings.model_validate(_general_settings()).enforce_fallback_model_access


@dataclass(frozen=True, slots=True)
class RouterFallbackAccessCheck:
    """
    `FallbackAccessCheck` for the proxy's router.

    The built-in access check is optional. Oso checks every target whenever its integration is
    enabled, including requests without usable identity, which fail closed.
    """

    is_enforced: Callable[[], bool]
    general_settings: Callable[[], Mapping[str, object]] = _general_settings
    oso_authorizer: OsoAuthorizer | None = None

    async def __call__(self, *, model: str, request_kwargs: Mapping[str, object], llm_router: Router) -> bool:
        valid_token: Final = _user_api_key_auth_from_request(request_kwargs)
        if self.is_enforced() and valid_token is not None:
            allowed_by_litellm: Final = await is_model_authorized_for_token(
                model=model,
                valid_token=valid_token,
                llm_router=llm_router,
            )
            if not allowed_by_litellm:
                return False
        try:
            await enforce_oso_model_authorization(
                general_settings=self.general_settings(),
                valid_token=valid_token or UserAPIKeyAuth(),
                model=model,
                route="/router/fallback",
                request_method="POST",
                authorizer=self.oso_authorizer,
            )
        except ProxyException as e:
            verbose_proxy_logger.info("Skipping fallback to model=%s: Oso authorization rejected it: %s", model, e)
            return False
        return True


router_fallback_access_check: Final = RouterFallbackAccessCheck(is_enforced=_enforced_by_general_settings)
