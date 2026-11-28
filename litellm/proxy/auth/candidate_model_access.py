"""
Authorize the deployments a request resolved to against the caller's key, team, user and agent model access.

Every model access check keys by a public name: a model group, a team's public model name or an access
group of them. A request may also name a deployment by its `litellm_params.model`, which no check keys
by, so auth only admits such a name when the caller is granted a deployment it selects, and the router
then asks `RouterCandidateAccessCheck` which of the deployments it resolved the caller may use. A
deployment is kept when the caller could have requested its public name directly.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.routing_common_utils import names_deployment_selector
from litellm.proxy._types import (
    LiteLLM_UserTable,
    ModelAccessDeniedProxyException,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    can_agent_call_model,
    can_key_call_resolved_model,
    can_user_call_model,
    deployment_public_name,
    get_user_object,
    key_model_aliases_for_auth_check,
)
from litellm.proxy.auth.auth_exception_handler import (
    _as_proxy_exception,  # pyright: ignore[reportPrivateUsage]  # one auth-failure mapping, shared with user_api_key_auth
)
from litellm.proxy.auth.fallback_model_access import (
    _user_api_key_auth_from_request,  # pyright: ignore[reportPrivateUsage]  # one reader of the request's key, shared with the fallback check
)
from litellm.proxy.auth.model_access_denied import model_access_denied_client_message
from litellm.router_api import RegistryReader, RoutingContext


async def key_owner(user_api_key_dict: UserAPIKeyAuth) -> LiteLLM_UserTable | None:
    if user_api_key_dict.team_id is not None or user_api_key_dict.user_id is None:
        return None
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    try:
        return await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # auth reads an unloadable user as no user limit, so /health does too
        return None


async def key_may_call_model(
    model_name: str,
    user_api_key_dict: UserAPIKeyAuth,
    owner: LiteLLM_UserTable | None,
    llm_router: RegistryReader | None,
) -> bool:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    await asyncio.sleep(0)
    try:
        await can_key_call_resolved_model(
            model=model_name, llm_model_list=None, valid_token=user_api_key_dict, llm_router=llm_router
        )
        await can_agent_call_model(
            model=model_name,
            valid_token=user_api_key_dict,
            llm_router=llm_router,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        await can_user_call_model(
            model=model_name,
            llm_router=llm_router,
            user_object=owner,
            key_model_aliases=key_model_aliases_for_auth_check(user_api_key_dict),
        )
    except ModelAccessDeniedProxyException:
        return False
    except ProxyException:
        raise
    except Exception as failure:
        if isinstance(failure, HTTPException) and failure.status_code == status.HTTP_403_FORBIDDEN:
            return False
        raise _as_proxy_exception(failure) from failure
    return True


async def _is_granted(
    model: str, valid_token: UserAPIKeyAuth, owner: LiteLLM_UserTable | None, llm_router: RegistryReader
) -> bool:
    try:
        return await key_may_call_model(model, valid_token, owner, llm_router)
    except Exception as e:  # noqa: BLE001  # fail closed: a lookup failure must not serve a deployment the caller may lack
        verbose_proxy_logger.warning("Withholding model=%s: authorization lookup failed: %s", model, e)
        return False


def deployment_selector_denied(model: str) -> ModelAccessDeniedProxyException:
    return ModelAccessDeniedProxyException(
        message=model_access_denied_client_message(model=model),
        internal_message=f"key not allowed to access model. No deployment of {model} is accessible to this key",
        type=ProxyErrorTypes.key_model_access_denied,
        param="model",
        code=status.HTTP_403_FORBIDDEN,
    )


async def refuse_ungranted_deployment_selector(
    request_kwargs: Mapping[str, object], caller: UserAPIKeyAuth | None, llm_router: RegistryReader
) -> None:
    """
    For a dispatch that skips the router's candidate stage: a name auth admitted only because it selects a
    deployment the caller is granted is refused there, since nothing would keep it to that deployment.
    """
    model: Final = request_kwargs.get("model")
    valid_token: Final = caller or _user_api_key_auth_from_request(request_kwargs)
    if valid_token is None or not isinstance(model, str):
        return
    if not names_deployment_selector(model, llm_router.registry()):
        return
    if not await _is_granted(model, valid_token, await key_owner(valid_token), llm_router):
        raise deployment_selector_denied(model)


@dataclass(frozen=True, slots=True)
class RouterCandidateAccessCheck:
    """
    `CandidateAccessCheck` for the proxy's router. A key that could have requested `model` itself keeps
    every candidate. Any other key keeps the candidates whose public name it could have requested.
    Requests that carry no key (for example internal health checks) are not restricted.
    """

    async def __call__(
        self, *, model: str, candidate_ids: Sequence[str], request_kwargs: Mapping[str, object], ctx: RoutingContext
    ) -> Sequence[str] | None:
        valid_token: Final = _user_api_key_auth_from_request(request_kwargs)
        if valid_token is None:
            return None
        owner: Final = await key_owner(valid_token)
        if await _is_granted(model, valid_token, owner, ctx):
            return None
        registry: Final = ctx.registry()
        public_names: Final = {
            candidate_id: deployment_public_name(registry.deployment_dict(candidate_id))
            for candidate_id in candidate_ids
        }
        granted: Final = frozenset(
            [
                name
                for name in dict.fromkeys(public_names.values())
                if name is not None and await _is_granted(name, valid_token, owner, ctx)
            ]
        )
        return tuple(candidate_id for candidate_id in candidate_ids if public_names[candidate_id] in granted)

    def denied(self, model: str) -> Exception:
        return deployment_selector_denied(model)


router_candidate_access_check: Final = RouterCandidateAccessCheck()
