"""
AUTO ROUTER MANAGEMENT ENDPOINTS

POST /auto_router/test_routing - Route one prompt through an unsaved complexity-router config
"""

from typing import TYPE_CHECKING, Annotated, Final

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _virtual_key_max_budget_check,
    can_key_call_resolved_model,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.repositories.team_repository import TeamRepository
from litellm.router_strategy.complexity_router import ComplexityRouter
from litellm.types.management_endpoints.auto_router_endpoints import (
    AutoRouterRoutingTestRequest,
    AutoRouterRoutingTestResponse,
    RequestComplexityRouterConfig,
)

if TYPE_CHECKING:
    from fastapi import APIRouter, Depends, HTTPException, status

    from litellm.router import Router
else:
    try:
        from fastapi import APIRouter, Depends, HTTPException, status
    except ImportError:
        # fastapi is only required for proxy, not for SDK usage
        pass

router: Final = APIRouter()


async def _authorize_routing_test(user_api_key_dict: UserAPIKeyAuth, team_id: str | None) -> None:
    """Allow exactly the callers who could create this router.

    Routing a prompt can spend money (an `llm` classifier config calls its classifier, a
    semantic config embeds the prompt), so this is gated like a write rather than a read:
    a proxy admin, or a team admin naming their own team, matching /model/new.
    """
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.proxy.proxy_server import premium_user, prisma_client

    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return

    if team_id is None:
        raise HTTPException(
            status_code=403,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping to keep this route's {"error": ...} response shape
                "error": f"User does not have permission to test an auto router. Your role={user_api_key_dict.user_role}. Test as a PROXY_ADMIN, or as a team admin by specifying a team_id."
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": CommonProxyErrors.db_not_connected_error.value
            },
        )

    team_row: Final = await TeamRepository(prisma_client).table.find_unique(
        where={"team_id": team_id},  # mutable-ok: Prisma query filters are dict-shaped
    )
    if team_row is None:
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": f"Team id={team_id} does not exist in db"
            },
        )

    ModelManagementAuthChecks.can_user_make_team_model_call(
        team_id=team_id,
        user_api_key_dict=user_api_key_dict,
        team_obj=LiteLLM_TeamTable.model_validate(team_row.model_dump()),
        premium_user=premium_user,
    )


def _models_this_test_can_call(config: RequestComplexityRouterConfig) -> tuple[str, ...]:
    """The models the routing test itself would send a request to, and so spend on.

    Excludes every tier's models: the prompt is never sent to the model it routed to.
    """
    return tuple(
        model
        for model in (
            config.classifier_llm_config.model
            if config.classifier_type == "llm" and config.classifier_llm_config is not None
            else None,
            config.embedding_model if config.semantic_keyword_matching else None,
        )
        if model is not None
    )


async def _authorize_models_this_test_can_call(
    config: RequestComplexityRouterConfig,
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router",
) -> None:
    """Hold a classifier or embedding call to the caller's model access and key budget.

    Those calls go through the router rather than through /v1/chat/completions, so the model
    checks a real request gets in user_api_key_auth would otherwise be skipped, letting a
    caller spend on a model their key cannot call, and this route is not an LLM API route, so
    the key's own budget is not checked either. Test Connection gets both for free by routing
    its calls through the proxy. Team and member budgets are already enforced on every route.
    """
    models: Final = _models_this_test_can_call(config)
    if not models:
        return

    from litellm.proxy.proxy_server import proxy_logging_obj

    for model in models:
        await can_key_call_resolved_model(
            model=model,
            llm_model_list=llm_router.model_list,
            valid_token=user_api_key_dict,
            llm_router=llm_router,
        )

    try:
        await _virtual_key_max_budget_check(
            valid_token=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )
    except BudgetExceededError as e:
        raise ProxyException(
            message=e.message,
            type=ProxyErrorTypes.budget_exceeded,
            param=None,
            code=status.HTTP_400_BAD_REQUEST,
        ) from e


@router.post(
    "/auto_router/test_routing",
    tags=["model management"],  # mutable-ok: fastapi's decorator signature types tags as a list
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: fastapi's decorator signature types dependencies as a list
    response_model=AutoRouterRoutingTestResponse,
    status_code=status.HTTP_200_OK,
)
async def preview_auto_router_routing(
    data: AutoRouterRoutingTestRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> AutoRouterRoutingTestResponse:
    """
    Route a single prompt through a complexity-router config and report where it landed.

    Answers "which model would this prompt get?" for a config that only exists in a form,
    so an auto router can be checked before it is created. The prompt is classified by the
    same pre-routing hook a live request runs, then dropped: nothing is sent to the model it
    routed to, and no auto router is created. A heuristic config therefore spends nothing, while
    an `llm` classifier or semantic keyword matching bills its classifier/embedding call to the
    calling key, like Test Connection does.

    **Example Request:**
    ```json
    {
        "prompt": "think step by step about how to shard this table",
        "complexity_router_config": {
            "tiers": {"SIMPLE": ["gpt-4o-mini"], "REASONING": ["o3"]},
            "classifier_type": "heuristic"
        }
    }
    ```
    """
    from litellm.proxy.proxy_server import llm_router

    await _authorize_routing_test(user_api_key_dict=user_api_key_dict, team_id=data.team_id)

    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": CommonProxyErrors.no_llm_router.value
            },
        )

    await _authorize_models_this_test_can_call(
        config=data.complexity_router_config,
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
    )

    complexity_router: Final = ComplexityRouter(
        model_name=data.router_name,
        litellm_router_instance=llm_router,
        complexity_router_config=data.complexity_router_config.model_dump(exclude_none=True),
        default_model=data.default_model,
        derive_savings_baseline=False,
    )

    request_kwargs: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data={"metadata": {}},  # mutable-ok: the request-metadata helper takes and returns request kwargs as a dict
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name="metadata",
    )

    try:
        hook_response: Final = await complexity_router.async_pre_routing_hook(
            model=data.router_name,
            request_kwargs=request_kwargs,
            messages=[  # mutable-ok: the routing hook's signature takes a list of message dicts
                {"role": "user", "content": data.prompt},  # mutable-ok: a message is dict-shaped
            ],
        )
    except Exception as e:  # noqa: BLE001 -- surfaces any classifier/plugin failure to the caller as a 400 instead of a 500, since the config under test is caller input
        verbose_proxy_logger.exception("Auto router routing test failed. Due to error - %s", e)
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": f"Could not route this prompt: {e}"
            },
        ) from e

    if hook_response is None or hook_response.routing_decision is None:
        raise HTTPException(
            status_code=400,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": "The router made no decision for this prompt. Check that at least one tier has a model."
            },
        )

    return AutoRouterRoutingTestResponse(
        routed_model=hook_response.model,
        routed_model_configured=hook_response.model in frozenset(llm_router.get_model_names()),
        routing_decision=hook_response.routing_decision,
    )
