"""
Proxy Success Callback - handles storing cost of a request in LiteLLM DB.

Updates cost for the following in LiteLLM DB:
    - spend logs 
    - virtual key spend 
    - internal user, team, external user spend
"""

import asyncio
import traceback

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import (
    _get_parent_otel_span_from_kwargs,
    get_litellm_metadata_from_kwargs,
    log_to_opentelemetry,
)


@log_to_opentelemetry
async def _PROXY_track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response: litellm.ModelResponse,  # response from completion
    start_time=None,
    end_time=None,  # start/end time for completion
):
    """
    Callback handles storing cost of a request in LiteLLM DB.

    Updates cost for the following in LiteLLM DB:
        - spend logs
        - virtual key spend
        - internal user, team, external user spend
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        update_cache,
        update_database,
    )

    verbose_proxy_logger.debug("INSIDE _PROXY_track_cost_callback")
    try:
        # check if it has collected an entire stream response
        verbose_proxy_logger.debug(
            "Proxy: In track_cost_callback for: kwargs=%s and completion_response: %s",
            kwargs,
            completion_response,
        )
        verbose_proxy_logger.debug(
            f"kwargs stream: {kwargs.get('stream', None)} + complete streaming response: {kwargs.get('complete_streaming_response', None)}"
        )
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs=kwargs)
        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs)
        user_id = metadata.get("user_api_key_user_id", None)
        team_id = metadata.get("user_api_key_team_id", None)
        org_id = metadata.get("user_api_key_org_id", None)
        key_alias = metadata.get("user_api_key_alias", None)
        end_user_max_budget = metadata.get("user_api_end_user_max_budget", None)
        if kwargs.get("response_cost", None) is not None:
            response_cost = kwargs["response_cost"]
            user_api_key = metadata.get("user_api_key", None)
            if kwargs.get("cache_hit", False) is True:
                response_cost = 0.0
                verbose_proxy_logger.info(
                    f"Cache Hit: response_cost {response_cost}, for user_id {user_id}"
                )

            verbose_proxy_logger.debug(
                f"user_api_key {user_api_key}, prisma_client: {prisma_client}"
            )
            if user_api_key is not None or user_id is not None or team_id is not None:
                ## UPDATE DATABASE
                await update_database(
                    token=user_api_key,
                    response_cost=response_cost,
                    user_id=user_id,
                    end_user_id=end_user_id,
                    team_id=team_id,
                    kwargs=kwargs,
                    completion_response=completion_response,
                    start_time=start_time,
                    end_time=end_time,
                    org_id=org_id,
                )

                # update cache
                asyncio.create_task(
                    update_cache(
                        token=user_api_key,
                        user_id=user_id,
                        end_user_id=end_user_id,
                        response_cost=response_cost,
                        team_id=team_id,
                        parent_otel_span=parent_otel_span,
                    )
                )

                await proxy_logging_obj.slack_alerting_instance.customer_spend_alert(
                    token=user_api_key,
                    key_alias=key_alias,
                    end_user_id=end_user_id,
                    response_cost=response_cost,
                    max_budget=end_user_max_budget,
                )
            else:
                raise Exception(
                    "User API key and team id and user id missing from custom callback."
                )
        else:
            if kwargs["stream"] is not True or (
                kwargs["stream"] is True and "complete_streaming_response" in kwargs
            ):
                cost_tracking_failure_debug_info = kwargs.get(
                    "response_cost_failure_debug_information"
                )
                model = kwargs.get("model")
                raise Exception(
                    f"Cost tracking failed for model={model}.\nDebug info - {cost_tracking_failure_debug_info}\nAdd custom pricing - https://docs.litellm.ai/docs/proxy/custom_pricing"
                )
    except Exception as e:
        error_msg = f"Error in tracking cost callback - {str(e)}\n Traceback:{traceback.format_exc()}"
        model = kwargs.get("model", "")
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        error_msg += f"\n Args to _PROXY_track_cost_callback\n model: {model}\n metadata: {metadata}\n"
        asyncio.create_task(
            proxy_logging_obj.failed_tracking_alert(
                error_message=error_msg,
                failing_model=model,
            )
        )
        verbose_proxy_logger.debug("error in tracking cost callback - %s", e)
