"""
Proxy Success Callback - handles storing cost of a request in LiteLLM DB.

Updates cost for the following in LiteLLM DB:
    - spend logs 
    - virtual key spend 
    - internal user, team, external user spend
"""

import asyncio
import traceback
from typing import Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.proxy.utils import log_db_metrics
from litellm.types.utils import StandardLoggingPayload


@log_db_metrics
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
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs=kwargs)
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_payload is None:
            raise ValueError(
                "standard_logging_payload is none in kwargs, cannot track cost without it"
            )
        end_user_id = standard_logging_payload.get("end_user")
        user_api_key = standard_logging_payload.get("metadata", {}).get(
            "user_api_key_hash"
        )
        user_id = standard_logging_payload.get("metadata", {}).get(
            "user_api_key_user_id"
        )
        team_id = standard_logging_payload.get("metadata", {}).get(
            "user_api_key_team_id"
        )
        org_id = standard_logging_payload.get("metadata", {}).get("user_api_key_org_id")
        key_alias = standard_logging_payload.get("metadata", {}).get(
            "user_api_key_alias"
        )
        end_user_max_budget = standard_logging_payload.get("metadata", {}).get(
            "user_api_end_user_max_budget"
        )
        response_cost: Optional[float] = standard_logging_payload.get("response_cost")

        if response_cost is not None:
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
            cost_tracking_failure_debug_info = standard_logging_payload.get(
                "response_cost_failure_debug_info"
            )
            model = kwargs.get("model")
            raise ValueError(
                f"Failed to write cost to DB, for model={model}.\nDebug info - {cost_tracking_failure_debug_info}\nAdd custom pricing - https://docs.litellm.ai/docs/proxy/custom_pricing"
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
