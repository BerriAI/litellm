import asyncio
import traceback
from typing import Optional, Union, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
    get_litellm_metadata_from_kwargs,
)
from litellm.proxy.auth.auth_checks import log_db_metrics
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_end_user_id_for_cost_tracking


@log_db_metrics
async def _PROXY_track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response: litellm.ModelResponse,  # response from completion
    start_time=None,
    end_time=None,  # start/end time for completion
):
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        update_cache,
        update_database,
    )

    verbose_proxy_logger.debug("INSIDE _PROXY_track_cost_callback")
    try:
        verbose_proxy_logger.debug(
            f"kwargs stream: {kwargs.get('stream', None)} + complete streaming response: {kwargs.get('complete_streaming_response', None)}"
        )
        parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs=kwargs)
        litellm_params = kwargs.get("litellm_params", {}) or {}
        end_user_id = get_end_user_id_for_cost_tracking(litellm_params)
        metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs)
        user_id = cast(Optional[str], metadata.get("user_api_key_user_id", None))
        team_id = cast(Optional[str], metadata.get("user_api_key_team_id", None))
        org_id = cast(Optional[str], metadata.get("user_api_key_org_id", None))
        key_alias = cast(Optional[str], metadata.get("user_api_key_alias", None))
        end_user_max_budget = metadata.get("user_api_end_user_max_budget", None)
        sl_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        response_cost = (
            sl_object.get("response_cost", None)
            if sl_object is not None
            else kwargs.get("response_cost", None)
        )

        if response_cost is not None:
            user_api_key = metadata.get("user_api_key", None)
            if kwargs.get("cache_hit", False) is True:
                response_cost = 0.0
                verbose_proxy_logger.info(
                    f"Cache Hit: response_cost {response_cost}, for user_id {user_id}"
                )

            verbose_proxy_logger.debug(
                f"user_api_key {user_api_key}, prisma_client: {prisma_client}"
            )
            if _should_track_cost_callback(
                user_api_key=user_api_key,
                user_id=user_id,
                team_id=team_id,
                end_user_id=end_user_id,
            ):
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
                if sl_object is not None:
                    cost_tracking_failure_debug_info: Union[dict, str] = (
                        sl_object["response_cost_failure_debug_info"]  # type: ignore
                        or "response_cost_failure_debug_info is None in standard_logging_object"
                    )
                else:
                    cost_tracking_failure_debug_info = (
                        "standard_logging_object not found"
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
        verbose_proxy_logger.exception("Error in tracking cost callback - %s", str(e))


def _should_track_cost_callback(
    user_api_key: Optional[str],
    user_id: Optional[str],
    team_id: Optional[str],
    end_user_id: Optional[str],
) -> bool:
    """
    Determine if the cost callback should be tracked based on the kwargs
    """
    if (
        user_api_key is not None
        or user_id is not None
        or team_id is not None
        or end_user_id is not None
    ):
        return True
    return False
