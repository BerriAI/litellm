import asyncio
import traceback
from datetime import datetime
from typing import Any, Optional, Union, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
    get_litellm_metadata_from_kwargs,
)
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import log_db_metrics
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.utils import ProxyUpdateSpend
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingUserAPIKeyMetadata,
)
from litellm.utils import get_end_user_id_for_cost_tracking


class _ProxyDBLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._PROXY_track_cost_callback(
            kwargs, response_obj, start_time, end_time
        )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ):
        request_route = user_api_key_dict.request_route
        if _ProxyDBLogger._should_track_errors_in_db() is False:
            return
        elif request_route is not None and not RouteChecks.is_llm_api_route(
            route=request_route
        ):
            return

        from litellm.proxy.proxy_server import proxy_logging_obj

        _metadata = dict(
            StandardLoggingUserAPIKeyMetadata(
                user_api_key_hash=user_api_key_dict.api_key,
                user_api_key_alias=user_api_key_dict.key_alias,
                user_api_key_spend=user_api_key_dict.spend,
                user_api_key_max_budget=user_api_key_dict.max_budget,
                user_api_key_budget_reset_at=user_api_key_dict.budget_reset_at.isoformat() if user_api_key_dict.budget_reset_at else None,
                user_api_key_user_email=user_api_key_dict.user_email,
                user_api_key_user_id=user_api_key_dict.user_id,
                user_api_key_team_id=user_api_key_dict.team_id,
                user_api_key_org_id=user_api_key_dict.org_id,
                user_api_key_team_alias=user_api_key_dict.team_alias,
                user_api_key_end_user_id=user_api_key_dict.end_user_id,
                user_api_key_request_route=user_api_key_dict.request_route,
            )
        )
        _metadata["user_api_key"] = user_api_key_dict.api_key
        _metadata["status"] = "failure"
        _metadata[
            "error_information"
        ] = StandardLoggingPayloadSetup.get_error_information(
            original_exception=original_exception,
            traceback_str=traceback_str,
        )

        existing_metadata: dict = request_data.get("metadata", None) or {}
        existing_metadata.update(_metadata)

        if "litellm_params" not in request_data:
            request_data["litellm_params"] = {}
        request_data["litellm_params"]["proxy_server_request"] = (
            request_data.get("proxy_server_request") or {}
        )
        request_data["litellm_params"]["metadata"] = existing_metadata
        await proxy_logging_obj.db_spend_update_writer.update_database(
            token=user_api_key_dict.api_key,
            response_cost=0.0,
            user_id=user_api_key_dict.user_id,
            end_user_id=user_api_key_dict.end_user_id,
            team_id=user_api_key_dict.team_id,
            kwargs=request_data,
            completion_response=original_exception,
            start_time=datetime.now(),
            end_time=datetime.now(),
            org_id=user_api_key_dict.org_id,
        )

    @log_db_metrics
    async def _PROXY_track_cost_callback(
        self,
        kwargs,  # kwargs to completion
        completion_response: Optional[
            Union[litellm.ModelResponse, Any]
        ],  # response from completion
        start_time=None,
        end_time=None,  # start/end time for completion
    ):
        from litellm.proxy.proxy_server import proxy_logging_obj, update_cache

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
                    verbose_proxy_logger.debug(
                        f"Cache Hit: response_cost {response_cost}, for user_id {user_id}"
                    )

                verbose_proxy_logger.debug(
                    f"user_api_key {user_api_key}, user_id {user_id}, team_id {team_id}, end_user_id {end_user_id}"
                )
                if _should_track_cost_callback(
                    user_api_key=user_api_key,
                    user_id=user_id,
                    team_id=team_id,
                    end_user_id=end_user_id,
                ):
                    ## UPDATE DATABASE
                    await proxy_logging_obj.db_spend_update_writer.update_database(
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
            metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs)
            litellm_metadata = kwargs.get("litellm_params", {}).get(
                "litellm_metadata", {}
            )
            old_metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            call_type = kwargs.get("call_type", "")
            error_msg += f"\n Args to _PROXY_track_cost_callback\n model: {model}\n chosen_metadata: {metadata}\n litellm_metadata: {litellm_metadata}\n old_metadata: {old_metadata}\n call_type: {call_type}\n"
            asyncio.create_task(
                proxy_logging_obj.failed_tracking_alert(
                    error_message=error_msg,
                    failing_model=model,
                )
            )

            verbose_proxy_logger.exception(
                "Error in tracking cost callback - %s", str(e)
            )

    @staticmethod
    def _should_track_errors_in_db():
        """
        Returns True if errors should be tracked in the database

        By default, errors are tracked in the database

        If users want to disable error tracking, they can set the disable_error_logs flag in the general_settings
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get("disable_error_logs") is True:
            return False
        return


def _should_track_cost_callback(
    user_api_key: Optional[str],
    user_id: Optional[str],
    team_id: Optional[str],
    end_user_id: Optional[str],
) -> bool:
    """
    Determine if the cost callback should be tracked based on the kwargs
    """

    # don't run track cost callback if user opted into disabling spend
    if ProxyUpdateSpend.disable_spend_updates() is True:
        return False

    if (
        user_api_key is not None
        or user_id is not None
        or team_id is not None
        or end_user_id is not None
    ):
        return True
    return False
