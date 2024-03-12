from typing import Optional
import litellm, traceback, sys
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm import ModelResponse
from datetime import datetime


class _PROXY_MaxParallelRequestsHandler(CustomLogger):
    user_api_key_cache = None

    # Class variables or attributes
    def __init__(self):
        pass

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except:
            pass

    async def check_key_in_limits(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
        max_parallel_requests: int,
        tpm_limit: int,
        rpm_limit: int,
        request_count_api_key: str,
    ):
        current = cache.get_cache(
            key=request_count_api_key
        )  # {"current_requests": 1, "current_tpm": 1, "current_rpm": 10}
        if current is None:
            new_val = {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 0,
            }
            cache.set_cache(request_count_api_key, new_val)
        elif (
            int(current["current_requests"]) < max_parallel_requests
            and current["current_tpm"] < tpm_limit
            and current["current_rpm"] < rpm_limit
        ):
            # Increase count for this token
            new_val = {
                "current_requests": current["current_requests"] + 1,
                "current_tpm": current["current_tpm"],
                "current_rpm": current["current_rpm"],
            }
            cache.set_cache(request_count_api_key, new_val)
        else:
            raise HTTPException(
                status_code=429, detail="Max parallel request limit reached."
            )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        self.print_verbose(f"Inside Max Parallel Request Pre-Call Hook")
        api_key = user_api_key_dict.api_key
        max_parallel_requests = user_api_key_dict.max_parallel_requests
        if max_parallel_requests is None:
            max_parallel_requests = sys.maxsize
        tpm_limit = getattr(user_api_key_dict, "tpm_limit", sys.maxsize)
        if tpm_limit is None:
            tpm_limit = sys.maxsize
        rpm_limit = getattr(user_api_key_dict, "rpm_limit", sys.maxsize)
        if rpm_limit is None:
            rpm_limit = sys.maxsize

        if api_key is None:
            return

        self.user_api_key_cache = cache  # save the api key cache for updating the value
        # ------------
        # Setup values
        # ------------

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%H")
        current_minute = datetime.now().strftime("%M")
        precise_minute = f"{current_date}-{current_hour}-{current_minute}"

        request_count_api_key = f"{api_key}::{precise_minute}::request_count"

        # CHECK IF REQUEST ALLOWED for key
        current = cache.get_cache(
            key=request_count_api_key
        )  # {"current_requests": 1, "current_tpm": 1, "current_rpm": 10}
        self.print_verbose(f"current: {current}")
        if (
            max_parallel_requests == sys.maxsize
            and tpm_limit == sys.maxsize
            and rpm_limit == sys.maxsize
        ):
            pass
        elif max_parallel_requests == 0 or tpm_limit == 0 or rpm_limit == 0:
            raise HTTPException(
                status_code=429, detail="Max parallel request limit reached."
            )
        elif current is None:
            new_val = {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 0,
            }
            cache.set_cache(request_count_api_key, new_val)
        elif (
            int(current["current_requests"]) < max_parallel_requests
            and current["current_tpm"] < tpm_limit
            and current["current_rpm"] < rpm_limit
        ):
            # Increase count for this token
            new_val = {
                "current_requests": current["current_requests"] + 1,
                "current_tpm": current["current_tpm"],
                "current_rpm": current["current_rpm"],
            }
            cache.set_cache(request_count_api_key, new_val)
        else:
            raise HTTPException(
                status_code=429, detail="Max parallel request limit reached."
            )

        # check if REQUEST ALLOWED for user_id
        user_id = user_api_key_dict.user_id
        _user_id_rate_limits = user_api_key_dict.user_id_rate_limits

        # get user tpm/rpm limits
        if _user_id_rate_limits is not None and isinstance(_user_id_rate_limits, dict):
            user_tpm_limit = _user_id_rate_limits.get("tpm_limit", None)
            user_rpm_limit = _user_id_rate_limits.get("rpm_limit", None)
            if user_tpm_limit is None:
                user_tpm_limit = sys.maxsize
            if user_rpm_limit is None:
                user_rpm_limit = sys.maxsize

            # now do the same tpm/rpm checks
            request_count_api_key = f"{user_id}::{precise_minute}::request_count"

            # print(f"Checking if {request_count_api_key} is allowed to make request for minute {precise_minute}")
            await self.check_key_in_limits(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
                max_parallel_requests=sys.maxsize,  # TODO: Support max parallel requests for a user
                request_count_api_key=request_count_api_key,
                tpm_limit=user_tpm_limit,
                rpm_limit=user_rpm_limit,
            )

        # TEAM RATE LIMITS
        ## get team tpm/rpm limits
        team_id = user_api_key_dict.team_id
        team_tpm_limit = getattr(user_api_key_dict, "team_tpm_limit", sys.maxsize)
        if team_tpm_limit is None:
            team_tpm_limit = sys.maxsize
        team_rpm_limit = getattr(user_api_key_dict, "team_rpm_limit", sys.maxsize)
        if team_rpm_limit is None:
            team_rpm_limit = sys.maxsize

        if team_tpm_limit is None:
            team_tpm_limit = sys.maxsize
        if team_rpm_limit is None:
            team_rpm_limit = sys.maxsize

        # now do the same tpm/rpm checks
        request_count_api_key = f"{team_id}::{precise_minute}::request_count"

        # print(f"Checking if {request_count_api_key} is allowed to make request for minute {precise_minute}")
        await self.check_key_in_limits(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type=call_type,
            max_parallel_requests=sys.maxsize,  # TODO: Support max parallel requests for a user
            request_count_api_key=request_count_api_key,
            tpm_limit=team_tpm_limit,
            rpm_limit=team_rpm_limit,
        )
        return

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.print_verbose(f"INSIDE parallel request limiter ASYNC SUCCESS LOGGING")
            user_api_key = kwargs["litellm_params"]["metadata"]["user_api_key"]
            user_api_key_user_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_user_id", None
            )
            user_api_key_team_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_team_id", None
            )

            if user_api_key is None:
                return

            if self.user_api_key_cache is None:
                return

            # ------------
            # Setup values
            # ------------

            current_date = datetime.now().strftime("%Y-%m-%d")
            current_hour = datetime.now().strftime("%H")
            current_minute = datetime.now().strftime("%M")
            precise_minute = f"{current_date}-{current_hour}-{current_minute}"

            total_tokens = 0

            if isinstance(response_obj, ModelResponse):
                total_tokens = response_obj.usage.total_tokens

            request_count_api_key = f"{user_api_key}::{precise_minute}::request_count"

            current = self.user_api_key_cache.get_cache(key=request_count_api_key) or {
                "current_requests": 1,
                "current_tpm": total_tokens,
                "current_rpm": 1,
            }

            # ------------
            # Update usage - API Key
            # ------------

            new_val = {
                "current_requests": max(current["current_requests"] - 1, 0),
                "current_tpm": current["current_tpm"] + total_tokens,
                "current_rpm": current["current_rpm"] + 1,
            }

            self.print_verbose(
                f"updated_value in success call: {new_val}, precise_minute: {precise_minute}"
            )
            self.user_api_key_cache.set_cache(
                request_count_api_key, new_val, ttl=60
            )  # store in cache for 1 min.

            # ------------
            # Update usage - User
            # ------------
            if user_api_key_user_id is not None:
                total_tokens = 0

                if isinstance(response_obj, ModelResponse):
                    total_tokens = response_obj.usage.total_tokens

                request_count_api_key = (
                    f"{user_api_key_user_id}::{precise_minute}::request_count"
                )

                current = self.user_api_key_cache.get_cache(
                    key=request_count_api_key
                ) or {
                    "current_requests": 1,
                    "current_tpm": total_tokens,
                    "current_rpm": 1,
                }

                new_val = {
                    "current_requests": max(current["current_requests"] - 1, 0),
                    "current_tpm": current["current_tpm"] + total_tokens,
                    "current_rpm": current["current_rpm"] + 1,
                }

                self.print_verbose(
                    f"updated_value in success call: {new_val}, precise_minute: {precise_minute}"
                )
                self.user_api_key_cache.set_cache(
                    request_count_api_key, new_val, ttl=60
                )  # store in cache for 1 min.

            # ------------
            # Update usage - Team
            # ------------
            if user_api_key_team_id is None:
                return

            total_tokens = 0

            if isinstance(response_obj, ModelResponse):
                total_tokens = response_obj.usage.total_tokens

            request_count_api_key = (
                f"{user_api_key_team_id}::{precise_minute}::request_count"
            )

            current = self.user_api_key_cache.get_cache(key=request_count_api_key) or {
                "current_requests": 1,
                "current_tpm": total_tokens,
                "current_rpm": 1,
            }

            new_val = {
                "current_requests": max(current["current_requests"] - 1, 0),
                "current_tpm": current["current_tpm"] + total_tokens,
                "current_rpm": current["current_rpm"] + 1,
            }

            self.print_verbose(
                f"updated_value in success call: {new_val}, precise_minute: {precise_minute}"
            )
            self.user_api_key_cache.set_cache(
                request_count_api_key, new_val, ttl=60
            )  # store in cache for 1 min.

        except Exception as e:
            self.print_verbose(e)  # noqa

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.print_verbose(f"Inside Max Parallel Request Failure Hook")
            user_api_key = kwargs["litellm_params"]["metadata"]["user_api_key"]
            if user_api_key is None:
                return

            if self.user_api_key_cache is None:
                return

            ## decrement call count if call failed
            if (
                hasattr(kwargs["exception"], "status_code")
                and kwargs["exception"].status_code == 429
                and "Max parallel request limit reached" in str(kwargs["exception"])
            ):
                pass  # ignore failed calls due to max limit being reached
            else:
                # ------------
                # Setup values
                # ------------

                current_date = datetime.now().strftime("%Y-%m-%d")
                current_hour = datetime.now().strftime("%H")
                current_minute = datetime.now().strftime("%M")
                precise_minute = f"{current_date}-{current_hour}-{current_minute}"

                request_count_api_key = (
                    f"{user_api_key}::{precise_minute}::request_count"
                )

                # ------------
                # Update usage
                # ------------

                current = self.user_api_key_cache.get_cache(
                    key=request_count_api_key
                ) or {
                    "current_requests": 1,
                    "current_tpm": 0,
                    "current_rpm": 0,
                }

                new_val = {
                    "current_requests": max(current["current_requests"] - 1, 0),
                    "current_tpm": current["current_tpm"],
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in failure call: {new_val}")
                self.user_api_key_cache.set_cache(
                    request_count_api_key, new_val, ttl=60
                )  # save in cache for up to 1 min.
        except Exception as e:
            print(f"An exception occurred - {str(e)}")  # noqa
