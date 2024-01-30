from typing import Optional
import litellm, traceback, sys
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm import ModelResponse
from datetime import datetime


class MaxParallelRequestsHandler(CustomLogger):
    user_api_key_cache = None

    # Class variables or attributes
    def __init__(self):
        pass

    def print_verbose(self, print_statement):
        verbose_proxy_logger.debug(print_statement)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        self.print_verbose(f"Inside Max Parallel Request Pre-Call Hook")
        api_key = user_api_key_dict.api_key
        max_parallel_requests = user_api_key_dict.max_parallel_requests or sys.maxsize
        tpm_limit = user_api_key_dict.tpm_limit or sys.maxsize
        rpm_limit = user_api_key_dict.rpm_limit or sys.maxsize

        if api_key is None:
            return

        if (
            max_parallel_requests == sys.maxsize
            and tpm_limit == sys.maxsize
            and rpm_limit == sys.maxsize
        ):
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

        # CHECK IF REQUEST ALLOWED
        current = cache.get_cache(
            key=request_count_api_key
        )  # {"current_requests": 1, "current_tpm": 1, "current_rpm": 10}
        self.print_verbose(f"current: {current}")
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

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.print_verbose(f"INSIDE parallel request limiter ASYNC SUCCESS LOGGING")
            user_api_key = kwargs["litellm_params"]["metadata"]["user_api_key"]
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
            # Update usage
            # ------------

            new_val = {
                "current_requests": current["current_requests"] - 1,
                "current_tpm": current["current_tpm"] + total_tokens,
                "current_rpm": current["current_rpm"] + 1,
            }

            self.print_verbose(f"updated_value in success call: {new_val}")
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
                    "current_requests": current["current_requests"] - 1,
                    "current_tpm": current["current_tpm"],
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in failure call: {new_val}")
                self.user_api_key_cache.set_cache(
                    request_count_api_key, new_val, ttl=60
                )  # save in cache for up to 1 min.
        except Exception as e:
            print(f"An exception occurred - {str(e)}")  # noqa
