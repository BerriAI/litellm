# What is this?
## Checks TPM/RPM Limits for a key/user/team on the proxy
## Works with Redis - if given

from typing import Optional, Literal
import litellm, traceback, sys
from litellm.caching import DualCache, RedisCache
from litellm.proxy._types import (
    UserAPIKeyAuth,
    LiteLLM_VerificationTokenView,
    LiteLLM_UserTable,
    LiteLLM_TeamTable,
)
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm import ModelResponse
from datetime import datetime


class _PROXY_MaxTPMRPMLimiter(CustomLogger):
    user_api_key_cache = None

    # Class variables or attributes
    def __init__(self, internal_cache: Optional[DualCache]):
        if internal_cache is None:
            self.internal_cache = DualCache()
        else:
            self.internal_cache = internal_cache

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except:
            pass

            ## check if admin has set tpm/rpm limits for this key/user/team

    def _check_limits_set(
        self,
        user_api_key_cache: DualCache,
        key: Optional[str],
        user_id: Optional[str],
        team_id: Optional[str],
    ) -> bool:
        ## key
        if key is not None:
            key_val = user_api_key_cache.get_cache(key=key)
            if isinstance(key_val, dict):
                key_val = LiteLLM_VerificationTokenView(**key_val)

            if isinstance(key_val, LiteLLM_VerificationTokenView):
                user_api_key_tpm_limit = key_val.tpm_limit

                user_api_key_rpm_limit = key_val.rpm_limit

                if (
                    user_api_key_tpm_limit is not None
                    or user_api_key_rpm_limit is not None
                ):
                    return True

        ## team
        if team_id is not None:
            team_val = user_api_key_cache.get_cache(key=team_id)
            if isinstance(team_val, dict):
                team_val = LiteLLM_TeamTable(**team_val)

            if isinstance(team_val, LiteLLM_TeamTable):
                team_tpm_limit = team_val.tpm_limit

                team_rpm_limit = team_val.rpm_limit

                if team_tpm_limit is not None or team_rpm_limit is not None:
                    return True

        ## user
        if user_id is not None:
            user_val = user_api_key_cache.get_cache(key=user_id)
            if isinstance(user_val, dict):
                user_val = LiteLLM_UserTable(**user_val)

            if isinstance(user_val, LiteLLM_UserTable):
                user_tpm_limit = user_val.tpm_limit

                user_rpm_limit = user_val.rpm_limit

                if user_tpm_limit is not None or user_rpm_limit is not None:
                    return True
        return False

    async def check_key_in_limits(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        current_minute_dict: dict,
        tpm_limit: int,
        rpm_limit: int,
        request_count_api_key: str,
        type: Literal["key", "user", "team"],
    ):

        if type == "key" and user_api_key_dict.api_key is not None:
            current = current_minute_dict["key"].get(user_api_key_dict.api_key, None)
        elif type == "user" and user_api_key_dict.user_id is not None:
            current = current_minute_dict["user"].get(user_api_key_dict.user_id, None)
        elif type == "team" and user_api_key_dict.team_id is not None:
            current = current_minute_dict["team"].get(user_api_key_dict.team_id, None)
        else:
            return
        if current is None:
            if tpm_limit == 0 or rpm_limit == 0:
                # base case
                raise HTTPException(
                    status_code=429, detail="Max tpm/rpm limit reached."
                )
        elif current["current_tpm"] < tpm_limit and current["current_rpm"] < rpm_limit:
            pass
        else:
            raise HTTPException(status_code=429, detail="Max tpm/rpm limit reached.")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        self.print_verbose(
            f"Inside Max TPM/RPM Limiter Pre-Call Hook - {user_api_key_dict}"
        )
        api_key = user_api_key_dict.api_key
        # check if REQUEST ALLOWED for user_id
        user_id = user_api_key_dict.user_id
        ## get team tpm/rpm limits
        team_id = user_api_key_dict.team_id

        self.user_api_key_cache = cache

        _set_limits = self._check_limits_set(
            user_api_key_cache=cache, key=api_key, user_id=user_id, team_id=team_id
        )

        self.print_verbose(f"_set_limits: {_set_limits}")

        if _set_limits == False:
            return

        # ------------
        # Setup values
        # ------------

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%H")
        current_minute = datetime.now().strftime("%M")
        precise_minute = f"{current_date}-{current_hour}-{current_minute}"
        cache_key = "usage:{}".format(precise_minute)
        current_minute_dict = await self.internal_cache.async_get_cache(
            key=cache_key
        )  # {"usage:{curr_minute}": {"key": {<api_key>: {"current_requests": 1, "current_tpm": 1, "current_rpm": 10}}}}

        if current_minute_dict is None:
            current_minute_dict = {"key": {}, "user": {}, "team": {}}

        if api_key is not None:
            tpm_limit = getattr(user_api_key_dict, "tpm_limit", sys.maxsize)
            if tpm_limit is None:
                tpm_limit = sys.maxsize
            rpm_limit = getattr(user_api_key_dict, "rpm_limit", sys.maxsize)
            if rpm_limit is None:
                rpm_limit = sys.maxsize
            request_count_api_key = f"{api_key}::{precise_minute}::request_count"
            await self.check_key_in_limits(
                user_api_key_dict=user_api_key_dict,
                current_minute_dict=current_minute_dict,
                request_count_api_key=request_count_api_key,
                tpm_limit=tpm_limit,
                rpm_limit=rpm_limit,
                type="key",
            )

        if user_id is not None:
            _user_id_rate_limits = user_api_key_dict.user_id_rate_limits

            # get user tpm/rpm limits
            if _user_id_rate_limits is not None and isinstance(
                _user_id_rate_limits, dict
            ):
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
                    current_minute_dict=current_minute_dict,
                    request_count_api_key=request_count_api_key,
                    tpm_limit=user_tpm_limit,
                    rpm_limit=user_rpm_limit,
                    type="user",
                )

        # TEAM RATE LIMITS
        if team_id is not None:
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
                current_minute_dict=current_minute_dict,
                request_count_api_key=request_count_api_key,
                tpm_limit=team_tpm_limit,
                rpm_limit=team_rpm_limit,
                type="team",
            )

        return

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.print_verbose(f"INSIDE TPM RPM Limiter ASYNC SUCCESS LOGGING")

            user_api_key = kwargs["litellm_params"]["metadata"]["user_api_key"]
            user_api_key_user_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_user_id", None
            )
            user_api_key_team_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_team_id", None
            )
            _limits_set = self._check_limits_set(
                user_api_key_cache=self.user_api_key_cache,
                key=user_api_key,
                user_id=user_api_key_user_id,
                team_id=user_api_key_team_id,
            )

            if _limits_set == False:  # don't waste cache calls if no tpm/rpm limits set
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

            """
            - get value from redis
            - increment requests + 1
            - increment tpm + 1
            - increment rpm + 1
            - update value in-memory + redis
            """
            cache_key = "usage:{}".format(precise_minute)
            if (
                self.internal_cache.redis_cache is not None
            ):  # get straight from redis if possible
                current_minute_dict = (
                    await self.internal_cache.redis_cache.async_get_cache(
                        key=cache_key,
                    )
                )  # {"usage:{current_minute}": {"key": {}, "team": {}, "user": {}}}
            else:
                current_minute_dict = await self.internal_cache.async_get_cache(
                    key=cache_key,
                )

            if current_minute_dict is None:
                current_minute_dict = {"key": {}, "user": {}, "team": {}}

            _cache_updated = False  # check if a cache update is required. prevent unnecessary rewrites.

            # ------------
            # Update usage - API Key
            # ------------

            if user_api_key is not None:
                _cache_updated = True
                ## API KEY ##
                if user_api_key in current_minute_dict["key"]:
                    current_key_usage = current_minute_dict["key"][user_api_key]
                    new_val = {
                        "current_tpm": current_key_usage["current_tpm"] + total_tokens,
                        "current_rpm": current_key_usage["current_rpm"] + 1,
                    }
                else:
                    new_val = {
                        "current_tpm": total_tokens,
                        "current_rpm": 1,
                    }

                current_minute_dict["key"][user_api_key] = new_val

                self.print_verbose(
                    f"updated_value in success call: {new_val}, precise_minute: {precise_minute}"
                )

            # ------------
            # Update usage - User
            # ------------
            if user_api_key_user_id is not None:
                _cache_updated = True
                total_tokens = 0

                if isinstance(response_obj, ModelResponse):
                    total_tokens = response_obj.usage.total_tokens

                if user_api_key_user_id in current_minute_dict["key"]:
                    current_key_usage = current_minute_dict["key"][user_api_key_user_id]
                    new_val = {
                        "current_tpm": current_key_usage["current_tpm"] + total_tokens,
                        "current_rpm": current_key_usage["current_rpm"] + 1,
                    }
                else:
                    new_val = {
                        "current_tpm": total_tokens,
                        "current_rpm": 1,
                    }

                current_minute_dict["user"][user_api_key_user_id] = new_val

            # ------------
            # Update usage - Team
            # ------------
            if user_api_key_team_id is not None:
                _cache_updated = True
                total_tokens = 0

                if isinstance(response_obj, ModelResponse):
                    total_tokens = response_obj.usage.total_tokens

                if user_api_key_team_id in current_minute_dict["key"]:
                    current_key_usage = current_minute_dict["key"][user_api_key_team_id]
                    new_val = {
                        "current_tpm": current_key_usage["current_tpm"] + total_tokens,
                        "current_rpm": current_key_usage["current_rpm"] + 1,
                    }
                else:
                    new_val = {
                        "current_tpm": total_tokens,
                        "current_rpm": 1,
                    }

                current_minute_dict["team"][user_api_key_team_id] = new_val

            if _cache_updated == True:
                await self.internal_cache.async_set_cache(
                    key=cache_key, value=current_minute_dict
                )

        except Exception as e:
            self.print_verbose("{}\n{}".format(e, traceback.format_exc()))  # noqa
