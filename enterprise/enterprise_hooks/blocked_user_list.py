# +------------------------------+
#
#        Blocked User List
#
# +------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
## This accepts a list of user id's for whom calls will be rejected


from typing import Optional, Literal
import litellm
from litellm.proxy.utils import PrismaClient
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth, LiteLLM_EndUserTable
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger
from fastapi import HTTPException


class _ENTERPRISE_BlockedUserList(CustomLogger):
    # Class variables or attributes
    def __init__(self, prisma_client: Optional[PrismaClient]):
        self.prisma_client = prisma_client

        blocked_user_list = litellm.blocked_user_list
        if blocked_user_list is None:
            self.blocked_user_list = None
            return

        if isinstance(blocked_user_list, list):
            self.blocked_user_list = blocked_user_list

        if isinstance(blocked_user_list, str):  # assume it's a filepath
            try:
                with open(blocked_user_list, "r") as file:
                    data = file.read()
                    self.blocked_user_list = data.split("\n")
            except FileNotFoundError:
                raise Exception(
                    f"File not found. blocked_user_list={blocked_user_list}"
                )
            except Exception as e:
                raise Exception(
                    f"An error occurred: {str(e)}, blocked_user_list={blocked_user_list}"
                )

    def print_verbose(self, print_statement, level: Literal["INFO", "DEBUG"] = "DEBUG"):
        if level == "INFO":
            verbose_proxy_logger.info(print_statement)
        elif level == "DEBUG":
            verbose_proxy_logger.debug(print_statement)

        if litellm.set_verbose is True:
            print(print_statement)  # noqa

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        try:
            """
            - check if user id part of call
            - check if user id part of blocked list
                - if blocked list is none or user not in blocked list
                - check if end-user in cache
                - check if end-user in db
            """
            self.print_verbose("Inside Blocked User List Pre-Call Hook")
            if "user_id" in data or "user" in data:
                user = data.get("user_id", data.get("user", ""))
                if (
                    self.blocked_user_list is not None
                    and user in self.blocked_user_list
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"User blocked from making LLM API Calls. User={user}"
                        },
                    )

                cache_key = f"litellm:end_user_id:{user}"
                end_user_cache_obj: Optional[LiteLLM_EndUserTable] = cache.get_cache(  # type: ignore
                    key=cache_key
                )
                if end_user_cache_obj is None and self.prisma_client is not None:
                    # check db
                    end_user_obj = (
                        await self.prisma_client.db.litellm_endusertable.find_unique(
                            where={"user_id": user}
                        )
                    )
                    if end_user_obj is None:  # user not in db - assume not blocked
                        end_user_obj = LiteLLM_EndUserTable(user_id=user, blocked=False)
                    cache.set_cache(key=cache_key, value=end_user_obj, ttl=60)
                    if end_user_obj is not None and end_user_obj.blocked is True:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": f"User blocked from making LLM API Calls. User={user}"
                            },
                        )
                elif (
                    end_user_cache_obj is not None
                    and end_user_cache_obj.blocked is True
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"User blocked from making LLM API Calls. User={user}"
                        },
                    )

        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.enterprise.enterprise_hooks.blocked_user_list::async_pre_call_hook - Exception occurred - {}".format(
                    str(e)
                )
            )
