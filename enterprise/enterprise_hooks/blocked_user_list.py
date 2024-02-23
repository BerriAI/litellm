# +------------------------------+
#
#        Blocked User List
#
# +------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
## This accepts a list of user id's for whom calls will be rejected


from typing import Optional, Literal
import litellm
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger
from fastapi import HTTPException
import json, traceback


class _ENTERPRISE_BlockedUserList(CustomLogger):
    # Class variables or attributes
    def __init__(self):
        blocked_user_list = litellm.blocked_user_list

        if blocked_user_list is None:
            raise Exception(
                "`blocked_user_list` can either be a list or filepath. None set."
            )

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
            """
            self.print_verbose(f"Inside Blocked User List Pre-Call Hook")
            if "user_id" in data:
                if data["user_id"] in self.blocked_user_list:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"User blocked from making LLM API Calls. User={data['user_id']}"
                        },
                    )
        except HTTPException as e:
            raise e
        except Exception as e:
            traceback.print_exc()
