# +------------------------------+
#
#        Banned Keywords
#
# +------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
## Reject a call / response if it contains certain keywords


from typing import Literal
import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import (
    is_text_content_call_type,
    iter_message_text,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger
from fastapi import HTTPException


class _ENTERPRISE_BannedKeywords(CustomLogger):
    # Class variables or attributes
    def __init__(self):
        banned_keywords_list = litellm.banned_keywords_list

        if banned_keywords_list is None:
            raise Exception(
                "`banned_keywords_list` can either be a list or filepath. None set."
            )

        if isinstance(banned_keywords_list, list):
            self.banned_keywords_list = banned_keywords_list

        if isinstance(banned_keywords_list, str):  # assume it's a filepath
            try:
                with open(banned_keywords_list, "r") as file:
                    data = file.read()
                    self.banned_keywords_list = data.split("\n")
            except FileNotFoundError:
                raise Exception(
                    f"File not found. banned_keywords_list={banned_keywords_list}"
                )
            except Exception as e:
                raise Exception(
                    f"An error occurred: {str(e)}, banned_keywords_list={banned_keywords_list}"
                )

    def print_verbose(self, print_statement, level: Literal["INFO", "DEBUG"] = "DEBUG"):
        if level == "INFO":
            verbose_proxy_logger.info(print_statement)
        elif level == "DEBUG":
            verbose_proxy_logger.debug(print_statement)

        if litellm.set_verbose is True:
            print(print_statement)  # noqa

    def test_violation(self, test_str: str):
        for word in self.banned_keywords_list:
            if word in test_str.lower():
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"Keyword banned. Keyword={word}"},
                )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,  # "completion", "embeddings", "image_generation", "moderation"
    ):
        try:
            """
            - check if user id part of call
            - check if user id part of blocked list
            """
            self.print_verbose("Inside Banned Keyword List Pre-Call Hook")
            if is_text_content_call_type(call_type):
                for text in iter_message_text(data):
                    self.test_violation(test_str=text)

        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.enterprise.enterprise_hooks.banned_keywords::async_pre_call_hook - Exception occurred - {}".format(
                    str(e)
                )
            )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        if not isinstance(response, litellm.ModelResponse):
            return

        for choice in response.choices:
            if not isinstance(choice, litellm.utils.Choices):
                continue
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                self.test_violation(test_str=content)

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        self.test_violation(test_str=response)
