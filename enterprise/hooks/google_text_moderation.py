# +-----------------------------------------------+
#
#            Google Text Moderation
#   https://cloud.google.com/natural-language/docs/moderating-text
#
# +-----------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan


from typing import Optional, Literal, Union
import litellm, traceback, sys, uuid
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.utils import (
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    StreamingChoices,
)
from datetime import datetime
import aiohttp, asyncio


class _ENTERPRISE_GoogleTextModeration(CustomLogger):
    user_api_key_cache = None

    # Class variables or attributes
    def __init__(self, mock_testing: bool = False):
        pass

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except:
            pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """
        - Calls Google's Text Moderation API
        - Rejects request if it fails safety check
        """
        pass
