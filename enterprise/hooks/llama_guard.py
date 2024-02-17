# +-------------------------------------------------------------+
#
#                   Llama Guard
#   https://huggingface.co/meta-llama/LlamaGuard-7b/tree/main
#
#           LLM for Content Moderation
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import sys, os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
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

litellm.set_verbose = True


class _ENTERPRISE_LlamaGuard(CustomLogger):
    # Class variables or attributes
    def __init__(self, model_name: Optional[str] = None):
        self.model = model_name or litellm.llamaguard_model_name

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except:
            pass

    async def async_moderation_hook(
        self,
        data: dict,
    ):
        """
        - Calls the Llama Guard Endpoint
        - Rejects request if it fails safety check

        The llama guard prompt template is applied automatically in factory.py
        """
        safety_check_messages = data["messages"][
            -1
        ]  # get the last response - llama guard has a 4k token limit
        response = await litellm.acompletion(
            model=self.model,
            messages=[safety_check_messages],
            hf_model_name="meta-llama/LlamaGuard-7b",
        )

        if "unsafe" in response.choices[0].message.content:
            raise HTTPException(
                status_code=400, detail={"error": "Violated content safety policy"}
            )

        return data
