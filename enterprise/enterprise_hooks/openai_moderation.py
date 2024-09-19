# +-------------------------------------------------------------+
#
#           Use OpenAI /moderations for your LLM calls
#
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
from litellm._logging import verbose_proxy_logger

litellm.set_verbose = True


class _ENTERPRISE_OpenAI_Moderation(CustomLogger):
    def __init__(self):
        self.model_name = (
            litellm.openai_moderations_model_name or "text-moderation-latest"
        )  # pass the model_name you initialized on litellm.Router()
        pass

    #### CALL HOOKS - proxy only ####

    async def async_moderation_hook(  ### 👈 KEY CHANGE ###
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ):
        if "messages" in data and isinstance(data["messages"], list):
            text = ""
            for m in data["messages"]:  # assume messages is a list
                if "content" in m and isinstance(m["content"], str):
                    text += m["content"]

        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return

        moderation_response = await llm_router.amoderation(
            model=self.model_name, input=text
        )

        verbose_proxy_logger.debug("Moderation response: %s", moderation_response)
        if moderation_response.results[0].flagged == True:
            raise HTTPException(
                status_code=403, detail={"error": "Violated content safety policy"}
            )
        pass
