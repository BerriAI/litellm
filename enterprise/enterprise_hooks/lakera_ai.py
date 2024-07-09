# +-------------------------------------------------------------+
#
#           Use lakeraAI /moderations for your LLM calls
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
from litellm.proxy.guardrails.init_guardrails import all_guardrails
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata

from datetime import datetime
import aiohttp, asyncio
from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
import httpx
import json

litellm.set_verbose = True

GUARDRAIL_NAME = "lakera_prompt_injection"


class _ENTERPRISE_lakeraAI_Moderation(CustomLogger):
    def __init__(self):
        self.async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
        self.lakera_api_key = os.environ["LAKERA_API_KEY"]
        pass

    #### CALL HOOKS - proxy only ####

    async def async_moderation_hook(  ### 👈 KEY CHANGE ###
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation"],
    ):

        if (
            await should_proceed_based_on_metadata(
                data=data,
                guardrail_name=GUARDRAIL_NAME,
            )
            is False
        ):
            return

        if "messages" in data and isinstance(data["messages"], list):
            text = ""
            for m in data["messages"]:  # assume messages is a list
                if "content" in m and isinstance(m["content"], str):
                    text += m["content"]

        # https://platform.lakera.ai/account/api-keys
        data = {"input": text}

        _json_data = json.dumps(data)

        """
        export LAKERA_GUARD_API_KEY=<your key>
        curl https://api.lakera.ai/v1/prompt_injection \
            -X POST \
            -H "Authorization: Bearer $LAKERA_GUARD_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{"input": "Your content goes here"}'
        """

        response = await self.async_handler.post(
            url="https://api.lakera.ai/v1/prompt_injection",
            data=_json_data,
            headers={
                "Authorization": "Bearer " + self.lakera_api_key,
                "Content-Type": "application/json",
            },
        )
        verbose_proxy_logger.debug("Lakera AI response: %s", response.text)
        if response.status_code == 200:
            # check if the response was flagged
            """
            Example Response from Lakera AI

            {
                "model": "lakera-guard-1",
                "results": [
                {
                    "categories": {
                    "prompt_injection": true,
                    "jailbreak": false
                    },
                    "category_scores": {
                    "prompt_injection": 1.0,
                    "jailbreak": 0.0
                    },
                    "flagged": true,
                    "payload": {}
                }
                ],
                "dev_info": {
                "git_revision": "784489d3",
                "git_timestamp": "2024-05-22T16:51:26+00:00"
                }
            }
            """
            _json_response = response.json()
            _results = _json_response.get("results", [])
            if len(_results) <= 0:
                return

            flagged = _results[0].get("flagged", False)

            if flagged == True:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated content safety policy",
                        "lakera_ai_response": _json_response,
                    },
                )

        pass
