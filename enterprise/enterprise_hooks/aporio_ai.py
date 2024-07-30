# +-------------------------------------------------------------+
#
#           Use AporioAI for your LLM calls
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
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from typing import List
from datetime import datetime
import aiohttp, asyncio
from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
import httpx
import json

litellm.set_verbose = True

GUARDRAIL_NAME = "aporio"


class _ENTERPRISE_Aporio(CustomLogger):
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )
        self.aporio_api_key = api_key or os.environ["APORIO_API_KEY"]
        self.aporio_api_base = api_base or os.environ["APORIO_API_BASE"]

    #### CALL HOOKS - proxy only ####
    def transform_messages(self, messages: List[dict]) -> List[dict]:
        supported_openai_roles = ["system", "user", "assistant"]
        default_role = "other"  # for unsupported roles - e.g. tool
        new_messages = []
        for m in messages:
            if m.get("role", "") in supported_openai_roles:
                new_messages.append(m)
            else:
                new_messages.append(
                    {
                        "role": default_role,
                        **{key: value for key, value in m.items() if key != "role"},
                    }
                )

        return new_messages

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

        new_messages: Optional[List[dict]] = None
        if "messages" in data and isinstance(data["messages"], list):
            new_messages = self.transform_messages(messages=data["messages"])

        if new_messages is not None:
            data = {"messages": new_messages, "validation_target": "prompt"}

            _json_data = json.dumps(data)

            """
            export APORIO_API_KEY=<your key>
            curl https://gr-prd-trial.aporia.com/some-id \
                -X POST \
                -H "X-APORIA-API-KEY: $APORIO_API_KEY" \
                -H "Content-Type: application/json" \
                -d '{
                    "messages": [
                        {
                        "role": "user",
                        "content": "This is a test prompt"
                        }
                    ],
                    }
'
            """

            response = await self.async_handler.post(
                url=self.aporio_api_base + "/validate",
                data=_json_data,
                headers={
                    "X-APORIA-API-KEY": self.aporio_api_key,
                    "Content-Type": "application/json",
                },
            )
            verbose_proxy_logger.debug("Aporio AI response: %s", response.text)
            if response.status_code == 200:
                # check if the response was flagged
                _json_response = response.json()
                action: str = _json_response.get(
                    "action"
                )  # possible values are modify, passthrough, block, rephrase
                if action == "block":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Violated guardrail policy",
                            "aporio_ai_response": _json_response,
                        },
                    )
