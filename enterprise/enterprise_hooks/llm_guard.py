# +------------------------+
#
#            LLM Guard
#   https://llm-guard.com/
#
# +------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
## This provides an LLM Guard Integration for content moderation on the proxy

from typing import Optional, Literal, Union
import litellm, traceback, sys, uuid, os
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


class _ENTERPRISE_LLMGuard(CustomLogger):
    # Class variables or attributes
    def __init__(
        self, mock_testing: bool = False, mock_redacted_text: Optional[dict] = None
    ):
        self.mock_redacted_text = mock_redacted_text
        if mock_testing == True:  # for testing purposes only
            return
        self.llm_guard_api_base = litellm.get_secret("LLM_GUARD_API_BASE", None)
        if self.llm_guard_api_base is None:
            raise Exception("Missing `LLM_GUARD_API_BASE` from environment")
        elif not self.llm_guard_api_base.endswith("/"):
            self.llm_guard_api_base += "/"

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except:
            pass

    async def moderation_check(self, text: str):
        """
        [TODO] make this more performant for high-throughput scenario
        """
        try:
            async with aiohttp.ClientSession() as session:
                if self.mock_redacted_text is not None:
                    redacted_text = self.mock_redacted_text
                else:
                    # Make the first request to /analyze
                    analyze_url = f"{self.llm_guard_api_base}analyze/prompt"
                    verbose_proxy_logger.debug(f"Making request to: {analyze_url}")
                    analyze_payload = {"prompt": text}
                    redacted_text = None
                    async with session.post(
                        analyze_url, json=analyze_payload
                    ) as response:
                        redacted_text = await response.json()
                verbose_proxy_logger.info(
                    f"LLM Guard: Received response - {redacted_text}"
                )
                if redacted_text is not None:
                    if (
                        redacted_text.get("is_valid", None) is not None
                        and redacted_text["is_valid"] == "True"
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail={"error": "Violated content safety policy"},
                        )
                    else:
                        pass
                else:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": f"Invalid content moderation response: {redacted_text}"
                        },
                    )
        except Exception as e:
            traceback.print_exc()
            raise e

    async def async_moderation_hook(
        self,
        data: dict,
    ):
        """
        - Calls the LLM Guard Endpoint
        - Rejects request if it fails safety check
        - Use the sanitized prompt returned
            - LLM Guard can handle things like PII Masking, etc.
        """
        return data

    async def async_post_call_streaming_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: str
    ):
        if response is not None:
            await self.moderation_check(text=response)

        return response


# llm_guard = _ENTERPRISE_LLMGuard()

# asyncio.run(
#     llm_guard.async_moderation_hook(
#         data={"messages": [{"role": "user", "content": "Hey how's it going?"}]}
#     )
# )
