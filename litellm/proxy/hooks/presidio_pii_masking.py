# +-----------------------------------------------+
# |                                               |
# |               PII Masking                     |
# |         with Microsoft Presidio               |
# |   https://github.com/BerriAI/litellm/issues/  |
# +-----------------------------------------------+
#
#  Tell us how we can improve! - Krrish & Ishaan


from typing import Optional
import litellm, traceback, sys
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm import ModelResponse
from datetime import datetime
import aiohttp, asyncio


class _OPTIONAL_PresidioPIIMasking(CustomLogger):
    user_api_key_cache = None

    # Class variables or attributes
    def __init__(self):
        self.presidio_analyzer_api_base = litellm.get_secret(
            "PRESIDIO_ANALYZER_API_BASE", None
        )
        self.presidio_anonymizer_api_base = litellm.get_secret(
            "PRESIDIO_ANONYMIZER_API_BASE", None
        )

        if self.presidio_analyzer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANALYZER_API_BASE` from environment")
        elif not self.presidio_analyzer_api_base.endswith("/"):
            self.presidio_analyzer_api_base += "/"

        if self.presidio_anonymizer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANONYMIZER_API_BASE` from environment")
        elif not self.presidio_anonymizer_api_base.endswith("/"):
            self.presidio_anonymizer_api_base += "/"

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except:
            pass

    async def check_pii(self, text: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                # Make the first request to /analyze
                analyze_url = f"{self.presidio_analyzer_api_base}/analyze"
                analyze_payload = {"text": text, "language": "en"}

                async with session.post(analyze_url, json=analyze_payload) as response:
                    analyze_results = await response.json()

                # Make the second request to /anonymize
                anonymize_url = f"{self.presidio_anonymizer_api_base}/anonymize"
                anonymize_payload = {
                    "text": "hello world, my name is Jane Doe. My number is: 034453334",
                    "analyzer_results": analyze_results,
                }

                async with session.post(
                    anonymize_url, json=anonymize_payload
                ) as response:
                    redacted_text = await response.json()

                return redacted_text["text"]
        except Exception as e:
            traceback.print_exc()
            raise e

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """
        - Take the request data
        - Call /analyze -> get the results
        - Call /anonymize w/ the analyze results -> get the redacted text

        For multiple messages in /chat/completions, we'll need to call them in parallel.
        """
        if call_type == "completion":  # /chat/completions requests
            messages = data["messages"]
            tasks = []
            for m in messages:
                if isinstance(m["content"], str):
                    tasks.append(self.check_pii(text=m["content"]))
            responses = await asyncio.gather(*tasks)
            for index, r in enumerate(responses):
                if isinstance(messages[index]["content"], str):
                    messages[index][
                        "content"
                    ] = r  # replace content with redacted string
        return data
