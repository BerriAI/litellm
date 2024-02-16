# +-----------------------------------------------+
# |                                               |
# |               PII Masking                     |
# |         with Microsoft Presidio               |
# |   https://github.com/BerriAI/litellm/issues/  |
# +-----------------------------------------------+
#
#  Tell us how we can improve! - Krrish & Ishaan


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


class _OPTIONAL_PresidioPIIMasking(CustomLogger):
    user_api_key_cache = None

    # Class variables or attributes
    def __init__(self, mock_testing: bool = False):
        self.pii_tokens: dict = (
            {}
        )  # mapping of PII token to original text - only used with Presidio `replace` operation
        if mock_testing == True:  # for testing purposes only
            return

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

    async def check_pii(self, text: str, output_parse_pii: bool) -> str:
        """
        [TODO] make this more performant for high-throughput scenario
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Make the first request to /analyze
                analyze_url = f"{self.presidio_analyzer_api_base}/analyze"
                analyze_payload = {"text": text, "language": "en"}
                redacted_text = None
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

                new_text = text
                if redacted_text is not None:
                    for item in redacted_text["items"]:
                        start = item["start"]
                        end = item["end"]
                        replacement = item["text"]  # replacement token
                        if item["operator"] == "replace" and output_parse_pii == True:
                            # check if token in dict
                            # if exists, add a uuid to the replacement token for swapping back to the original text in llm response output parsing
                            if replacement in self.pii_tokens:
                                replacement = replacement + uuid.uuid4()

                            self.pii_tokens[replacement] = new_text[
                                start:end
                            ]  # get text it'll replace

                        new_text = new_text[:start] + replacement + new_text[end:]
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
        permissions = user_api_key_dict.permissions

        if permissions.get("pii", True) == False:  # allow key to turn off pii masking
            return data

        output_parse_pii = permissions.get(
            "output_parse_pii", litellm.output_parse_pii
        )  # allow key to turn on/off output parsing for pii

        if call_type == "completion":  # /chat/completions requests
            messages = data["messages"]
            tasks = []

            for m in messages:
                if isinstance(m["content"], str):
                    tasks.append(
                        self.check_pii(
                            text=m["content"], output_parse_pii=output_parse_pii
                        )
                    )
            responses = await asyncio.gather(*tasks)
            for index, r in enumerate(responses):
                if isinstance(messages[index]["content"], str):
                    messages[index][
                        "content"
                    ] = r  # replace content with redacted string
        return data

    async def async_post_call_success_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
    ):
        """
        Output parse the response object to replace the masked tokens with user sent values
        """
        verbose_proxy_logger.debug(
            f"PII Masking Args: litellm.output_parse_pii={litellm.output_parse_pii}; type of response={type(response)}"
        )
        if litellm.output_parse_pii == False:
            return response

        if isinstance(response, ModelResponse) and not isinstance(
            response.choices[0], StreamingChoices
        ):  # /chat/completions requests
            if isinstance(response.choices[0].message.content, str):
                verbose_proxy_logger.debug(
                    f"self.pii_tokens: {self.pii_tokens}; initial response: {response.choices[0].message.content}"
                )
                for key, value in self.pii_tokens.items():
                    response.choices[0].message.content = response.choices[
                        0
                    ].message.content.replace(key, value)
        return response
