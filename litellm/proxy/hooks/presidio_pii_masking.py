# +-----------------------------------------------+
# |                                               |
# |               PII Masking                     |
# |         with Microsoft Presidio               |
# |   https://github.com/BerriAI/litellm/issues/  |
# +-----------------------------------------------+
#
#  Tell us how we can improve! - Krrish & Ishaan


import asyncio
import json
import traceback
import uuid
from typing import Any, List, Optional, Tuple, Union

import aiohttp
from fastapi import HTTPException

import litellm  # noqa: E401
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    StreamingChoices,
    get_formatted_prompt,
)


class _OPTIONAL_PresidioPIIMasking(CustomLogger):
    user_api_key_cache = None
    ad_hoc_recognizers = None

    # Class variables or attributes
    def __init__(
        self,
        logging_only: Optional[bool] = None,
        mock_testing: bool = False,
        mock_redacted_text: Optional[dict] = None,
    ):
        self.pii_tokens: dict = (
            {}
        )  # mapping of PII token to original text - only used with Presidio `replace` operation

        self.mock_redacted_text = mock_redacted_text
        self.logging_only = logging_only
        if mock_testing is True:  # for testing purposes only
            return

        ad_hoc_recognizers = litellm.presidio_ad_hoc_recognizers
        if ad_hoc_recognizers is not None:
            try:
                with open(ad_hoc_recognizers, "r") as file:
                    self.ad_hoc_recognizers = json.load(file)
            except FileNotFoundError:
                raise Exception(f"File not found. file_path={ad_hoc_recognizers}")
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Error decoding JSON file: {str(e)}, file_path={ad_hoc_recognizers}"
                )
            except Exception as e:
                raise Exception(
                    f"An error occurred: {str(e)}, file_path={ad_hoc_recognizers}"
                )

        self.validate_environment()

    def validate_environment(self):
        self.presidio_analyzer_api_base: Optional[str] = litellm.get_secret(
            "PRESIDIO_ANALYZER_API_BASE", None
        )  # type: ignore
        self.presidio_anonymizer_api_base: Optional[str] = litellm.get_secret(
            "PRESIDIO_ANONYMIZER_API_BASE", None
        )  # type: ignore

        if self.presidio_analyzer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANALYZER_API_BASE` from environment")
        if not self.presidio_analyzer_api_base.endswith("/"):
            self.presidio_analyzer_api_base += "/"
        if not (
            self.presidio_analyzer_api_base.startswith("http://")
            or self.presidio_analyzer_api_base.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.presidio_analyzer_api_base = (
                "http://" + self.presidio_analyzer_api_base
            )

        if self.presidio_anonymizer_api_base is None:
            raise Exception("Missing `PRESIDIO_ANONYMIZER_API_BASE` from environment")
        if not self.presidio_anonymizer_api_base.endswith("/"):
            self.presidio_anonymizer_api_base += "/"
        if not (
            self.presidio_anonymizer_api_base.startswith("http://")
            or self.presidio_anonymizer_api_base.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.presidio_anonymizer_api_base = (
                "http://" + self.presidio_anonymizer_api_base
            )

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass

    async def check_pii(self, text: str, output_parse_pii: bool) -> str:
        """
        [TODO] make this more performant for high-throughput scenario
        """
        try:
            async with aiohttp.ClientSession() as session:
                if self.mock_redacted_text is not None:
                    redacted_text = self.mock_redacted_text
                else:
                    # Make the first request to /analyze
                    analyze_url = f"{self.presidio_analyzer_api_base}analyze"
                    verbose_proxy_logger.debug("Making request to: %s", analyze_url)
                    analyze_payload = {"text": text, "language": "en"}
                    if self.ad_hoc_recognizers is not None:
                        analyze_payload["ad_hoc_recognizers"] = self.ad_hoc_recognizers
                    redacted_text = None
                    async with session.post(
                        analyze_url, json=analyze_payload
                    ) as response:
                        analyze_results = await response.json()

                    # Make the second request to /anonymize
                    anonymize_url = f"{self.presidio_anonymizer_api_base}anonymize"
                    verbose_proxy_logger.debug("Making request to: %s", anonymize_url)
                    anonymize_payload = {
                        "text": text,
                        "analyzer_results": analyze_results,
                    }

                    async with session.post(
                        anonymize_url, json=anonymize_payload
                    ) as response:
                        redacted_text = await response.json()

                new_text = text
                if redacted_text is not None:
                    verbose_proxy_logger.debug("redacted_text: %s", redacted_text)
                    for item in redacted_text["items"]:
                        start = item["start"]
                        end = item["end"]
                        replacement = item["text"]  # replacement token
                        if item["operator"] == "replace" and output_parse_pii is True:
                            # check if token in dict
                            # if exists, add a uuid to the replacement token for swapping back to the original text in llm response output parsing
                            if replacement in self.pii_tokens:
                                replacement = replacement + str(uuid.uuid4())

                            self.pii_tokens[replacement] = new_text[
                                start:end
                            ]  # get text it'll replace

                        new_text = new_text[:start] + replacement + new_text[end:]
                    return redacted_text["text"]
                else:
                    raise Exception(f"Invalid anonymizer response: {redacted_text}")
        except Exception as e:
            verbose_proxy_logger.error(
                "litellm.proxy.hooks.presidio_pii_masking.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_proxy_logger.debug(traceback.format_exc())
            raise e

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """
        - Check if request turned off pii
            - Check if user allowed to turn off pii (key permissions -> 'allow_pii_controls')

        - Take the request data
        - Call /analyze -> get the results
        - Call /anonymize w/ the analyze results -> get the redacted text

        For multiple messages in /chat/completions, we'll need to call them in parallel.
        """
        try:
            if (
                self.logging_only is True
            ):  # only modify the logging obj data (done by async_logging_hook)
                return data
            permissions = user_api_key_dict.permissions
            output_parse_pii = permissions.get(
                "output_parse_pii", litellm.output_parse_pii
            )  # allow key to turn on/off output parsing for pii
            no_pii = permissions.get(
                "no-pii", None
            )  # allow key to turn on/off pii masking (if user is allowed to set pii controls, then they can override the key defaults)

            if no_pii is None:
                # check older way of turning on/off pii
                no_pii = not permissions.get("pii", True)

            content_safety = data.get("content_safety", None)
            verbose_proxy_logger.debug("content_safety: %s", content_safety)
            ## Request-level turn on/off PII controls ##
            if content_safety is not None and isinstance(content_safety, dict):
                # pii masking ##
                if (
                    content_safety.get("no-pii", None) is not None
                    and content_safety.get("no-pii") is True
                ):
                    # check if user allowed to turn this off
                    if permissions.get("allow_pii_controls", False) is False:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "Not allowed to set PII controls per request"
                            },
                        )
                    else:  # user allowed to turn off pii masking
                        no_pii = content_safety.get("no-pii")
                        if not isinstance(no_pii, bool):
                            raise HTTPException(
                                status_code=400,
                                detail={"error": "no_pii needs to be a boolean value"},
                            )
                ## pii output parsing ##
                if content_safety.get("output_parse_pii", None) is not None:
                    # check if user allowed to turn this off
                    if permissions.get("allow_pii_controls", False) is False:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "Not allowed to set PII controls per request"
                            },
                        )
                    else:  # user allowed to turn on/off pii output parsing
                        output_parse_pii = content_safety.get("output_parse_pii")
                        if not isinstance(output_parse_pii, bool):
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": "output_parse_pii needs to be a boolean value"
                                },
                            )

            if no_pii is True:  # turn off pii masking
                return data

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
                verbose_proxy_logger.info(
                    f"Presidio PII Masking: Redacted pii message: {data['messages']}"
                )
            return data
        except Exception as e:
            verbose_proxy_logger.info(
                "An error occurred -",
            )
            raise e

    async def async_logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """
        Masks the input before logging to langfuse, datadog, etc.
        """
        if (
            call_type == "completion" or call_type == "acompletion"
        ):  # /chat/completions requests
            messages: Optional[List] = kwargs.get("messages", None)
            tasks = []

            if messages is None:
                return kwargs, result

            for m in messages:
                text_str = ""
                if m["content"] is None:
                    continue
                if isinstance(m["content"], str):
                    text_str = m["content"]
                    tasks.append(
                        self.check_pii(text=text_str, output_parse_pii=False)
                    )  # need to pass separately b/c presidio has context window limits
            responses = await asyncio.gather(*tasks)
            for index, r in enumerate(responses):
                if isinstance(messages[index]["content"], str):
                    messages[index][
                        "content"
                    ] = r  # replace content with redacted string
            verbose_proxy_logger.info(
                f"Presidio PII Masking: Redacted pii message: {messages}"
            )
            kwargs["messages"] = messages

        return kwargs, result

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
    ):
        """
        Output parse the response object to replace the masked tokens with user sent values
        """
        verbose_proxy_logger.debug(
            f"PII Masking Args: litellm.output_parse_pii={litellm.output_parse_pii}; type of response={type(response)}"
        )
        if litellm.output_parse_pii is False:
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
