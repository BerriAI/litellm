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
import uuid
from typing import Any, List, Optional, Tuple, Union

import aiohttp
from pydantic import BaseModel

import litellm  # noqa: E401
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    StreamingChoices,
)


class PresidioPerRequestConfig(BaseModel):
    """
    presdio params that can be controlled per request, api key
    """

    language: Optional[str] = None


class _OPTIONAL_PresidioPIIMasking(CustomGuardrail):
    user_api_key_cache = None
    ad_hoc_recognizers = None

    # Class variables or attributes
    def __init__(
        self,
        mock_testing: bool = False,
        mock_redacted_text: Optional[dict] = None,
        presidio_analyzer_api_base: Optional[str] = None,
        presidio_anonymizer_api_base: Optional[str] = None,
        output_parse_pii: Optional[bool] = False,
        presidio_ad_hoc_recognizers: Optional[str] = None,
        logging_only: Optional[bool] = None,
        **kwargs,
    ):
        if logging_only is True:
            self.logging_only = True
            kwargs["event_hook"] = GuardrailEventHooks.logging_only
        super().__init__(**kwargs)
        self.pii_tokens: dict = (
            {}
        )  # mapping of PII token to original text - only used with Presidio `replace` operation
        self.mock_redacted_text = mock_redacted_text
        self.output_parse_pii = output_parse_pii or False
        if mock_testing is True:  # for testing purposes only
            return

        ad_hoc_recognizers = presidio_ad_hoc_recognizers
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
        self.validate_environment(
            presidio_analyzer_api_base=presidio_analyzer_api_base,
            presidio_anonymizer_api_base=presidio_anonymizer_api_base,
        )

    def validate_environment(
        self,
        presidio_analyzer_api_base: Optional[str] = None,
        presidio_anonymizer_api_base: Optional[str] = None,
    ):
        self.presidio_analyzer_api_base: Optional[
            str
        ] = presidio_analyzer_api_base or get_secret(
            "PRESIDIO_ANALYZER_API_BASE", None
        )  # type: ignore
        self.presidio_anonymizer_api_base: Optional[
            str
        ] = presidio_anonymizer_api_base or litellm.get_secret(
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

    async def check_pii(
        self,
        text: str,
        output_parse_pii: bool,
        presidio_config: Optional[PresidioPerRequestConfig],
        request_data: dict,
    ) -> str:
        """
        [TODO] make this more performant for high-throughput scenario
        """
        try:
            async with aiohttp.ClientSession() as session:
                if self.mock_redacted_text is not None:
                    redacted_text = self.mock_redacted_text
                else:
                    # Make the first request to /analyze
                    # Construct Request 1
                    analyze_url = f"{self.presidio_analyzer_api_base}analyze"
                    analyze_payload = {"text": text, "language": "en"}
                    if presidio_config and presidio_config.language:
                        analyze_payload["language"] = presidio_config.language
                    if self.ad_hoc_recognizers is not None:
                        analyze_payload["ad_hoc_recognizers"] = self.ad_hoc_recognizers
                    # End of constructing Request 1
                    analyze_payload.update(
                        self.get_guardrail_dynamic_request_body_params(
                            request_data=request_data
                        )
                    )
                    redacted_text = None
                    verbose_proxy_logger.debug(
                        "Making request to: %s with payload: %s",
                        analyze_url,
                        analyze_payload,
                    )
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
            raise e

    @log_guardrail_information
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
            content_safety = data.get("content_safety", None)
            verbose_proxy_logger.debug("content_safety: %s", content_safety)
            presidio_config = self.get_presidio_settings_from_request_data(data)

            if call_type == "completion":  # /chat/completions requests
                messages = data["messages"]
                tasks = []

                for m in messages:
                    if isinstance(m["content"], str):
                        tasks.append(
                            self.check_pii(
                                text=m["content"],
                                output_parse_pii=self.output_parse_pii,
                                presidio_config=presidio_config,
                                request_data=data,
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
                data["messages"] = messages
            return data
        except Exception as e:
            raise e

    @log_guardrail_information
    def logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        from concurrent.futures import ThreadPoolExecutor

        def run_in_new_loop():
            """Run the coroutine in a new event loop within this thread."""
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(
                    self.async_logging_hook(
                        kwargs=kwargs, result=result, call_type=call_type
                    )
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)

        try:
            # First, try to get the current event loop
            _ = asyncio.get_running_loop()
            # If we're already in an event loop, run in a separate thread
            # to avoid nested event loop issues
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()

        except RuntimeError:
            # No running event loop, we can safely run in this thread
            return run_in_new_loop()

    @log_guardrail_information
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

            presidio_config = self.get_presidio_settings_from_request_data(kwargs)

            for m in messages:
                text_str = ""
                if m["content"] is None:
                    continue
                if isinstance(m["content"], str):
                    text_str = m["content"]
                    tasks.append(
                        self.check_pii(
                            text=text_str,
                            output_parse_pii=False,
                            presidio_config=presidio_config,
                            request_data=kwargs,
                        )
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

    @log_guardrail_information
    async def async_post_call_success_hook(  # type: ignore
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
    ):
        """
        Output parse the response object to replace the masked tokens with user sent values
        """
        verbose_proxy_logger.debug(
            f"PII Masking Args: self.output_parse_pii={self.output_parse_pii}; type of response={type(response)}"
        )

        if self.output_parse_pii is False and litellm.output_parse_pii is False:
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

    def get_presidio_settings_from_request_data(
        self, data: dict
    ) -> Optional[PresidioPerRequestConfig]:
        if "metadata" in data:
            _metadata = data["metadata"]
            _guardrail_config = _metadata.get("guardrail_config")
            if _guardrail_config:
                _presidio_config = PresidioPerRequestConfig(**_guardrail_config)
                return _presidio_config

        return None

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass
