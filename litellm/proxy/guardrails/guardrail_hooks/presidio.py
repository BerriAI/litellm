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
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import aiohttp

import litellm  # noqa: E401
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

from litellm._uuid import uuid
from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import (
    GuardrailEventHooks,
    LitellmParams,
    PiiAction,
    PiiEntityType,
    PresidioPerRequestConfig,
)
from litellm.types.proxy.guardrails.guardrail_hooks.presidio import (
    PresidioAnalyzeRequest,
    PresidioAnalyzeResponseItem,
)
from litellm.types.utils import GuardrailStatus
from litellm.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


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
        apply_to_output: bool = False,
        presidio_ad_hoc_recognizers: Optional[str] = None,
        logging_only: Optional[bool] = None,
        pii_entities_config: Optional[
            Dict[Union[PiiEntityType, str], PiiAction]
        ] = None,
        presidio_language: Optional[str] = None,
        presidio_score_thresholds: Optional[
            Dict[Union[PiiEntityType, str], float]
        ] = None,
        **kwargs,
    ):
        if logging_only is True:
            self.logging_only = True
            kwargs["event_hook"] = GuardrailEventHooks.logging_only
        super().__init__(**kwargs)
        self.guardrail_provider = "presidio"
        self.pii_tokens: dict = (
            {}
        )  # mapping of PII token to original text - only used with Presidio `replace` operation
        self.mock_redacted_text = mock_redacted_text
        self.output_parse_pii = output_parse_pii or False
        self.apply_to_output = apply_to_output
        self.pii_entities_config: Dict[Union[PiiEntityType, str], PiiAction] = (
            pii_entities_config or {}
        )
        self.presidio_score_thresholds: Dict[Union[PiiEntityType, str], float] = (
            presidio_score_thresholds or {}
        )
        self.presidio_language = presidio_language or "en"
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

    def _get_presidio_analyze_request_payload(
        self,
        text: str,
        presidio_config: Optional[PresidioPerRequestConfig],
        request_data: dict,
    ) -> PresidioAnalyzeRequest:
        """
        Construct the payload for the Presidio analyze request

        API Ref: https://microsoft.github.io/presidio/api-docs/api-docs.html#tag/Analyzer/paths/~1analyze/post
        """
        analyze_payload: PresidioAnalyzeRequest = PresidioAnalyzeRequest(
            text=text,
            language=self.presidio_language,
        )
        ##################################################################
        ###### Check if user has configured any params for this guardrail
        ################################################################
        if self.ad_hoc_recognizers is not None:
            analyze_payload["ad_hoc_recognizers"] = self.ad_hoc_recognizers

        if self.pii_entities_config:
            analyze_payload["entities"] = list(self.pii_entities_config.keys())

        ##################################################################
        ######### End of adding config params
        ##################################################################

        # Check if client side request passed any dynamic params
        if presidio_config and presidio_config.language:
            analyze_payload["language"] = presidio_config.language

        casted_analyze_payload: dict = cast(dict, analyze_payload)
        casted_analyze_payload.update(
            self.get_guardrail_dynamic_request_body_params(request_data=request_data)
        )
        return cast(PresidioAnalyzeRequest, casted_analyze_payload)

    async def analyze_text(
        self,
        text: str,
        presidio_config: Optional[PresidioPerRequestConfig],
        request_data: dict,
    ) -> Union[List[PresidioAnalyzeResponseItem], Dict]:
        """
        Send text to the Presidio analyzer endpoint and get analysis results
        """
        try:
            # Skip empty or whitespace-only text to avoid Presidio errors
            # Common in tool/function calling where assistant content is empty
            if not text or len(text.strip()) == 0:
                verbose_proxy_logger.debug(
                    "Skipping Presidio analysis for empty/whitespace-only text"
                )
                return []

            async with aiohttp.ClientSession() as session:
                if self.mock_redacted_text is not None:
                    return self.mock_redacted_text

                # Make the request to /analyze
                analyze_url = f"{self.presidio_analyzer_api_base}analyze"

                analyze_payload: PresidioAnalyzeRequest = (
                    self._get_presidio_analyze_request_payload(
                        text=text,
                        presidio_config=presidio_config,
                        request_data=request_data,
                    )
                )

                verbose_proxy_logger.debug(
                    "Making request to: %s with payload: %s",
                    analyze_url,
                    analyze_payload,
                )

                async with session.post(analyze_url, json=analyze_payload) as response:
                    analyze_results = await response.json()
                    verbose_proxy_logger.debug("analyze_results: %s", analyze_results)

                    # Handle error responses from Presidio (e.g., {'error': 'No text provided'})
                    # Presidio may return a dict instead of a list when errors occur
                    if isinstance(analyze_results, dict):
                        if "error" in analyze_results:
                            verbose_proxy_logger.warning(
                                "Presidio analyzer returned error: %s, returning empty list",
                                analyze_results.get("error")
                            )
                            return []
                        # If it's a dict but not an error, try to process it as a single item
                        verbose_proxy_logger.debug(
                            "Presidio returned dict (not list), attempting to process as single item"
                        )
                        try:
                            return [PresidioAnalyzeResponseItem(**analyze_results)]
                        except Exception as e:
                            verbose_proxy_logger.warning(
                                "Failed to parse Presidio dict response: %s, returning empty list",
                                e
                            )
                            return []

                    # Normal case: list of results
                    final_results = []
                    for item in analyze_results:
                        try:
                            final_results.append(PresidioAnalyzeResponseItem(**item))
                        except TypeError as te:
                            # Handle case where item is not a dict (shouldn't happen, but be defensive)
                            verbose_proxy_logger.warning(
                                "Skipping invalid Presidio result item: %s (error: %s)",
                                item,
                                te,
                            )
                            continue
                    return final_results
        except Exception as e:
            raise e

    async def anonymize_text(
        self,
        text: str,
        analyze_results: Any,
        output_parse_pii: bool,
        masked_entity_count: Dict[str, int],
    ) -> str:
        """
        Send analysis results to the Presidio anonymizer endpoint to get redacted text
        """
        try:
            # If there are no detections after filtering, return the original text
            if isinstance(analyze_results, list) and len(analyze_results) == 0:
                return text

            async with aiohttp.ClientSession() as session:
                # Make the request to /anonymize
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
                        entity_type = item.get("entity_type", None)
                        if entity_type is not None:
                            masked_entity_count[entity_type] = (
                                masked_entity_count.get(entity_type, 0) + 1
                            )
                    return redacted_text["text"]
                else:
                    raise Exception(f"Invalid anonymizer response: {redacted_text}")
        except Exception as e:
            raise e

    def filter_analyze_results_by_score(
        self, analyze_results: Union[List[PresidioAnalyzeResponseItem], Dict]
    ) -> Union[List[PresidioAnalyzeResponseItem], Dict]:
        """
        Drop detections that fall below configured per-entity score thresholds.
        """
        if not self.presidio_score_thresholds:
            return analyze_results

        if not isinstance(analyze_results, list):
            return analyze_results

        filtered_results: List[PresidioAnalyzeResponseItem] = []
        for item in analyze_results:
            entity_type = item.get("entity_type")
            score = item.get("score")

            threshold = None
            if entity_type is not None:
                threshold = self.presidio_score_thresholds.get(entity_type)
            if threshold is None:
                threshold = self.presidio_score_thresholds.get("ALL")

            if threshold is not None:
                if score is None or score < threshold:
                    continue

            filtered_results.append(item)

        return filtered_results

    def raise_exception_if_blocked_entities_detected(
        self, analyze_results: Union[List[PresidioAnalyzeResponseItem], Dict]
    ):
        """
        Raise an exception if blocked entities are detected
        """
        if self.pii_entities_config is None:
            return

        if isinstance(analyze_results, Dict):
            # if mock testing is enabled, analyze_results is a dict
            # we don't need to raise an exception in this case
            return

        for result in analyze_results:
            entity_type = result.get("entity_type")

            if entity_type:
                # Check if entity_type is in config (supports both enum and string)
                if (
                    entity_type in self.pii_entities_config
                    and self.pii_entities_config[entity_type] == PiiAction.BLOCK
                ):
                    raise BlockedPiiEntityError(
                        entity_type=entity_type,
                        guardrail_name=self.guardrail_name,
                    )

    async def check_pii(
        self,
        text: str,
        output_parse_pii: bool,
        presidio_config: Optional[PresidioPerRequestConfig],
        request_data: dict,
    ) -> str:
        """
        Calls Presidio Analyze + Anonymize endpoints for PII Analysis + Masking
        """
        start_time = datetime.now()
        analyze_results: Optional[Union[List[PresidioAnalyzeResponseItem], Dict]] = None
        status: GuardrailStatus = "success"
        masked_entity_count: Dict[str, int] = {}
        exception_str: str = ""
        try:
            if self.mock_redacted_text is not None:
                redacted_text = self.mock_redacted_text
            else:
                # First get analysis results
                analyze_results = await self.analyze_text(
                    text=text,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )

                verbose_proxy_logger.debug("analyze_results: %s", analyze_results)

                # Apply score threshold filtering if configured
                analyze_results = self.filter_analyze_results_by_score(
                    analyze_results=analyze_results
                )

                ####################################################
                # Blocked Entities check
                ####################################################
                self.raise_exception_if_blocked_entities_detected(
                    analyze_results=analyze_results
                )

                # Then anonymize the text using the analysis results
                return await self.anonymize_text(
                    text=text,
                    analyze_results=analyze_results,
                    output_parse_pii=output_parse_pii,
                    masked_entity_count=masked_entity_count,
                )
            return redacted_text["text"]
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            raise e
        finally:
            ####################################################
            # Create Guardrail Trace for logging on Langfuse, Datadog, etc.
            ####################################################
            guardrail_json_response: Union[Exception, str, dict, List[dict]] = {}
            if status == "success":
                if isinstance(analyze_results, List):
                    guardrail_json_response = [dict(item) for item in analyze_results]
            else:
                guardrail_json_response = exception_str
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response=guardrail_json_response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                masked_entity_count=masked_entity_count,
            )

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
            messages = data.get("messages", None)
            if messages is None:
                return data
            tasks = []
            task_mappings: List[
                Tuple[int, Optional[int]]
            ] = []  # Track (message_index, content_index) for each task

            for msg_idx, m in enumerate(messages):
                content = m.get("content", None)
                if content is None:
                    continue
                if isinstance(content, str):
                    tasks.append(
                        self.check_pii(
                            text=content,
                            output_parse_pii=self.output_parse_pii,
                            presidio_config=presidio_config,
                            request_data=data,
                        )
                    )
                    task_mappings.append(
                        (msg_idx, None)
                    )  # None indicates string content
                elif isinstance(content, list):
                    for content_idx, c in enumerate(content):
                        text_str = c.get("text", None)
                        if text_str is None:
                            continue
                        tasks.append(
                            self.check_pii(
                                text=text_str,
                                output_parse_pii=self.output_parse_pii,
                                presidio_config=presidio_config,
                                request_data=data,
                            )
                        )
                        task_mappings.append((msg_idx, int(content_idx)))

            responses = await asyncio.gather(*tasks)

            # Map responses back to the correct message and content item
            for task_idx, r in enumerate(responses):
                mapping = task_mappings[task_idx]
                msg_idx = cast(int, mapping[0])
                content_idx_optional = cast(Optional[int], mapping[1])
                content = messages[msg_idx].get("content", None)
                if content is None:
                    continue
                if isinstance(content, str) and content_idx_optional is None:
                    messages[msg_idx][
                        "content"
                    ] = r  # replace content with redacted string
                elif isinstance(content, list) and content_idx_optional is not None:
                    messages[msg_idx]["content"][content_idx_optional]["text"] = r

            verbose_proxy_logger.debug(
                f"Presidio PII Masking: Redacted pii message: {data['messages']}"
            )
            data["messages"] = messages
            return data
        except Exception as e:
            raise e

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
            task_mappings: List[
                Tuple[int, Optional[int]]
            ] = []  # Track (message_index, content_index) for each task

            if messages is None:
                return kwargs, result

            presidio_config = self.get_presidio_settings_from_request_data(kwargs)

            for msg_idx, m in enumerate(messages):
                content = m.get("content", None)
                if content is None:
                    continue
                if isinstance(content, str):
                    tasks.append(
                        self.check_pii(
                            text=content,
                            output_parse_pii=False,
                            presidio_config=presidio_config,
                            request_data=kwargs,
                        )
                    )  # need to pass separately b/c presidio has context window limits
                    task_mappings.append(
                        (msg_idx, None)
                    )  # None indicates string content
                elif isinstance(content, list):
                    for content_idx, c in enumerate(content):
                        text_str = c.get("text", None)
                        if text_str is None:
                            continue
                        tasks.append(
                            self.check_pii(
                                text=text_str,
                                output_parse_pii=False,
                                presidio_config=presidio_config,
                                request_data=kwargs,
                            )
                        )
                        task_mappings.append((msg_idx, int(content_idx)))

            responses = await asyncio.gather(*tasks)

            # Map responses back to the correct message and content item
            for task_idx, r in enumerate(responses):
                mapping = task_mappings[task_idx]
                msg_idx = cast(int, mapping[0])
                content_idx_optional = cast(Optional[int], mapping[1])
                content = messages[msg_idx].get("content", None)
                if content is None:
                    continue
                if isinstance(content, str) and content_idx_optional is None:
                    messages[msg_idx][
                        "content"
                    ] = r  # replace content with redacted string
                elif isinstance(content, list) and content_idx_optional is not None:
                    messages[msg_idx]["content"][content_idx_optional]["text"] = r

            verbose_proxy_logger.debug(
                f"Presidio PII Masking: Redacted pii message: {messages}"
            )
            kwargs["messages"] = messages

        return kwargs, result

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

        if self.apply_to_output is True:
            return await self._mask_output_response(
                response=response, request_data=data
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

    async def _mask_output_response(
        self,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
        request_data: dict,
    ):
        """
        Apply Presidio masking on model responses (non-streaming).
        """
        if not isinstance(response, ModelResponse):
            return response

        # skip streaming here; handled in async_post_call_streaming_iterator_hook
        if response.choices and isinstance(response.choices[0], StreamingChoices):
            return response

        presidio_config = self.get_presidio_settings_from_request_data(
            request_data or {}
        )

        for choice in response.choices:
            # Type narrowing: StreamingChoices doesn't have .message attribute
            if not hasattr(choice, "message"):
                continue
            content = getattr(choice.message, "content", None)
            if content is None:
                continue
            if isinstance(content, str):
                choice.message.content = await self.check_pii(
                    text=content,
                    output_parse_pii=False,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text_value = item.get("text")
                    if text_value is None:
                        continue
                    item["text"] = await self.check_pii(
                        text=text_value,
                        output_parse_pii=False,
                        presidio_config=presidio_config,
                        request_data=request_data,
                    )

        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Process streaming response chunks to unmask PII tokens when needed.

        If PII processing is enabled, this collects all chunks, applies PII unmasking,
        and returns a reconstructed stream. Otherwise, it passes through the original stream.
        """
        # If we need to mask model output, collect the full stream, apply masking, and replay it.
        if self.apply_to_output:
            from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
            from litellm.types.utils import Choices, Message

            try:
                collected_content = ""
                last_chunk = None

                async for chunk in response:
                    last_chunk = chunk

                    if (
                        hasattr(chunk, "choices")
                        and chunk.choices
                        and hasattr(chunk.choices[0], "delta")
                        and hasattr(chunk.choices[0].delta, "content")
                        and isinstance(chunk.choices[0].delta.content, str)
                    ):
                        collected_content += chunk.choices[0].delta.content

                if not last_chunk:
                    async for chunk in response:
                        yield chunk
                    return

                presidio_config = self.get_presidio_settings_from_request_data(
                    request_data or {}
                )
                masked_content = await self.check_pii(
                    text=collected_content,
                    output_parse_pii=False,
                    presidio_config=presidio_config,
                    request_data=request_data,
                )

                mock_response = MockResponseIterator(
                    model_response=ModelResponse(
                        id=last_chunk.id,
                        object=last_chunk.object,
                        created=last_chunk.created,
                        model=last_chunk.model,
                        choices=[
                            Choices(
                                message=Message(
                                    role="assistant",
                                    content=masked_content,
                                ),
                                index=0,
                                finish_reason="stop",
                            )
                        ],
                    ),
                    json_mode=False,
                )

                async for chunk in mock_response:
                    yield chunk
                return

            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error masking streaming PII output: {str(e)}"
                )
                async for chunk in response:
                    yield chunk
                return

        # If PII unmasking not needed, just pass through the original stream
        if not (self.output_parse_pii and self.pii_tokens):
            async for chunk in response:
                yield chunk
            return

        # Import here to avoid circular imports
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.types.utils import Choices, Message

        try:
            # Collect all chunks to process them together
            collected_content = ""
            last_chunk = None

            async for chunk in response:
                last_chunk = chunk

                # Extract content safely with proper attribute checks
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and hasattr(chunk.choices[0], "delta")
                    and hasattr(chunk.choices[0].delta, "content")
                    and isinstance(chunk.choices[0].delta.content, str)
                ):
                    collected_content += chunk.choices[0].delta.content

            # No need to proceed if we didn't capture a valid chunk
            if not last_chunk:
                async for chunk in response:
                    yield chunk
                return

            # Apply PII unmasking to the complete content
            for token, original_text in self.pii_tokens.items():
                collected_content = collected_content.replace(token, original_text)

            # Reconstruct the response with unmasked content
            mock_response = MockResponseIterator(
                model_response=ModelResponse(
                    id=last_chunk.id,
                    object=last_chunk.object,
                    created=last_chunk.created,
                    model=last_chunk.model,
                    choices=[
                        Choices(
                            message=Message(
                                role="assistant",
                                content=collected_content,
                            ),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                ),
                json_mode=False,
            )

            # Return the reconstructed stream
            async for chunk in mock_response:
                yield chunk

        except Exception as e:
            verbose_proxy_logger.error(f"Error in PII streaming processing: {str(e)}")
            # Fallback to original stream on error
            async for chunk in response:
                yield chunk

    def get_presidio_settings_from_request_data(
        self, data: dict
    ) -> Optional[PresidioPerRequestConfig]:
        if "metadata" in data:
            _metadata = data.get("metadata", None)
            if _metadata is None:
                return None
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

    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """
        UI will call this function to check:
            1. If the connection to the guardrail is working
            2. When Testing the guardrail with some text, this function will be called with the input text and returns a text after applying the guardrail
        """
        texts = inputs.get("texts", [])

        new_texts = []
        for text in texts:
            modified_text = await self.check_pii(
                text=text,
                output_parse_pii=self.output_parse_pii,
                presidio_config=None,
                request_data=request_data or {},
            )
            new_texts.append(modified_text)
        inputs["texts"] = new_texts
        return inputs

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """
        Update the guardrails litellm params in memory
        """
        super().update_in_memory_litellm_params(litellm_params)
        if litellm_params.pii_entities_config:
            self.pii_entities_config = litellm_params.pii_entities_config
        if litellm_params.presidio_score_thresholds:
            self.presidio_score_thresholds = litellm_params.presidio_score_thresholds
