import os
from typing import List, Literal, Optional, Union

from acuvity import Acuvity, GuardConfig, GuardName, Matches, ResponseMatch, ScanResponseMatch, Security
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.utils import EmbeddingResponse, ImageResponse, ModelResponse, StreamingChoices

GUARDRAIL_NAME = "acuvity"

class AcuvityGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None,
        **kwargs,
    ):
        try:
            if kwargs.get("event_hook") == "pre_call":
                self.pre_guard_config_dict = kwargs.pop("vendor_params", None)  # None as default if not found
                self.pre_guard_config = GuardConfig(self.pre_guard_config_dict)
            if kwargs.get("event_hook") == "during_call":
                self.during_guard_config_dict = kwargs.pop("vendor_params", None)  # None as default if not found
                self.during_guard_config = GuardConfig(self.during_guard_config_dict)
                if self.during_guard_config.redaction_keys:
                    raise ValueError("acuvity guard config cannot do redaction for during_call mode, please add redactions in the pre_call")
            if kwargs.get("event_hook") == "post_call":
                self.post_guard_config_dict = kwargs.pop("vendor_params", None)  # None as default if not found
                self.post_guard_config = GuardConfig(self.post_guard_config_dict)
        except Exception as e:
            raise ValueError("Acuvity guard config cannot be parsed") from e

        self.acuvity_client = Acuvity(Security(token=(api_key or os.environ["ACUVITY_TOKEN"])))
        super().__init__(**kwargs)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs on request before the LLM API call

        Returns:
            The processed request, potentially with redacted content

        Raises:
            HTTPException: If guardrail policies are violated
        """
        redacted = False
        _messages = data.get("messages")
        if _messages:
            msgs = [message.get("content") for message in _messages if message.get("content") is not None]

            verbose_proxy_logger.debug("Acuvity pre_call data: %s", msgs)
            resp = await self.acuvity_client.apex.scan_async(*msgs, guard_config=self.pre_guard_config_dict)

            redaction_set = set(self.pre_guard_config.redaction_keys)
            for index, extraction in enumerate(resp.scan_response.extractions):
                # Skip if no match
                if resp.match_details[index].response_match != ResponseMatch.YES:
                    continue

                if self._check_violations(resp, redaction_set, index):
                    raise HTTPException(
                                status_code=400, detail={
                                    "error": "Violated guardrail policy",
                                    "guard": self.matched_guards(resp.match_details[index])
                                }
                            )
                #If no violations then, check if we need to replace the redacted response.
                if isinstance(_messages[index]["content"], str):
                    if _messages[index]["content"] != extraction.data:
                        redacted = True
                        _messages[index]["content"] = extraction.data

            data["messages"] = _messages
            verbose_proxy_logger.info(
                f"Acuvity pre call processed message: {data['messages']}, redaction applied {redacted}"
            )
        return data

    @log_guardrail_information
    async def async_moderation_hook(
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
        """
        For any triggered guards we raise exceptions.
        """
        _messages = data.get("messages")
        if _messages:
            msgs = [message.get("content") for message in _messages if message.get("content") is not None]

            verbose_proxy_logger.debug("Acuvity during_call data: %s", msgs)
            resp = await self.acuvity_client.apex.scan_async(*msgs, guard_config=self.during_guard_config_dict)

            if resp.scan_response.extractions:
                    for index, r in enumerate(resp.scan_response.extractions):
                        if resp.match_details[index].response_match == ResponseMatch.YES:
                            raise HTTPException(
                                status_code=400, detail={
                                    "error": "Violated guardrail policy",
                                    "guard": self.matched_guards(resp.match_details[index])
                                }
                            )
        pass

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
    ):
        """
        Runs on response from LLM API call

        Returns:
            The processed response, potentially with redacted content

        Raises:
            HTTPException: If guardrail policies are violated
        """

        # we are intrested/supported only in text msgs and non streaming.
        _messages = self._extract_response_messages(response)
        redacted = False
        if _messages:
            verbose_proxy_logger.debug("Acuvity post_call data: %s", _messages)
            resp = await self.acuvity_client.apex.scan_async(_messages, request_type= "Output" ,guard_config=self.post_guard_config_dict)

            redaction_set = set(self.post_guard_config.redaction_keys)
            for index, extraction in enumerate(resp.scan_response.extractions):
                # Skip if no match
                if resp.match_details[index].response_match != ResponseMatch.YES:
                    continue

                if self._check_violations(resp, redaction_set, index):
                    raise HTTPException(
                                status_code=400, detail={
                                    "error": "Violated guardrail policy",
                                    "guard": self.matched_guards(resp.match_details[index])
                                }
                            )

                #If no violations then, check if we need to replace the redacted response.
                if isinstance(response, ModelResponse) and not isinstance(response.choices[0], StreamingChoices):
                    if response.choices[0].message.content != extraction.data:
                        redacted = True
                        response.choices[0].message.content = extraction.data

            verbose_proxy_logger.info(
                    f"Acuvity post call processed message: {response.choices[0]}, redaction applied {redacted}"
                )
        return response

    def _extract_response_messages(self, response_obj: Union[ModelResponse, EmbeddingResponse, ImageResponse]) -> Optional[str]:
        """
        Extracts the first non-streaming text message from a response object.

        Args:
            response_obj: The response object to process

        Returns:
            The first message content if available, None otherwise
        """
        if not isinstance(response_obj, ModelResponse):
            return None

        for choice in response_obj.choices:
            if (not isinstance(choice, StreamingChoices) and
                choice.message.content and
                isinstance(choice.message.content, str)):
                return choice.message.content
        return None

    def matched_guards(self, all_guard_list: Matches)-> List[str]:
        matched_guard_list : List[str] = []
        for m in all_guard_list.matched_checks:
            matched_guard_list.append(m.guard_name.name)
        return matched_guard_list

    def _check_violations(self, resp: ScanResponseMatch, redaction_set: set ,index: int) -> bool:
        # find all the detected PII guards.
        detected_pii_detectors = [match
                for detector in [
                resp.guard_match(guard=GuardName.PII_DETECTOR, msg_index=index),
                resp.guard_match(guard=GuardName.SECRETS_DETECTOR, msg_index=index),
                resp.guard_match(guard=GuardName.KEYWORD_DETECTOR, msg_index=index)
                ]
                for match in detector
                if match.response_match == ResponseMatch.YES
                ]

        detected_guards = resp.match_details[index].matched_checks
        detected_pii_extraction_vals : list[str] = []
        for m in detected_guards:
            detected_pii_extraction_vals.extend(m.match_values)

        # we reject the call if either of the following condition meets
        # 1. we found more violations than the textual detections.
        # 2. we found any textual detections which are not part of redaction set.
        #   2a. example: if pii guard has detect ssn but no redaction was set, then we flag it for violations.
        if len(detected_guards) > len(detected_pii_detectors) or \
                len([item for item in detected_pii_extraction_vals if item not in redaction_set]) > 0:
            return True
