import os
from typing import Literal, Optional, Union

from acuvity import Acuvity, GuardConfig, GuardName, ResponseMatch, Security
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_str,
)
from litellm.proxy._types import UserAPIKeyAuth

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
        Runs before the LLM API call
        Runs on only Input
        We just update the data here if there is a pii detection else if more detections are deteected then we raise exception.
        """

        _messages = data.get("messages")
        if _messages:
            msgs = [message.get("content") for message in _messages if message.get("content") is not None]

            verbose_proxy_logger.debug("Acuvity pre_call data: %s", msgs)
            resp = await self.acuvity_client.apex.scan_async(*msgs, guard_config=self.pre_guard_config_dict)

            # List of PII detectors to exclude from validation
            pii_detectors = [
                match
                for detector in [
                resp.guard_match(guard=GuardName.PII_DETECTOR),
                resp.guard_match(guard=GuardName.SECRETS_DETECTOR),
                resp.guard_match(guard=GuardName.KEYWORD_DETECTOR)
                ]
                for match in detector
                if match.response_match == ResponseMatch.YES
            ]

            for index, extraction in enumerate(resp.scan_response.extractions):
                # Skip if no match
                if resp.match_details[index].response_match != ResponseMatch.YES:
                    continue

                # Check for violations
                matched_checks = resp.match_details[index].matched_checks
                if len(matched_checks) > len(pii_detectors):
                    # Found violation beyond basic detectors
                    violation = matched_checks[0].to_dict()
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Violated guardrail policy",
                            "guard": violation
                        }
                    )

                if isinstance(_messages[index]["content"], str):
                    _messages[index]["content"] = extraction.data

                data["messages"] = _messages
                verbose_proxy_logger.info(
                    f"Acuvity pre call processed message: {data['messages']}"
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
        We decide here based on the config whether to reject the call based on violations.
        """
        _messages = data.get("messages")
        if _messages:
            msgs = [message.get("content") for message in _messages if message.get("content") is not None]

            verbose_proxy_logger.debug("Acuvity during_call data: %s", msgs)
            resp = await self.acuvity_client.apex.scan_async(*msgs, guard_config=self.during_guard_config_dict)

            if resp.scan_response.extractions:
                    for index, r in enumerate(resp.scan_response.extractions):
                        if resp.match_details[index].response_match == ResponseMatch.YES:
                            # return the 1st detection
                            raise HTTPException(
                                status_code=400, detail={
                                    "error": "Violated guardrail policy",
                                    "guard": resp.match_details[index].matched_checks[0].to_dict()
                                }
                            )
        pass

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        It can be used to reject a response
        """

        response_str: Optional[str] = convert_litellm_response_object_to_str(response)
        if response_str is not None:
            verbose_proxy_logger.debug("Acuvity post_call data: %s", response_str)
            resp = await self.acuvity_client.apex.scan_async(response_str, request_type= "Output" ,guard_config=self.post_guard_config_dict)

            if resp.scan_response.extractions:
                for index, r in enumerate(resp.scan_response.extractions):
                    if resp.match_details[index].response_match == ResponseMatch.YES:
                        # return the 1st detection
                        raise HTTPException(
                                status_code=400, detail={
                                    "error": "Violated guardrail policy",
                                    "guard": resp.match_details[index].matched_checks[0].to_dict()
                                }
                            )
        pass
