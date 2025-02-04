import os
from typing import Any, Dict, List, Literal, Optional, Union

import litellm
from acuvity import Acuvity, Security
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_str,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailEventHooks

GUARDRAIL_NAME = "acuvity"

class AcuvityGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None,
        analyzer_names: Optional[List[str]] = None,
        redact_entities: Optional[List[str]] = None,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs
        self.acuvity_client = Acuvity(Security(token=(api_key or os.environ["ACUVITY_TOKEN"])))
        # Extract 'analyzer_name' and store it
        self.analyzer_names = analyzer_names
        self.redact_entities = redact_entities
        super().__init__(**kwargs)

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
        Use this if you want to MODIFY the input
        """

        # In this guardrail, if a user inputs `litellm` we will mask it and then send it to the LLM
        _messages = data.get("messages")
        if _messages:
            msgs = [message.get("content") for message in _messages if message.get("content") is not None]

            verbose_proxy_logger.debug("**** ACUVITY ANALYZER: %s", self.analyzer_names)
            resp = await self.acuvity_client.apex.scan_async(*msgs, analyzers=self.analyzer_names, redactions=self.redact_entities)
            if resp.extractions is not None:
                resp_msgs = [extr.data for extr in resp.extractions if extr.data is not None]
                for (message, resp_message) in zip(_messages, resp_msgs):
                    message["content"] = resp_message

        verbose_proxy_logger.debug(
            "********************  ACUVITY: async_pre_call_hook: Message after masking %s", _messages
        )

        return data

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
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """

        # this works the same as async_pre_call_hook, but just runs in parallel as the LLM API Call
        # In this guardrail, if a user inputs `litellm` we will mask it.
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "litellm" in _content.lower():
                        raise ValueError("Guardrail failed words - `litellm` detected")

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
        verbose_proxy_logger.debug("********************  ACUVITY: async_post_call_hook response: %s", response)
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug("SKIPPING GUARDRAIL CHECK")
            return

        response_str: Optional[str] = convert_litellm_response_object_to_str(response)
        if response_str is not None:
            verbose_proxy_logger.debug("\n calling with analyzer %s and redactions %s", self.analyzer_names, self.redact_entities)
            resp = await self.acuvity_client.apex.scan_async(response_str, analyzers=self.analyzer_names, redactions=self.redact_entities)
            print("\n\n *** RESP: ", resp)
            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )
