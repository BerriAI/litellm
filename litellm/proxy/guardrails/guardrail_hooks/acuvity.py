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
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs
        self.acuvity_client = Acuvity(Security(token=(api_key or os.environ["ACUVITY_TOKEN"])))
        # Extract 'analyzer_name' and store it
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

            verbose_proxy_logger.debug("**** ACUVITY GUARDS: %s", self.optional_params.get("vendor_specific_params"))
            resp = await self.acuvity_client.apex.scan_async(*msgs, guard_config=self.optional_params.get("vendor_specific_params"))
            print(resp.matches())

        verbose_proxy_logger.debug(
            "********************  ACUVITY: async_pre_call_hook: Message after masking %s", _messages
        )

        return data

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
            verbose_proxy_logger.debug("**** ACUVITY GUARDS: %s", self.optional_params.get("vendor_specific_params"))
            resp = await self.acuvity_client.apex.scan_async(response_str, guard_config=self.optional_params.get("vendor_specific_params"))
            print("\n *** RESPONSE *** ",resp.matches())
