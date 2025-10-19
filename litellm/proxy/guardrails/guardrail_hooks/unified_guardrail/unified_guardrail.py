"""
Unified Guardrail, leveraging LiteLLM's /applyGuardrail endpoint

1. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_pre_call_hook
2. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_success_hook
3. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_streaming_iterator_hook
"""

import asyncio
import os
from datetime import datetime
from typing import (
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

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GuardrailStatus, ModelResponseStream

GUARDRAIL_NAME = "unified_llm_guardrails"


class UnifiedLLMGuardrails(CustomLogger):
    def __init__(
        self,
        **kwargs,
    ):

        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "UnifiedLLMGuardrails initialized with optional_params: %s",
            self.optional_params,
        )

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
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Union[Exception, str, dict, None]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """
        verbose_proxy_logger.debug("Running UnifiedLLMGuardrails pre-call hook")

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)
        if guardrail_to_apply is None:
            return data

        _messages = data.get("messages")
        if _messages is None:
            return data
        tasks = []
        task_mappings: List[Tuple[int, Optional[int]]] = (
            []
        )  # Track (message_index, content_index) for each task

        for msg_idx, m in enumerate(_messages):
            content = m.get("content", None)
            if content is None:
                continue
            if isinstance(content, str):
                tasks.append(guardrail_to_apply.apply_guardrail(text=content))
                task_mappings.append((msg_idx, None))  # None indicates string content
            elif isinstance(content, list):
                for content_idx, c in enumerate(content):
                    text_str = c.get("text", None)
                    if text_str is None:
                        continue
                    tasks.append(guardrail_to_apply.apply_guardrail(text=text_str))
                    task_mappings.append((msg_idx, int(content_idx)))

            responses = await asyncio.gather(*tasks)

            # Map responses back to the correct message and content item
            for task_idx, r in enumerate(responses):
                mapping = task_mappings[task_idx]
                msg_idx = cast(int, mapping[0])
                content_idx_optional = cast(Optional[int], mapping[1])
                content = _messages[msg_idx].get("content", None)
                if content is None:
                    continue
                if isinstance(content, str) and content_idx_optional is None:
                    _messages[msg_idx][
                        "content"
                    ] = r  # replace content with redacted string
                elif isinstance(content, list) and content_idx_optional is not None:
                    _messages[msg_idx]["content"][content_idx_optional]["text"] = r

            verbose_proxy_logger.debug(
                f"UnifiedLLMGuardrails: Redacted message: {_messages}"
            )
            data["messages"] = _messages
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

        Uses Enkrypt AI guardrails to check the response for policy violations, PII, and injection attacks
        """
        pass
        # from litellm.proxy.common_utils.callback_utils import (
        #     add_guardrail_to_applied_guardrails_header,
        # )
        # from litellm.types.guardrails import GuardrailEventHooks

        # if (
        #     self.should_run_guardrail(
        #         data=data, event_type=GuardrailEventHooks.post_call
        #     )
        #     is not True
        # ):
        #     return

        # verbose_proxy_logger.debug(
        #     "async_post_call_success_hook response: %s", response
        # )

        # # Check if the ModelResponse has text content in its choices
        # # to avoid sending empty content to EnkryptAI (e.g., during tool calls)
        # if isinstance(response, litellm.ModelResponse):
        #     has_text_content = False
        #     for choice in response.choices:
        #         if isinstance(choice, litellm.Choices):
        #             if choice.message.content and isinstance(
        #                 choice.message.content, str
        #             ):
        #                 has_text_content = True
        #                 break

        #     if not has_text_content:
        #         verbose_proxy_logger.warning(
        #             "EnkryptAI: not running guardrail. No output text in response"
        #         )
        #         return

        #     for choice in response.choices:
        #         if isinstance(choice, litellm.Choices):
        #             verbose_proxy_logger.debug(
        #                 "async_post_call_success_hook choice: %s", choice
        #             )
        #             if choice.message.content and isinstance(
        #                 choice.message.content, str
        #             ):
        #                 result = await self._call_enkryptai_guardrails(
        #                     prompt=choice.message.content,
        #                     request_data=data,
        #                 )

        #                 verbose_proxy_logger.debug(
        #                     "Guardrails async_post_call_success_hook result: %s", result
        #                 )

        #                 # Process the guardrails response
        #                 processed_result = self._process_enkryptai_guardrails_response(
        #                     result
        #                 )
        #                 attacks_detected = processed_result["attacks_detected"]

        #                 # If any attacks are detected, raise an error
        #                 if attacks_detected:
        #                     error_message = self._create_error_message(processed_result)
        #                     raise ValueError(error_message)

        # # Add guardrail to applied guardrails header
        # add_guardrail_to_applied_guardrails_header(
        #     request_data=data, guardrail_name=self.guardrail_name
        # )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        See Aim guardrail implementation for an example - https://github.com/BerriAI/litellm/blob/d0e022cfacb8e9ebc5409bb652059b6fd97b45c0/litellm/proxy/guardrails/guardrail_hooks/aim.py#L168

        Triggered by mode: 'post_call'
        """
        async for item in response:
            yield item
