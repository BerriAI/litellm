import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import json
import sys
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailEventHooks


class myCustomGuardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

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
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        # In this guardrail, if a user inputs `litellm` we will mask it.
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "litellm" in _content.lower():
                        _content = _content.replace("litellm", "********")
                        message["content"] = _content

        verbose_proxy_logger.debug(
            "async_pre_call_hook: Message after masking %s", _messages
        )

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation"],
    ):
        """
        Runs in parallel to LLM API call
        Runs on only Input
        """

        # this works the same as async_pre_call_hook, but just runs in parallel as the LLM API Call
        # In this guardrail, if a user inputs `litellm` we will mask it.
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "litellm" in _content.lower():
                        _content = _content.replace("litellm", "********")
                        message["content"] = _content

            verbose_proxy_logger.debug(
                "async_pre_call_hook: Message after masking %s", _messages
            )
        pass

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        If a response contains the word "coffee" -> we will raise an exception
        """
        verbose_proxy_logger.debug("async_pre_call_hook response: %s", response)
        if isinstance(response, litellm.ModelResponse):
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    verbose_proxy_logger.debug("async_pre_call_hook choice: %s", choice)
                    if (
                        choice.message.content
                        and isinstance(choice.message.content, str)
                        and "coffee" in choice.message.content
                    ):
                        raise ValueError("Guardrail failed Coffee Detected")
