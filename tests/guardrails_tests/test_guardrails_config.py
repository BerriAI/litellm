# What is this?
## Unit Tests for guardrails config
import asyncio
import inspect
import os
import sys
import time
import traceback
from litellm._uuid import uuid
from datetime import datetime

import pytest
from pydantic import BaseModel

import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging

sys.path.insert(0, os.path.abspath("../.."))
from typing import Any, List, Literal, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Cache, completion, embedding
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import LiteLLMCommonStrings


class CustomLoggingIntegration(CustomLogger):
    def __init__(self) -> None:
        super().__init__()

    def logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        input: Optional[Any] = kwargs.get("input", None)
        messages: Optional[List] = kwargs.get("messages", None)
        if call_type == "completion":
            # assume input is of type messages
            if input is not None and isinstance(input, list):
                input[0]["content"] = "Hey, my name is [NAME]."
            if messages is not None and isinstance(messages, List):
                messages[0]["content"] = "Hey, my name is [NAME]."

            kwargs["input"] = input
            kwargs["messages"] = messages
        return kwargs, result


def test_guardrail_masking_logging_only():
    """
    Assert response is unmasked.

    Assert logged response is masked.
    """
    callback = CustomLoggingIntegration()

    with patch.object(callback, "log_success_event", new=MagicMock()) as mock_call:
        litellm.callbacks = [callback]
        messages = [{"role": "user", "content": "Hey, my name is Peter."}]
        response = completion(
            model="gpt-3.5-turbo", messages=messages, mock_response="Hi Peter!"
        )

        assert response.choices[0].message.content == "Hi Peter!"  # type: ignore

        time.sleep(3)
        mock_call.assert_called_once()

        print(mock_call.call_args.kwargs["kwargs"]["messages"][0]["content"])

        assert (
            mock_call.call_args.kwargs["kwargs"]["messages"][0]["content"]
            == "Hey, my name is [NAME]."
        )


def test_guardrail_list_of_event_hooks():
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

    cg = CustomGuardrail(
        guardrail_name="custom-guard", event_hook=["pre_call", "post_call"]
    )

    data = {"model": "gpt-3.5-turbo", "metadata": {"guardrails": ["custom-guard"]}}
    assert cg.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call)

    assert cg.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call)

    assert not cg.should_run_guardrail(
        data=data, event_type=GuardrailEventHooks.during_call
    )


def test_guardrail_info_response():
    from litellm.types.guardrails import (
        GuardrailInfoResponse,
        LitellmParams,
    )

    guardrail_info = GuardrailInfoResponse(
        guardrail_name="aporia-pre-guard",
        litellm_params=LitellmParams(
            guardrail="aporia",
            mode="pre_call",
        ),
        guardrail_info={
            "guardrail_name": "aporia-pre-guard",
            "litellm_params": {
                "guardrail": "aporia",
                "mode": "always_on",
            },
        },
    )

    assert guardrail_info.litellm_params.default_on == False
