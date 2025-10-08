import os
import sys
import traceback
from litellm._uuid import uuid
import pytest
from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import io
import os
import time
import json

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.router import Router
import asyncio
from typing import Optional
from litellm.types.utils import StandardLoggingPayload, Usage, ModelInfoBase
from litellm.integrations.custom_logger import CustomLogger


class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.recorded_usage: Optional[Usage] = None
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload = kwargs.get("standard_logging_object")
        self.standard_logging_payload = standard_logging_payload
        print(
            "standard_logging_payload",
            json.dumps(standard_logging_payload, indent=4, default=str),
        )

        pass

@pytest.mark.asyncio
@pytest.mark.parametrize("model", [
    None,
    "omni-moderation-latest",
    "router-internal-moderation-model"
])

async def test_moderations_api_logging(model):
    """
    When moderations API is called, it should log the event on standard_logging_payload
    """
    custom_logger = TestCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)


    MODEL_GROUP = "internal-moderation-model"
    router = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "openai/omni-moderation-latest",
                },
            }
        ]
    )

    input_content = "Hello, how are you?"
    if model == "router-internal-moderation-model":
        response = await router.amoderation(
            input=input_content,
            model=MODEL_GROUP,
        )
    else:
        response = await litellm.amoderation(
            input=input_content,
            model=model,
        )

    print("response", json.dumps(response, indent=4, default=str))

    await asyncio.sleep(2)

    assert custom_logger.standard_logging_payload is not None

    # validate the standard_logging_payload
    standard_logging_payload: StandardLoggingPayload = custom_logger.standard_logging_payload
    assert standard_logging_payload["call_type"] == litellm.utils.CallTypes.amoderation.value
    assert standard_logging_payload["status"] == "success"
    assert standard_logging_payload["custom_llm_provider"] == litellm.LlmProviders.OPENAI.value


    # assert the logged input == input
    assert standard_logging_payload["messages"][0]["content"] == input_content

    # assert the logged response == response user received client side 
    assert dict(standard_logging_payload["response"]) == response.model_dump()


    # if router used, validate model_group is logged as expected
    if model == "router-internal-moderation-model":
        assert standard_logging_payload["model_group"] == MODEL_GROUP

