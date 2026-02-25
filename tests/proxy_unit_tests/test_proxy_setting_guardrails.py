import json
import os
import sys
from unittest import mock

from dotenv import load_dotenv

load_dotenv()
import asyncio
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import openai
import pytest
from fastapi import Response
from fastapi.testclient import TestClient

import litellm
from litellm.proxy.proxy_server import (  # Replace with the actual module where your FastAPI router is defined
    initialize,
    router,
    save_worker_config,
)


@pytest.fixture
def client():
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_guardrails_config.yaml"
    with mock.patch("litellm.proxy.proxy_server.premium_user", True):
        asyncio.run(initialize(config=config_fp))
    from litellm.proxy.proxy_server import app

    return TestClient(app)


# raise openai.AuthenticationError
def test_active_callbacks(client):
    response = client.get("/active/callbacks")

    print("response", response)
    print("response.text", response.text)
    print("response.status_code", response.status_code)

    json_response = response.json()
    print(f"json_response={json_response}")
    _active_callbacks = json_response["litellm.callbacks"]

    expected_callback_names = [
        "lakeraAI_Moderation",
        "_OPTIONAL_PromptInjectionDetectio",
        "_ENTERPRISE_SecretDetection",
    ]

    for callback_name in expected_callback_names:
        # check if any of the callbacks have callback_name as a substring
        found_match = False
        for callback in _active_callbacks:
            if callback_name in callback:
                found_match = True
                break
        assert (
            found_match is True
        ), f"{callback_name} not found in _active_callbacks={_active_callbacks}"

    assert not any(
        "_ENTERPRISE_OpenAI_Moderation" in callback for callback in _active_callbacks
    ), f"_ENTERPRISE_OpenAI_Moderation should not be in _active_callbacks={_active_callbacks}"
