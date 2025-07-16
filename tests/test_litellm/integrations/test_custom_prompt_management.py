import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
import litellm
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage
from litellm.types.utils import StandardCallbackDynamicParams


@pytest.fixture(autouse=True)
def setup_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-some-key")

@pytest.fixture(autouse=False)
def setup_custom_prompt_id_skip(monkeypatch):
    monkeypatch.setenv("CUSTOM_PROMPT_PROMPT_ID_CHECK", "false")



class CustomPromptManagementForTest(CustomPromptManagement):
    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        print(
            "CustomPromptManagementForTest: running get_chat_completion_prompt for prompt_id: ",
            prompt_id,
        )
        if prompt_id == "test_prompt_id":
            messages = [
                ChatCompletionUserMessage(role="user", content="This is the prompt for test_prompt_id"),
            ]
            return model, messages, non_default_params
        elif prompt_id == "prompt_with_variables":
            content = "Hello, {name}! You are {age} years old and live in {city}."
            content_with_variables = content.format(**(prompt_variables or {}))
            messages = [
                ChatCompletionUserMessage(role="user", content=content_with_variables),
            ]
            return model, messages, non_default_params
        else:
            return model, messages, non_default_params

class CustomPromptManagementForSkippingPromptIdCheck(CustomPromptManagement):
    def get_chat_completion_prompt(
            self,
            model: str,
            messages: List[AllMessageValues],
            non_default_params: dict,
            prompt_id: Optional[str],
            prompt_variables: Optional[dict],
            dynamic_callback_params: StandardCallbackDynamicParams,
            prompt_label: Optional[str] = None,
            prompt_version: Optional[int] = None,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        print(
            "CustomPromptManagementForSkippingPromptIdCheck: running get_chat_completion_prompt for prompt_id: ",
            prompt_id,
        )
        custom_prompt_prompt_id_check = os.getenv("CUSTOM_PROMPT_PROMPT_ID_CHECK", "true")
        if custom_prompt_prompt_id_check == "false":
            messages = [
                ChatCompletionUserMessage(role="user", content="This is the prompt when prompt_id is None"),
            ]

        return model, messages, non_default_params


@pytest.mark.asyncio
async def test_custom_prompt_management_with_prompt_id(monkeypatch):
    custom_prompt_management = CustomPromptManagementForTest()
    litellm.callbacks = [custom_prompt_management]

    # Mock AsyncHTTPHandler.post method
    client = AsyncHTTPHandler()
    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        await litellm.acompletion(
            model="anthropic/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            client=client,
            prompt_id="test_prompt_id",
        )

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        request_body = mock_post.call_args.kwargs["json"]
        print("request_body: ", json.dumps(request_body, indent=4))

        assert request_body["model"] == "claude-3-5-sonnet"
        # the message gets applied to the prompt from the custom prompt management callback
        assert (
            request_body["messages"][0]["content"][0]["text"]
            == "This is the prompt for test_prompt_id"
        )


@pytest.mark.asyncio
async def test_custom_prompt_management_with_prompt_id_and_prompt_variables():
    custom_prompt_management = CustomPromptManagementForTest()
    litellm.callbacks = [custom_prompt_management]

    # Mock AsyncHTTPHandler.post method
    client = AsyncHTTPHandler()
    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        await litellm.acompletion(
            model="anthropic/claude-3-5-sonnet",
            messages=[],
            client=client,
            prompt_id="prompt_with_variables",
            prompt_variables={"name": "John", "age": 30, "city": "New York"},
        )

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        request_body = mock_post.call_args.kwargs["json"]
        print("request_body: ", json.dumps(request_body, indent=4))

        assert request_body["model"] == "claude-3-5-sonnet"
        # the message gets applied to the prompt from the custom prompt management callback
        assert (
            request_body["messages"][0]["content"][0]["text"]
            == "Hello, John! You are 30 years old and live in New York."
        )


@pytest.mark.asyncio
async def test_custom_prompt_management_without_prompt_id():
    custom_prompt_management = CustomPromptManagementForTest()
    litellm.callbacks = [custom_prompt_management]

    # Mock AsyncHTTPHandler.post method
    client = AsyncHTTPHandler()
    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        await litellm.acompletion(
            model="anthropic/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            client=client,
        )

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        request_body = mock_post.call_args.kwargs["json"]
        print("request_body: ", json.dumps(request_body, indent=4))

        assert request_body["model"] == "claude-3-5-sonnet"
        # the message does not get applied to the prompt from the custom prompt management callback since we did not pass a prompt_id
        assert (
            request_body["messages"][0]["content"][0]["text"] == "Hello, how are you?"
        )


@pytest.mark.asyncio
async def test_custom_prompt_management_with_prompt_id_check_off(setup_custom_prompt_id_skip):
    custom_prompt_management = CustomPromptManagementForSkippingPromptIdCheck()
    litellm.callbacks = [custom_prompt_management]

    # Mock AsyncHTTPHandler.post method
    client = AsyncHTTPHandler()
    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        await litellm.acompletion(
            model="anthropic/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            client=client
        )

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        request_body = mock_post.call_args.kwargs["json"]
        print("request_body: ", json.dumps(request_body, indent=4))

        assert request_body["model"] == "claude-3-5-sonnet"
        # the message gets applied to the prompt from the custom prompt management callback
        assert (
                request_body["messages"][0]["content"][0]["text"]
                == "This is the prompt when prompt_id is None"
        )


@pytest.mark.asyncio
async def test_custom_prompt_management_with_prompt_id_check_on():
    custom_prompt_management = CustomPromptManagementForSkippingPromptIdCheck()
    litellm.callbacks = [custom_prompt_management]

    # Mock AsyncHTTPHandler.post method
    client = AsyncHTTPHandler()
    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        await litellm.acompletion(
            model="anthropic/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            client=client
        )

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        request_body = mock_post.call_args.kwargs["json"]
        print("request_body: ", json.dumps(request_body, indent=4))

        assert request_body["model"] == "claude-3-5-sonnet"
        # the message gets applied to the prompt from the custom prompt management callback
        assert (
                request_body["messages"][0]["content"][0]["text"]
                == "Hello, how are you?"
        )