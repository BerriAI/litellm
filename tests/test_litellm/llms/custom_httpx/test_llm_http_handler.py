import io
import os
import pathlib
import ssl
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler


def test_prepare_fake_stream_request():
    # Initialize the BaseLLMHTTPHandler
    handler = BaseLLMHTTPHandler()

    # Test case 1: fake_stream is True
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that "stream" key is removed from data
    assert "stream" not in result_data
    # Verify other data remains unchanged
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 2: fake_stream is False
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = False

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream remains True
    assert result_stream is True
    # Verify that data remains unchanged
    assert "stream" in result_data
    assert result_data["stream"] is True
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 3: data doesn't have stream key but fake_stream is True
    stream = True
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that data remains unchanged (since there was no stream key to remove)
    assert "stream" not in result_data
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]
