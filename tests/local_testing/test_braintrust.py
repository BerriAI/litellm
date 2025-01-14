# What is this?
## This tests the braintrust integration

import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def test_braintrust_logging():
    import litellm

    litellm.set_verbose = True

    http_client = HTTPHandler()

    with patch.object(
        litellm.integrations.braintrust_logging.global_braintrust_sync_http_handler,
        "post",
        new=MagicMock(),
    ) as mock_client:
        # set braintrust as a callback, litellm will send the data to braintrust
        litellm.callbacks = ["braintrust"]

        # openai call
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
        )

        time.sleep(2)
        mock_client.assert_called()

def test_braintrust_logging_specific_project_id():
    import litellm

    litellm.set_verbose = True

    with patch.object(
        litellm.integrations.braintrust_logging.global_braintrust_sync_http_handler,
        "post",
        new=MagicMock(),
    ) as mock_client:
        # set braintrust as a callback, litellm will send the data to braintrust
        litellm.callbacks = ["braintrust"]

        response = litellm.completion(model="openai/gpt-4o", messages=[{ "content": "Hello, how are you?","role": "user"}], metadata={"project_id": "123"})

        time.sleep(2)
        
        # Check that the log was inserted into the correct project
        mock_client.assert_called()
        _, kwargs = mock_client.call_args
        assert 'url' in kwargs
        assert kwargs['url'] == "https://api.braintrustdata.com/v1/project_logs/123/insert"

def test_braintrust_span_attributes():
    import litellm

    litellm.set_verbose = True

    with patch.object(
        litellm.integrations.braintrust_logging.global_braintrust_sync_http_handler,
        "post",
        new=MagicMock(),
    ) as mock_client:
        # set braintrust as a callback
        litellm.callbacks = ["braintrust"]

        # Test 1: Setting span attributes via metadata
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - test span attributes"}],
            metadata={
                "span_attributes": {
                    "name": "Custom Span",
                    "type": "custom_type"
                }
            }
        )

        time.sleep(2)
        
        # Verify the span attributes were set correctly
        mock_client.assert_called()
        _, kwargs = mock_client.call_args
        assert 'json' in kwargs
        event = kwargs['json']['events'][0]
        assert event['span_attributes'] == {
            "name": "Custom Span",
            "type": "custom_type"
        }

        # Reset mock
        mock_client.reset_mock()

        # Test 2: Setting span attributes via headers
        headers = {
            "braintrust_span_attributes_name": "Header Span",
            "braintrust_span_attributes_type": "header_type"
        }
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - test header span attributes"}],
            litellm_params={
                "proxy_server_request": {
                    "headers": headers
                }
            }
        )

        time.sleep(2)
        
        # Verify the span attributes were set correctly from headers
        mock_client.assert_called()
        _, kwargs = mock_client.call_args
        assert 'json' in kwargs
        event = kwargs['json']['events'][0]
        assert event['span_attributes'] == {
            "name": "Header Span",
            "type": "header_type"
        }

        # Test 3: Default values when no span attributes provided
        mock_client.reset_mock()
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - test default span attributes"}],
        )

        time.sleep(2)
        
        # Verify default span attributes are used
        mock_client.assert_called()
        _, kwargs = mock_client.call_args
        assert 'json' in kwargs
        event = kwargs['json']['events'][0]
        assert event['span_attributes'] == {
            "name": "Chat Completion",
            "type": "llm"
        }

