import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlparse, urlunparse


sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import pytest
from litellm.types.llms.openai import Run
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.azure_passthrough_logging_handler import (
    AzurePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)


@pytest.fixture
def azure_handler():
    handler = AzurePassthroughLoggingHandler()
    return handler

@pytest.fixture
def mock_run():
    run_dict = {
        "id": "run_id",
        "thread_id": "thread_id",
        "assistant_id": "assistant_id",
        "created_at": 100,
        "instructions": "hi",
        "model": "example_model",
        "object": "thread.run",
        "parallel_tool_calls": False,
        "status": "queued",
        "tools": [],
    }
    return Run(**run_dict)

@pytest.fixture
def mock_run_response():
    run_dict = {
        "id": "run_id",
        "thread_id": "thread_id",
        "assistant_id": "assistant_id",
        "created_at": 100,
        "instructions": "hi",
        "model": "example_model",
        "object": "thread.run",
        "parallel_tool_calls": False,
        "status": "queued",
        "tools": [],
    }
    return httpx.Response(
        status_code=200,
        content=json.dumps(run_dict).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

@pytest.fixture
def post_run_url():
    return "https://dep.openai.azure.com/openai/threads/thread_id/runs?api-version=2024-02-12-preview"

def test_should_log_request():
    handler = AzurePassthroughLoggingHandler()
    assert handler._should_log_request("POST") == True
    assert handler._should_log_request("GET") == False


def test_get_retrieve_url(azure_handler, mock_run, post_run_url):
    url =  azure_handler._get_retrieve_url(mock_run, post_run_url)
    parsed_post_run_url = urlparse(post_run_url)
    assert url == "https://" + parsed_post_run_url.hostname + "/openai/threads/thread_id/runs/run_id?api-version=2024-02-12-preview"

def test_get_populated_run(azure_handler, mock_run, post_run_url):
    """
    Test that the _get_populated_run method calls GET /openai/thread/{thread_id}/run/{run_id}
    and uses the test key returned by the mocked get_credentials.
    """
    # Patch get_credentials to return "test-key"
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with patch("httpx.get") as mock_get:
            mock_get.return_value.json.return_value = mock_run
            mock_get.return_value.raise_for_status.return_value = None
            run = azure_handler._get_populated_run(mock_run, post_run_url)
            parsed_post_run_url = urlparse(post_run_url)
            assert run == mock_run

            mock_get.assert_called_once_with(
                "https://" + parsed_post_run_url.hostname + "/openai/threads/thread_id/runs/run_id?api-version=2024-02-12-preview",
                headers={
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                },
            )


def test_poll_unfinished_run(
    azure_handler, mock_run, mock_run_response, post_run_url
):
    """
    Test that the _poll_unfinished_run method returns the correct run response
    """
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
        return_value="test-key",
    ):
        with patch("httpx.get") as mock_get:
            mock_get.return_value.json.return_value = mock_run
            mock_get.return_value.raise_for_status.return_value = None

            # Override polling settings for faster test
            azure_handler.polling_interval = 0.01
            azure_handler.max_polling_attempts = 2

            run = azure_handler._poll_unfinished_run(
                mock_run_response,
                post_run_url,
            )
            assert run == mock_run


def test_is_azureai_route():
    """
    Test that the is_azureai_route method correctly identifies AssemblyAI routes
    """
    handler = PassThroughEndpointLogging()

    assert (
        handler.is_azure_route("https://dep.openai.azure.com/openai/threads/runs") == False
    )
    assert (
        handler.is_azure_route("https://dep.openai.azure.com/openai/deployments/dep/threads/runs?api-version=2025-02-12-preview") == True
    )
    assert handler.is_azure_route("") == False