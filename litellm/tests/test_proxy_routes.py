import os
import sys

from dotenv import load_dotenv

load_dotenv()
import io
import os

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

import pytest

import litellm
from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.auth.auth_utils import is_openai_route
from litellm.proxy.proxy_server import app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the desired logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def test_routes_on_litellm_proxy():
    """
    Goal of this test: Test that we have all the critical OpenAI Routes on the Proxy server Fast API router


    this prevents accidentelly deleting /threads, or /batches etc
    """
    _all_routes = []
    for route in app.routes:

        _path_as_str = str(route.path)
        if ":path" in _path_as_str:
            # remove the :path
            _path_as_str = _path_as_str.replace(":path", "")
        _all_routes.append(_path_as_str)

    print("ALL ROUTES on LiteLLM Proxy:", _all_routes)
    print("\n\n")
    print("ALL OPENAI ROUTES:", LiteLLMRoutes.openai_routes.value)

    for route in LiteLLMRoutes.openai_routes.value:
        assert route in _all_routes


@pytest.mark.parametrize(
    "route,expected",
    [
        # Test exact matches
        ("/chat/completions", True),
        ("/v1/chat/completions", True),
        ("/embeddings", True),
        ("/v1/models", True),
        ("/utils/token_counter", True),
        # Test routes with placeholders
        ("/engines/gpt-4/chat/completions", True),
        ("/openai/deployments/gpt-3.5-turbo/chat/completions", True),
        ("/threads/thread_49EIN5QF32s4mH20M7GFKdlZ", True),
        ("/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ", True),
        ("/threads/thread_49EIN5QF32s4mH20M7GFKdlZ/messages", True),
        ("/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ/runs", True),
        ("/v1/batches123456", True),
        # Test non-OpenAI routes
        ("/some/random/route", False),
        ("/v2/chat/completions", False),
        ("/threads/invalid/format", False),
        ("/v1/non_existent_endpoint", False),
    ],
)
def test_is_openai_route(route: str, expected: bool):
    assert is_openai_route(route) == expected


# Test case for routes that are similar but should return False
@pytest.mark.parametrize(
    "route",
    [
        "/v1/threads/thread_id/invalid",
        "/threads/thread_id/invalid",
        "/v1/batches/123/invalid",
        "/engines/model/invalid/completions",
    ],
)
def test_is_openai_route_similar_but_false(route: str):
    assert is_openai_route(route) == False
