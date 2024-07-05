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
from litellm.proxy.proxy_server import router

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
    for route in router.routes:

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
