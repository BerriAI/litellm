#### What this tests ####
# This tests litellm router

import asyncio
import os
import sys
import time
import traceback

import openai
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from dotenv import load_dotenv

import litellm
from litellm import Router
from litellm._logging import verbose_logger

verbose_logger.setLevel(logging.DEBUG)


load_dotenv()


@pytest.mark.asyncio()
async def test_router_free_paid_tier():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                },
                "model_info": {"id": "very-cheap-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["free"]},
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-expensive-model"
