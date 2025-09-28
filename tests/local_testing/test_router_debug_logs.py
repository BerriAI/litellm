import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import asyncio
import logging

import litellm
from litellm import Router

# this tests debug logs from litellm router and litellm proxy server
from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger


# this tests debug logs from litellm router and litellm proxy server
def test_async_fallbacks(caplog):
    # THIS IS A PROD TEST - DO NOT DELETE THIS. Used for testing if litellm proxy verbose logs are human readable
    litellm.set_verbose = False
    litellm.success_callback = []
    litellm.failure_callback = []
    verbose_router_logger.setLevel(level=logging.INFO)
    verbose_logger.setLevel(logging.CRITICAL + 1)
    verbose_proxy_logger.setLevel(logging.CRITICAL + 1)
    model_list = [
        {
            "model_name": "azure/gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/gpt-4.1-nano",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "mock_response": "Hello world",
            },
            "tpm": 240000,
            "rpm": 1800,
        },
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "bad-key",
            },
            "tpm": 1000000,
            "rpm": 9000,
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"gpt-3.5-turbo": ["azure/gpt-3.5-turbo"]}],
        num_retries=1,
    )

    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]

    async def _make_request():
        try:
            await router.acompletion(
                model="gpt-3.5-turbo", messages=messages, max_tokens=1
            )
            router.reset()
        except litellm.Timeout:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        finally:
            router.reset()

    asyncio.run(_make_request())
    captured_logs = [rec.message for rec in caplog.records]

    # on circle ci the captured logs get some async task exception logs - filter them out "Task exception was never retrieved"
    captured_logs = [
        log
        for log in captured_logs
        if "Task exception was never retrieved" not in log
        and "get_available_deployment" not in log
        and "in the Langfuse queue" not in log
    ]

    print("\n Captured caplog records - ", captured_logs)

    # Define the expected log messages
    # - error request, falling back notice, success notice
    expected_logs = [
        "Falling back to model_group = azure/gpt-3.5-turbo",
        "litellm.acompletion(model=azure/gpt-4.1-nano)\x1b[32m 200 OK\x1b[0m",
        "Successful fallback b/w models.",
    ]

    # Assert that the captured logs match the expected log messages
    assert captured_logs[-3:] == expected_logs
