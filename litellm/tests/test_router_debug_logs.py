import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm, asyncio, logging
from litellm import Router

# this tests debug logs from litellm router and litellm proxy server
from litellm._logging import verbose_router_logger, verbose_logger, verbose_proxy_logger


# this tests debug logs from litellm router and litellm proxy server
def test_async_fallbacks(caplog):
    # THIS IS A PROD TEST - DO NOT DELETE THIS. Used for testing if litellm proxy verbose logs are human readable
    litellm.set_verbose = False
    verbose_router_logger.setLevel(level=logging.INFO)
    verbose_logger.setLevel(logging.CRITICAL + 1)
    verbose_proxy_logger.setLevel(logging.CRITICAL + 1)
    model_list = [
        {
            "model_name": "azure/gpt-3.5-turbo",
            "litellm_params": {
                "model": "azure/chatgpt-v-2",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_version": os.getenv("AZURE_API_VERSION"),
                "api_base": os.getenv("AZURE_API_BASE"),
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
    )

    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]

    async def _make_request():
        try:
            response = await router.acompletion(
                model="gpt-3.5-turbo", messages=messages, max_tokens=1
            )
            router.reset()
        except litellm.Timeout as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        finally:
            router.reset()

    asyncio.run(_make_request())
    captured_logs = [rec.message for rec in caplog.records]

    # on circle ci the captured logs get some async task exception logs - filter them out
    "Task exception was never retrieved"
    captured_logs = [
        log
        for log in captured_logs
        if "Task exception was never retrieved" not in log
        and "get_available_deployment" not in log
    ]

    print("\n Captured caplog records - ", captured_logs)

    # Define the expected log messages
    # - error request, falling back notice, success notice
    expected_logs = [
        "litellm.acompletion(model=gpt-3.5-turbo)\x1b[31m Exception OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: bad-key. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}\x1b[0m",
        "Falling back to model_group = azure/gpt-3.5-turbo",
        "litellm.acompletion(model=azure/chatgpt-v-2)\x1b[32m 200 OK\x1b[0m",
    ]

    # Assert that the captured logs match the expected log messages
    assert captured_logs == expected_logs
