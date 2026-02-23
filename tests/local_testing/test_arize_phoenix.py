import asyncio
import logging
import pytest
from dotenv import load_dotenv

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.integrations.arize.arize_phoenix import ArizePhoenixConfig, ArizePhoenixLogger

load_dotenv()


@pytest.mark.asyncio()
async def test_async_otel_callback():
    litellm.set_verbose = True

    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_logger.setLevel(logging.DEBUG)
    litellm.success_callback = ["arize_phoenix"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "this is arize phoenix"}],
        mock_response="hello",
        temperature=0.1,
        user="OTEL_USER",
    )

    await asyncio.sleep(2)
