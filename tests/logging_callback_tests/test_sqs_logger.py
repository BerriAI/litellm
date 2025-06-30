import asyncio
from unittest.mock import AsyncMock

import litellm
import pytest

from litellm.integrations.sqs import SQSLogger


@pytest.mark.asyncio
async def test_async_sqs_logger_flush():
    sqs_logger = SQSLogger(
        sqs_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
        sqs_region_name="us-east-1",
        sqs_flush_interval=1,
    )
    sqs_logger.async_send_message = AsyncMock()
    litellm.callbacks = [sqs_logger]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="hi",
    )

    await asyncio.sleep(2)

    sqs_logger.async_send_message.assert_called()
