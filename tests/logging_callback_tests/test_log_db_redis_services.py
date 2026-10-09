import io


import gzip
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.proxy.utils import ServiceTypes


@pytest.mark.asyncio
async def test_dd_log_db_spend_failure_metrics():
    from litellm._service_logger import ServiceLogging
    from litellm.integrations.datadog.datadog import DataDogLogger

    dd_logger = DataDogLogger()
    with patch.object(dd_logger, "async_service_failure_hook", new_callable=AsyncMock):
        service_logging_obj = ServiceLogging()

        litellm.service_callback = [dd_logger]

        await service_logging_obj.async_service_failure_hook(
            service=ServiceTypes.DB,
            call_type="test_call_type",
            error="test_error",
            duration=1.0,
        )
