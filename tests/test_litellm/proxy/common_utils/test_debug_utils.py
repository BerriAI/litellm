import os
import socket

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.debug_utils import get_memory_summary


@pytest.mark.asyncio
async def test_memory_summary_names_the_host_and_worker_that_answered() -> None:
    summary = await get_memory_summary(UserAPIKeyAuth())

    assert summary["hostname"] == socket.gethostname()
    assert summary["worker_pid"] == os.getpid()
    assert summary["memory"]["ram_usage_mb"] > 0
