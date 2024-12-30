import asyncio
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.abspath("../.."))
import pytest
import litellm
from litellm.integrations.pagerduty.pagerduty import PagerDutyAlerting, AlertingConfig


@pytest.mark.asyncio
async def test_pagerduty_alerting():
    pagerduty = PagerDutyAlerting(
        alerting_config=AlertingConfig(
            failure_threshold=1, failure_threshold_window_seconds=10
        )
    )
    litellm.callbacks = [pagerduty]

    try:
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="litellm.RateLimitError",
        )
    except litellm.RateLimitError:
        pass

    await asyncio.sleep(2)
