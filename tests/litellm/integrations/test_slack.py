import os
import sys
from unittest.mock import patch, AsyncMock

from litellm.integrations.SlackAlerting.slack_alerting import (
    SlackAlerting,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

# Calling update_values with alerting args should try to start the periodic task which will fail
@patch("asyncio.create_task")
def test_update_values_starts_periodic_task(mock_create_task):
    # Make it do nothing (or return a dummy future)
    mock_create_task.return_value = AsyncMock()  # prevents awaiting errors

    sa = SlackAlerting()
    assert(sa.periodic_started == False)

    sa.update_values(alerting_args={"slack_alerting": "True"})
    assert(sa.periodic_started == True)


