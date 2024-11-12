import os
import sys
import threading
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from unittest.mock import Mock, patch, AsyncMock


def test_dynamic_logging():
    """
    Scenario 1:
    - Test if global callbacks are called when dynamic logging is set.

    Scenario 2:
    - Test if dynamic callbacks override the global callback for the same integration
    """
    import litellm

    dynamic_sync_callback = Mock()
    global_callback = Mock()
    litellm.success_callback = [global_callback]
    litellm_logging_obj = LiteLLMLogging(
        model="my-test-gpt",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
        dynamic_success_callbacks=[dynamic_sync_callback],
    )

    callbacks = LiteLLMLogging._get_success_callback_list(litellm_logging_obj)
    assert len(callbacks) == 2
