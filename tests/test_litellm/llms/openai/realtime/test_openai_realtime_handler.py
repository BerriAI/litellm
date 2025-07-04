import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


@pytest.mark.parametrize(
    "api_base", ["https://api.openai.com/v1", "https://api.openai.com"]
)
def test_openai_realtime_handler_url_construction(api_base):
    from litellm.llms.openai.realtime.handler import OpenAIRealtime

    handler = OpenAIRealtime()
    url = handler._construct_url(
        api_base=api_base, model="gpt-4o-realtime-preview-2024-10-01"
    )
    assert (
        url
        == f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    )
