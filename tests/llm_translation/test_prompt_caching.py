import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse
from litellm.types.utils import PromptTokensDetails


@pytest.mark.asyncio
async def test_prompt_caching():
    """
    Tests that:
    - prompt_tokens_details is correctly handled and returned as PromptTokensDetails type
    """
    response1 = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    print("response1", response1)
    print("response1.usage", response1.usage)
    print("type of prompt_tokens_details", type(response1.usage.prompt_tokens_details))
    assert isinstance(response1.usage.prompt_tokens_details, PromptTokensDetails)
