import json
import os
import sys
import traceback
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.chat.o_series_transformation import AzureOpenAIO1Config


@pytest.mark.asyncio
async def test_azure_chat_o_series_transformation():
    provider_config = AzureOpenAIO1Config()
    model = "o_series/web-interface-o1-mini"
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    optional_params = {}
    litellm_params = {}
    headers = {}

    response = await provider_config.async_transform_request(
        model, messages, optional_params, litellm_params, headers
    )
    print(response)
    assert response["model"] == "web-interface-o1-mini"
