import asyncio
import json
import os
import sys
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock

import pytest

from litellm.proxy.auth.user_api_key_auth import get_api_key


def test_get_api_key():
    bearer_token = "Bearer sk-12345678"
    api_key = "sk-12345678"
    passed_in_key = "Bearer sk-12345678"
    assert get_api_key(
        custom_litellm_key_header=None,
        api_key=bearer_token,
        azure_api_key_header=None,
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        pass_through_endpoints=None,
        route="",
        request=MagicMock(),
    ) == (api_key, passed_in_key)


@pytest.mark.parametrize(
    "custom_litellm_key_header, api_key, passed_in_key",
    [
        ("Bearer sk-12345678", "sk-12345678", "Bearer sk-12345678"),
        ("Basic sk-12345678", "sk-12345678", "Basic sk-12345678"),
        ("bearer sk-12345678", "sk-12345678", "bearer sk-12345678"),
        ("sk-12345678", "sk-12345678", "sk-12345678"),
    ],
)
def test_get_api_key_with_custom_litellm_key_header(
    custom_litellm_key_header, api_key, passed_in_key
):
    assert get_api_key(
        custom_litellm_key_header=custom_litellm_key_header,
        api_key=None,
        azure_api_key_header=None,
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        pass_through_endpoints=None,
        route="",
        request=MagicMock(),
    ) == (api_key, passed_in_key)
