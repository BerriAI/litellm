import copy
import logging
from unittest import mock

from dotenv import load_dotenv

from litellm.types.utils import StandardCallbackDynamicParams

load_dotenv()
import os

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, headers
from litellm.utils import (
    get_llm_provider,
    get_supported_openai_params,
    get_token_count,
    trim_messages,
    validate_environment,
)
from litellm.llms.openai_like.json_loader import JSONProviderRegistry


# Assuming your trim_messages, shorten_message_to_fit_limit, and get_token_count functions are all in a module named 'message_utils'
@pytest.fixture(autouse=True)
def reset_mock_cache():
    from litellm.utils import _model_cache

    _model_cache.flush_cache()


# Test 1: Check trimming of normal message


# test_basic_trimming()


# test_basic_trimming_no_max_tokens_specified()


# test_multiple_messages_trimming()


# test_multiple_messages_no_trimming()


# test_large_trimming()


# test_get_valid_models()


# test validate environment


from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.utils import get_applied_guardrails
from unittest.mock import Mock


def test_get_whitelisted_models():
    """
    Snapshot of all bedrock models as of 12/24/2024.

    Enforce any new bedrock chat model to be added as `bedrock_converse` unless explicitly whitelisted.

    Create whitelist to prevent naming regressions for older litellm versions.
    """
    whitelisted_models = []
    for model, info in litellm.model_cost.items():
        if info.get("litellm_provider") == "bedrock" and info.get("mode") == "chat":
            whitelisted_models.append(model)

        # Write to a local file
    with open("whitelisted_bedrock_models.txt", "w") as file:
        for model in whitelisted_models:
            file.write(f"{model}\n")

    print("whitelisted_models written to whitelisted_bedrock_models.txt")
