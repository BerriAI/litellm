import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponseStream


@pytest.fixture
def initialized_custom_stream_wrapper() -> CustomStreamWrapper:
    streaming_handler = CustomStreamWrapper(
        completion_stream=None,
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )
    return streaming_handler


def test_is_chunk_non_empty(initialized_custom_stream_wrapper: CustomStreamWrapper):
    """Unit test if non-empty when reasoning_content is present"""
    chunk = {
        "id": "e89b6501-8ac2-464c-9550-7cd3daf94350",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_5417b77867_prod0225",
        "choices": [
            {
                "index": 0,
                "delta": {"content": None, "reasoning_content": "."},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    assert initialized_custom_stream_wrapper.is_chunk_non_empty(
        completion_obj=MagicMock(),
        model_response=ModelResponseStream(**chunk),
        response_obj=MagicMock(),
    )
