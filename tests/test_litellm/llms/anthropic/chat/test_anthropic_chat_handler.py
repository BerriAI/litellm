import json
import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.anthropic.chat.handler import ModelResponseIterator


def test_redacted_thinking_content_block_delta():
    chunk = {
        "type": "content_block_start",
        "index": 58,
        "content_block": {
            "type": "redacted_thinking",
            "data": "EuoBCoYBGAIiQJ/SxkPAgqxhKok29YrpJHRUJ0OT8ahCHKAwyhmRuUhtdmDX9+mn4gDzKNv3fVpQdB01zEPMzNY3QuTCd+1bdtEqQK6JuKHqdndbwpr81oVWb4wxd1GqF/7Jkw74IlQa27oobX+KuRkopr9Dllt/RDe7Se0sI1IkU7tJIAQCoP46OAwSDF51P09q67xhHlQ3ihoM2aOVlkghq/X0w8NlIjBMNvXYNbjhyrOcIg6kPFn2ed/KK7Cm5prYAtXCwkb4Wr5tUSoSHu9T5hKdJRbr6WsqEc7Lle7FULqMLZGkhqXyc3BA",
        },
    }
    model_response_iterator = ModelResponseIterator(
        streaming_response=MagicMock(), sync_stream=False, json_mode=False
    )
    model_response = model_response_iterator.chunk_parser(chunk=chunk)
    print(f"\n\nmodel_response: {model_response}\n\n")
    assert model_response.choices[0].delta.thinking_blocks is not None
    assert len(model_response.choices[0].delta.thinking_blocks) == 1
    print(
        f"\n\nmodel_response.choices[0].delta.thinking_blocks[0]: {model_response.choices[0].delta.thinking_blocks[0]}\n\n"
    )
    assert (
        model_response.choices[0].delta.thinking_blocks[0]["type"]
        == "redacted_thinking"
    )

    assert model_response.choices[0].delta.provider_specific_fields is not None
    assert "thinking_blocks" in model_response.choices[0].delta.provider_specific_fields
