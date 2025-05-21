import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder


def test_transform_thinking_blocks_with_redacted_content():
    thinking_block = {"redactedContent": "This is a redacted content"}
    decoder = AWSEventStreamDecoder(model="test")
    transformed_thinking_blocks = decoder.translate_thinking_blocks(thinking_block)
    assert len(transformed_thinking_blocks) == 1
    assert transformed_thinking_blocks[0]["type"] == "redacted_thinking"
    assert transformed_thinking_blocks[0]["data"] == "This is a redacted content"
