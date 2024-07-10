# What is this?
## Unit tests for Anthropic Adapter

import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import adapter_completion
from litellm.adapters.anthropic_adapter import anthropic_adapter


def test_anthropic_completion():
    litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response = adapter_completion(
        model="gpt-3.5-turbo", messages=messages, adapter_id="anthropic"
    )

    print(response)
