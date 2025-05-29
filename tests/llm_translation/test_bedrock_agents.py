import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types

load_dotenv()
import io
import os
import json

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, Mock, patch

import pytest

async def test_bedrock_agents():
    response = litellm.completion(
            model="bedrock/agent/L1RT58GYRW/MFPSBCXYTW",
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
        )
    pass