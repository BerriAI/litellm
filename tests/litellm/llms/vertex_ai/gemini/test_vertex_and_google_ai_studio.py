import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)


def test_top_logprobs():
    non_default_params = {
        "top_logprobs": 2,
        "logprobs": True,
    }
    optional_params = {}
    model = "gemini"

    v = VertexGeminiConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert v["responseLogprobs"] is non_default_params["logprobs"]
    assert v["logprobs"] is non_default_params["top_logprobs"]
