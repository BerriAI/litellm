import json
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import litellm

from litellm.llms.triton.embedding.transformation import TritonEmbeddingConfig

from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE


@pytest.mark.asyncio
async def test_triton_embeddings():
    try:
        litellm.set_verbose = True
        response = await litellm.aembedding(
            model="triton/my-triton-model",
            api_base=f"{FAKE_OPENAI_API_BASE}/triton/embeddings",
            input=["good morning from litellm"],
        )
        print(f"response: {response}")

        # stubbed endpoint is setup to return this
        assert response.data[0]["embedding"] == [0.1, 0.2]
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
