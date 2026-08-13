import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import concurrent

from dotenv import load_dotenv
import asyncio
import litellm


@pytest.mark.asyncio
async def test_acompletion_fallbacks_basic():
    response = await litellm.acompletion(
        model="openai/unknown-model",
        messages=[{"role": "user", "content": "Hello, world!"}],
        fallbacks=["openai/gpt-4o-mini"],
    )
    print(response)
    assert response is not None


@pytest.mark.asyncio
async def test_acompletion_fallbacks_bad_models():
    """
    Test that the acompletion call times out after 10 seconds - if no fallbacks work
    """
    try:
        # Wrap the acompletion call with asyncio.wait_for to enforce a timeout
        response = await asyncio.wait_for(
            litellm.acompletion(
                model="openai/unknown-model",
                messages=[{"role": "user", "content": "Hello, world!"}],
                fallbacks=["openai/bad-model", "openai/unknown-model"],
            ),
            timeout=5.0,  # Timeout after 5 seconds
        )
        assert response is not None
    except asyncio.TimeoutError:
        pytest.fail("Test timed out - possible infinite loop in fallbacks")
    except Exception as e:
        print(e)
        pass


@pytest.mark.asyncio
async def test_acompletion_fallbacks_with_dict_config():
    """
    Test fallbacks with dictionary configuration that includes model-specific settings
    """
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        api_key="very-bad-api-key",
        fallbacks=[{"api_key": os.getenv("OPENAI_API_KEY")}],
    )
    assert response is not None


@pytest.mark.asyncio
async def test_acompletion_fallbacks_empty_list():
    """
    Test behavior when fallbacks list is empty
    """
    try:
        response = await litellm.acompletion(
            model="openai/unknown-model",
            messages=[{"role": "user", "content": "Hello, world!"}],
            fallbacks=[],
        )
    except Exception as e:
        assert isinstance(e, litellm.NotFoundError)


@pytest.mark.asyncio
async def test_acompletion_fallbacks_none_response():
    """
    Test handling when a fallback model returns None
    Should continue to next fallback rather than returning None
    """
    response = await litellm.acompletion(
        model="openai/unknown-model",
        messages=[{"role": "user", "content": "Hello, world!"}],
        fallbacks=["gpt-3.5-turbo"],  # replace with a model you know works
    )
    assert response is not None


async def test_completion_fallbacks_sync():
    response = litellm.completion(
        model="openai/unknown-model",
        messages=[{"role": "user", "content": "Hello, world!"}],
        fallbacks=["openai/gpt-4o-mini"],
    )
    print(response)
    assert response is not None
