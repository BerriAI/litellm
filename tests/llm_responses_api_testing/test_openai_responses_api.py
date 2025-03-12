import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm
import json


@pytest.mark.asyncio
async def test_basic_openai_responses_api():
    litellm._turn_on_debug()
    response = await litellm.aresponses(
        model="gpt-4o", input="Tell me a three sentence bedtime story about a unicorn."
    )
    print("litellm response=", json.dumps(response, indent=4, default=str))

    # validate_responses_api_response()
