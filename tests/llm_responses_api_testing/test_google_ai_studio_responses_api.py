
import os
import sys
import pytest
sys.path.insert(0, os.path.abspath("../.."))
import litellm
import json

@pytest.mark.asyncio
async def test_basic_google_ai_studio_responses_api_with_tools():
    litellm._turn_on_debug()
    litellm.set_verbose = True
    request_model = "gemini/gemini-2.5-flash"
    response = await litellm.aresponses(
        model=request_model,
        input="what is the latest version of supabase python package and when was it released?",
        tools=[
            {
                "type": "web_search_preview",
                "search_context_size": "low"
            }
        ]
    )
    print("litellm response=", json.dumps(response, indent=4, default=str))