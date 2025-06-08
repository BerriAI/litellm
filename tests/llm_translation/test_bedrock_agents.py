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

@pytest.mark.asyncio
async def test_bedrock_agents():
    litellm._turn_on_debug()
    response = litellm.completion(
            model="bedrock/agent/L1RT58GYRW/MFPSBCXYTW",
            messages=[
                {
                    "role": "user",
                    "content": "Hi just respond with a ping message"
                }
            ],
        )

    #########################################################
    #########################################################
    print("response from agent=", response.model_dump_json(indent=4))

    # assert that the message content has a response with some length
    assert len(response.choices[0].message.content) > 0


    # assert we were able to get the response cost
    assert response._hidden_params["response_cost"] is not None and response._hidden_params["response_cost"] > 0

    pass

@pytest.mark.asyncio
async def test_bedrock_agents_with_streaming():
    # litellm._turn_on_debug()
    response = litellm.completion(
        model="bedrock/agent/L1RT58GYRW/MFPSBCXYTW",
        messages=[
            {
                "role": "user",
                "content": "Hi who is ishaan cto of litellm, tell me 10 things about him"
            }
        ],
        stream=True,
    )

    for chunk in response:
        print("final chunk=", chunk)

    pass