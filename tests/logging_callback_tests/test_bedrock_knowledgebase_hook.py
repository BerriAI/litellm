import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.rag_hooks.bedrock_knowledgebase import BedrockKnowledgeBaseHook


@pytest.mark.asyncio
async def test_basic_bedrock_knowledgebase_retrieval():

    bedrock_knowledgebase_hook = BedrockKnowledgeBaseHook()
    response = await bedrock_knowledgebase_hook.make_bedrock_kb_retrieve_request(
        knowledge_base_id="test_knowledge_base_id",
        query="what is litellm?",
    )
    assert response is not None


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_completion():
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "what is litellm?"}],
        knowledge_bases = [
            "T37J8R4WTM"
        ]
        
    )
    assert response is not None

