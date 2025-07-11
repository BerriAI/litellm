#!/usr/bin/env python3
"""
Test OpenAI Moderation Guardrail
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../../.."))

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.openai.moderations import (
    OpenAIModerationGuardrail,
)
from litellm.types.llms.openai import OpenAIModerationResponse, OpenAIModerationResult


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_init():
    """Test OpenAI moderation guardrail initialization"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )
        
        assert guardrail.guardrail_name == "test-openai-moderation"
        assert guardrail.api_key == "test-key"
        assert guardrail.model == "omni-moderation-latest"
        assert guardrail.api_base == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_safe_content():
    """Test OpenAI moderation guardrail with safe content"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )
        
        # Mock safe moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=False,
                    categories={
                        "sexual": False,
                        "hate": False,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.001,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": [],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    }
                )
            ]
        )
        
        with patch.object(guardrail, 'async_make_request', return_value=mock_response):
            # Test pre-call hook with safe content
            user_api_key_dict = UserAPIKeyAuth(api_key="test")
            data = {
                "messages": [
                    {"role": "user", "content": "Hello, how are you today?"}
                ]
            }
            
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=None,
                data=data,
                call_type="completion"
            )
            
            # Should return the original data unchanged
            assert result == data


@pytest.mark.asyncio 
async def test_openai_moderation_guardrail_harmful_content():
    """Test OpenAI moderation guardrail with harmful content"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )
        
        # Mock harmful moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=True,
                    categories={
                        "sexual": False,
                        "hate": True,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.95,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": ["text"],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    }
                )
            ]
        )
        
        with patch.object(guardrail, 'async_make_request', return_value=mock_response):
            # Test pre-call hook with harmful content
            user_api_key_dict = UserAPIKeyAuth(api_key="test")
            data = {
                "messages": [
                    {"role": "user", "content": "This is hateful content"}
                ]
            }
            
            # Should raise HTTPException
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=None,
                    data=data,
                    call_type="completion"
                )
            
            assert exc_info.value.status_code == 400
            assert "Violated OpenAI moderation policy" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_streaming_safe_content():
    """Test OpenAI moderation guardrail with streaming safe content"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )
        
        # Mock safe moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=False,
                    categories={
                        "sexual": False,
                        "hate": False,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.001,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": [],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    }
                )
            ]
        )
        
        # Mock streaming chunks
        async def mock_stream():
            # Simulate streaming chunks with safe content
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="world"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))])
            ]
            for chunk in chunks:
                yield chunk
        
        # Mock the stream_chunk_builder to return a proper ModelResponse
        mock_model_response = MagicMock()
        mock_model_response.choices = [
            MagicMock(message=MagicMock(content="Hello world!"))
        ]
        
        with patch.object(guardrail, 'async_make_request', return_value=mock_response), \
             patch('litellm.main.stream_chunk_builder', return_value=mock_model_response), \
             patch('litellm.llms.base_llm.base_model_iterator.MockResponseIterator') as mock_iterator:
            
            # Mock the iterator to yield the original chunks
            async def mock_yield_chunks():
                chunks = [
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="world"))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))])
                ]
                for chunk in chunks:
                    yield chunk
            
            mock_iterator.return_value.__aiter__ = lambda self: mock_yield_chunks()
            
            user_api_key_dict = UserAPIKeyAuth(api_key="test")
            request_data = {
                "messages": [
                    {"role": "user", "content": "Hello, how are you today?"}
                ]
            }
            
            # Test streaming hook with safe content
            result_chunks = []
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=mock_stream(),
                request_data=request_data
            ):
                result_chunks.append(chunk)
            
            # Should return all chunks without blocking
            assert len(result_chunks) == 3


@pytest.mark.asyncio
async def test_openai_moderation_guardrail_streaming_harmful_content():
    """Test OpenAI moderation guardrail with streaming harmful content"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        guardrail = OpenAIModerationGuardrail(
            guardrail_name="test-openai-moderation",
        )
        
        # Mock harmful moderation response
        mock_response = OpenAIModerationResponse(
            id="modr-123",
            model="omni-moderation-latest",
            results=[
                OpenAIModerationResult(
                    flagged=True,
                    categories={
                        "sexual": False,
                        "hate": True,
                        "harassment": False,
                        "self-harm": False,
                        "violence": False,
                    },
                    category_scores={
                        "sexual": 0.001,
                        "hate": 0.95,
                        "harassment": 0.001,
                        "self-harm": 0.001,
                        "violence": 0.001,
                    },
                    category_applied_input_types={
                        "sexual": [],
                        "hate": ["text"],
                        "harassment": [],
                        "self-harm": [],
                        "violence": [],
                    }
                )
            ]
        )
        
        # Mock streaming chunks with harmful content
        async def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="This is "))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="harmful content"))])
            ]
            for chunk in chunks:
                yield chunk
        
        # Mock the stream_chunk_builder to return a ModelResponse with harmful content
        mock_model_response = MagicMock()
        mock_model_response.choices = [
            MagicMock(message=MagicMock(content="This is harmful content"))
        ]
        
        with patch.object(guardrail, 'async_make_request', return_value=mock_response), \
             patch('litellm.main.stream_chunk_builder', return_value=mock_model_response):
            
            user_api_key_dict = UserAPIKeyAuth(api_key="test")
            request_data = {
                "messages": [
                    {"role": "user", "content": "Generate harmful content"}
                ]
            }
            
            # Should raise HTTPException when processing streaming harmful content
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                result_chunks = []
                async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=mock_stream(),
                    request_data=request_data
                ):
                    result_chunks.append(chunk)
            
            assert exc_info.value.status_code == 400
            assert "Violated OpenAI moderation policy" in str(exc_info.value.detail) 