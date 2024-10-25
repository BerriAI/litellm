import os
import sys
import traceback
import uuid
import datetime as dt
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import io
import os
import time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

from fastapi import HTTPException, Request
import pytest
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth


# Mock objects and functions
class MockRequest:
    def __init__(self, query_params=None):
        self.query_params = query_params or {}


# Test is_llm_api_route
def test_is_llm_api_route():
    assert RouteChecks.is_llm_api_route("/v1/chat/completions") is True
    assert RouteChecks.is_llm_api_route("/v1/completions") is True
    assert RouteChecks.is_llm_api_route("/v1/embeddings") is True
    assert RouteChecks.is_llm_api_route("/v1/images/generations") is True
    assert RouteChecks.is_llm_api_route("/v1/threads/thread_12345") is True
    assert RouteChecks.is_llm_api_route("/bedrock/model/invoke") is True
    assert RouteChecks.is_llm_api_route("/vertex-ai/text") is True
    assert RouteChecks.is_llm_api_route("/gemini/generate") is True
    assert RouteChecks.is_llm_api_route("/cohere/generate") is True

    # check non-matching routes
    assert RouteChecks.is_llm_api_route("/some/random/route") is False
    assert RouteChecks.is_llm_api_route("/key/regenerate/82akk800000000jjsk") is False
    assert RouteChecks.is_llm_api_route("/key/82akk800000000jjsk/delete") is False


# Test _route_matches_pattern
def test_route_matches_pattern():
    # check matching routes
    assert (
        RouteChecks._route_matches_pattern(
            "/threads/thread_12345", "/threads/{thread_id}"
        )
        is True
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/key/regenerate/82akk800000000jjsk", "/key/{token_id}/regenerate"
        )
        is False
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/chat/completions", "/v1/chat/completions"
        )
        is True
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/models/gpt-4", "/v1/models/{model_name}"
        )
        is True
    )

    # check non-matching routes
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/chat/completionz/thread_12345", "/v1/chat/completions/{thread_id}"
        )
        is False
    )
    assert (
        RouteChecks._route_matches_pattern(
            "/v1/{thread_id}/messages", "/v1/messages/thread_2345"
        )
        is False
    )
