import os
from datetime import datetime, timezone

from litellm import Router
import pytest


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "tpm": 1000,  # Add TPM limit so async method doesn't return early
                "rpm": 100,  # Add RPM limit so async method doesn't return early
            },
            "model_info": {
                "access_groups": ["group1", "group2"],
            },
        },
        {
            "model_name": "gpt-5.5",
            "litellm_params": {
                "model": "gpt-5.5",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-image-1",
            "litellm_params": {
                "model": "gpt-image-1",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "claude-*",
            "litellm_params": {
                "model": "anthropic/*",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
        },
    ]


def _rpm_tpm_router(model_id: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake", "tpm": 1000, "rpm": 100},
                "model_info": {"id": model_id},
            }
        ]
    )


@pytest.fixture
def router_minute_pinned(monkeypatch):
    pinned = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr("litellm.router.get_utc_datetime", lambda: pinned)


# def test_pattern_match_deployments(model_list):
#     from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
#     import re

#     patter_router = PatternMatchRouter()

#     request = "fo::hi::static::hello"
#     model_name = "fo::*:static::*"

#     model_name_regex = patter_router._pattern_to_regex(model_name)

#     # Match against the request
#     match = re.match(model_name_regex, request)

#     print(f"match: {match}")
#     print(f"match.end: {match.end()}")
#     if match is None:
#         raise ValueError("Match not found")
#     updated_model = patter_router.set_deployment_model_name(
#         matched_pattern=match, litellm_deployment_litellm_model="openai/*"
#     )
#     assert updated_model == "openai/fo::hi:static::hello"
