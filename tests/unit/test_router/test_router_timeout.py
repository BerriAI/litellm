from __future__ import annotations

import asyncio
import os
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm import Router
import time
from unittest.mock import patch, MagicMock


@pytest.fixture
def restore_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "request_timeout", litellm.request_timeout)


@pytest.mark.parametrize(
    "num_retries, expected_call_count",
    [(0, 1), (1, 2), (2, 3), (3, 4)],
)
@pytest.mark.usefixtures("fake_provider_credentials", "restore_request_timeout")
def test_router_timeout_with_retries_anthropic_model(num_retries, expected_call_count):
    """
    If request hits custom timeout, ensure it's retried.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.num_retries = num_retries
    litellm.request_timeout = 0.000001

    router = Router(
        model_list=[
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": f"anthropic/{os.environ.get('CI_CD_DEFAULT_ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')}",
                },
            }
        ],
    )

    custom_client = HTTPHandler()

    with patch.object(custom_client, "post", new=MagicMock()) as mock_client:
        try:

            def delayed_response(*args, **kwargs):
                time.sleep(0.01)  # Exceeds the 0.000001 timeout
                raise TimeoutError("Request timed out.")

            mock_client.side_effect = delayed_response

            router.completion(
                model="claude-3-haiku",
                messages=[{"role": "user", "content": "hello, who are u"}],
                client=custom_client,
            )
        except litellm.Timeout:
            pass

        assert mock_client.call_count == expected_call_count


@pytest.mark.parametrize(
    "stream",
    [
        True,
        False,
    ],
)
def test_unit_test_streaming_timeout(stream):
    import os
    from dotenv import load_dotenv
    import litellm
    from litellm.router import Router, RetryPolicy, AllowedFailsPolicy

    litellm.set_verbose = True

    model_list = [
        {
            "model_name": "llama3",
            "litellm_params": {
                "model": "watsonx/meta-llama/llama-3-1-8b-instruct",
                "api_base": os.getenv("WATSONX_URL_US_SOUTH"),
                "api_key": os.getenv("WATSONX_API_KEY"),
                "project_id": os.getenv("WATSONX_PROJECT_ID_US_SOUTH"),
                "timeout": 0.01,
                "stream_timeout": 0.0000001,
            },
        },
        {
            "model_name": "bedrock-anthropic",
            "litellm_params": {
                "model": "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
                "timeout": 0.01,
                "stream_timeout": 0.0000001,
            },
        },
        {
            "model_name": "llama3-fallback",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ]

    router = Router(model_list=model_list)

    stream_timeout = 0.0000001
    normal_timeout = 0.01

    args = {
        "kwargs": {"stream": stream},
        "data": {"timeout": normal_timeout, "stream_timeout": stream_timeout},
    }

    assert router._get_stream_timeout(**args) == stream_timeout

    assert router._get_non_stream_timeout(**args) == normal_timeout

    stream_timeout_val = router._get_timeout(
        kwargs={"stream": stream},
        data={"timeout": normal_timeout, "stream_timeout": stream_timeout},
    )

    if stream:
        assert stream_timeout_val == stream_timeout
    else:
        assert stream_timeout_val == normal_timeout


@pytest.mark.parametrize(
    "request_kwargs, expected_timeout",
    [
        ({"service_tier": "flex"}, 900.0),
        ({"service_tier": "default"}, 610.0),
        ({}, 610.0),
        ({"service_tier": "flex", "timeout": 20}, 20),
        ({"service_tier": "flex", "stream": True}, 950.0),
        ({"service_tier": "default", "stream": True}, 610.0),
        ({"service_tier": "flex", "stream": True, "stream_timeout": 30}, 30),
    ],
)
def test_the_deployment_timeout_for_the_requests_service_tier_wins_over_its_general_timeout(
    request_kwargs: dict[str, object], expected_timeout: float
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai-gpt-5.1",
                "litellm_params": {
                    "model": "openai/gpt-5.1",
                    "api_key": "sk-test",
                    "timeout": 610,
                    "stream_timeout": 610,
                    "timeout_by_service_tier": {"flex": 900},
                    "stream_timeout_by_service_tier": {"flex": 950},
                },
            }
        ]
    )
    deployment_params: Final = router.model_list[0]["litellm_params"]

    assert router._get_timeout(kwargs=request_kwargs, data=deployment_params) == expected_timeout


def test_a_stream_keeps_the_deployment_stream_timeout_when_only_the_request_timeout_map_names_its_tier() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai-gpt-5.1",
                "litellm_params": {
                    "model": "openai/gpt-5.1",
                    "api_key": "sk-test",
                    "timeout": 610,
                    "stream_timeout": 30,
                    "timeout_by_service_tier": {"flex": 900},
                },
            }
        ]
    )
    deployment_params: Final = router.model_list[0]["litellm_params"]

    assert router._get_timeout(kwargs={"service_tier": "flex", "stream": True}, data=deployment_params) == 30.0
    assert router._get_timeout(kwargs={"service_tier": "flex"}, data=deployment_params) == 900.0
