from typing import Final

import pytest

from litellm import Router


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
