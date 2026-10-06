from typing import Final

from litellm import Router


def test_get_settings_returns_the_routing_and_retry_settings_the_router_was_built_with():
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-4.1-mini", "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": "fake-key"}}
        ],
        routing_strategy="latency-based-routing",
        routing_strategy_args={"ttl": 10},
        num_retries=3,
        retry_after=5,
        allowed_fails=1,
        cooldown_time=30,
    )

    settings: Final = router.get_settings()

    assert settings["routing_strategy"] == "latency-based-routing"
    assert settings["routing_strategy_args"]["ttl"] == 10
    assert settings["allowed_fails"] == 1
    assert settings["num_retries"] == 3
    assert settings["retry_after"] == 5
    assert settings["cooldown_time"] == 30
