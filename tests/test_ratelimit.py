# %%
import asyncio
import os
import pytest
import random
from typing import Any

from pydantic import BaseModel
from litellm import utils, Router

COMPLETION_TOKENS = 5
base_model_list = [
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": COMPLETION_TOKENS,
        },
    }
]


class RouterConfig(BaseModel):
    rpm: int
    tpm: int


@pytest.fixture(scope="function")
def router_factory():
    def create_router(rpm, tpm):
        model_list = base_model_list.copy()
        model_list[0]["rpm"] = rpm
        model_list[0]["tpm"] = tpm
        return Router(
            model_list=model_list,
            routing_strategy="usage-based-routing",
            debug_level="DEBUG",
        )

    return create_router


def generate_list_of_messages(num_messages):
    return [
        [{"role": "user", "content": f"{i}. Hey, how's it going? {random.random()}"}]
        for i in range(num_messages)
    ]


def calculate_limits(list_of_messages):
    rpm = len(list_of_messages)
    tpm = sum((utils.token_counter(messages=m) + COMPLETION_TOKENS for m in list_of_messages))
    return rpm, tpm


async def async_call(router: Router, list_of_messages) -> Any:
    tasks = [router.acompletion(model="gpt-3.5-turbo", messages=m) for m in list_of_messages]
    return await asyncio.gather(*tasks)


def sync_call(router: Router, list_of_messages) -> Any:
    return [router.completion(model="gpt-3.5-turbo", messages=m) for m in list_of_messages]


class ExpectNoException(Exception):
    pass


@pytest.mark.parametrize(
    "num_messages, num_rate_limits",
    [
        # (2, 30),  # No exception expected
        # (2, 3),  # No exception expected
        (2, 2),  # No exception expected
        (3, 2),  # Expect ValueError
        # (6, 5),  # Expect ValueError
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])  # Use parametrization for sync/async
def test_rate_limit(router_factory, num_messages, num_rate_limits, sync_mode):
    expected_exception = ExpectNoException if num_messages <= num_rate_limits else ValueError

    list_of_messages = generate_list_of_messages(max(num_messages, num_rate_limits))
    rpm, tpm = calculate_limits(list_of_messages[:num_rate_limits])
    list_of_messages = list_of_messages[:num_messages]
    router = router_factory(rpm, tpm)

    with pytest.raises(expected_exception) as excinfo:
        if sync_mode:
            results = sync_call(router, list_of_messages)
        else:
            results = asyncio.run(async_call(router, list_of_messages))
            if len([i for i in results if i is not None]) != num_messages:
                raise ValueError(
                    "No deployments available for selected model"
                )  # not all results got returned
        raise ExpectNoException

    if expected_exception is not ExpectNoException:
        assert "No deployments available for selected model" in str(excinfo.value)
    else:
        assert len([i for i in results if i is not None]) == num_messages
