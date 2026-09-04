from unittest.mock import Mock

import pytest

import litellm
from litellm.rust_bridge import responses


@pytest.fixture(autouse=True)
def reset_responses_endpoint():
    responses._RESPONSES.reset()
    yield
    responses._RESPONSES.reset()


def test_unavailable_responses_bridge_does_not_prepare_request() -> None:
    prepare = Mock(side_effect=AssertionError("unavailable Rust must not prepare a request"))
    responses._RESPONSES.override(sync=None)
    litellm.rust(True)

    result = responses.responses(
        prepare=prepare,
        model="gpt-5",
        provider="openai",
        request_override=True,
    )

    assert result is None
    prepare.assert_not_called()


@pytest.mark.asyncio
async def test_unavailable_async_responses_bridge_does_not_prepare_request(
) -> None:
    prepare = Mock(side_effect=AssertionError("unavailable Rust must not prepare a request"))
    responses._RESPONSES.override(asynchronous=None)
    litellm.rust(True)

    result = await responses.aresponses(
        prepare=prepare,
        model="gpt-5",
        provider="openai",
        request_override=True,
    )

    assert result is None
    prepare.assert_not_called()
