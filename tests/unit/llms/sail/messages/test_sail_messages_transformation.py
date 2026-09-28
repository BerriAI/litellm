from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.anthropic.pass_through.messages.transformation import AnthropicMessagesConfig
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig
from litellm.llms.sail.common_utils import billed_service_tier
from litellm.llms.sail.messages.transformation import SailAnthropicMessagesConfig
from tests.unit.llms.sail.helpers import MODEL, SAIL_API_BASE, messages_body, sent_body

MESSAGES: Final = [{"role": "user", "content": "hi"}]


@pytest.fixture
def messages_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{SAIL_API_BASE}/messages").mock(return_value=httpx.Response(200, json=messages_body()))


@pytest.mark.parametrize("window", ["flex", "balanced", "asap"])
@pytest.mark.asyncio
async def test_sail_messages_sends_caller_window(sail_env: None, messages_route: respx.Route, window: str) -> None:
    await litellm.anthropic_messages(
        model=MODEL, messages=MESSAGES, max_tokens=16, metadata={"completion_window": window}
    )

    body: Final = sent_body(messages_route)
    assert body["metadata"] == {"completion_window": window}
    assert "service_tier" not in body


@pytest.mark.parametrize("drop_params", ["false", True])
@pytest.mark.asyncio
async def test_sail_messages_handles_unknown_window_drop_params(
    sail_env: None, messages_route: respx.Route, drop_params: bool | str
) -> None:
    if drop_params == "false":
        with pytest.raises(litellm.UnsupportedParamsError, match=r"metadata\.completion_window"):
            await litellm.anthropic_messages(
                model=MODEL,
                messages=MESSAGES,
                max_tokens=16,
                metadata={"completion_window": "unknown"},
                drop_params=drop_params,
            )
        assert not messages_route.called
        return

    await litellm.anthropic_messages(
        model=MODEL,
        messages=MESSAGES,
        max_tokens=16,
        metadata={"completion_window": "unknown"},
        drop_params=drop_params,
    )

    body: Final = sent_body(messages_route)
    assert "completion_window" not in body["metadata"]


def test_sail_messages_logging_optional_params_bill_selected_window() -> None:
    logging_optional_params: Final = {"metadata": {"completion_window": "flex"}, "service_tier": "balanced"}

    assert billed_service_tier(logging_optional_params) == "flex"


def test_native_anthropic_messages_filter_provider_metadata() -> None:
    assert AnthropicMessagesConfig().request_metadata({"user_id": "user-1", "trace_id": "internal"}) == {
        "user_id": "user-1"
    }


def test_messages_metadata_filtering_is_provider_owned() -> None:
    meta_provider: Final = JSONProviderRegistry.get("meta")
    sail_provider: Final = JSONProviderRegistry.get("sail")
    assert meta_provider is not None
    assert sail_provider is not None
    openai_like: Final = JSONProviderAnthropicMessagesConfig(meta_provider)
    sail: Final = SailAnthropicMessagesConfig(sail_provider)
    metadata: Final = {"completion_window": "flex", "user_id": "u"}

    assert openai_like.request_metadata(metadata) == {"user_id": "u"}
    assert sail.request_metadata(metadata) == metadata
