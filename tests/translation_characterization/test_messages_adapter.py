"""Pin the /v1/messages surface, both flavours:

1. The anthropic<->openai adapter (anthropic-in -> non-anthropic model):
   ``LiteLLMAnthropicMessagesAdapter.translate_anthropic_to_openai`` and
   ``translate_openai_response_to_anthropic``.
2. The native ``AnthropicMessagesConfig.transform_anthropic_messages_request``
   quirks, despite its "No transformation is needed" docstring: max_tokens is
   popped/required, reasoning_effort is rewritten to ``thinking``/adaptive
   params, and advisor blocks are stripped when the advisor tool is absent.
"""

import copy

import pytest

from ._helpers import FIXTURES_DIR, assert_snapshot, load_json

_ADAPTER_DIR = FIXTURES_DIR / "adapter"


def _ids(sub: str) -> list:
    return sorted(p.stem for p in (_ADAPTER_DIR / sub).glob("*.json"))


@pytest.mark.parametrize("fixture_id", _ids("requests"))
def test_adapter_anthropic_request_to_openai(
    fixture_id: str, snapshot_update: bool
) -> None:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    anthropic_request = load_json(_ADAPTER_DIR / "requests" / f"{fixture_id}.json")
    openai_request, tool_name_mapping = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_message_request=anthropic_request
    )
    assert_snapshot(
        f"adapter/requests/{fixture_id}.json",
        {"openai_request": openai_request, "tool_name_mapping": tool_name_mapping},
        snapshot_update,
    )


@pytest.mark.parametrize("fixture_id", _ids("responses"))
def test_adapter_openai_response_to_anthropic(
    fixture_id: str, snapshot_update: bool
) -> None:
    import litellm
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    openai_response = litellm.ModelResponse(
        **load_json(_ADAPTER_DIR / "responses" / f"{fixture_id}.json")
    )
    anthropic_response = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=openai_response
    )
    assert_snapshot(
        f"adapter/responses/{fixture_id}.json", anthropic_response, snapshot_update
    )


@pytest.mark.parametrize("fixture_id", _ids("native"))
def test_native_messages_request_transform(
    fixture_id: str, snapshot_update: bool
) -> None:
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )
    from litellm.types.router import GenericLiteLLMParams

    case = load_json(_ADAPTER_DIR / "native" / f"{fixture_id}.json")
    result = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model=case["model"],
        messages=copy.deepcopy(case["messages"]),
        anthropic_messages_optional_request_params=copy.deepcopy(case["params"]),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert_snapshot(f"adapter/native/{fixture_id}.json", result, snapshot_update)


def test_native_messages_requires_max_tokens() -> None:
    """The documented sad path: max_tokens missing -> AnthropicError(400)."""
    from litellm.llms.anthropic.common_utils import AnthropicError
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )
    from litellm.types.router import GenericLiteLLMParams

    with pytest.raises(AnthropicError) as excinfo:
        AnthropicMessagesConfig().transform_anthropic_messages_request(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert excinfo.value.status_code == 400
    assert "max_tokens is required" in str(excinfo.value)
