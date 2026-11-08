from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm import ModelResponse
from litellm.llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig
from litellm.utils import get_optional_params
import os


@pytest.mark.parametrize(
    "model_name,expected",
    [
        # Valid O-series models
        ("o1", True),
        ("o3", True),
        ("o4-mini", True),
        ("o3-mini", True),
        # Valid O-series models with provider prefix
        ("openai/o1", True),
        ("openai/o3", True),
        ("openai/o4-mini", True),
        ("openai/o3-mini", True),
        # Non-O-series models
        ("gpt-4", False),
        ("gpt-3.5-turbo", False),
        ("claude-3-opus", False),
        # Non-O-series models with provider prefix
        ("openai/gpt-4", False),
        ("openai/gpt-3.5-turbo", False),
        ("anthropic/claude-3-opus", False),
        # Edge cases
        ("o", False),  # Too short
        ("o5", False),  # Not a valid O-series model
        ("o1-", False),  # Invalid suffix
        ("o3_", False),  # Invalid suffix
    ],
)
def test_is_model_o_series_model(model_name: str, expected: bool):
    """
    Test that is_model_o_series_model correctly identifies O-series models.

    Args:
        model_name: The model name to test
        expected: The expected result (True if it should be identified as an O-series model)
    """
    config = OpenAIOSeriesConfig()
    assert config.is_model_o_series_model(model_name) == expected, (
        f"Expected {model_name} to be {'an O-series model' if expected else 'not an O-series model'}"
    )


@pytest.mark.parametrize("model", ["o1"])
@pytest.mark.asyncio
async def test_o1_handle_system_role(model):
    """
    Tests that:
    - max_tokens is translated to 'max_completion_tokens'
    - role 'system' is translated to 'user'
    """
    from openai import AsyncOpenAI
    from litellm.utils import supports_system_messages

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.set_verbose = True

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client, "post"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model=model,
                max_tokens=10,
                messages=[{"role": "system", "content": "Be a good bot!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs["body"]

        print("request_body: ", request_body)

        assert request_body["model"] == model
        assert request_body["max_completion_tokens"] == 10
        if supports_system_messages(model, "openai"):
            assert request_body["messages"] == [
                {"role": "system", "content": "Be a good bot!"}
            ]
        else:
            assert request_body["messages"] == [
                {"role": "user", "content": "Be a good bot!"}
            ]


@pytest.mark.parametrize(
    "model, expected_tool_calling_support",
    [("o1", True)],
)
@pytest.mark.asyncio
async def test_o1_handle_tool_calling_optional_params(
    model, expected_tool_calling_support
):
    """
    Tests that:
    - max_tokens is translated to 'max_completion_tokens'
    - role 'system' is translated to 'user'
    """
    from litellm.utils import ProviderConfigManager
    from litellm.types.utils import LlmProviders

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    config = ProviderConfigManager.get_provider_chat_config(
        model=model, provider=LlmProviders.OPENAI
    )

    supported_params = config.get_supported_openai_params(model=model)

    assert expected_tool_calling_support == ("tools" in supported_params)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4", "gpt-4-0613"])
async def test_o1_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed directly to OpenAI chat completion models
    """
    from openai import AsyncOpenAI

    litellm.set_verbose = True

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client, "post"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs["body"]

        print("request_body: ", request_body)

        assert request_body["model"] == model
        assert request_body["max_completion_tokens"] == 10
        assert request_body["messages"] == [{"role": "user", "content": "Hello!"}]


def test_litellm_responses():
    """
    ensures that type of completion_tokens_details is correctly handled / returned
    """
    from litellm.types.utils import CompletionTokensDetails

    response = ModelResponse(
        usage={
            "completion_tokens": 436,
            "prompt_tokens": 14,
            "total_tokens": 450,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    print("response: ", response)

    assert isinstance(response.usage.completion_tokens_details, CompletionTokensDetails)


def _o3_row_with(flags: Mapping[str, bool]) -> dict[str, object]:
    return {
        **{
            field: value
            for field, value in litellm.model_cost["o3"].items()
            if not (field.startswith("supports_") and field.endswith("_reasoning_effort"))
        },
        **flags,
    }


@pytest.mark.parametrize(
    ("flags", "level", "forwarded"),
    [
        ({}, "minimal", True),
        ({"supports_minimal_reasoning_effort": False}, "minimal", False),
        ({"supports_minimal_reasoning_effort": True}, "minimal", True),
        ({"supports_none_reasoning_effort": False}, "none", False),
        ({"supports_minimal_reasoning_effort": False}, "low", True),
        ({"supports_xhigh_reasoning_effort": False}, "xhigh", False),
        ({}, "xhigh", True),
    ],
)
def test_the_o_series_row_flags_alone_decide_whether_a_chat_level_is_forwarded(
    local_model_cost_map: None,
    monkeypatch: pytest.MonkeyPatch,
    flags: dict[str, bool],
    level: str,
    forwarded: bool,
):
    monkeypatch.setitem(litellm.model_cost, "o3", _o3_row_with(flags))
    config: Final = OpenAIOSeriesConfig()
    dropped: Final = config.map_openai_params(
        non_default_params={"reasoning_effort": level, "max_tokens": 10},
        optional_params={},
        model="o3",
        drop_params=True,
    )
    assert dropped.get("reasoning_effort") == (level if forwarded else None)
    assert dropped["max_completion_tokens"] == 10
    if forwarded:
        assert (
            config.map_openai_params(
                non_default_params={"reasoning_effort": level},
                optional_params={},
                model="o3",
                drop_params=False,
            )["reasoning_effort"]
            == level
        )
    else:
        with pytest.raises(litellm.utils.UnsupportedParamsError, match=f"reasoning_effort={level} is not supported"):
            config.map_openai_params(
                non_default_params={"reasoning_effort": level},
                optional_params={},
                model="o3",
                drop_params=False,
            )


@pytest.mark.parametrize("model", ["o3", "o3-2025-04-16"])
def test_o3_rows_turn_minimal_and_none_off(local_model_cost_map: None, model: str):
    """Pinned so the row-flag test above cannot go vacuous: OpenAI's chat completions endpoint answered
    400 for reasoning_effort minimal and none on o3 and 200 for low, medium, high and xhigh, both through
    a local proxy and straight from api.openai.com on 2026-11-07. When OpenAI adds a level, flip the rows
    and this assertion together."""
    row: Final = litellm.model_cost[model]
    assert row["supports_minimal_reasoning_effort"] is False
    assert row["supports_none_reasoning_effort"] is False


@pytest.mark.parametrize(
    ("non_default_params", "optional_params"),
    [
        ({"reasoning_effort": {"effort": "minimal", "summary": "auto"}}, {}),
        ({}, {"reasoning_effort": "minimal"}),
    ],
)
def test_o3_judges_every_form_of_a_requested_level(
    local_model_cost_map: None, non_default_params: dict[str, object], optional_params: dict[str, object]
):
    assert "reasoning_effort" not in OpenAIOSeriesConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="o3",
        drop_params=True,
    )


@pytest.mark.parametrize("custom_llm_provider", ["openai", "azure"])
def test_o3_chat_request_drops_minimal_under_drop_params(local_model_cost_map: None, custom_llm_provider: str):
    assert "reasoning_effort" not in get_optional_params(
        model="o3", custom_llm_provider=custom_llm_provider, reasoning_effort="minimal", drop_params=True
    )


@pytest.mark.parametrize("custom_llm_provider", ["openai", "azure"])
def test_o3_chat_request_refuses_minimal_without_drop_params(local_model_cost_map: None, custom_llm_provider: str):
    with pytest.raises(litellm.utils.UnsupportedParamsError, match="reasoning_effort=minimal is not supported"):
        get_optional_params(model="o3", custom_llm_provider=custom_llm_provider, reasoning_effort="minimal")
