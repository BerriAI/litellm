from collections.abc import Mapping
from typing import Final, cast

import pytest
from pydantic import JsonValue

import litellm
from litellm.responses.bridge_check import deployment_routing_mode, responses_api_bridge_check

_CHAT: Final[Mapping[str, JsonValue]] = {"id": "dep-1", "mode": "chat"}
_RESPONSES: Final[Mapping[str, JsonValue]] = {"id": "dep-1", "mode": "responses"}


def _pin_shared_mode(monkeypatch: pytest.MonkeyPatch, model: str, mode: str) -> None:
    cost_map: Final = cast(dict[str, dict[str, object]], litellm.model_cost)
    monkeypatch.setitem(cost_map, model, {**cost_map.get(model, {}), "mode": mode})


@pytest.mark.parametrize(
    "deployment_model_info, custom_llm_provider, expected",
    [
        (_CHAT, "bedrock_mantle", "chat"),
        (_RESPONSES, "bedrock_mantle", "responses"),
        (_CHAT, "openai", "chat"),
        (_CHAT, "chatgpt", None),
        (_CHAT, "github_copilot", None),
        ({"id": "dep-1", "mode": "embedding"}, "bedrock_mantle", None),
        ({"id": "dep-1"}, "bedrock_mantle", None),
        (None, "bedrock_mantle", None),
    ],
)
def test_deployment_routing_mode_pins_chat_or_responses_except_on_responses_only_backends(
    deployment_model_info: Mapping[str, JsonValue] | None, custom_llm_provider: str, expected: str | None
) -> None:
    assert deployment_routing_mode(deployment_model_info, custom_llm_provider) == expected


def test_a_chat_pinned_mantle_deployment_stays_on_chat_completions_while_the_shared_row_says_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: Final = "bedrock_mantle/openai.gpt-5.6-sol"
    _pin_shared_mode(monkeypatch, model, "responses")
    shared_info, _ = responses_api_bridge_check(model=model, custom_llm_provider="bedrock_mantle")
    pinned_info, pinned_model = responses_api_bridge_check(
        model=model, custom_llm_provider="bedrock_mantle", deployment_model_info=_CHAT
    )
    assert shared_info["mode"] == "responses"
    assert (pinned_info["mode"], pinned_model) == ("chat", model)


def test_a_chat_pinned_chatgpt_alias_still_bridges_to_the_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_shared_mode(monkeypatch, "chatgpt/gpt-5.4", "responses")
    info, _ = responses_api_bridge_check(model="gpt-5.4", custom_llm_provider="chatgpt", deployment_model_info=_CHAT)
    assert info["mode"] == "responses"
