"""Contract: Together AI reasoning, tools and replayed reasoning; Databricks serving endpoint auth, URL and tool shapes.

Together AI (`TogetherAIChatConfig`, `litellm/llms/together_ai/chat/transformation.py`) is an OpenAI-compatible chat
client posting to `<api_base>/chat/completions`. Its own rules: models the price map does not know keep `tools`
and `response_format` (Together validates them), replayed assistant `reasoning_content` stays in the history
(Together uses it for preserved thinking), and `reasoning_effort` is reshaped per model family: hybrid models
take `reasoning: {"enabled": false}` for `none` and translate `minimal`/`xhigh` to `low`/`high`, while
DeepSeek-V4-Pro only takes `high` and `max`.

Databricks (`DatabricksConfig`, `litellm/llms/databricks/chat/transformation.py`, `common_utils.py`) posts to
`<api_base>/chat/completions` where the base is the workspace `/serving-endpoints` route, with `Authorization:
Bearer <PAT>` and a `User-Agent` of `litellm/<version>` (or `<partner>_litellm/<version>` when a user agent is
configured). Models whose name holds two or more dots go through the AI Gateway at `/ai-gateway/mlflow/v1`.
Claude models get tools rebuilt from their Anthropic form (`name`, `parameters`, `description` only), other
models get tools as sent; `max_completion_tokens` is sent as `max_tokens`; params outside its list are refused.
The large Databricks suites (`test_databricks_*_wire.py`) cover JSON schema refs, content flattening, agents,
OAuth and decisions.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_otherprov2_support import upstream_error
from integration.providers.dec1_otherprov_support import (
    PROVIDER_KEY,
    TOOL,
    chat,
    deployment,
    error_message,
    marker,
    peer,
    served,
)
from pydantic import JsonValue

_TOGETHER_UNMAPPED: Final = "together_ai/dec1-org/dec1-unmapped-chat"
_TOGETHER_HYBRID: Final = "together_ai/nvidia/nemotron-3-ultra-550b-a55b"
_TOGETHER_DSV4: Final = "together_ai/deepseek-ai/DeepSeek-V4-Pro"
_DBX_LLAMA: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
_DBX_CLAUDE: Final = "databricks/databricks-claude-sonnet-4-5"


def test_together_unmapped_model_keeps_tools_and_response_format(gateway: Gateway) -> None:
    """`together_ai/dec1-org/dec1-unmapped-chat` has no price map row, so the function calling and structured output
    verdicts are unknown and both are passed through for Together to validate (`_tool_params_to_drop`,
    `_drop_response_format`, `together_ai/chat/transformation.py:88-102`, `:149-161`): Together receives the tool,
    `tool_choice: "auto"` and `response_format: {"type": "json_object"}`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _TOGETHER_UNMAPPED, api_base_suffix="/v1")
        served(chat(gateway, model, tools=[TOOL], tool_choice="auto", response_format={"type": "json_object"}))
        sent: Final = provider.only().body
        assert (sent["tool_choice"], sent["response_format"]) == ("auto", {"type": "json_object"}), sent
        assert len(list_value(sent["tools"])) == 1, sent


def test_together_replayed_reasoning_content_stays_in_the_history(gateway: Gateway) -> None:
    """An assistant turn replayed with `reasoning_content: "dec1 earlier thought"`. Together consumes it for
    preserved thinking, so only litellm-internal fields are stripped and the assistant message reaches Together
    with the same `reasoning_content` (`_transform_messages`, `together_ai/chat/transformation.py:198-210`)."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"role": "assistant", "content": "earlier", "reasoning_content": "dec1 earlier thought"},
        {"role": "user", "content": marker()},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _TOGETHER_HYBRID, api_base_suffix="/v1")
        served(chat(gateway, model, messages=messages))
        assistant: Final = object_value(list_value(provider.only().body["messages"])[1])
        assert assistant["reasoning_content"] == "dec1 earlier thought", assistant


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("none", {"reasoning": {"enabled": False}}),
        ("minimal", {"reasoning_effort": "low"}),
        ("xhigh", {"reasoning_effort": "high"}),
        ("medium", {"reasoning_effort": "medium"}),
    ],
)
def test_together_hybrid_model_reasoning_effort_mapping(
    gateway: Gateway, effort: str, expected: dict[str, JsonValue]
) -> None:
    """`nvidia/nemotron-3-ultra-550b-a55b` is a hybrid reasoning model with no declared efforts in the price map.
    `none` turns reasoning off with `reasoning: {"enabled": false}`, `minimal` and `xhigh` become `low` and `high`
    (`EFFORT_TRANSLATION`), and `medium` passes as is (`_reasoning_effort_payload`, `together_ai/chat/
    transformation.py:62`, `:134-146`). The body carries exactly the mapped reasoning keys."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _TOGETHER_HYBRID, api_base_suffix="/v1")
        served(chat(gateway, model, reasoning_effort=effort))
        sent: Final = provider.only().body
        assert {key: sent[key] for key in ("reasoning_effort", "reasoning") if key in sent} == expected, sent


@pytest.mark.parametrize(
    ("effort", "expected"), [("low", "high"), ("medium", "high"), ("xhigh", "max"), ("high", "high")]
)
def test_together_deepseek_v4_pro_only_takes_high_or_max(gateway: Gateway, effort: str, expected: str) -> None:
    """`deepseek-ai/DeepSeek-V4-Pro` (no price map row) starts with `HIGH_MAX_EFFORT_MODEL_PREFIX`, so `low` and
    `medium` become `high`, `xhigh` becomes `max`, and `high` stays (`HIGH_MAX_EFFORT_TRANSLATION`,
    `together_ai/chat/transformation.py:61-65`, `:144-145`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _TOGETHER_DSV4, api_base_suffix="/v1")
        served(chat(gateway, model, reasoning_effort=effort))
        assert provider.only().body["reasoning_effort"] == expected


def test_together_rate_limit_reaches_the_caller_as_429(gateway: Gateway) -> None:
    """Together answers 429 with `dec1 together busy`. The caller gets 429 with that message after exactly one
    upstream call (Together is in `openai_compatible_providers`, `exception_mapping_utils.py:2733-2749`)."""
    with peer(lambda call: upstream_error(429, "dec1 together busy")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _TOGETHER_UNMAPPED, api_base_suffix="/v1")
        response: Final = chat(gateway, model)
        assert response.status_code == 429, response.text
        assert "dec1 together busy" in error_message(response)
        assert len(provider.posts()) == 1


def test_databricks_posts_to_the_serving_endpoints_chat_route_with_the_pat_and_user_agent(gateway: Gateway) -> None:
    """A Databricks deployment whose base is the peer plus `/serving-endpoints` and whose key is the PAT.

    Databricks receives `POST /serving-endpoints/chat/completions` (`get_complete_url`, `databricks/chat/
    transformation.py:468-478`) with `authorization: Bearer dec1-otherprov-provider-key`, `content-type:
    application/json`, a `user-agent` starting with `litellm/` (`_build_user_agent`, `databricks/common_utils.py:
    229-269`), and `model: databricks-meta-llama-3-3-70b-instruct` in the body.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _DBX_LLAMA, api_base_suffix="/serving-endpoints")
        served(chat(gateway, model))
        call: Final = provider.only()
        assert call.path == "/serving-endpoints/chat/completions", call.path
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}", call.headers
        assert call.headers["user-agent"].startswith("litellm/"), call.headers
        assert call.body["model"] == "databricks-meta-llama-3-3-70b-instruct", call.body


def test_databricks_partner_user_agent_is_prefixed_and_kept_out_of_the_body(gateway: Gateway) -> None:
    """The deployment sets `user_agent: "dec1partner/9.9.9"`. Databricks receives `user-agent:
    dec1partner_litellm/<litellm version>` (the partner's version is ignored) and no `user_agent` key in the body,
    since `validate_environment` pops it from the params (`databricks/chat/transformation.py:445-454`,
    `common_utils.py:256-269`). UNVERIFIED: that a deployment-level `user_agent` reaches `optional_params` rather
    than only `litellm_params`; either source is read, so the header is pinned, and the body check holds for both."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, _DBX_LLAMA, api_base_suffix="/serving-endpoints", user_agent="dec1partner/9.9.9"
        )
        served(chat(gateway, model))
        call: Final = provider.only()
        assert call.headers["user-agent"].startswith("dec1partner_litellm/"), call.headers
        assert "9.9.9" not in call.headers["user-agent"], call.headers
        assert "user_agent" not in call.body, call.body


def test_databricks_invalid_partner_name_falls_back_to_the_plain_user_agent(gateway: Gateway) -> None:
    """The deployment sets `user_agent: "dec1 partner!/1.0"`; the partner name has a space and `!`, which fail the
    alphanumeric check, so the header is the plain `litellm/<version>` (`databricks/common_utils.py:259-269`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, _DBX_LLAMA, api_base_suffix="/serving-endpoints", user_agent="dec1 partner!/1.0"
        )
        served(chat(gateway, model))
        assert provider.only().headers["user-agent"].startswith("litellm/"), provider.only().headers


def test_databricks_dotted_model_on_a_bare_host_goes_through_the_ai_gateway(gateway: Gateway) -> None:
    """A model `databricks/system.ai.dec1-llm` has two dots, so it uses the AI Gateway (`uses_ai_gateway`,
    `databricks/common_utils.py:199-201`). With a bare base (the peer, no path) the gateway route is joined on
    (`_get_api_base`, `:289-299`): Databricks receives `POST /ai-gateway/mlflow/v1/chat/completions`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "databricks/system.ai.dec1-llm")
        served(chat(gateway, model))
        assert provider.only().path == "/ai-gateway/mlflow/v1/chat/completions"


def test_databricks_claude_tools_are_rebuilt_without_strict(gateway: Gateway) -> None:
    """A strict tool on `databricks-claude-sonnet-4-5`. Claude tools are converted to Anthropic tools and back
    (`_map_openai_to_dbrx_tool`, `convert_anthropic_tool_to_databricks_tool`, `databricks/chat/transformation.py:
    506-545`), so Databricks receives `{"type": "function", "function": {"name", "parameters", "description"}}`
    and the function has no `strict` key."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _DBX_CLAUDE, api_base_suffix="/serving-endpoints")
        served(chat(gateway, model, tools=[TOOL]))
        tool: Final = object_value(list_value(provider.only().body["tools"])[0])
        function: Final = object_value(tool["function"])
        assert tool["type"] == "function", tool
        assert sorted(function) == ["description", "name", "parameters"], function
        assert function["name"] == "get_weather", function


def test_databricks_non_claude_tools_are_sent_as_given(gateway: Gateway) -> None:
    """The same strict tool on the Llama endpoint is sent unchanged ("if not claude, send as is",
    `databricks/chat/transformation.py:525-527`): the function keeps `strict: true`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _DBX_LLAMA, api_base_suffix="/serving-endpoints")
        served(chat(gateway, model, tools=[TOOL]))
        function: Final = object_value(object_value(list_value(provider.only().body["tools"])[0])["function"])
        assert function["strict"] is True, function


def test_databricks_sends_max_completion_tokens_as_max_tokens(gateway: Gateway) -> None:
    """`max_completion_tokens: 64` reaches Databricks as `max_tokens: 64` with no `max_completion_tokens`
    (`databricks/chat/transformation.py:593-597`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _DBX_LLAMA, api_base_suffix="/serving-endpoints")
        served(chat(gateway, model, max_completion_tokens=64))
        sent: Final = provider.only().body
        assert sent["max_tokens"] == 64 and "max_completion_tokens" not in sent, sent


def test_databricks_refuses_a_param_outside_its_list(gateway: Gateway) -> None:
    """`presence_penalty` is not in `DatabricksConfig.get_supported_openai_params`
    (`databricks/chat/transformation.py:480-495`), so `presence_penalty: 0.5` is answered 400 with `databricks does
    not support parameters: ['presence_penalty']` before Databricks is called."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _DBX_LLAMA, api_base_suffix="/serving-endpoints")
        response: Final = chat(gateway, model, presence_penalty=0.5)
        assert response.status_code == 400, response.text
        assert "databricks does not support parameters: ['presence_penalty']" in error_message(response)
        assert provider.posts() == ()


def test_databricks_service_tier_is_forwarded(gateway: Gateway) -> None:
    """`service_tier` is in the Databricks supported list (`databricks/chat/transformation.py:494`), so
    `service_tier: "priority"` reaches the serving endpoint as sent."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _DBX_LLAMA, api_base_suffix="/serving-endpoints")
        served(chat(gateway, model, service_tier="priority"))
        assert provider.only().body["service_tier"] == "priority"
