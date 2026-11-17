"""Contract: chat completions on the large OpenAI-compatible providers, through the proxy, against a scripted peer.

Groq, DeepSeek, xAI, Mistral, OpenRouter, Together AI, Fireworks AI, Baseten and hosted vLLM all speak the OpenAI
chat dialect, so a deployment with an `api_base` sends `POST <api_base>/chat/completions` with
`Authorization: Bearer <api_key>`. Each provider's config (`litellm/llms/<provider>/chat/transformation.py`)
then changes a few things, and those differences are the contract here:

- base URL shapes: Mistral appends `/v1` when the base lacks it (`ensure_api_route(base, "v1")`), DeepSeek appends
  `chat/completions` only when missing, the others append `/chat/completions` to whatever base is given;
- auth: hosted vLLM sends `Bearer fake-api-key` when no key is configured, OpenRouter adds `HTTP-Referer:
  https://litellm.ai` and `X-Title: liteLLM`;
- param mapping: Mistral sends `max_completion_tokens` as `max_tokens`, `seed` as `random_seed`, and
  `tool_choice: "required"` as `"any"`; xAI sends `max_completion_tokens` as `max_tokens` and strips `strict` from
  tools and `name` from messages; DeepSeek turns `reasoning_effort` into `thinking: {"type": ...}`; Together maps
  efforts per model family; OpenRouter always asks for `usage: {"include": true}` and lifts `transforms`,
  `models`, `route` to the top level; hosted vLLM drops an empty `tools` list and strips `strict` and
  `additionalProperties: false` from tool schemas; Groq emulates `json_schema` with a forced `json_tool_call` tool
  on models without native structured outputs;
- params the provider does not take are refused (400) or dropped under `drop_params`, and `user` is silently
  left out for configs that do not list it;
- usage and cost: spend uses the deployment's prices; OpenRouter's `usage.cost` is reported as the response cost.

The `/v1/messages` bridge to Fireworks and Baseten is covered in `messages_endpoint/test_dec1_msgprov_openai_compatible.py`;
the bridge to the other providers is in `test_dec1_openai_messages_bridge_wire.py`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    CALL_COST,
    PROVIDER_KEY,
    Call,
    chat,
    chat_json,
    chat_stream,
    deployment,
    error_message,
    first_message,
    json_reply,
    marker,
    peer,
    served,
    spend_row_by_group,
    sse,
    stream_lines,
)
from pydantic import JsonValue

_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

_GROQ_SERVICE_TIER_BUG: Final = (
    "BUG: a Groq reply without service_tier fails with AttributeError on ModelResponse.service_tier"
    " (groq/chat/transformation.py:329 calls getattr without a default)"
)
_TOGETHER_PRICE_BUG: Final = (
    "BUG: a together_ai deployment's own prices are ignored; the cost uses the size-category price"
    " (cost_calculator.py:2305-2308 checks only the registry key for together_ai/<model>)"
)
_XAI_FINISH_REASON_BUG: Final = (
    "BUG: xAI's empty finish_reason is mapped to stop before the tool_calls fix can see it"
    " (core_helpers.py:493-498 runs first, xai/chat/transformation.py:281-289 then never fires)"
)

_PROVIDERS: Final = (
    ("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
    ("deepseek/deepseek-chat", "deepseek-chat"),
    ("xai/grok-3", "grok-3"),
    ("mistral/mistral-large-latest", "mistral-large-latest"),
    ("openrouter/openai/gpt-4o-mini", "openai/gpt-4o-mini"),
    ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    (
        "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
    ),
    ("baseten/zai-org/GLM-4.6", "zai-org/GLM-4.6"),
    ("hosted_vllm/dec1-served-model", "dec1-served-model"),
)


@pytest.mark.parametrize(("model_path", "upstream"), _PROVIDERS)
def test_each_provider_posts_chat_completions_under_the_v1_base_with_the_bearer_key(
    gateway: Gateway, model_path: str, upstream: str
) -> None:
    """A deployment of each provider with `api_base` `<peer>/v1` and `api_key` `dec1-openai-provider-key`.

    The provider receives one `POST /v1/chat/completions` with `authorization: Bearer dec1-openai-provider-key`,
    the upstream model name without the provider prefix, and the caller's user message. This is the one shape all
    of them share and what a Rust OpenAI-compatible client must produce. The Groq cell is skipped as a product
    bug, see `_GROQ_SERVICE_TIER_BUG`.
    """
    if model_path.startswith("groq/"):
        pytest.skip(_GROQ_SERVICE_TIER_BUG)
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, model_path)
        served(chat(gateway, model, text=text))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/chat/completions"), call.path
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body["model"] == upstream, call.body
        assert list_value(call.body["messages"])[-1] == {"role": "user", "content": text}


@pytest.mark.parametrize(("model_path", "upstream"), _PROVIDERS)
def test_each_provider_call_is_billed_at_the_deployment_prices(
    gateway: Gateway, model_path: str, upstream: str
) -> None:
    """The provider reports 11 prompt and 7 completion tokens; the deployment prices them at 0.001 and 0.002.

    The caller's `x-litellm-response-cost` header and the one SpendLogs row both say 0.025, and the row's
    `custom_llm_provider` is the provider prefix. OpenRouter is the exception only when it reports its own cost
    (see the OpenRouter cost test). The Groq and Together cells are skipped as product bugs, see
    `_GROQ_SERVICE_TIER_BUG` and `_TOGETHER_PRICE_BUG` (Together billed 1.44e-05 from its 41.1b-80b category).
    """
    if model_path.startswith("groq/"):
        pytest.skip(_GROQ_SERVICE_TIER_BUG)
    if model_path.startswith("together_ai/"):
        pytest.skip(_TOGETHER_PRICE_BUG)
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, model_path)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST), response.headers
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST), row
        assert row["custom_llm_provider"] == model_path.split("/")[0], row


def test_mistral_appends_v1_to_a_bare_api_base(gateway: Gateway) -> None:
    """A Mistral deployment whose `api_base` is the bare peer URL (no `/v1`).

    `get_mistral_chat_credentials` runs `ensure_api_route(base, "v1")`, so the provider still receives
    `POST /v1/chat/completions`. A deployment that already ends in `/v1` is not doubled (see the shared test).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/mistral-large-latest", api_base_suffix="")
        served(chat(gateway, model))
        assert provider.only().path == "/v1/chat/completions"


def test_deepseek_does_not_double_a_full_chat_completions_base(gateway: Gateway) -> None:
    """A DeepSeek deployment whose `api_base` already ends in `/beta/chat/completions`.

    `get_complete_url` uses `ensure_api_route(base, "chat/completions")`, so the provider receives
    `POST /beta/chat/completions`, never `/beta/chat/completions/chat/completions`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "deepseek/deepseek-chat", api_base_suffix="/beta/chat/completions"
        )
        served(chat(gateway, model))
        assert provider.only().path == "/beta/chat/completions"


def test_groq_with_a_bare_base_posts_chat_completions_at_the_root(gateway: Gateway) -> None:
    """A Groq deployment whose `api_base` is `<peer>/openai/v1`, Groq's real layout.

    The provider receives `POST /openai/v1/chat/completions`: the base path is kept and only `/chat/completions`
    is appended. Skipped as a product bug until fixed, see `_GROQ_SERVICE_TIER_BUG`.
    """
    pytest.skip(_GROQ_SERVICE_TIER_BUG)
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "groq/llama-3.3-70b-versatile", api_base_suffix="/openai/v1")
        served(chat(gateway, model))
        assert provider.only().path == "/openai/v1/chat/completions"


def test_hosted_vllm_without_a_key_sends_the_fake_key(gateway: Gateway) -> None:
    """A hosted vLLM deployment with an `api_base` and no `api_key` (and no `HOSTED_VLLM_API_KEY` on the proxy).

    The provider receives `authorization: Bearer fake-api-key` (`_get_openai_compatible_provider_info`), since
    most vLLM servers run without auth but the OpenAI client needs a key.
    UNVERIFIED: assumes the shared proxy has no `HOSTED_VLLM_API_KEY` in its environment.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="hosted_vllm/dec1-served-model", api_base=f"{provider.url}/v1", api_key=None, num_retries=0
        )
        served(chat(gateway, model))
        assert provider.only().headers["authorization"] == "Bearer fake-api-key"


def test_hosted_vllm_drops_an_empty_tools_list_with_its_tool_choice(gateway: Gateway) -> None:
    """The caller sends `tools: []` with `tool_choice: "auto"` and `parallel_tool_calls: true` (#50812).

    vLLM rejects an empty tools list, so the provider body has none of `tools`, `tool_choice` or
    `parallel_tool_calls`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "hosted_vllm/dec1-served-model")
        served(chat(gateway, model, tools=[], tool_choice="auto", parallel_tool_calls=True))
        sent: Final = provider.only().body
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & set(sent), sent


def test_hosted_vllm_strips_strict_and_closed_schemas_from_tools(gateway: Gateway) -> None:
    """A tool with `strict: true` and `additionalProperties: false` sent to hosted vLLM.

    The provider's tool has neither `strict` nor `additionalProperties` anywhere in it
    (`remove_additional_properties`, `remove_strict_from_schema`), while name, description and properties stay.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "hosted_vllm/dec1-served-model")
        served(chat(gateway, model, tools=[_TOOL]))
        function: Final = object_value(object_value(list_value(provider.only().body["tools"])[0])["function"])
        assert "strict" not in function, function
        assert "additionalProperties" not in object_value(function["parameters"]), function
        assert function["name"] == "get_weather"


def test_hosted_vllm_turns_anthropic_style_thinking_into_reasoning_effort(gateway: Gateway) -> None:
    """The caller sends `thinking: {"type": "enabled", "budget_tokens": 1024}` to hosted vLLM without a
    `reasoning_effort`.

    The provider receives a `reasoning_effort` string and no `thinking` key (`reasoning_effort_from_thinking`).
    A 1024 budget is the `low` bucket (`DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET` is 1024).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "hosted_vllm/dec1-served-model")
        served(chat(gateway, model, thinking={"type": "enabled", "budget_tokens": 1024}))
        sent: Final = provider.only().body
        assert "thinking" not in sent and sent.get("reasoning_effort") == "low", sent


def test_hosted_vllm_leaves_out_user_because_its_config_does_not_list_it(gateway: Gateway) -> None:
    """`user` is only in the supported list for OpenAI catalog models, and `user` is exempt from the unsupported
    check (`PROVIDER_UNVALIDATED_PARAMS`), so on hosted vLLM it is neither refused nor forwarded.

    The call succeeds and the provider body has no `user` key. Pinned current behaviour: a port that forwards
    `user` everywhere would change what self-hosted servers see.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "hosted_vllm/dec1-served-model")
        served(chat(gateway, model, user="dec1-end-user"))
        assert "user" not in provider.only().body


def test_hosted_vllm_maps_a_reasoning_field_in_the_answer_to_reasoning_content(gateway: Gateway) -> None:
    """The vLLM server answers a message with `reasoning: "thought dec1"` (newer vLLM naming).

    The caller's message has `reasoning_content: "thought dec1"`. UNVERIFIED: the non-stream mapping; the stream
    mapping is in `OpenAIChatCompletionStreamingHandler._map_reasoning_to_reasoning_content`.
    """

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, extra_message={"reasoning": "thought dec1"}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "hosted_vllm/dec1-served-model")
        assert first_message(served(chat(gateway, model))).get("reasoning_content") == "thought dec1"


def test_mistral_maps_max_completion_tokens_seed_and_required_tool_choice(gateway: Gateway) -> None:
    """The caller sends `max_completion_tokens: 40`, `seed: 9`, a tool and `tool_choice: "required"` to Mistral.

    The provider body has `max_tokens: 40` and no `max_completion_tokens`, `random_seed: 9` and no `seed`, and
    `tool_choice: "any"` (Mistral's word for required) (`MistralConfig.map_openai_params`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/mistral-large-latest")
        served(chat(gateway, model, max_completion_tokens=40, seed=9, tools=[_TOOL], tool_choice="required"))
        sent: Final = provider.only().body
        assert (sent["max_tokens"], sent["random_seed"], sent["tool_choice"]) == (40, 9, "any"), sent
        assert "max_completion_tokens" not in sent and "seed" not in sent, sent


def test_mistral_keeps_strict_on_tools(gateway: Gateway) -> None:
    """A tool with `strict: true` reaches Mistral with `strict: true`, its name and its properties. Pinned current
    behaviour: `_clean_tool_schema_for_mistral` only strips `$ref`/`$defs` (`mistral/chat/transformation.py:517-542`,
    its docstring still lists `strict`), and the Mistral API takes `strict` on a function."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/mistral-large-latest")
        served(chat(gateway, model, tools=[_TOOL]))
        function: Final = object_value(object_value(list_value(provider.only().body["tools"])[0])["function"])
        assert function["strict"] is True and function["name"] == "get_weather", function


@pytest.mark.parametrize(("name", "value"), [("frequency_penalty", 0.2), ("logprobs", True), ("n", 2)])
def test_mistral_refuses_params_it_does_not_take(gateway: Gateway, name: str, value: JsonValue) -> None:
    """Mistral's supported list has no `frequency_penalty`, `logprobs` or `n`.

    Each is answered 400 with `mistral does not support parameters: ['<name>']` and the provider is not called.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/mistral-large-latest")
        response: Final = chat(gateway, model, **{name: value})
        assert response.status_code == 400, response.text
        assert f"mistral does not support parameters: ['{name}']" in error_message(response), response.text
        assert provider.posts() == ()


def test_mistral_drops_unsupported_params_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true`, `frequency_penalty: 0.2` on Mistral is removed and `temperature: 0.4` sent with it
    is kept."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/mistral-large-latest", drop_params=True)
        served(chat(gateway, model, frequency_penalty=0.2, temperature=0.4))
        sent: Final = provider.only().body
        assert "frequency_penalty" not in sent and sent["temperature"] == 0.4, sent


def test_xai_maps_max_completion_tokens_strips_strict_and_keeps_a_user_name(gateway: Gateway) -> None:
    """The caller sends `max_completion_tokens: 30`, a strict tool, and a user message with `name: "alice"` to
    `xai/grok-3`.

    The provider body has `max_tokens: 30` and no `max_completion_tokens`, its tool has no `strict`, and the user
    message keeps `name: "alice"`: `strip_name_from_messages` drops `name` only outside its `allowed_name_roles`,
    which default to `["user"]` (`prompt_templates/common_utils.py:392-405`, called at
    `xai/chat/transformation.py:277`).
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-3")
        served(
            chat(
                gateway,
                model,
                messages=[{"role": "user", "content": text, "name": "alice"}],
                max_completion_tokens=30,
                tools=[_TOOL],
            )
        )
        sent: Final = provider.only().body
        assert sent["max_tokens"] == 30 and "max_completion_tokens" not in sent, sent
        assert "strict" not in object_value(object_value(list_value(sent["tools"])[0])["function"]), sent
        assert list_value(sent["messages"]) == [{"role": "user", "content": text, "name": "alice"}], sent


def test_xai_grok_4_refuses_stop(gateway: Gateway) -> None:
    """`grok-4` models do not take stop sequences (`XAIChatConfig._supports_stop_reason`), so `stop: ["END"]` on
    `xai/grok-4.3` is answered 400 with `xai does not support parameters: ['stop']`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-4.3")
        response: Final = chat(gateway, model, stop=["END"])
        assert response.status_code == 400, response.text
        assert "xai does not support parameters: ['stop']" in error_message(response), response.text
        assert provider.posts() == ()


def test_xai_empty_finish_reason_with_tool_calls_becomes_tool_calls(gateway: Gateway) -> None:
    """xAI answers `finish_reason: ""` when it returns tool calls. The caller's choice has
    `finish_reason == "tool_calls"` (`_fix_choice_finish_reason_for_tool_calls`). Skipped as a product bug until
    fixed, see `_XAI_FINISH_REASON_BUG`: the caller got `stop` with `native_finish_reason: ""`."""
    pytest.skip(_XAI_FINISH_REASON_BUG)
    tool_call: Final[dict[str, JsonValue]] = {
        "id": "call_dec1xai",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
    }

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(call.body, content=None, finish_reason="", extra_message={"tool_calls": [tool_call]})
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-3")
        body: Final = served(chat(gateway, model, tools=[_TOOL]))
        assert object_value(list_value(body["choices"])[0])["finish_reason"] == "tool_calls", body


@pytest.mark.parametrize(
    ("sent_params", "expected"),
    [
        ({"reasoning_effort": "high"}, {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        ({"reasoning_effort": "none"}, {"thinking": {"type": "disabled"}}),
        ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, {"thinking": {"type": "enabled"}}),
    ],
)
def test_deepseek_maps_reasoning_to_its_thinking_switch(
    gateway: Gateway, sent_params: dict[str, JsonValue], expected: dict[str, JsonValue]
) -> None:
    """DeepSeek takes `thinking` only as enabled or disabled with no budget (`deepseek_reasoning_params`).

    `reasoning_effort: "high"` becomes `thinking: {"type": "enabled"}` plus `reasoning_effort: "high"`; `"none"`
    becomes `thinking: {"type": "disabled"}` with no effort; an Anthropic-style thinking budget loses its
    `budget_tokens`. The provider body carries exactly those reasoning keys.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "deepseek/deepseek-chat")
        served(chat(gateway, model, **sent_params))
        sent: Final = provider.only().body
        assert {key: sent[key] for key in ("thinking", "reasoning_effort") if key in sent} == expected, sent


def test_deepseek_collapses_a_text_content_list_to_a_string(gateway: Gateway) -> None:
    """`deepseek-chat` has no vision, so a user message with content `[{"type": "text", "text": "<a>"}, {"type":
    "text", "text": "<b>"}]` reaches DeepSeek as one string content (`convert_content_list_to_str`).
    UNVERIFIED: the separator between parts; this pins only that it is a string containing both texts."""
    first: Final = marker()
    second: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "deepseek/deepseek-chat")
        served(
            chat(
                gateway,
                model,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": first}, {"type": "text", "text": second}]}
                ],
            )
        )
        content: Final = object_value(list_value(provider.only().body["messages"])[0])["content"]
        assert isinstance(content, str) and first in content and second in content, content


def test_deepseek_reasoning_content_in_the_answer_reaches_the_caller(gateway: Gateway) -> None:
    """DeepSeek answers a message with `reasoning_content: "thought dec1"`; the caller's message carries the same
    `reasoning_content`, which clients must replay on the next turn."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, extra_message={"reasoning_content": "thought dec1"}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "deepseek/deepseek-reasoner")
        assert first_message(served(chat(gateway, model)))["reasoning_content"] == "thought dec1"


def test_openrouter_sends_its_attribution_headers_and_asks_for_usage(gateway: Gateway) -> None:
    """An OpenRouter deployment with no `OR_SITE_URL` or `OR_APP_NAME` on the proxy.

    The provider request carries `http-referer: https://litellm.ai` and `x-title: liteLLM`, and the body has
    `usage: {"include": true}` so OpenRouter reports cost (`OpenrouterConfig.transform_request`).
    UNVERIFIED: assumes the shared proxy's environment has neither variable.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openrouter/openai/gpt-4o-mini")
        served(chat(gateway, model))
        call: Final = provider.only()
        assert (call.headers["http-referer"], call.headers["x-title"]) == ("https://litellm.ai", "liteLLM")
        assert call.body["usage"] == {"include": True}, call.body


def test_openrouter_routing_fields_reach_the_top_level(gateway: Gateway) -> None:
    """The caller sends OpenRouter's `transforms: ["middle-out"]`, `models: ["a/b", "c/d"]` and `route: "fallback"`.

    All three reach the provider body at the top level with the same values and there is no `extra_body` key.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openrouter/openai/gpt-4o-mini")
        served(chat(gateway, model, transforms=["middle-out"], models=["a/b", "c/d"], route="fallback"))
        sent: Final = provider.only().body
        assert (sent["transforms"], sent["models"], sent["route"]) == (["middle-out"], ["a/b", "c/d"], "fallback")
        assert "extra_body" not in sent


def test_openrouter_max_effort_is_sent_as_xhigh(gateway: Gateway) -> None:
    """OpenRouter has no `max` level, so `reasoning_effort: "max"` reaches it as `reasoning_effort: "xhigh"`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openrouter/anthropic/claude-sonnet-4.5")
        served(chat(gateway, model, reasoning_effort="max"))
        assert provider.only().body["reasoning_effort"] == "xhigh"


def test_openrouter_reported_cost_is_the_response_cost(gateway: Gateway) -> None:
    """OpenRouter answers `usage.cost: 0.0042` alongside the token counts.

    The caller's `x-litellm-response-cost` is 0.0042, OpenRouter's own figure, rather than the 0.025 the
    deployment prices would give. The deployment is unpriced: deployment prices win over a provider-reported cost
    (`not custom_pricing` in `litellm_logging.py:2544-2552`, `not caller_set_prices` in `cost_calculator.py:2284`).
    """

    def respond(call: Call) -> Reply:
        payload: Final = chat_json(call.body)
        return json_reply({**payload, "usage": {**object_value(payload["usage"]), "cost": 0.0042}})

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openrouter/openai/gpt-4o-mini", priced=False)
        response: Final = chat(gateway, model)
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.0042), response.headers


def test_openrouter_error_inside_a_200_answer_is_raised(gateway: Gateway) -> None:
    """OpenRouter answers HTTP 200 whose only choice has `finish_reason: "error"` and `error: {"code": 502,
    "message": "dec1 upstream died"}` (#50880).

    The caller gets HTTP 502 with `dec1 upstream died` in the message, not a 200 with empty content.
    """

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(
                call.body,
                content="",
                finish_reason="error",
                extra_choice={"error": {"code": 502, "message": "dec1 upstream died"}},
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openrouter/openai/gpt-4o-mini")
        response: Final = chat(gateway, model)
        assert response.status_code == 502, response.text
        assert "dec1 upstream died" in error_message(response)


@pytest.mark.parametrize(
    ("upstream", "effort", "expected"),
    [
        ("openai/gpt-oss-120b", "none", {"reasoning_effort": "low"}),
        ("openai/gpt-oss-120b", "xhigh", {"reasoning_effort": "high"}),
        ("moonshotai/Kimi-K3", "none", {"reasoning": {"enabled": False}}),
    ],
)
def test_together_maps_reasoning_effort_per_model_family(
    gateway: Gateway, upstream: str, effort: str, expected: dict[str, JsonValue]
) -> None:
    """Together's gpt-oss models cannot turn reasoning off and have no xhigh, so `none` becomes `low` and `xhigh`
    becomes `high`; hybrid models such as Kimi-K3 turn reasoning off with `reasoning: {"enabled": false}`
    (`_reasoning_effort_payload`). The provider body carries exactly the mapped keys."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"together_ai/{upstream}")
        served(chat(gateway, model, reasoning_effort=effort))
        sent: Final = provider.only().body
        assert {key: sent[key] for key in ("reasoning_effort", "reasoning") if key in sent} == expected, sent


def test_groq_emulates_json_schema_with_a_forced_tool_and_returns_the_json_as_content(gateway: Gateway) -> None:
    """`groq/llama-3.3-70b-versatile` has no `supports_response_schema` row, so a `json_schema` response_format is
    sent as one tool `json_tool_call` whose parameters are the schema, with `tool_choice` naming it and no
    `response_format` (`GroqChatConfig.map_openai_params`).

    The provider answers with a call to `json_tool_call` whose arguments are `{"city": "Oslo"}`, and the caller
    gets that JSON as the message `content` with no tool calls (json mode). Skipped as a product bug until fixed,
    see `_GROQ_SERVICE_TIER_BUG`.
    """
    pytest.skip(_GROQ_SERVICE_TIER_BUG)
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    def respond(call: Call) -> Reply:
        tool_call: Final[dict[str, JsonValue]] = {
            "id": "call_dec1groq",
            "type": "function",
            "function": {"name": "json_tool_call", "arguments": '{"city": "Oslo"}'},
        }
        return json_reply(
            chat_json(call.body, content=None, finish_reason="tool_calls", extra_message={"tool_calls": [tool_call]})
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "groq/llama-3.3-70b-versatile")
        body: Final = served(
            chat(
                gateway,
                model,
                response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
            )
        )
        sent: Final = provider.only().body
        assert "response_format" not in sent, sent
        assert sent["tool_choice"] == {"type": "function", "function": {"name": "json_tool_call"}}, sent
        tool: Final = object_value(object_value(list_value(sent["tools"])[0])["function"])
        assert (tool["name"], tool["parameters"]) == ("json_tool_call", schema), sent
        message: Final = first_message(body)
        assert message["content"] == '{"city": "Oslo"}' and not message.get("tool_calls"), message


def test_groq_refuses_json_schema_emulation_together_with_caller_tools(gateway: Gateway) -> None:
    """The same Groq model with a `json_schema` response_format AND the caller's own tools is answered 400 with
    `does not support native structured outputs`, because the tool workaround cannot coexist with real tools."""
    schema: Final[dict[str, JsonValue]] = {"type": "object", "properties": {}}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "groq/llama-3.3-70b-versatile")
        response: Final = chat(
            gateway,
            model,
            tools=[_TOOL],
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
        )
        assert response.status_code == 400, response.text
        assert "does not support native structured outputs" in error_message(response)
        assert provider.posts() == ()


def test_groq_streaming_reasoning_field_reaches_the_caller_as_reasoning_content(gateway: Gateway) -> None:
    """Groq streams `delta.reasoning: "think"` before its text. The caller's chunks carry it as
    `delta.reasoning_content: "think"` (`GroqChatCompletionStreamingHandler.chunk_parser`)."""

    def respond(call: Call) -> Reply:
        plain: Final = chat_stream(call.body)
        assert plain.chunks is not None
        reasoning: Final = sse(
            {
                "id": "chatcmpl-dec1groq",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": call.body.get("model"),
                "choices": [{"index": 0, "delta": {"reasoning": "think"}, "finish_reason": None}],
            }
        )
        return Reply(content_type="text/event-stream", chunks=(reasoning, *plain.chunks))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "groq/llama-3.3-70b-versatile")
        _, lines = stream_lines(
            gateway,
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker()}], "stream": True},
        )
        joined: Final = "\n".join(lines)
        assert '"reasoning_content":"think"' in joined.replace(" ", ""), joined
