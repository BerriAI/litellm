"""Contract: OpenAI reasoning models (o-series and the gpt-5 family) on `POST /v1/chat/completions`.

Reasoning models accept a narrower parameter set than gpt-4o, and the proxy rewrites or refuses the rest before
the provider sees them. The rules live in `litellm/llms/openai/chat/o_series_transformation.py` (o1, o3, o3-mini,
o4-mini) and `litellm/llms/openai/chat/gpt_reasoning_transformation.py` (gpt-5, gpt-5.1, gpt-5.4 and later), and
read capability flags from the model cost map rows (`supports_none_reasoning_effort`,
`supports_xhigh_reasoning_effort`, `supports_minimal_reasoning_effort`, `default_reasoning_effort`,
`supports_system_messages`, `supports_parallel_function_calling`):

- `max_tokens` is always sent as `max_completion_tokens`;
- `temperature` other than 1 is refused (400) or dropped under `drop_params`, except on a gpt-5.x model whose
  reasoning effort resolves to `none` (explicitly, or through `default_reasoning_effort: "none"` in its row);
- `logprobs`, `top_logprobs`, `top_p` follow the same `none` rule on gpt-5.1+, and are never supported on o-series;
- `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias` are unsupported on gpt-5;
- a `reasoning_effort` level the row disables (or an opt-in level like `xhigh` the row does not enable) is
  refused with `reasoning_effort=<level> is not supported for <model>`, or dropped under `drop_params`;
- a `reasoning_effort` dict (`{"effort": "high", "summary": "detailed"}`, the Responses shape) is flattened to
  the effort string;
- o-series models without `supports_system_messages` get their system messages sent as user messages.

The Rust router port will need all of these tables. Each test uses a fresh deployment on an owned peer and a
unique prompt; expected messages are copied from the product source.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    Call,
    chat,
    chat_json,
    deployment,
    json_reply,
    marker,
    peer,
    served,
    usage,
)
from pydantic import JsonValue


def _refused(response_status: int, message: str, *fragments: str) -> None:
    assert response_status == 400, message
    for fragment in fragments:
        assert fragment in message, message


@pytest.mark.parametrize("upstream", ["o3-mini", "gpt-5", "gpt-5.1"])
def test_max_tokens_is_sent_as_max_completion_tokens(gateway: Gateway, upstream: str) -> None:
    """The caller sends `max_tokens: 100` to an o-series or gpt-5 deployment.

    The provider body has `max_completion_tokens: 100` and no `max_tokens` key, because OpenAI rejects
    `max_tokens` on reasoning models. Clients written for gpt-4 still send `max_tokens`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"openai/{upstream}")
        served(chat(gateway, model, max_tokens=100))
        sent: Final = provider.only().body
        assert sent["max_completion_tokens"] == 100 and "max_tokens" not in sent, sent


def test_max_completion_tokens_is_forwarded_as_sent_on_a_reasoning_model(gateway: Gateway) -> None:
    """`max_completion_tokens: 64` on `o4-mini` reaches the provider unchanged."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o4-mini")
        served(chat(gateway, model, max_completion_tokens=64))
        assert provider.only().body["max_completion_tokens"] == 64


def test_o_series_refuses_a_temperature_other_than_one(gateway: Gateway) -> None:
    """`temperature: 0.5` on an `openai/o3-mini` deployment without `drop_params` is answered 400 with
    `O-series models don't support temperature=0.5. Only temperature=1 is supported.` and the provider is not
    called (`o_series_transformation.py:map_openai_params`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o3-mini")
        response: Final = chat(gateway, model, temperature=0.5)
        _refused(
            response.status_code,
            response.text,
            "O-series models don't support temperature=0.5. Only temperature=1 is supported.",
        )
        assert provider.posts() == ()


def test_o_series_forwards_temperature_one(gateway: Gateway) -> None:
    """`temperature: 1` is the only accepted value, so it reaches the `o3-mini` provider as `temperature: 1`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o3-mini")
        served(chat(gateway, model, temperature=1))
        assert provider.only().body["temperature"] == 1


def test_o_series_drops_temperature_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true` on the deployment, `temperature: 0.5` on `o3-mini` is removed: the call succeeds
    and the provider body has no `temperature` key."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o3-mini", drop_params=True)
        served(chat(gateway, model, temperature=0.5))
        assert "temperature" not in provider.only().body


@pytest.mark.parametrize(("name", "value"), [("logprobs", True), ("top_p", 0.5), ("presence_penalty", 0.1)])
def test_o_series_refuses_sampling_params_it_does_not_support(gateway: Gateway, name: str, value: JsonValue) -> None:
    """`o1` lists `logprobs`, `top_p`, `presence_penalty`, `frequency_penalty` and `top_logprobs` as unsupported.

    Each is answered 400 with `openai does not support parameters: ['<name>'], for model=o1` and the provider is
    not called.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o1")
        response: Final = chat(gateway, model, **{name: value})
        _refused(response.status_code, response.text, f"openai does not support parameters: ['{name}'], for model=o1")
        assert provider.posts() == ()


def test_o_series_drops_logprobs_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true`, `logprobs: true` on `o1` is removed and the call is served."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o1", drop_params=True)
        served(chat(gateway, model, logprobs=True))
        assert "logprobs" not in provider.only().body


def test_o3_refuses_parallel_tool_calls(gateway: Gateway) -> None:
    """The `o3` row has `supports_parallel_function_calling: false`, so `parallel_tool_calls: false` with a tool is
    answered 400 naming `parallel_tool_calls`. A caller that always sends the flag must set `drop_params`."""
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}},
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o3")
        response: Final = chat(gateway, model, tools=[tool], parallel_tool_calls=False)
        _refused(response.status_code, response.text, "['parallel_tool_calls']", "for model=o3")
        assert provider.posts() == ()


def test_o_series_forwards_reasoning_effort(gateway: Gateway) -> None:
    """`reasoning_effort: "high"` is an o-series-only param and reaches the `o3-mini` provider unchanged."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o3-mini")
        served(chat(gateway, model, reasoning_effort="high"))
        assert provider.only().body["reasoning_effort"] == "high"


def test_o3_mini_system_message_is_sent_as_a_user_message(gateway: Gateway) -> None:
    """The `o3-mini` row has no `supports_system_messages`, so `supports_system_messages` answers false and the
    system message is rewritten as `{"role": "user", "content": <same text>}` while its position is kept."""
    system_text: Final = f"system {marker()}"
    user_text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o3-mini")
        served(
            chat(
                gateway,
                model,
                messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
            )
        )
        assert provider.only().body["messages"] == [
            {"role": "user", "content": system_text},
            {"role": "user", "content": user_text},
        ]


def test_o1_keeps_its_system_message(gateway: Gateway) -> None:
    """The `o1` row has `supports_system_messages: true`, so a system message reaches the provider as `system`."""
    system_text: Final = f"system {marker()}"
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o1")
        served(
            chat(
                gateway,
                model,
                messages=[{"role": "system", "content": system_text}, {"role": "user", "content": marker()}],
            )
        )
        first: Final = object_value(list_value(provider.only().body["messages"])[0])
        assert first == {"role": "system", "content": system_text}


def test_o_series_keeps_the_developer_role(gateway: Gateway) -> None:
    """O-series models accept `developer` natively (`translate_developer_role_to_system_role` is a no-op there), so
    a developer message reaches `o3-mini` with `role: "developer"`."""
    text: Final = f"developer {marker()}"
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/o3-mini")
        served(
            chat(
                gateway, model, messages=[{"role": "developer", "content": text}, {"role": "user", "content": marker()}]
            )
        )
        assert object_value(list_value(provider.only().body["messages"])[0]) == {"role": "developer", "content": text}


def test_gpt_5_refuses_a_temperature_while_reasoning_is_active(gateway: Gateway) -> None:
    """`gpt-5` cannot turn reasoning off (`supports_none_reasoning_effort: false`), so `temperature: 0.2` is
    answered 400 with `gpt-5 doesn't support temperature=0.2 while reasoning is active. Only temperature=1 is
    supported` and the provider is not called."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5")
        response: Final = chat(gateway, model, temperature=0.2)
        _refused(
            response.status_code,
            response.text,
            "gpt-5 doesn't support temperature=0.2 while reasoning is active. Only temperature=1 is supported",
        )
        assert provider.posts() == ()


def test_gpt_5_drops_temperature_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true`, `temperature: 0.2` on `gpt-5` is removed and the call is served."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5", drop_params=True)
        served(chat(gateway, model, temperature=0.2))
        assert "temperature" not in provider.only().body


def test_gpt_5_1_accepts_a_temperature_when_its_default_effort_is_none(gateway: Gateway) -> None:
    """The `gpt-5.1` row declares `default_reasoning_effort: "none"`, so a request without `reasoning_effort` resolves
    to no reasoning and `temperature: 0.2` reaches the provider unchanged."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.1")
        served(chat(gateway, model, temperature=0.2))
        assert provider.only().body["temperature"] == 0.2


def test_gpt_5_1_refuses_a_temperature_with_an_explicit_high_effort(gateway: Gateway) -> None:
    """On `gpt-5.1`, `reasoning_effort: "high"` turns reasoning on, so `temperature: 0.2` alongside it is answered
    400 with `doesn't support temperature=0.2 while reasoning is active` and the provider is not called."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.1")
        response: Final = chat(gateway, model, temperature=0.2, reasoning_effort="high")
        _refused(response.status_code, response.text, "doesn't support temperature=0.2 while reasoning is active")
        assert provider.posts() == ()


def test_gpt_5_1_forwards_logprobs_when_effort_resolves_to_none(gateway: Gateway) -> None:
    """`logprobs: true, top_logprobs: 2` on `gpt-5.1` without an effort (default `none`) reach the provider."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.1")
        served(chat(gateway, model, logprobs=True, top_logprobs=2))
        sent: Final = provider.only().body
        assert (sent["logprobs"], sent["top_logprobs"]) == (True, 2), sent


def test_gpt_5_1_refuses_logprobs_with_reasoning_on(gateway: Gateway) -> None:
    """`logprobs: true` with `reasoning_effort: "medium"` on `gpt-5.1` is answered 400 with `gpt-5.1 only supports
    logprobs, top_p, top_logprobs when reasoning_effort resolves to 'none'`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.1")
        response: Final = chat(gateway, model, logprobs=True, reasoning_effort="medium")
        _refused(
            response.status_code,
            response.text,
            "gpt-5.1 only supports logprobs, top_p, top_logprobs when reasoning_effort resolves to 'none'",
        )
        assert provider.posts() == ()


def test_gpt_5_1_drops_sampling_params_with_reasoning_on_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true`, `top_p: 0.5` and `reasoning_effort: "medium"` on `gpt-5.1` are served with
    `reasoning_effort: "medium"` kept and `top_p` removed."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.1", drop_params=True)
        served(chat(gateway, model, top_p=0.5, reasoning_effort="medium"))
        sent: Final = provider.only().body
        assert sent["reasoning_effort"] == "medium" and "top_p" not in sent, sent


@pytest.mark.parametrize("name", ["stop", "presence_penalty", "frequency_penalty", "logit_bias"])
def test_gpt_5_refuses_params_the_family_does_not_take(gateway: Gateway, name: str) -> None:
    """`gpt-5` lists `stop`, `presence_penalty`, `frequency_penalty` and `logit_bias` as unsupported; each is answered
    400 with `openai does not support parameters: ['<name>'], for model=gpt-5`."""
    values: Final[dict[str, JsonValue]] = {
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "logit_bias": {"1": 1},
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5")
        response: Final = chat(gateway, model, **{name: values[name]})
        _refused(
            response.status_code, response.text, f"openai does not support parameters: ['{name}'], for model=gpt-5"
        )
        assert provider.posts() == ()


def test_gpt_5_forwards_minimal_effort_and_verbosity(gateway: Gateway) -> None:
    """The `gpt-5` row enables `minimal`; `reasoning_effort: "minimal"` and `verbosity: "low"` (a gpt-5-only param)
    reach the provider unchanged."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5")
        served(chat(gateway, model, reasoning_effort="minimal", verbosity="low"))
        sent: Final = provider.only().body
        assert (sent["reasoning_effort"], sent["verbosity"]) == ("minimal", "low"), sent


@pytest.mark.parametrize("level", ["none", "xhigh"])
def test_gpt_5_refuses_an_effort_level_its_row_does_not_enable(gateway: Gateway, level: str) -> None:
    """The `gpt-5` row sets `supports_none_reasoning_effort: false` and `supports_xhigh_reasoning_effort: false`.

    `reasoning_effort: "<level>"` is answered 400 with `reasoning_effort=<level> is not supported for gpt-5. To drop
    unsupported params set `litellm.drop_params = True`` (`unsupported_reasoning_effort_message`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5")
        response: Final = chat(gateway, model, reasoning_effort=level)
        _refused(response.status_code, response.text, f"reasoning_effort={level} is not supported for gpt-5")
        assert provider.posts() == ()


def test_gpt_5_drops_a_refused_effort_level_under_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true`, `reasoning_effort: "xhigh"` on `gpt-5` is removed: the call is served and the
    provider body has no `reasoning_effort` key, so the provider applies its own default."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5", drop_params=True)
        served(chat(gateway, model, reasoning_effort="xhigh"))
        assert "reasoning_effort" not in provider.only().body


def test_gpt_5_4_forwards_xhigh_because_its_row_enables_it(gateway: Gateway) -> None:
    """The `gpt-5.4` row sets `supports_xhigh_reasoning_effort: true`, so `reasoning_effort: "xhigh"` reaches the
    provider unchanged. A port with a hardcoded level table would refuse it."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.4")
        served(chat(gateway, model, reasoning_effort="xhigh"))
        assert provider.only().body["reasoning_effort"] == "xhigh"


def test_a_declared_effort_level_in_model_info_is_forwarded(gateway: Gateway) -> None:
    """A `gpt-5` deployment declares `model_info.reasoning_effort_levels: ["none", "low"]`.

    `reasoning_effort: "none"`, refused by the cost map row, is forwarded because the deployment declares it
    (`declared_reasoning_efforts` bypasses the gate). UNVERIFIED: the key name `reasoning_effort_levels` comes from
    `refused_reasoning_effort_warning`; the wiring from `model_info` to `declared_reasoning_efforts` was not traced.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, "openai/gpt-5", model_info={"reasoning_effort_levels": ["none", "low"]}
        )
        served(chat(gateway, model, reasoning_effort="none"))
        assert provider.only().body["reasoning_effort"] == "none"


def test_a_responses_shaped_effort_dict_is_flattened_to_its_effort(gateway: Gateway) -> None:
    """The caller sends `reasoning_effort: {"effort": "high", "summary": "detailed"}` (the Responses API shape) to
    `gpt-5`. Chat Completions takes only a string, so the provider receives `reasoning_effort: "high"`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5")
        served(chat(gateway, model, reasoning_effort={"effort": "high", "summary": "detailed"}))
        assert provider.only().body["reasoning_effort"] == "high"


def test_reasoning_tokens_in_usage_are_relayed_to_the_caller(gateway: Gateway) -> None:
    """The `gpt-5` provider answers usage with `completion_tokens_details.reasoning_tokens: 5`.

    The caller's `usage.completion_tokens_details.reasoning_tokens` is 5, and `completion_tokens` stays the
    provider's 7 (reasoning is part of completion tokens on OpenAI, not added on top).
    """

    def respond(call: Call) -> Reply:
        payload: Final = chat_json(call.body)
        return json_reply({**payload, "usage": {**usage(), "completion_tokens_details": {"reasoning_tokens": 5}}})

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5")
        body: Final = served(chat(gateway, model))
        reported: Final = object_value(body["usage"])
        assert reported["completion_tokens"] == 7, reported
        assert object_value(reported["completion_tokens_details"])["reasoning_tokens"] == 5, reported
