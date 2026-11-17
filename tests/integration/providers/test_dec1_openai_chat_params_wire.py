"""Contract: `POST /v1/chat/completions` on an `openai/` deployment, from the caller's body to the provider's wire.

OpenAI is the reference dialect, so the proxy's job on this route is mostly to forward: every OpenAI parameter the
model supports reaches the provider with the same name and value, the public model name is replaced by the
upstream one, and the deployment key replaces the caller's virtual key. The interesting parts are the edges:

- a parameter the model does not support (from `OpenAIGPTConfig.get_supported_openai_params`, for example
  `reasoning_effort` on `gpt-4o-mini` or `response_format` on `gpt-4`) is refused with HTTP 400
  `litellm.UnsupportedParamsError` before the provider is called, unless the deployment sets `drop_params`
  (then it is removed) or the caller lists it in `allowed_openai_params` (then it is forwarded);
- a field that is not an OpenAI parameter at all is put in the OpenAI client's `extra_body`, which merges it into
  the top level of the provider body (`litellm/utils.py:add_provider_specific_params_to_optional_params`);
- `additional_drop_params` removes named fields;
- a streaming call always asks the provider for a usage chunk (`stream_options.include_usage: true`) because the
  proxy needs usage for spend, and strips that chunk from the caller's stream if the caller did not ask for it
  (`litellm/proxy/common_request_processing.py:stream_usage_tracking_updates`);
- a message without a role is refused with 400 before any provider call (#51124, landed 2026-11-11).

Each test registers a fresh deployment on the shared proxy pointing at an owned peer, and sends a unique prompt so
the shared Redis response cache never answers. The OpenAI SDK parameter set as one request is already pinned in
`test_openai_chat_wire.py`; here each behaviour is pinned on its own.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    PROVIDER_KEY,
    REPLY_TEXT,
    Call,
    chat,
    chat_json,
    chat_reply,
    data_events,
    deployment,
    error_message,
    error_of,
    first_message,
    json_reply,
    marker,
    peer,
    served,
    stream_lines,
)
from pydantic import JsonValue

_UPSTREAM: Final = "gpt-4o-mini"
_MODEL: Final = f"openai/{_UPSTREAM}"
_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_JSON_SCHEMA_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
            "required": ["city", "celsius"],
            "additionalProperties": False,
        },
    },
}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("temperature", 0.3),
        ("top_p", 0.9),
        ("frequency_penalty", 0.5),
        ("presence_penalty", -0.5),
        ("logit_bias", {"50256": -100}),
        ("stop", ["END", "STOP"]),
        ("max_tokens", 32),
        ("max_completion_tokens", 48),
        ("seed", 42),
        ("user", "dec1-end-user"),
        ("service_tier", "flex"),
        ("prompt_cache_key", "dec1-cache-key"),
        ("safety_identifier", "dec1-safety-id"),
        ("store", True),
        ("top_logprobs", 3),
    ],
)
def test_each_common_openai_param_reaches_the_provider_with_the_same_name_and_value(
    gateway: Gateway, name: str, value: JsonValue
) -> None:
    """A caller sends one optional OpenAI parameter on an `openai/gpt-4o-mini` deployment.

    The provider must receive exactly `{"model": "gpt-4o-mini", "messages": [...], <name>: <value>}`: no rename,
    no type change, and nothing added. `top_logprobs` alone is forwarded as sent even without `logprobs`, since
    both are supported params and validation is the provider's job. A rewrite that maps params through a
    narrower typed struct would silently lose rare ones such as `logit_bias` or `service_tier`.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, text=text, **{name: value}))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/v1/chat/completions")
        assert call.body == {"model": _UPSTREAM, "messages": [{"role": "user", "content": text}], name: value}


def test_the_provider_sees_the_deployment_key_and_never_the_virtual_key(gateway: Gateway) -> None:
    """A virtual key calls an OpenAI deployment configured with its own `api_key`.

    The provider request carries `authorization: Bearer dec1-openai-provider-key` and the virtual key appears in
    no header value sent upstream, so proxy credentials never leak to a provider.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        key: Final = scenario.key(models=[model])
        served(chat(gateway, model, key=key))
        call: Final = provider.only()
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert all(key not in value for value in call.headers.values()), dict(call.headers)


def test_n_two_returns_both_provider_choices_in_index_order(gateway: Gateway) -> None:
    """The caller sends `n: 2` and the provider answers two choices.

    The provider body carries `n: 2`, and the caller's body has exactly two choices with `index` 0 and 1, each
    with the provider's content. A port that only reads `choices[0]` would drop the second answer.
    """

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, choices=2))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model, n=2))
        assert provider.only().body["n"] == 2
        choices: Final = list_value(body["choices"])
        assert [object_value(choice)["index"] for choice in choices] == [0, 1], body
        assert [object_value(object_value(choice)["message"])["content"] for choice in choices] == [REPLY_TEXT] * 2


def test_n_one_is_forwarded_as_sent(gateway: Gateway) -> None:
    """`n: 1` is the OpenAI default but the caller sent it explicitly.

    `_check_valid_arg` exempts `n == 1` from the unsupported check, and OpenAI supports `n`, so the provider body
    still carries `n: 1`. UNVERIFIED: `pre_process_non_default_params` could treat 1 as a default and leave it
    out; see `litellm/utils.py:get_optional_params`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, n=1))
        assert provider.only().body.get("n") == 1


def test_logprobs_are_requested_and_the_provider_logprobs_come_back_unchanged(gateway: Gateway) -> None:
    """The caller asks for `logprobs: true, top_logprobs: 2` and the provider answers a choice with a logprobs
    block of one token.

    The provider body carries both params, and the caller's first choice has `logprobs.content[0]` with the same
    `token`, `logprob` and two `top_logprobs` entries. Evaluation tools read these fields directly.
    """
    logprobs: Final[dict[str, JsonValue]] = {
        "content": [
            {
                "token": "dec1",
                "logprob": -0.25,
                "bytes": [100, 101, 99, 49],
                "top_logprobs": [
                    {"token": "dec1", "logprob": -0.25, "bytes": [100, 101, 99, 49]},
                    {"token": "dec2", "logprob": -1.5, "bytes": [100, 101, 99, 50]},
                ],
            }
        ],
        "refusal": None,
    }

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, extra_choice={"logprobs": logprobs}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model, logprobs=True, top_logprobs=2))
        sent: Final = provider.only().body
        assert (sent["logprobs"], sent["top_logprobs"]) == (True, 2)
        returned: Final = object_value(object_value(list_value(body["choices"])[0])["logprobs"])
        token: Final = object_value(list_value(returned["content"])[0])
        assert (token["token"], token["logprob"]) == ("dec1", -0.25), returned
        assert [object_value(entry)["token"] for entry in list_value(token["top_logprobs"])] == ["dec1", "dec2"]


def test_seed_is_forwarded_and_system_fingerprint_comes_back(gateway: Gateway) -> None:
    """The caller sends `seed: 7` and the provider answers with `system_fingerprint: "fp_dec1"`.

    The provider body carries `seed: 7` and the caller's body has `system_fingerprint == "fp_dec1"`, which
    reproducibility tooling compares across calls.
    """

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, extra_body={"system_fingerprint": "fp_dec1"}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model, seed=7))
        assert provider.only().body["seed"] == 7
        assert body["system_fingerprint"] == "fp_dec1", body


def test_parallel_tool_calls_false_reaches_the_provider_with_the_tools(gateway: Gateway) -> None:
    """The caller sends one function tool, `tool_choice: "auto"` and `parallel_tool_calls: false`.

    The provider body has `tools == [<the tool>]`, `tool_choice == "auto"` and `parallel_tool_calls is False`.
    `False` is a meaningful value here; a rewrite that drops falsy params would turn parallel calls back on.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, tools=[_TOOL], tool_choice="auto", parallel_tool_calls=False))
        sent: Final = provider.only().body
        assert (sent["tools"], sent["tool_choice"], sent["parallel_tool_calls"]) == ([_TOOL], "auto", False)


def test_a_tool_call_answer_comes_back_with_its_id_name_and_arguments(gateway: Gateway) -> None:
    """The provider answers `finish_reason: "tool_calls"` with one call `call_dec1` to `get_weather` whose
    arguments are the JSON string `{"city": "Oslo"}`.

    The caller's message has `content: null`, one tool call with the same `id`, `type: "function"`, name and
    arguments string (not parsed), and the choice's `finish_reason` is `tool_calls`.
    """
    tool_call: Final[dict[str, JsonValue]] = {
        "id": "call_dec1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
    }

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(call.body, content=None, finish_reason="tool_calls", extra_message={"tool_calls": [tool_call]})
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model, tools=[_TOOL]))
        message: Final = first_message(body)
        assert message["content"] is None, message
        returned: Final = object_value(list_value(message["tool_calls"])[0])
        assert (returned["id"], returned["type"]) == ("call_dec1", "function"), returned
        assert object_value(returned["function"]) == {"name": "get_weather", "arguments": '{"city": "Oslo"}'}
        assert object_value(list_value(body["choices"])[0])["finish_reason"] == "tool_calls"


def test_a_strict_json_schema_response_format_is_forwarded_exactly(gateway: Gateway) -> None:
    """The caller sends `response_format` of type `json_schema` with `strict: true` and
    `additionalProperties: false`.

    The provider receives the same object byte for byte (no `strict` removed, no schema rewritten, no `$defs`
    inlined), since OpenAI enforces strict mode only when the flag arrives. Other providers rewrite this field,
    OpenAI must not.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, response_format=_JSON_SCHEMA_FORMAT))
        assert provider.only().body["response_format"] == _JSON_SCHEMA_FORMAT


def test_json_object_response_format_is_forwarded(gateway: Gateway) -> None:
    """`response_format: {"type": "json_object"}` reaches the provider unchanged and the provider's JSON string
    content comes back as a string, not parsed into an object."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, content='{"ok": true}'))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model, response_format={"type": "json_object"}))
        assert provider.only().body["response_format"] == {"type": "json_object"}
        assert first_message(body)["content"] == '{"ok": true}'


def test_response_format_on_gpt_4_is_refused_before_the_provider(gateway: Gateway) -> None:
    """`gpt-4` is listed as not supporting `response_format` (`gpt_transformation.py:297`).

    A call with `response_format: {"type": "json_object"}` on an `openai/gpt-4` deployment without
    `drop_params` is answered 400 with `UnsupportedParamsError` naming `response_format`, and the provider is
    never called.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-4")
        response: Final = chat(gateway, model, response_format={"type": "json_object"})
        assert response.status_code == 400, response.text
        message: Final = error_message(response)
        assert "UnsupportedParamsError" in message and "['response_format']" in message, message
        assert provider.posts() == ()


def test_response_format_on_gpt_4_is_dropped_when_the_deployment_drops_params(gateway: Gateway) -> None:
    """The same `gpt-4` call on a deployment with `drop_params: true` succeeds, and the provider body has no
    `response_format` key."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-4", drop_params=True)
        served(chat(gateway, model, response_format={"type": "json_object"}))
        assert "response_format" not in provider.only().body


def test_reasoning_effort_on_gpt_4o_mini_is_refused_with_the_drop_params_hint(gateway: Gateway) -> None:
    """`gpt-4o-mini` is a catalog model whose config (`OpenAIGPTConfig`) has no `reasoning_effort`.

    A call with `reasoning_effort: "high"` is answered 400 and the message reads
    `openai does not support parameters: ['reasoning_effort'], for model=gpt-4o-mini. To drop these, set
    `litellm.drop_params=True`` (`litellm/utils.py:_check_valid_arg`). The provider is never called.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(gateway, model, reasoning_effort="high")
        assert response.status_code == 400, response.text
        message: Final = error_message(response)
        assert "openai does not support parameters: ['reasoning_effort'], for model=gpt-4o-mini" in message, message
        assert "drop_params" in message
        assert error_of(response)["type"] == "invalid_request_error", response.text
        assert provider.posts() == ()


def test_reasoning_effort_on_gpt_4o_mini_is_removed_under_deployment_drop_params(gateway: Gateway) -> None:
    """With `drop_params: true` on the deployment the same call succeeds and the provider body has no
    `reasoning_effort` key; every other field is untouched."""
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, drop_params=True)
        served(chat(gateway, model, text=text, reasoning_effort="high", temperature=0.2))
        assert provider.only().body == {
            "model": _UPSTREAM,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.2,
        }


def test_drop_params_sent_by_the_caller_removes_the_unsupported_param(gateway: Gateway) -> None:
    """The caller sends `drop_params: true` in the body together with `reasoning_effort: "low"` on a deployment
    without `drop_params`.

    The call succeeds and the provider body has neither `reasoning_effort` nor `drop_params`; `drop_params` is a
    litellm control, not a provider field.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, reasoning_effort="low", drop_params=True))
        sent: Final = provider.only().body
        assert "reasoning_effort" not in sent and "drop_params" not in sent, sent


def test_allowed_openai_params_lets_an_unsupported_param_through(gateway: Gateway) -> None:
    """The caller sends `reasoning_effort: "low"` with `allowed_openai_params: ["reasoning_effort"]`.

    The supported list is extended with the allowed names, so the call is not refused and the provider body
    carries `reasoning_effort: "low"` while `allowed_openai_params` itself is not forwarded.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, reasoning_effort="low", allowed_openai_params=["reasoning_effort"]))
        sent: Final = provider.only().body
        assert sent["reasoning_effort"] == "low", sent
        assert "allowed_openai_params" not in sent, sent


def test_an_unknown_openai_model_name_forwards_reasoning_effort(gateway: Gateway) -> None:
    """A deployment `openai/dec1-private-model` names a model the catalog does not know, which usually means an
    OpenAI-compatible server or another proxy.

    `OpenAIUnknownModelConfig` adds `reasoning_effort` to the supported list, so `reasoning_effort: "medium"`
    reaches the provider instead of being refused.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/dec1-private-model")
        served(chat(gateway, model, reasoning_effort="medium"))
        sent: Final = provider.only().body
        assert (sent["model"], sent["reasoning_effort"]) == ("dec1-private-model", "medium"), sent


def test_an_unknown_top_level_field_reaches_openai_at_the_top_level(gateway: Gateway) -> None:
    """The caller adds `dec1_vendor_flag: {"mode": "x"}`, which is not an OpenAI parameter.

    For OpenAI the field goes into the client's `extra_body`, which is merged into the request body, so the
    provider sees `dec1_vendor_flag` at the top level with the same value and no `extra_body` key.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, dec1_vendor_flag={"mode": "x"}))
        sent: Final = provider.only().body
        assert sent["dec1_vendor_flag"] == {"mode": "x"}, sent
        assert "extra_body" not in sent, sent


def test_extra_body_members_are_merged_into_the_provider_body(gateway: Gateway) -> None:
    """The caller sends `extra_body: {"dec1_nested": 3}` as the OpenAI SDK does for vendor fields.

    The provider body has `dec1_nested: 3` at the top level and no `extra_body` key.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, extra_body={"dec1_nested": 3}))
        sent: Final = provider.only().body
        assert sent.get("dec1_nested") == 3 and "extra_body" not in sent, sent


def test_additional_drop_params_on_the_deployment_removes_a_named_field(gateway: Gateway) -> None:
    """The deployment sets `additional_drop_params: ["user", "dec1_vendor_flag"]`.

    A call with `user` and `dec1_vendor_flag` succeeds, and the provider body has neither key, while `seed` sent
    alongside is kept.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, additional_drop_params=["user", "dec1_vendor_flag"])
        served(chat(gateway, model, user="dec1-user", dec1_vendor_flag=True, seed=3))
        sent: Final = provider.only().body
        assert "user" not in sent and "dec1_vendor_flag" not in sent, sent
        assert sent["seed"] == 3


def test_proxy_metadata_and_tags_never_reach_the_provider(gateway: Gateway) -> None:
    """The caller sends `metadata: {"tags": ["dec1"]}` and `tags: ["dec1-tag"]`, both proxy-side controls.

    For a catalog OpenAI model `metadata` is consumed by the proxy for logging, so the provider body has no
    `metadata` and no `tags` key. UNVERIFIED: OpenAI accepts a `metadata` field with `store: true`; the proxy
    currently treats the field as its own (`litellm_pre_call_utils.py`), confirmed only for tags.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, metadata={"tags": ["dec1"]}, tags=["dec1-tag"]))
        sent: Final = provider.only().body
        assert "tags" not in sent and "metadata" not in sent, sent


def test_a_message_without_a_role_is_refused_before_the_provider(gateway: Gateway) -> None:
    """The second message has no `role` (#51124, landed 2026-11-11).

    The caller gets HTTP 400 whose message contains `Missing required parameter: 'messages[1].role'.` and the
    provider receives nothing. Before the fix the message was sent upstream as an assistant turn and the call
    returned 200, so a port must not reintroduce the silent fill.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = chat(
            gateway, model, messages=[{"role": "user", "content": marker()}, {"content": "no role here"}]
        )
        assert response.status_code == 400, response.text
        assert "Missing required parameter: 'messages[1].role'." in error_message(response), response.text
        assert provider.posts() == ()


def test_a_role_less_turn_with_tool_calls_is_still_sent_as_assistant(gateway: Gateway) -> None:
    """A message without `role` but with `tool_calls` can only be an assistant turn, so #51124 keeps filling it.

    The provider receives that message with `role: "assistant"` and the call succeeds.
    """
    tool_call: Final[dict[str, JsonValue]] = {
        "id": "call_dec1fill",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {"content": None, "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_dec1fill", "content": "sunny"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, messages=history, tools=[_TOOL]))
        sent_messages: Final = list_value(provider.only().body["messages"])
        assert object_value(sent_messages[1])["role"] == "assistant", sent_messages


def test_a_stream_without_stream_options_still_asks_the_provider_for_usage(gateway: Gateway) -> None:
    """The caller streams with no `stream_options`.

    The proxy injects `stream_options: {"include_usage": true}` upstream so it can bill the call
    (`stream_usage_tracking_updates`), and strips the usage chunk from the caller's stream: no chunk the caller
    receives has a `usage` object and every chunk has at least one choice. The text still adds up to the
    provider's reply.
    """
    with peer(lambda call: chat_reply(call.body)) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(
            gateway,
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker()}], "stream": True},
        )
        assert provider.only().body["stream_options"] == {"include_usage": True}
        events: Final = data_events(lines)
        assert all(not event.get("usage") for event in events), events
        assert all(list_value(event["choices"]) for event in events), events
        text: Final = "".join(
            str(object_value(object_value(list_value(event["choices"])[0])["delta"]).get("content") or "")
            for event in events
        )
        assert text == REPLY_TEXT
        assert lines[-1] == "data: [DONE]"


def test_include_usage_false_is_overridden_upstream_and_hidden_from_the_caller(gateway: Gateway) -> None:
    """The caller streams with `stream_options: {"include_usage": false}`.

    The provider receives `stream_options: {"include_usage": true}` (the caller's dict merged with the proxy's
    flag), while the caller still sees no usage chunk. UNVERIFIED: read from `stream_usage_tracking_updates`,
    which only skips injection when the caller already asked for `true`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker()}],
                "stream": True,
                "stream_options": {"include_usage": False},
            },
        )
        assert provider.only().body["stream_options"] == {"include_usage": True}
        assert all(not event.get("usage") for event in data_events(lines))


def test_include_usage_true_is_forwarded_once_and_the_usage_chunk_reaches_the_caller(gateway: Gateway) -> None:
    """The caller streams with `stream_options: {"include_usage": true}`.

    The provider receives exactly `{"include_usage": true}` (nothing extra), and the caller's last data chunk
    before `[DONE]` has `choices: []` and `usage` with 11 prompt and 7 completion tokens.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker()}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert provider.only().body["stream_options"] == {"include_usage": True}
        last: Final = data_events(lines)[-1]
        assert last["choices"] == [], last
        assert {name: object_value(last["usage"])[name] for name in ("prompt_tokens", "completion_tokens")} == {
            "prompt_tokens": 11,
            "completion_tokens": 7,
        }


def test_a_caller_openai_project_header_is_not_forwarded_by_default(gateway: Gateway) -> None:
    """The caller sends `OpenAI-Project: proj_dec1` and `OpenAI-Organization: org-dec1` with no forwarding
    settings on the shared proxy.

    Neither header reaches the provider: a caller must not be able to bill another OpenAI project or org through
    the proxy's key. `forward_openai_org_id` is covered in the configuration suite.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, headers={"OpenAI-Project": "proj_dec1", "OpenAI-Organization": "org-dec1"}))
        headers: Final = provider.only().headers
        assert "openai-project" not in headers and "openai-organization" not in headers, dict(headers)


def test_a_deployment_extra_headers_project_reaches_openai(gateway: Gateway) -> None:
    """The deployment sets `extra_headers: {"OpenAI-Project": "proj_dec1_deployment"}`, the only way to pin a
    project since litellm has no `project` param.

    The provider request carries `openai-project: proj_dec1_deployment`. UNVERIFIED: `extra_headers` in
    `litellm_params` is passed to the OpenAI client's `extra_headers` (`litellm/main.py`); not traced end to end.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, extra_headers={"OpenAI-Project": "proj_dec1_deployment"})
        served(chat(gateway, model))
        assert provider.only().headers["openai-project"] == "proj_dec1_deployment"


def test_the_caller_sees_the_public_model_name_and_the_provider_id(gateway: Gateway) -> None:
    """The provider answers `id: chatcmpl-dec1fixed` with `model: gpt-4o-mini`.

    The caller's body keeps the provider id unchanged and reports `model` as the public model group name
    the caller asked for, and `object` is `chat.completion`.
    """

    def respond(call: Call) -> Reply:
        payload: Final = chat_json(call.body)
        return Reply(body=json.dumps({**payload, "id": "chatcmpl-dec1fixed"}).encode())

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model))
        assert (body["id"], body["model"], body["object"]) == ("chatcmpl-dec1fixed", model, "chat.completion"), body
        assert provider.only().body["model"] == _UPSTREAM
