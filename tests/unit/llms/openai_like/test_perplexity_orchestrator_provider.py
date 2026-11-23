import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

ORCHESTRATOR_BASE: Final = "https://api.perplexity.ai/orchestrator/v1"
HAIKU: Final = "perplexity_orchestrator/anthropic/claude-haiku-5-5"
GPT_LUNA: Final = "perplexity_orchestrator/openai/gpt-6-luna"
AUTO: Final = "perplexity_orchestrator/perplexity/auto"


def _chat_response(model: str, usage: Mapping[str, object], service_tier: str | None = None) -> dict[str, object]:
    return {
        "id": "chatcmpl_orchestrator",
        "object": "chat.completion",
        "created": 1_795_300_000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": dict(usage),
        **({"service_tier": service_tier} if service_tier is not None else {}),
    }


def _row(model: str) -> Mapping[str, float]:
    return litellm.model_cost[model]


def _messages_response(usage: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": "msg_orchestrator",
        "type": "message",
        "role": "assistant",
        "model": "anthropic/claude-haiku-5-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": dict(usage),
    }


_RESPONSES_RESPONSE: Final = {
    "id": "resp_orchestrator",
    "object": "response",
    "created_at": 1_795_300_000,
    "model": "openai/gpt-6-luna",
    "status": "completed",
    "output": [
        {
            "id": "msg_orchestrator",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
}


@pytest.mark.parametrize("key_variable", ["PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY"])
def test_provider_resolution_keeps_the_vendor_prefix_the_orchestrator_requires(
    monkeypatch: pytest.MonkeyPatch, key_variable: str
):
    monkeypatch.delenv("PERPLEXITYAI_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv(key_variable, "pplx-env-key")

    model, provider, api_key, api_base = get_llm_provider(model=HAIKU)

    assert model == "anthropic/claude-haiku-5-5"
    assert provider == "perplexity_orchestrator"
    assert api_key == "pplx-env-key"
    assert api_base == ORCHESTRATOR_BASE


@pytest.mark.asyncio
@pytest.mark.parametrize("key_variable", ["PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY"])
async def test_either_perplexity_key_variable_authenticates_chat_responses_and_messages(
    monkeypatch: pytest.MonkeyPatch, key_variable: str
):
    monkeypatch.delenv("PERPLEXITYAI_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv(key_variable, "pplx-env-key")
    with respx.mock() as upstream:
        routes: Final = (
            upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
                200, json=_chat_response("openai/gpt-6-luna", {"prompt_tokens": 4, "completion_tokens": 1})
            ),
            upstream.post(f"{ORCHESTRATOR_BASE}/responses").respond(200, json=_RESPONSES_RESPONSE),
            upstream.post(f"{ORCHESTRATOR_BASE}/messages").respond(
                200, json=_messages_response({"input_tokens": 4, "output_tokens": 1})
            ),
        )
        litellm.completion(model=GPT_LUNA, messages=[{"role": "user", "content": "hi"}])
        litellm.responses(model=GPT_LUNA, input="hi")
        await litellm.anthropic.messages.acreate(
            model=HAIKU,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()),
        )

    assert [route.calls.last.request.headers["authorization"] for route in routes] == ["Bearer pplx-env-key"] * 3


def test_with_both_perplexity_key_variables_set_the_orchestrator_uses_the_key_the_perplexity_provider_uses(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "pplx-ai-key")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-key")

    assert get_llm_provider(model=HAIKU)[2] == get_llm_provider(model="perplexity/sonar-pro")[2]


def test_chat_request_sends_tools_and_maps_cache_control_to_the_orchestrator_breakpoint():
    tools: Final = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("anthropic/claude-haiku-5-5", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        litellm.completion(
            model=HAIKU,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}],
                },
                {"role": "user", "content": "hi"},
            ],
            tools=tools,
            api_key="pplx-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer pplx-test-key"
    assert body["model"] == "anthropic/claude-haiku-5-5"
    assert body["tools"] == tools
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "rules", "prompt_cache_breakpoint": {"mode": "explicit"}}
    ]


def test_auto_routing_keeps_tools_although_it_has_no_cost_map_row():
    tools: Final = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("openai/gpt-6-luna", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        litellm.completion(
            model=AUTO,
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
            api_key="pplx-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "perplexity/auto"
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_a_param_the_orchestrator_rejects_answers_400_or_drops_under_drop_params():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("openai/gpt-6-luna", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.completion(
                model=GPT_LUNA, messages=[{"role": "user", "content": "hi"}], seed=7, api_key="pplx-test-key"
            )
        assert route.call_count == 0

        litellm.completion(
            model=GPT_LUNA,
            messages=[{"role": "user", "content": "hi"}],
            seed=7,
            temperature=0.2,
            drop_params=True,
            api_key="pplx-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert "seed" not in body
    assert body["temperature"] == 0.2


def test_chat_forwards_user_to_the_orchestrator():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("openai/gpt-6-luna", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        litellm.completion(
            model=GPT_LUNA, messages=[{"role": "user", "content": "hi"}], user="end-user-42", api_key="pplx-test-key"
        )

    assert json.loads(route.calls.last.request.content)["user"] == "end-user-42"


@pytest.mark.parametrize(
    ("param", "rejected", "accepted"),
    [
        ("n", 2, 1),
        ("frequency_penalty", 0.5, 0),
        ("presence_penalty", 0.5, 0),
        ("logprobs", True, False),
        ("store", True, False),
    ],
)
def test_a_value_the_orchestrator_rejects_answers_400_or_drops_under_drop_params_while_its_default_passes(
    param: str, rejected: object, accepted: object
):
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("openai/gpt-6-luna", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.completion(
                model=GPT_LUNA,
                messages=[{"role": "user", "content": "hi"}],
                api_key="pplx-test-key",
                **{param: rejected},
            )
        assert route.call_count == 0

        litellm.completion(
            model=GPT_LUNA,
            messages=[{"role": "user", "content": "hi"}],
            drop_params=True,
            api_key="pplx-test-key",
            **{param: rejected},
        )
        dropped: Final = json.loads(route.calls.last.request.content)

        litellm.completion(
            model=GPT_LUNA, messages=[{"role": "user", "content": "hi"}], api_key="pplx-test-key", **{param: accepted}
        )
        kept: Final = json.loads(route.calls.last.request.content)

    assert param not in dropped
    assert kept.get(param, accepted) == accepted


def test_flex_service_tier_answers_400_or_drops_on_claude_while_auto_passes_and_openai_models_keep_it():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("anthropic/claude-haiku-5-5", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        with pytest.raises(litellm.UnsupportedParamsError) as rejection:
            litellm.completion(
                model=HAIKU, messages=[{"role": "user", "content": "hi"}], service_tier="flex", api_key="pplx-test-key"
            )
        assert route.call_count == 0

        litellm.completion(
            model=HAIKU,
            messages=[{"role": "user", "content": "hi"}],
            service_tier="flex",
            drop_params=True,
            api_key="pplx-test-key",
        )
        dropped: Final = json.loads(route.calls.last.request.content)

        litellm.completion(
            model=HAIKU, messages=[{"role": "user", "content": "hi"}], service_tier="auto", api_key="pplx-test-key"
        )
        auto: Final = json.loads(route.calls.last.request.content)

        litellm.completion(
            model=GPT_LUNA, messages=[{"role": "user", "content": "hi"}], service_tier="flex", api_key="pplx-test-key"
        )
        openai_flex: Final = json.loads(route.calls.last.request.content)

    assert rejection.value.status_code == 400
    assert "service_tier" not in dropped
    assert auto["service_tier"] == "auto"
    assert openai_flex["service_tier"] == "flex"


@pytest.mark.parametrize(("param", "value"), [("n", 2), ("frequency_penalty", 0.5), ("service_tier", "flex")])
def test_allowed_openai_params_sends_a_value_the_orchestrator_would_otherwise_reject_on_chat(param: str, value: object):
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("anthropic/claude-haiku-5-5", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        litellm.completion(
            model=HAIKU,
            messages=[{"role": "user", "content": "hi"}],
            allowed_openai_params=[param],
            api_key="pplx-test-key",
            **{param: value},
        )

    assert json.loads(route.calls.last.request.content)[param] == value


def test_auto_routing_keeps_the_callers_cache_marker_on_chat_and_responses():
    marked_text: Final = {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    with respx.mock() as upstream:
        chat_route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("anthropic/claude-haiku-5-5", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        responses_route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/responses").respond(200, json=_RESPONSES_RESPONSE)
        litellm.completion(
            model=AUTO,
            messages=[{"role": "system", "content": [marked_text]}, {"role": "user", "content": "hi"}],
            api_key="pplx-test-key",
        )
        litellm.responses(
            model=AUTO,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "rules", "cache_control": {"type": "ephemeral"}}],
                }
            ],
            api_key="pplx-test-key",
        )

    breakpoint: Final = {"mode": "explicit"}
    chat_body: Final = json.loads(chat_route.calls.last.request.content)
    responses_body: Final = json.loads(responses_route.calls.last.request.content)
    assert chat_body["messages"][0]["content"][0]["prompt_cache_breakpoint"] == breakpoint
    assert responses_body["input"][0]["content"][0]["prompt_cache_breakpoint"] == breakpoint


def test_a_claude_model_the_cost_map_does_not_list_yet_keeps_the_callers_cache_marker():
    marked_text: Final = {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    with respx.mock() as upstream:
        chat_route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("anthropic/claude-sonnet-9", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        responses_route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/responses").respond(200, json=_RESPONSES_RESPONSE)
        litellm.completion(
            model="perplexity_orchestrator/anthropic/claude-sonnet-9",
            messages=[{"role": "system", "content": [marked_text]}, {"role": "user", "content": "hi"}],
            api_key="pplx-test-key",
        )
        litellm.responses(
            model="perplexity_orchestrator/anthropic/claude-sonnet-9",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "rules", "cache_control": {"type": "ephemeral"}}],
                }
            ],
            api_key="pplx-test-key",
        )

    breakpoint: Final = {"mode": "explicit"}
    chat_body: Final = json.loads(chat_route.calls.last.request.content)
    responses_body: Final = json.loads(responses_route.calls.last.request.content)
    assert chat_body["messages"][0]["content"][0]["prompt_cache_breakpoint"] == breakpoint
    assert responses_body["input"][0]["content"][0]["prompt_cache_breakpoint"] == breakpoint


@pytest.mark.parametrize("model", [HAIKU, "perplexity_orchestrator/anthropic/claude-sonnet-9"])
def test_json_object_on_a_claude_model_drops_under_drop_params_and_otherwise_reaches_the_orchestrator(model: str):
    json_object: Final = {"type": "json_object"}
    with respx.mock() as upstream:
        chat_route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("anthropic/claude-haiku-5-5", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        responses_route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/responses").respond(200, json=_RESPONSES_RESPONSE)
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            response_format=json_object,
            drop_params=True,
            api_key="pplx-test-key",
        )
        dropped_chat: Final = json.loads(chat_route.calls.last.request.content)
        litellm.responses(
            model=model,
            input="hi",
            text={"format": json_object, "verbosity": "low"},
            drop_params=True,
            api_key="pplx-test-key",
        )
        dropped_responses: Final = json.loads(responses_route.calls.last.request.content)
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            response_format=json_object,
            api_key="pplx-test-key",
        )
        forwarded_chat: Final = json.loads(chat_route.calls.last.request.content)

    assert "response_format" not in dropped_chat
    assert dropped_responses["text"] == {"verbosity": "low"}
    assert forwarded_chat["response_format"] == json_object


def test_json_object_on_a_model_that_accepts_it_stays_under_drop_params():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("openai/gpt-6-luna", {"prompt_tokens": 4, "completion_tokens": 1})
        )
        litellm.completion(
            model=GPT_LUNA,
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
            drop_params=True,
            api_key="pplx-test-key",
        )

    assert json.loads(route.calls.last.request.content)["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("param", "rejected", "accepted"),
    [("store", True, False), ("background", True, False), ("previous_response_id", "resp_1", None)],
)
def test_a_responses_field_the_orchestrator_rejects_answers_400_or_drops_under_drop_params(
    param: str, rejected: object, accepted: object
):
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/responses").respond(200, json=_RESPONSES_RESPONSE)
        with pytest.raises(litellm.UnsupportedParamsError) as rejection:
            litellm.responses(model=GPT_LUNA, input="hi", api_key="pplx-test-key", **{param: rejected})
        assert route.call_count == 0

        litellm.responses(
            model=GPT_LUNA, input="hi", temperature=0.2, drop_params=True, api_key="pplx-test-key", **{param: rejected}
        )
        dropped: Final = json.loads(route.calls.last.request.content)

        litellm.responses(
            model=GPT_LUNA,
            input="hi",
            api_key="pplx-test-key",
            **({} if accepted is None else {param: accepted}),
        )
        kept: Final = json.loads(route.calls.last.request.content)

    assert rejection.value.status_code == 400
    assert param not in dropped
    assert kept.get(param) == accepted


@pytest.mark.parametrize(("param", "value"), [("store", True), ("background", True), ("service_tier", "flex")])
def test_allowed_openai_params_sends_a_value_the_orchestrator_would_otherwise_reject_on_responses(
    param: str, value: object
):
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/responses").respond(200, json=_RESPONSES_RESPONSE)
        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.responses(model=HAIKU, input="hi", api_key="pplx-test-key", **{param: value})
        assert route.call_count == 0

        litellm.responses(
            model=HAIKU, input="hi", allowed_openai_params=[param], api_key="pplx-test-key", **{param: value}
        )

    assert json.loads(route.calls.last.request.content)[param] == value


def test_responses_request_goes_to_the_orchestrator_responses_route():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/responses").respond(200, json=_RESPONSES_RESPONSE)
        response: Final = litellm.responses(model=GPT_LUNA, input="hi", api_key="pplx-test-key")

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer pplx-test-key"
    assert json.loads(request.content)["model"] == "openai/gpt-6-luna"
    assert response.output[0].content[0].text == "ok"


@pytest.mark.asyncio
async def test_messages_request_keeps_the_one_hour_cache_ttl_and_bills_it_at_the_one_hour_rate():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/messages").respond(
            200,
            json=_messages_response(
                {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 4500,
                    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4500},
                    "cache_read_input_tokens": 0,
                    "output_tokens": 20,
                }
            ),
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=HAIKU,
            system=[{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=32,
            api_key="pplx-test-key",
            client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()),
        )

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer pplx-test-key"
    assert json.loads(request.content)["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    row: Final = _row(HAIKU)
    cost: Final = litellm.completion_cost(completion_response=response, model=HAIKU)
    assert cost == pytest.approx(
        10 * row["input_cost_per_token"]
        + 4500 * row["cache_creation_input_token_cost_above_1hr"]
        + 20 * row["output_cost_per_token"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "sent_thinking_type"),
    [
        ("perplexity_orchestrator/anthropic/claude-haiku-5-5", "adaptive"),
        ("perplexity_orchestrator/anthropic/claude-opus-4-6", "enabled"),
    ],
)
async def test_messages_thinking_enabled_reaches_an_adaptive_only_claude_model_as_adaptive_thinking(
    model: str, sent_thinking_type: str
):
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ORCHESTRATOR_BASE}/messages").respond(
            200, json=_messages_response({"input_tokens": 10, "output_tokens": 20})
        )
        await litellm.anthropic.messages.acreate(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=32000,
            thinking={"type": "enabled", "budget_tokens": 31999},
            api_key="pplx-test-key",
            client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()),
        )

    assert json.loads(route.calls.last.request.content)["thinking"]["type"] == sent_thinking_type


def test_chat_cost_above_the_long_context_threshold_uses_the_tier_rates_for_every_token_class():
    usage: Final = {
        "prompt_tokens": 150_000,
        "completion_tokens": 200,
        "total_tokens": 150_200,
        "prompt_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 30_000},
    }
    with respx.mock() as upstream:
        upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("anthropic/claude-haiku-5-5", usage)
        )
        response: Final = litellm.completion(
            model=HAIKU, messages=[{"role": "user", "content": "hi"}], api_key="pplx-test-key"
        )

    row: Final = _row(HAIKU)
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        20_000 * row["input_cost_per_token_above_100k_tokens"]
        + 100_000 * row["cache_read_input_token_cost_above_100k_tokens"]
        + 30_000 * row["cache_creation_input_token_cost_above_100k_tokens"]
        + 200 * row["output_cost_per_token_above_100k_tokens"]
    )


def test_flex_service_tier_is_billed_at_the_flex_rates():
    usage: Final = {"prompt_tokens": 2_000, "completion_tokens": 300, "total_tokens": 2_300}
    with respx.mock() as upstream:
        upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("openai/gpt-6-luna", usage, service_tier="flex")
        )
        response: Final = litellm.completion(
            model=GPT_LUNA,
            messages=[{"role": "user", "content": "hi"}],
            service_tier="flex",
            api_key="pplx-test-key",
        )

    row: Final = _row(GPT_LUNA)
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        2_000 * row["input_cost_per_token_flex"] + 300 * row["output_cost_per_token_flex"]
    )


def test_auto_routing_is_billed_at_the_rates_of_the_model_that_served_it():
    usage: Final = {"prompt_tokens": 1_000, "completion_tokens": 100, "total_tokens": 1_100}
    served: Final = "perplexity_orchestrator/openai/gpt-6.1-sol"
    with respx.mock() as upstream:
        upstream.post(f"{ORCHESTRATOR_BASE}/chat/completions").respond(
            200, json=_chat_response("openai/gpt-6.1-sol", usage)
        )
        response: Final = litellm.completion(
            model=AUTO, messages=[{"role": "user", "content": "hi"}], api_key="pplx-test-key"
        )

    row: Final = _row(served)
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        1_000 * row["input_cost_per_token"] + 100 * row["output_cost_per_token"]
    )


def test_every_orchestrator_row_is_listed_and_declares_the_endpoints_the_provider_serves():
    providers_json: Final = json.loads(
        (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
    )
    declared: Final = providers_json["perplexity_orchestrator"]["supported_endpoints"]
    rows: Final = {
        name: info
        for name, info in litellm.model_cost.items()
        if info.get("litellm_provider") == "perplexity_orchestrator"
    }

    assert HAIKU in rows
    assert set(litellm.get_valid_models(custom_llm_provider="perplexity_orchestrator")) == set(rows)
    assert all(row["supported_endpoints"] == declared for row in rows.values())
