"""Contract: newer OpenAI chat completions features on a native `openai/` deployment.

`OpenAIGPTConfig.get_supported_openai_params` (`litellm/llms/openai/chat/gpt_transformation.py:262`) lists
`modalities`, `audio`, `prediction`, `web_search_options`, `prompt_cache_retention` and the rest for every
OpenAI model, so each reaches `<api_base>/chat/completions` under its own name. The answer side matters as much:
an audio answer (`message.audio` with `id`, `data`, `expires_at`, `transcript`) and the usage details that
predicted outputs and audio add (`accepted_prediction_tokens`, `rejected_prediction_tokens`, `audio_tokens`)
reach the caller unchanged, and audio tokens are billed at the row's audio rates.

Per-deployment identity: a deployment `organization` reaches OpenAI as the `openai-organization` header through
the OpenAI SDK client (`litellm/main.py:2823`), and two deployments of the same provider keep their own orgs.

Data residency: the regional uplift (`regional_processing_uplift_multiplier_eu`, 1.1 on the `gpt-5.4` family) is
inferred only from an `api_base` host of `eu.api.openai.com` (`litellm/llms/openai/data_residency.py:20`). A local
peer cannot answer for that host without TLS interception or DNS control, and the run must never reach the real
host, so that contract is written but skipped as a harness gap. The params already pinned in
`test_dec1_openai_chat_params_wire.py` (`prompt_cache_key`, `safety_identifier`, `store`, `service_tier`, ...)
are not repeated.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    Call,
    chat,
    chat_json,
    data_events,
    deployment,
    first_message,
    json_reply,
    marker,
    peer,
    served,
    spend_row_by_group,
    sse,
    stream_lines,
    usage,
)
from pydantic import JsonValue

_MODEL: Final = "openai/gpt-4o-mini"
_UPSTREAM: Final = "gpt-4o-mini"
_AUDIO_MODEL: Final = "openai/gpt-audio"
_AUDIO: Final[dict[str, JsonValue]] = {
    "id": "audio_dec1openai2",
    "data": "UklGRgAAAABXQVZF",
    "expires_at": 1893456000,
    "transcript": "dec1 spoken reply",
}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("prompt_cache_retention", "24h"),
        ("prediction", {"type": "content", "content": "def add(a, b):\n    return a + b\n"}),
        ("prediction", {"type": "content", "content": [{"type": "text", "text": "class Dec1: pass"}]}),
        ("modalities", ["text", "audio"]),
        ("audio", {"voice": "alloy", "format": "wav"}),
        (
            "web_search_options",
            {"search_context_size": "low", "user_location": {"type": "approximate", "approximate": {"country": "GB"}}},
        ),
    ],
    ids=[
        "prompt_cache_retention",
        "prediction_string",
        "prediction_parts",
        "modalities",
        "audio",
        "web_search_options",
    ],
)
def test_each_newer_chat_param_reaches_openai_unchanged(gateway: Gateway, name: str, value: JsonValue) -> None:
    """The caller sends one newer chat parameter on an `openai/gpt-4o-mini` deployment.

    The provider body is exactly `{"model": "gpt-4o-mini", "messages": [...], <name>: <value>}`: predicted
    output content (string or text parts), the audio voice and format, and web search options keep their shape.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, text=text, **{name: value}))
        assert provider.only().body == {
            "model": _UPSTREAM,
            "messages": [{"role": "user", "content": text}],
            name: value,
        }


def test_modalities_and_audio_travel_together_to_an_audio_model(gateway: Gateway) -> None:
    """The caller asks `openai/gpt-audio` for `modalities: ["text", "audio"]` and `audio: {"voice": "verse",
    "format": "mp3"}`.

    The provider body carries both fields together with `model: gpt-audio`; a port that only forwards
    `modalities` would get a text answer back.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _AUDIO_MODEL, priced=False)
        served(chat(gateway, model, modalities=["text", "audio"], audio={"voice": "verse", "format": "mp3"}))
        body: Final = provider.only().body
        assert (body["model"], body["modalities"], body["audio"]) == (
            "gpt-audio",
            ["text", "audio"],
            {"voice": "verse", "format": "mp3"},
        ), body


def test_an_audio_answer_reaches_the_caller_with_its_id_data_and_transcript(gateway: Gateway) -> None:
    """The provider answers with `message.content: null` and `message.audio` carrying id, base64 data,
    `expires_at` and transcript.

    The caller's `choices[0].message.audio` has the same four values and `content` stays null, so a caller can
    play the audio and reference it by id in the next turn.
    """

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, content=None, extra_message={"audio": _AUDIO}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _AUDIO_MODEL, priced=False)
        message: Final = first_message(
            served(chat(gateway, model, modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"}))
        )
        audio: Final = object_value(message["audio"])
        assert {key: audio[key] for key in _AUDIO} == _AUDIO, message
        assert message.get("content") is None, message


def test_a_previous_audio_turn_is_sent_back_by_its_audio_id(gateway: Gateway) -> None:
    """The caller replays an assistant turn as `{"role": "assistant", "audio": {"id": "audio_dec1openai2"}}`
    followed by a new user turn.

    The provider receives the assistant message with the same `audio.id` reference; dropping it would make
    OpenAI lose the spoken context of the conversation.
    """
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "audio": {"id": "audio_dec1openai2"}},
        {"role": "user", "content": marker()},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _AUDIO_MODEL, priced=False)
        served(
            chat(
                gateway,
                model,
                messages=history,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
            )
        )
        sent: Final = provider.only().body["messages"]
        assert isinstance(sent, list) and object_value(sent[1])["audio"] == {"id": "audio_dec1openai2"}, sent


def test_audio_token_usage_details_reach_the_caller(gateway: Gateway) -> None:
    """The provider reports `prompt_tokens_details.audio_tokens: 4` and `completion_tokens_details.audio_tokens: 5`
    inside usage 11 + 7.

    The caller's usage keeps both audio counts under the same keys, so clients can show audio consumption.
    """

    def respond(call: Call) -> Reply:
        details: Final[dict[str, JsonValue]] = {
            **usage(),
            "prompt_tokens_details": {"audio_tokens": 4, "cached_tokens": 0},
            "completion_tokens_details": {"audio_tokens": 5, "reasoning_tokens": 0},
        }
        return json_reply(
            chat_json(call.body, content=None, extra_message={"audio": _AUDIO}, extra_body={"usage": details})
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _AUDIO_MODEL, priced=False)
        body: Final = served(
            chat(gateway, model, modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"})
        )
        reported: Final = object_value(body["usage"])
        assert object_value(reported["prompt_tokens_details"])["audio_tokens"] == 4, reported
        assert object_value(reported["completion_tokens_details"])["audio_tokens"] == 5, reported


def test_audio_tokens_are_billed_at_the_audio_rates_of_the_row(gateway: Gateway) -> None:
    """An unpriced `openai/gpt-audio` deployment answers with usage 11 prompt tokens (4 of them audio) and 7
    completion tokens (5 of them audio).

    The SpendLogs row bills 7 text input tokens at 2.5e-06, 4 audio input tokens at 3.2e-05, 2 text output tokens
    at 1e-05 and 5 audio output tokens at 6.4e-05 (the `gpt-audio` row of the cost map): 0.0004795.
    UNVERIFIED: the split of text tokens as total minus audio follows `generic_cost_per_token` in
    `litellm/litellm_core_utils/llm_cost_calc/utils.py`.
    """

    def respond(call: Call) -> Reply:
        details: Final[dict[str, JsonValue]] = {
            **usage(),
            "prompt_tokens_details": {"audio_tokens": 4, "cached_tokens": 0},
            "completion_tokens_details": {"audio_tokens": 5, "reasoning_tokens": 0},
        }
        return json_reply(
            chat_json(call.body, content=None, extra_message={"audio": _AUDIO}, extra_body={"usage": details})
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _AUDIO_MODEL, priced=False)
        served(chat(gateway, model, modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"}))
        row: Final = spend_row_by_group(gateway, model)
        expected: Final = 7 * 2.5e-06 + 4 * 3.2e-05 + 2 * 1e-05 + 5 * 6.4e-05
        assert float(str(row["spend"])) == pytest.approx(expected)


def test_streamed_audio_deltas_reach_the_caller(gateway: Gateway) -> None:
    """The provider streams two chunks whose `delta.audio` carries `id`, a `data` piece and a `transcript` piece,
    then a finish chunk and `[DONE]`.

    The caller's chunks carry the same `delta.audio` objects in order. UNVERIFIED: the streaming handler keeps
    `delta.audio` (`litellm/litellm_core_utils/streaming_handler.py`); not traced end to end.
    """
    pieces: Final[tuple[dict[str, JsonValue], ...]] = (
        {"id": "audio_dec1stream", "data": "UklG", "transcript": "dec1 "},
        {"data": "RgAA", "transcript": "spoken"},
    )

    def respond(call: Call) -> Reply:
        base: Final[dict[str, JsonValue]] = {
            "id": "chatcmpl-dec1audio",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": call.body.get("model"),
        }
        return Reply(
            content_type="text/event-stream",
            chunks=(
                *(
                    sse({**base, "choices": [{"index": 0, "delta": {"audio": piece}, "finish_reason": None}]})
                    for piece in pieces
                ),
                sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                sse({**base, "choices": [], "usage": usage()}),
                b"data: [DONE]\n\n",
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _AUDIO_MODEL, priced=False)
        _, lines = stream_lines(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker()}],
                "stream": True,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "pcm16"},
            },
        )
        audio: Final = tuple(
            object_value(object_value(object_value(choice)["delta"])["audio"])
            for event in data_events(lines)
            for choice in (event.get("choices") or [])
            if isinstance(choice, dict) and isinstance(choice.get("delta"), dict) and "audio" in choice["delta"]
        )
        assert [piece.get("data") for piece in audio] == ["UklG", "RgAA"], audio
        assert [piece.get("transcript") for piece in audio] == ["dec1 ", "spoken"], audio


def test_predicted_output_token_counts_reach_the_caller(gateway: Gateway) -> None:
    """The caller sends a `prediction` and the provider reports `accepted_prediction_tokens: 5` and
    `rejected_prediction_tokens: 2` under `completion_tokens_details`.

    The caller's usage has both counts unchanged; they tell the caller whether the prediction paid off.
    """

    def respond(call: Call) -> Reply:
        details: Final[dict[str, JsonValue]] = {
            **usage(),
            "completion_tokens_details": {"accepted_prediction_tokens": 5, "rejected_prediction_tokens": 2},
        }
        return json_reply(chat_json(call.body, extra_body={"usage": details}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(chat(gateway, model, prediction={"type": "content", "content": "x = 1"}))
        details: Final = object_value(object_value(body["usage"])["completion_tokens_details"])
        assert (details["accepted_prediction_tokens"], details["rejected_prediction_tokens"]) == (5, 2), details


def test_a_predicted_output_call_is_billed_on_completion_tokens_including_rejected_ones(gateway: Gateway) -> None:
    """The provider reports usage 11 + 7 with `rejected_prediction_tokens: 2` (OpenAI already counts rejected
    prediction tokens inside `completion_tokens`).

    The SpendLogs row bills exactly 11 * 0.001 + 7 * 0.002 = 0.025: rejected tokens are neither added again nor
    subtracted.
    """

    def respond(call: Call) -> Reply:
        details: Final[dict[str, JsonValue]] = {
            **usage(),
            "completion_tokens_details": {"accepted_prediction_tokens": 3, "rejected_prediction_tokens": 2},
        }
        return json_reply(chat_json(call.body, extra_body={"usage": details}))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model, prediction={"type": "content", "content": "x = 1"}))
        assert float(str(spend_row_by_group(gateway, model)["spend"])) == pytest.approx(0.025)


def test_a_deployment_organization_reaches_openai_as_the_organization_header(gateway: Gateway) -> None:
    """The deployment sets `organization: org-dec1-openai2` and the caller sends a chat request.

    The provider request carries `openai-organization: org-dec1-openai2`, set by the OpenAI SDK client the proxy
    builds for that deployment.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, organization="org-dec1-openai2")
        served(chat(gateway, model))
        assert provider.only().headers["openai-organization"] == "org-dec1-openai2"


def test_two_deployments_keep_their_own_organization_and_project(gateway: Gateway) -> None:
    """Two OpenAI deployments share one peer: A sets `organization: org-dec1-a` and
    `extra_headers: {"OpenAI-Project": "proj_dec1_a"}`, B sets `organization: org-dec1-b` and `proj_dec1_b`.

    A call to A carries A's org and project, then a call to B carries B's: the OpenAI client cache is keyed per
    deployment, so one deployment's identity never leaks onto the other's requests.
    """
    with peer() as provider, gateway.scenario() as scenario:
        first: Final = deployment(
            scenario, provider, _MODEL, organization="org-dec1-a", extra_headers={"OpenAI-Project": "proj_dec1_a"}
        )
        second: Final = deployment(
            scenario, provider, _MODEL, organization="org-dec1-b", extra_headers={"OpenAI-Project": "proj_dec1_b"}
        )
        served(chat(gateway, first))
        served(chat(gateway, second))
        calls: Final = provider.posts()
        assert [(call.headers.get("openai-organization"), call.headers.get("openai-project")) for call in calls] == [
            ("org-dec1-a", "proj_dec1_a"),
            ("org-dec1-b", "proj_dec1_b"),
        ], calls


def test_a_deployment_without_organization_sends_no_organization_header(gateway: Gateway) -> None:
    """A deployment with no `organization` serves a chat request.

    The provider request has no `openai-organization` header. UNVERIFIED: the shared proxy has no
    `OPENAI_ORGANIZATION` in its environment (`litellm/main.py:2825` would otherwise fill it in).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(chat(gateway, model))
        assert "openai-organization" not in provider.only().headers


def test_an_eu_data_residency_deployment_is_billed_with_the_regional_uplift(gateway: Gateway) -> None:
    """An unpriced `openai/gpt-5.4` deployment whose `api_base` host is `eu.api.openai.com` answers usage 11 + 7.

    The SpendLogs row is the list price times `regional_processing_uplift_multiplier_eu` 1.1:
    (11 * 2.5e-06 + 7 * 1.5e-05) * 1.1 = 0.00014575, because `infer_openai_data_residency` maps that host to
    `eu`; the same call on any other host is billed 0.0001325. The body below uses the local peer as a stand-in
    and would need the peer reachable under the EU host name (for example through a resolver override) to run.
    """
    pytest.skip(
        "HARNESS GAP: data residency is inferred only from the eu.api.openai.com host; a local peer cannot answer "
        "for it without TLS interception or DNS control, and the run must never reach the real host"
    )
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.4", priced=False)
        served(chat(gateway, model))
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx((11 * 2.5e-06 + 7 * 1.5e-05) * 1.1)


def test_a_non_regional_openai_host_is_billed_at_list_price(gateway: Gateway) -> None:
    """An unpriced `openai/gpt-5.4` deployment on a local peer (not a regional OpenAI host) answers usage 11 + 7.

    The SpendLogs row is exactly the list price 11 * 2.5e-06 + 7 * 1.5e-05 = 0.0001325, with no uplift: the uplift
    must never apply on a host that is not `eu.`, `us.` or `gov.api.openai.com`. This is the half of the data
    residency contract a peer can pin.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-5.4", priced=False)
        served(chat(gateway, model))
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(11 * 2.5e-06 + 7 * 1.5e-05)
