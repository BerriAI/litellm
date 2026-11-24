import json
from collections.abc import Callable, Mapping
from typing import Final

import httpx
import pytest

import litellm
import litellm.completion_extras
from litellm.completion_extras.litellm_interactions_transformation import InteractionsToCompletionBridgeHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.interactions.transformation import VertexAIInteractionsConfig
from litellm.types.utils import Choices, ModelResponse, StreamingChoices

MESSAGES: Final = ({"role": "user", "content": "Write a long story"},)
MODEL: Final = "gemini/gemini-4-neon"
API_BASE: Final = "https://generativelanguage.googleapis.com"


def _interaction(
    text: str, status: str, token: str | None = None, interaction_id: str = "v1_first"
) -> Mapping[str, object]:
    return {
        "id": interaction_id,
        "status": status,
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_by_modality": [{"modality": "text", "tokens": 5}],
        },
        **({"continuation_token": token} if token is not None else {}),
    }


def _sse(interaction: Mapping[str, object]) -> httpx.Response:
    steps: Final = interaction["steps"]
    assert isinstance(steps, list)
    text: Final = steps[0]["content"][0]["text"]
    events: Final = (
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": text}},
        {
            "event_type": "interaction.completed",
            "interaction": {key: value for key, value in interaction.items() if key != "steps"},
        },
    )
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def _scripted(
    replies: tuple[httpx.Response, ...],
) -> tuple[list[Mapping[str, object]], Callable[[httpx.Request], httpx.Response]]:
    bodies: Final[list[Mapping[str, object]]] = []  # mutable-ok: transport spy
    remaining: Final = list(replies)  # mutable-ok: scripted replies are consumed in order

    def responder(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return remaining.pop(0)

    return bodies, responder


def _message_of(response: object) -> tuple[Choices, str | None]:
    assert isinstance(response, ModelResponse)
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices)
    return choice, (choice.message.provider_specific_fields or {}).get("continuation_token")


def test_unary_bridge_resends_with_the_continuation_token_until_the_answer_completes() -> None:
    bodies, responder = _scripted(
        (
            httpx.Response(200, json=_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(200, json=_interaction("upon ", "requires_continuation", "tok-2", "v1_second")),
            httpx.Response(200, json=_interaction("a time.", "completed", interaction_id="v1_third")),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model=MODEL,
            messages=list(MESSAGES),
            max_tokens=1000,
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )

    choice, token = _message_of(response)
    assert choice.message.content == "Once upon a time."
    assert choice.finish_reason == "stop"
    assert token is None
    assert [(body.get("continuation_token"), body.get("previous_interaction_id")) for body in bodies] == [
        (None, None),
        ("tok-1", "v1_first"),
        ("tok-2", "v1_second"),
    ]
    assert bodies[1]["input"] == bodies[0]["input"]
    usage: Final = response.usage  # pyright: ignore[reportAttributeAccessIssue]  # ModelResponse sets usage dynamically
    assert (usage.prompt_tokens, usage.completion_tokens) == (30, 15)
    assert usage.completion_tokens_details.text_tokens == 15


@pytest.mark.asyncio
async def test_async_unary_bridge_returns_the_partial_answer_and_token_when_a_hop_fails() -> None:
    bodies, responder = _scripted(
        (
            httpx.Response(200, json=_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(503, json={"error": {"message": "overloaded"}}),
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as raw_client:
        handler: Final = AsyncHTTPHandler()
        handler.client = raw_client
        response: Final = await litellm.acompletion(
            model=MODEL, messages=list(MESSAGES), max_tokens=1000, api_key="k", api_base=API_BASE, client=handler
        )

    choice, token = _message_of(response)
    assert choice.message.content == "Once "
    assert choice.finish_reason == "length"
    assert token == "tok-1"
    assert len(bodies) == 2


def test_answer_cut_off_by_the_output_cap_is_not_resent_and_hands_the_token_to_the_caller() -> None:
    bodies, responder = _scripted((httpx.Response(200, json=_interaction("Once ", "incomplete", "tok-cap")),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        client: Final = HTTPHandler(client=raw_client)
        response: Final = litellm.completion(
            model=MODEL, messages=list(MESSAGES), max_tokens=5, api_key="k", api_base=API_BASE, client=client
        )

    choice, token = _message_of(response)
    assert choice.finish_reason == "length"
    assert token == "tok-cap"
    assert len(bodies) == 1


def test_caller_continuation_token_is_forwarded_as_an_unstored_resume_since_chat_cannot_name_the_interaction() -> None:
    bodies, responder = _scripted((httpx.Response(200, json=_interaction("a time.", "completed")),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        _ = litellm.completion(
            model=MODEL,
            max_tokens=1000,
            messages=list(MESSAGES),
            continuation_token="tok-cap",
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )

    assert (bodies[0]["continuation_token"], bodies[0]["store"]) == ("tok-cap", False)
    assert "previous_interaction_id" not in bodies[0]


def _drain(stream: object) -> tuple[str, list[str], list[str], tuple[int, int] | None]:
    chunks: Final = list(stream)  # pyright: ignore[reportArgumentType,reportUnknownArgumentType,reportUnknownVariableType]  # CustomStreamWrapper is iterable
    choices: Final = [chunk.choices[0] for chunk in chunks if chunk.choices]
    assert all(isinstance(choice, StreamingChoices) for choice in choices)
    usages: Final = [chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None]
    return (
        "".join(choice.delta.content or "" for choice in choices),
        [choice.finish_reason for choice in choices if choice.finish_reason],
        [
            token
            for choice in choices
            if (token := (choice.delta.provider_specific_fields or {}).get("continuation_token"))
        ],
        (usages[-1].prompt_tokens, usages[-1].completion_tokens) if usages else None,
    )


def test_streaming_bridge_continues_into_the_next_hop_as_one_stream() -> None:
    bodies, responder = _scripted(
        (
            _sse(_interaction("Once ", "requires_continuation", "tok-1")),
            _sse(_interaction("upon a time.", "completed", interaction_id="v1_second")),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        stream: Final = litellm.completion(
            model=MODEL,
            max_tokens=1000,
            messages=list(MESSAGES),
            stream=True,
            stream_options={"include_usage": True},
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )
        text, finish_reasons, tokens, usage = _drain(stream)

    assert text == "Once upon a time."
    assert finish_reasons == ["stop"]
    assert tokens == []
    assert usage == (20, 10)
    assert (bodies[1]["continuation_token"], bodies[1].get("previous_interaction_id")) == ("tok-1", "v1_first")
    assert bodies[1]["input"] == bodies[0]["input"]


def test_streaming_bridge_ends_with_length_and_the_token_when_a_hop_fails() -> None:
    _, responder = _scripted(
        (
            _sse(_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(503, json={"error": {"message": "overloaded"}}),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        stream: Final = litellm.completion(
            model=MODEL,
            max_tokens=1000,
            messages=list(MESSAGES),
            stream=True,
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )
        text, finish_reasons, tokens, _ = _drain(stream)

    assert text == "Once "
    assert finish_reasons == ["length"]
    assert tokens == ["tok-1"]


def test_streaming_bridge_ends_with_length_and_the_token_when_a_hop_stream_reports_a_provider_error() -> None:
    quota_event: Final = {"event_type": "error", "error": {"code": "resource_exhausted", "message": "quota"}}
    _, responder = _scripted(
        (
            _sse(_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(
                200, text=f"data: {json.dumps(quota_event)}\n\n", headers={"content-type": "text/event-stream"}
            ),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        stream: Final = litellm.completion(
            model=MODEL,
            max_tokens=1000,
            messages=list(MESSAGES),
            stream=True,
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )
        text, finish_reasons, tokens, _ = _drain(stream)

    assert text == "Once "
    assert finish_reasons == ["length"]
    assert tokens == ["tok-1"]


@pytest.mark.parametrize(
    ("ending", "raised"),
    (
        (({"event_type": "error", "error": {"code": "unavailable", "message": "dropped"}},), "dropped"),
        ((), "ended before the interaction completed"),
    ),
    ids=("error-event", "no-completion"),
)
def test_streaming_bridge_raises_when_a_hop_fails_after_it_delivered_text_so_the_cut_is_not_passed_off_as_a_finish(
    ending: tuple[dict[str, object], ...], raised: str
) -> None:
    events: Final = (
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "upon "}},
        *ending,
    )
    _, responder = _scripted(
        (
            _sse(_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(
                200,
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
                headers={"content-type": "text/event-stream"},
            ),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        stream: Final = litellm.completion(
            model=MODEL,
            max_tokens=1000,
            messages=list(MESSAGES),
            stream=True,
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )
        seen: Final = [next(stream), next(stream)]
        with pytest.raises(litellm.exceptions.MidStreamFallbackError, match=raised):
            _ = next(stream)

    assert "".join(chunk.choices[0].delta.content or "" for chunk in seen) == "Once upon "
    assert [chunk.choices[0].finish_reason for chunk in seen] == [None, None]
    first_hop_usage: Final = _interaction("Once ", "requires_continuation", "tok-1")["usage"]
    billed: Final = stream.partial_response().usage
    assert (billed.prompt_tokens, billed.completion_tokens) == (
        first_hop_usage["total_input_tokens"],
        first_hop_usage["total_output_tokens"],
    )


@pytest.mark.parametrize(
    ("store", "resends_input", "previous_interaction_id"),
    ((True, False, "v1_first"), (None, False, "v1_first"), (False, True, None)),
    ids=("stored", "store left to Google's default", "unstored"),
)
def test_vertex_resume_sends_the_input_again_unless_a_stored_interaction_holds_it(
    store: bool | None,
    resends_input: bool,
    previous_interaction_id: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        litellm.completion_extras,
        "interactions_api_bridge",
        InteractionsToCompletionBridgeHandler(
            provider_config=VertexAIInteractionsConfig(
                mint_access_token=lambda _creds, project: ("unit-token", project or "")
            )
        ),
    )
    bodies, responder = _scripted(
        (
            httpx.Response(200, json=_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(200, json=_interaction("upon a time.", "completed", interaction_id="v1_second")),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model="vertex_ai/gemini-4-neon",
            max_tokens=1000,
            messages=list(MESSAGES),
            vertex_project="unit-project",
            vertex_location="global",
            store=store,
            client=HTTPHandler(client=raw_client),
        )

    choice, _ = _message_of(response)
    assert choice.message.content == "Once upon a time."
    assert "input" in bodies[0]
    assert ("input" in bodies[1]) is resends_input
    assert (bodies[1]["continuation_token"], bodies[1].get("previous_interaction_id")) == (
        "tok-1",
        previous_interaction_id,
    )


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    (
        ("failed", {"code": "resource_exhausted", "message": "quota"}, litellm.RateLimitError),
        ("failed", {"code": "invalid_argument", "message": "bad input"}, litellm.BadRequestError),
        ("cancelled", None, litellm.APIConnectionError),
    ),
)
def test_unary_interaction_that_failed_or_was_cancelled_raises_instead_of_answering(
    status: str, error: Mapping[str, object] | None, expected: type[Exception]
) -> None:
    failed: Final = {**_interaction("partial", status), **({"error": error} if error is not None else {})}
    _, responder = _scripted((httpx.Response(200, json=failed),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client, pytest.raises(expected):
        litellm.completion(
            model=MODEL,
            messages=list(MESSAGES),
            api_key="k",
            api_base=API_BASE,
            num_retries=0,
            client=HTTPHandler(client=raw_client),
        )


def test_continuation_hop_that_fails_returns_the_answer_so_far_with_its_token() -> None:
    _, responder = _scripted(
        (
            httpx.Response(200, json=_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(200, json=_interaction("", "failed", interaction_id="v1_second")),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        choice, token = _message_of(
            litellm.completion(
                model=MODEL,
                max_tokens=1000,
                messages=list(MESSAGES),
                api_key="k",
                api_base=API_BASE,
                client=HTTPHandler(client=raw_client),
            )
        )

    assert (choice.message.content, choice.finish_reason, token) == ("Once ", "length", "tok-1")


def test_streamed_images_and_audio_reach_the_caller() -> None:
    events: Final = (
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "model_output", "content": [{"type": "image", "mime_type": "image/png", "data": "QQ=="}]},
        },
        {"event_type": "step.delta", "index": 0, "delta": {"type": "image", "mime_type": "image/png", "data": "Qg=="}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "audio", "mime_type": "audio/wav", "data": "Qw=="}},
        {"event_type": "interaction.completed", "interaction": {"id": "v1_first", "status": "completed"}},
    )
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    _, responder = _scripted((httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        chunks: Final = list(
            litellm.completion(
                model=MODEL,
                messages=list(MESSAGES),
                stream=True,
                api_key="k",
                api_base=API_BASE,
                client=HTTPHandler(client=raw_client),
            )
        )

    deltas: Final = [chunk.choices[0].delta for chunk in chunks if chunk.choices]
    images: Final = [image for delta in deltas for image in (getattr(delta, "images", None) or [])]
    audio: Final = [delta.audio["data"] for delta in deltas if getattr(delta, "audio", None)]
    assert [(image["index"], image["image_url"]["url"]) for image in images] == [
        (0, "data:image/png;base64,QQ=="),
        (1, "data:image/png;base64,Qg=="),
    ]
    assert audio == ["Qw=="]


@pytest.mark.parametrize(
    ("response_format", "capped"),
    (
        (None, True),
        ({"type": "text", "mime_type": "application/json"}, True),
        ({"type": "audio"}, False),
        ([{"type": "text"}, {"type": "image"}], False),
    ),
    ids=("text", "json", "audio", "text-and-image"),
)
def test_default_output_budget_only_caps_text_answers(
    response_format: object, capped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.completion_extras.litellm_interactions_transformation.continuation import (
        DEFAULT_OUTPUT_BUDGET_ENV,
        with_default_output_budget,
    )

    monkeypatch.setenv(DEFAULT_OUTPUT_BUDGET_ENV, "100")
    body: Final = {"input": "hi", **({"response_format": response_format} if response_format is not None else {})}

    budgeted: Final = with_default_output_budget(body, lambda: 4096)

    assert budgeted.get("generation_config") == ({"max_output_tokens": 100} if capped else None)


def _cited(text: str, url: str, status: str, token: str | None = None) -> Mapping[str, object]:
    cites: Final = [{"type": "url_citation", "start_index": 0, "end_index": len(text), "url": url, "title": url}]
    return {
        **_interaction(text, status, token),
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text, "annotations": cites}]}],
    }


def _cited_sse(text: str, url: str, status: str, token: str | None = None) -> httpx.Response:
    interaction: Final = _cited(text, url, status, token)
    steps: Final = interaction["steps"]
    assert isinstance(steps, list)
    events: Final = (
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": text}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text_annotation_delta", "annotations": steps[0]["content"][0]["annotations"]},
        },
        {
            "event_type": "interaction.completed",
            "interaction": {key: value for key, value in interaction.items() if key != "steps"},
        },
    )
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize("stream", (False, True), ids=("unary", "stream"))
def test_citations_of_a_continued_answer_point_into_the_joined_text(stream: bool) -> None:
    hops: Final = (
        ("Once ", "https://a.example", "requires_continuation", "tok-1"),
        ("upon.", "https://b.example", "completed", None),
    )
    _, responder = _scripted(
        tuple(_cited_sse(*hop) if stream else httpx.Response(200, json=_cited(*hop)) for hop in hops)
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        answer: Final = litellm.completion(
            model=MODEL,
            max_tokens=1000,
            messages=list(MESSAGES),
            stream=stream,
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )
        carriers: Final = (
            [chunk.choices[0].delta for chunk in answer] if stream else [answer.choices[0].message]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # completion returns an untyped union
        )

    text: Final = "".join(carrier.content or "" for carrier in carriers)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]  # untyped union above
    spans: Final = [
        (text[cite["url_citation"]["start_index"] : cite["url_citation"]["end_index"]], cite["url_citation"]["url"])
        for carrier in carriers  # pyright: ignore[reportUnknownVariableType]  # untyped union above
        for cite in getattr(carrier, "annotations", None) or ()
    ]
    assert spans == [("Once ", "https://a.example"), ("upon.", "https://b.example")]


def test_streamed_citations_sent_in_a_later_step_with_no_text_point_at_the_answer_before_them() -> None:
    cites: Final = [
        {"type": "url_citation", "start_index": 0, "end_index": 5, "url": "https://b.example", "title": "b"}
    ]
    events: Final = (
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "upon."}},
        {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": 1, "delta": {"type": "text_annotation_delta", "annotations": cites}},
        {"event_type": "interaction.completed", "interaction": {"id": "v1_second", "status": "completed"}},
    )
    _, responder = _scripted(
        (
            _sse(_interaction("Once ", "requires_continuation", "tok-1")),
            httpx.Response(
                200,
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            ),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        deltas: Final = [
            chunk.choices[0].delta
            for chunk in litellm.completion(
                model=MODEL,
                max_tokens=1000,
                messages=list(MESSAGES),
                stream=True,
                api_key="k",
                api_base=API_BASE,
                client=HTTPHandler(client=raw_client),
            )
        ]

    spans: Final = [
        (cite["url_citation"]["start_index"], cite["url_citation"]["end_index"])
        for delta in deltas
        for cite in getattr(delta, "annotations", None) or ()
    ]
    assert spans == [(5, 10)]


@pytest.mark.parametrize(
    "cited_events",
    [
        pytest.param(
            (
                {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
                {
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {
                        "type": "text",
                        "text": "é Paris",
                        "annotations": [{"type": "url_citation", "start_index": 3, "end_index": 8, "url": "https://p"}],
                    },
                },
            ),
            id="on the text delta",
        ),
        pytest.param(
            (
                {
                    "event_type": "step.start",
                    "index": 0,
                    "step": {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "text",
                                "text": "é Paris",
                                "annotations": [
                                    {"type": "url_citation", "start_index": 3, "end_index": 8, "url": "https://p"}
                                ],
                            }
                        ],
                    },
                },
            ),
            id="on the step start",
        ),
        pytest.param(
            (
                {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
                {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "é Paris"}},
                {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}},
                {
                    "event_type": "step.delta",
                    "index": 1,
                    "delta": {
                        "type": "text_annotation_delta",
                        "annotations": [{"type": "url_citation", "start_index": 3, "end_index": 8, "url": "https://p"}],
                    },
                },
            ),
            id="in a later step",
        ),
    ],
)
def test_streamed_citations_given_as_byte_spans_select_the_cited_words_after_non_ascii_text(
    cited_events: tuple[dict[str, object], ...],
) -> None:
    events: Final = (
        *cited_events,
        {"event_type": "interaction.completed", "interaction": {"id": "v1_second", "status": "completed"}},
    )
    _, responder = _scripted(
        (
            _sse(_interaction("né ", "requires_continuation", "tok-1")),
            httpx.Response(
                200,
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            ),
        )
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        deltas: Final = [
            chunk.choices[0].delta
            for chunk in litellm.completion(
                model=MODEL,
                max_tokens=1000,
                messages=list(MESSAGES),
                stream=True,
                api_key="k",
                api_base=API_BASE,
                client=HTTPHandler(client=raw_client),
            )
        ]

    answer: Final = "".join(delta.content or "" for delta in deltas)
    cited_words: Final = [
        answer[cite["url_citation"]["start_index"] : cite["url_citation"]["end_index"]]
        for delta in deltas
        for cite in getattr(delta, "annotations", None) or ()
    ]
    assert cited_words == ["Paris"]


def test_unary_citations_given_as_byte_spans_select_the_cited_words_after_non_ascii_text() -> None:
    cited: Final = {
        "id": "v1_first",
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "né "},
                    {
                        "type": "text",
                        "text": "é Paris",
                        "annotations": [{"type": "url_citation", "start_index": 3, "end_index": 8, "url": "https://p"}],
                    },
                ],
            }
        ],
    }
    _, responder = _scripted((httpx.Response(200, json=cited),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        message: Final = (
            litellm.completion(
                model=MODEL,
                max_tokens=1000,
                messages=list(MESSAGES),
                api_key="k",
                api_base=API_BASE,
                client=HTTPHandler(client=raw_client),
            )
            .choices[0]
            .message
        )

    assert [
        message.content[cite["url_citation"]["start_index"] : cite["url_citation"]["end_index"]]
        for cite in message.annotations
    ] == ["Paris"]


_GROUNDED_CALLS: Final = (
    {"type": "google_search_call", "id": "s1", "arguments": {"queries": ["capital of France"]}},
    {"type": "google_search_result", "call_id": "s1", "result": [{"search_suggestions": "<div>chips</div>"}]},
    {
        "type": "url_context_result",
        "call_id": "u1",
        "result": [
            {"url": "https://a.example", "status": "success"},
            {"url": "https://b.example", "status": "error"},
        ],
    },
)
_GROUNDED_CITES: Final = (
    {"type": "url_citation", "start_index": 17, "end_index": 22, "url": "https://a.example", "title": "a"},
    {"type": "url_citation", "start_index": 17, "end_index": 22, "url": "https://b.example", "title": "b"},
)
_GROUNDED_TEXT: Final = "Café capital is Paris."
_GROUNDED_USAGE: Final = {
    "total_input_tokens": 5,
    "total_output_tokens": 4,
    "total_tokens": 9,
    "grounding_tool_count": [{"type": "google_search", "count": 2}, {"type": "google_maps", "count": 3}],
}
_EXPECTED_GROUNDING: Final = [
    {
        "searchEntryPoint": {"renderedContent": "<div>chips</div>"},
        "groundingChunks": [
            {"web": {"uri": "https://a.example", "title": "a"}},
            {"web": {"uri": "https://b.example", "title": "b"}},
        ],
        "groundingSupports": [
            {"segment": {"startIndex": 17, "endIndex": 22, "text": "Paris"}, "groundingChunkIndices": [0, 1]}
        ],
        "webSearchQueries": ["capital of France"],
    }
]
_EXPECTED_URL_CONTEXT: Final = [
    {
        "urlMetadata": [
            {"retrievedUrl": "https://a.example", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"},
            {"retrievedUrl": "https://b.example", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_ERROR"},
        ]
    }
]


def _grounded_sse(*, whole_steps_on_start: bool = False) -> httpx.Response:
    events: Final = (
        *(
            event
            for position, call in enumerate(_GROUNDED_CALLS)
            for event in (
                *(
                    (
                        {
                            "event_type": "step.start",
                            "index": position,
                            "step": {"type": call["type"], "arguments": {}},
                        },
                        {"event_type": "step.start", "index": position, "step": call},
                    )
                    if whole_steps_on_start
                    else ()
                ),
                {
                    "event_type": "step.delta",
                    "index": position,
                    "delta": {key: value for key, value in call.items() if key not in ("id", "call_id")}
                    if whole_steps_on_start
                    else call,
                },
            )
        ),
        {"event_type": "step.start", "index": 3, "step": {"type": "model_output"}},
        {"event_type": "step.delta", "index": 3, "delta": {"type": "text", "text": _GROUNDED_TEXT[:8]}},
        {"event_type": "step.delta", "index": 3, "delta": {"type": "text", "text": _GROUNDED_TEXT[8:]}},
        {
            "event_type": "step.delta",
            "index": 3,
            "delta": {"type": "text_annotation_delta", "annotations": list(_GROUNDED_CITES)},
        },
        {
            "event_type": "interaction.completed",
            "interaction": {"id": "v1_grounded", "status": "completed", "usage": _GROUNDED_USAGE},
        },
    )
    return httpx.Response(
        200,
        text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def test_unary_grounded_answer_carries_the_search_and_url_context_fields_generate_content_callers_read() -> None:
    grounded: Final = {
        "id": "v1_grounded",
        "status": "completed",
        "steps": [
            *_GROUNDED_CALLS,
            {
                "type": "model_output",
                "content": [{"type": "text", "text": _GROUNDED_TEXT, "annotations": list(_GROUNDED_CITES)}],
            },
        ],
        "usage": _GROUNDED_USAGE,
    }
    _, responder = _scripted((httpx.Response(200, json=grounded),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model=MODEL,
            messages=list(MESSAGES),
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )

    assert isinstance(response, ModelResponse)
    assert response.vertex_ai_grounding_metadata == _EXPECTED_GROUNDING
    assert response.vertex_ai_url_context_metadata == _EXPECTED_URL_CONTEXT
    assert response.vertex_ai_safety_results == []
    assert response.vertex_ai_citation_metadata == []
    details: Final = response.usage.prompt_tokens_details
    assert (details.web_search_requests, details.google_maps_grounding_requests) == (2, 3)


_RECITED: Final = {"type": "url_citation", "start_index": 0, "end_index": 5, "url": ""}
_RECITED_AS_READ: Final = {"startIndex": 0, "endIndex": 5, "uri": ""}


@pytest.mark.parametrize(
    ("tool_steps", "cites", "expected_recited", "expected_grounding"),
    [
        pytest.param(
            (),
            (_RECITED, *_GROUNDED_CITES),
            [
                _RECITED_AS_READ,
                {"startIndex": 17, "endIndex": 22, "uri": "https://a.example", "title": "a"},
                {"startIndex": 17, "endIndex": 22, "uri": "https://b.example", "title": "b"},
            ],
            [],
            id="no_search_ran_so_every_cited_span_is_a_recitation",
        ),
        pytest.param(
            _GROUNDED_CALLS,
            ({**_RECITED, "end_index": 3}, _RECITED, *_GROUNDED_CITES),
            [_RECITED_AS_READ],
            _EXPECTED_GROUNDING,
            id="search_ran_so_only_the_longest_span_with_no_url_is_a_recitation",
        ),
    ],
)
def test_unary_recited_passages_are_returned_as_citation_metadata(
    tool_steps: tuple[dict[str, object], ...],
    cites: tuple[dict[str, object], ...],
    expected_recited: list[dict[str, object]],
    expected_grounding: list[dict[str, object]],
) -> None:
    recited: Final = {
        "id": "v1_recited",
        "status": "completed",
        "steps": [
            *tool_steps,
            {"type": "model_output", "content": [{"type": "text", "text": _GROUNDED_TEXT, "annotations": list(cites)}]},
        ],
        "usage": _GROUNDED_USAGE,
    }
    _, responder = _scripted((httpx.Response(200, json=recited),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model=MODEL,
            messages=list(MESSAGES),
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )

    assert isinstance(response, ModelResponse)
    assert response.vertex_ai_citation_metadata == [{"citations": expected_recited}]
    assert response.vertex_ai_grounding_metadata == expected_grounding


@pytest.mark.parametrize("whole_steps_on_start", [False, True])
def test_streamed_grounded_answer_carries_the_search_and_url_context_fields_on_its_last_chunk(
    whole_steps_on_start: bool,
) -> None:
    _, responder = _scripted((_grounded_sse(whole_steps_on_start=whole_steps_on_start),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        chunks: Final = list(
            litellm.completion(
                model=MODEL,
                messages=list(MESSAGES),
                stream=True,
                stream_options={"include_usage": True},
                api_key="k",
                api_base=API_BASE,
                client=HTTPHandler(client=raw_client),
            )
        )

    grounded: Final = [chunk for chunk in chunks if getattr(chunk, "vertex_ai_grounding_metadata", None)]
    assert [chunk.vertex_ai_grounding_metadata for chunk in grounded] == [_EXPECTED_GROUNDING]
    assert grounded[0].vertex_ai_url_context_metadata == _EXPECTED_URL_CONTEXT
    usages: Final = [chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None]
    assert usages[-1].prompt_tokens_details.google_maps_grounding_requests == 3


def test_plain_answer_keeps_the_empty_grounding_lists_generate_content_returns() -> None:
    _, responder = _scripted((httpx.Response(200, json=_interaction("Hello", "completed", None)),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model=MODEL,
            messages=list(MESSAGES),
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )

    assert isinstance(response, ModelResponse)
    assert (response.vertex_ai_grounding_metadata, response.vertex_ai_url_context_metadata) == ([], [])


_RECITATION_BLOCK: Final = {"error": {"message": "Request blocked due to recitation.", "code": "recitation"}}


@pytest.mark.parametrize("stream", (False, True), ids=("unary", "stream"))
def test_continued_answer_blocked_for_recitation_ends_as_content_filter_with_no_token_to_resume(stream: bool) -> None:
    first: Final = _interaction("Once ", "requires_continuation", "tok-1")
    _, responder = _scripted(
        ((_sse(first) if stream else httpx.Response(200, json=first)), httpx.Response(400, json=_RECITATION_BLOCK))
    )
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model=MODEL,
            max_tokens=1000,
            messages=list(MESSAGES),
            stream=stream,
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )
        if stream:
            text, finish_reasons, tokens, _ = _drain(response)
        else:
            choice, token = _message_of(response)
            text, finish_reasons, tokens = choice.message.content, [choice.finish_reason], [token] if token else []

    assert text == "Once "
    assert finish_reasons == ["content_filter"]
    assert tokens == []


def test_stream_flag_nested_in_extra_body_does_not_turn_a_unary_request_into_a_provider_stream() -> None:
    bodies, responder = _scripted((httpx.Response(200, json=_interaction("Once upon a time.", "completed")),))
    with httpx.Client(transport=httpx.MockTransport(responder)) as raw_client:
        response: Final = litellm.completion(
            model=MODEL,
            messages=list(MESSAGES),
            stream=False,
            extra_body={"stream": True},
            api_key="k",
            api_base=API_BASE,
            client=HTTPHandler(client=raw_client),
        )

    choice, _ = _message_of(response)
    assert choice.message.content == "Once upon a time."
    assert "stream" not in bodies[0]
