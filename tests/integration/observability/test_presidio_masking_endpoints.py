import json
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import accumulate, chain
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, list_value, object_value
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

CARD: Final = "4111 1111 1111 1111"
PHONE: Final = "+1 415-555-0134"
MASKED_CARD: Final = "<CREDIT_CARD>"
MASKED_EMAIL: Final = "<EMAIL_ADDRESS>"
MASKED_PHONE: Final = "<PHONE_NUMBER>"

REPEAT_REQUEST: Final = "Repeat this sentence back to me exactly, word for word: "
THREE_LINE_REQUEST: Final = (
    "Reply with exactly three lines and nothing else. "
    "Line 1: the part of the email address before the @ sign. "
    "Line 2: the full email address. "
    "Line 3: the phone number."
)
CARD_REQUEST: Final = "Reply with only the well known Visa sandbox test card number that starts with 4111."

EMAIL_PATTERN: Final = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE_PATTERN: Final = re.compile(r"(?:\+1 )?\d{3}-\d{3}-\d{4}")
CARD_PATTERN: Final = re.compile(r"\b(?:4\d{3}|5[0-5]\d{2}|6\d{3}|1\d{3}|3\d{3})[- ]?\d{3,4}[- ]?\d{3,4}[- ]?\d{3,5}\b")
RECOGNIZERS: Final = {
    "EMAIL_ADDRESS": (EMAIL_PATTERN, 1.0),
    "PHONE_NUMBER": (PHONE_PATTERN, 0.4),
    "CREDIT_CARD": (CARD_PATTERN, 1.0),
}
JSON: Final = TypeAdapter(JsonValue)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _AnalyzeRequest(_Frozen):
    text: str = ""
    language: str = ""
    entities: tuple[str, ...] | None = None


class _Span(_Frozen):
    entity_type: str
    start: int
    end: int
    score: float


class _AnonymizeRequest(_Frozen):
    text: str
    analyzer_results: tuple[_Span, ...]


class _Part(_Frozen):
    text: str


class _Message(_Frozen):
    content: str | tuple[_Part, ...]


class _ModelRequest(_Frozen):
    messages: tuple[_Message, ...] = ()
    input: str | tuple[_Message, ...] | None = None
    stream: bool = False

    @property
    def prompt(self) -> str:
        turns: Final = self.messages if self.input is None else self.input
        if isinstance(turns, str):
            return turns
        content: Final = turns[-1].content
        return content if isinstance(content, str) else "".join(part.text for part in content)


class _StreamDelta(_Frozen):
    content: str | None = None


class _StreamChoice(_Frozen):
    delta: _StreamDelta


class _ChatChunk(_Frozen):
    choices: tuple[_StreamChoice, ...] = ()


class _AnthropicDelta(_Frozen):
    type: str | None = None
    text: str | None = None


class _AnthropicEvent(_Frozen):
    type: str
    delta: _AnthropicDelta | None = None


class _LoggedEntity(_Frozen):
    entity_type: str
    start: int
    end: int
    score: float


def _passes_luhn(digits: str) -> bool:
    checksum: Final = sum(
        digit if position % 2 == 0 else (digit * 2 - 9 if digit * 2 > 9 else digit * 2)
        for position, digit in enumerate(int(char) for char in reversed(digits))
    )
    return checksum % 10 == 0


def _detections(entity: str, text: str) -> tuple[_Span, ...]:
    pattern, score = RECOGNIZERS[entity]
    return tuple(
        _Span(entity_type=entity, start=match.start(), end=match.end(), score=score)
        for match in pattern.finditer(text)
        if entity != "CREDIT_CARD" or _passes_luhn(re.sub(r"\D", "", match.group()))
    )


def _analyze(request: Request) -> Reply:
    assert request.target == "/analyze", request.target
    body: Final = _AnalyzeRequest.model_validate_json(request.body)
    if not body.text or not body.language:
        return Reply(status=400, body=json.dumps({"error": "No text provided"}).encode())
    requested: Final = body.entities or tuple(RECOGNIZERS)
    findings: Final = chain.from_iterable(
        _detections(entity, body.text) for entity in requested if entity in RECOGNIZERS
    )
    return Reply(body=json.dumps([span.model_dump() for span in sorted(findings, key=lambda s: s.start)]).encode())


def _anonymize(request: Request) -> Reply:
    assert request.target == "/anonymize", request.target
    body: Final = _AnonymizeRequest.model_validate_json(request.body)
    spans: Final = sorted(body.analyzer_results, key=lambda span: span.start)
    masks: Final = [f"<{span.entity_type}>" for span in spans]
    shifts: Final = tuple(
        accumulate((len(mask) - (span.end - span.start) for mask, span in zip(masks, spans)), initial=0)
    )
    items: Final = [
        {
            "start": span.start + shift,
            "end": span.start + shift + len(mask),
            "entity_type": span.entity_type,
            "text": mask,
            "operator": "replace",
        }
        for span, mask, shift in zip(spans, masks, shifts)
    ]
    pieces: Final = [
        body.text[(spans[index - 1].end if index else 0) : span.start] + masks[index]
        for index, span in enumerate(spans)
    ]
    tail: Final = body.text[spans[-1].end :] if spans else body.text
    return Reply(body=json.dumps({"text": "".join(pieces) + tail, "items": items}).encode())


def _model_answer(prompt: str) -> str:
    if REPEAT_REQUEST in prompt:
        return prompt.split(REPEAT_REQUEST, 1)[1]
    if THREE_LINE_REQUEST in prompt:
        email: Final = EMAIL_PATTERN.search(prompt)
        phone: Final = PHONE_PATTERN.search(prompt)
        assert email is not None and phone is not None, prompt
        return "\n".join((email.group().split("@")[0], email.group(), phone.group()))
    assert CARD_REQUEST in prompt, prompt
    return CARD


def _chat_completion(answer: str) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
    }


def _chat_frames(answer: str) -> tuple[bytes, ...]:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"

    def frame(delta: dict[str, JsonValue], finish: str | None = None) -> bytes:
        payload: Final = {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    return (
        frame({"role": "assistant", "content": ""}),
        *(frame({"content": word}) for word in re.findall(r"\S+\s*", answer)),
        frame({}, "stop"),
        b"data: [DONE]\n\n",
    )


def _response_object(answer: str, identity: str, item: str, *, completed: bool = True) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed" if completed else "in_progress",
        "model": "gpt-4o-mini",
        "output": [_message_item(answer, item)] if completed else [],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        if completed
        else None,
    }


def _message_item(answer: str, item: str, *, completed: bool = True) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": item,
        "status": "completed" if completed else "in_progress",
        "role": "assistant",
        "content": [_output_text(answer)] if completed else [],
    }


def _output_text(text: str) -> dict[str, JsonValue]:
    return {"type": "output_text", "text": text, "annotations": []}


def _response_frames(answer: str) -> tuple[bytes, ...]:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    item: Final = f"msg_{uuid.uuid4().hex}"
    location: Final[dict[str, JsonValue]] = {"item_id": item, "output_index": 0, "content_index": 0}
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "response.created", "response": _response_object("", identity, item, completed=False)},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": _message_item("", item, completed=False),
        },
        {"type": "response.content_part.added", **location, "part": _output_text("")},
        *({"type": "response.output_text.delta", **location, "delta": word} for word in re.findall(r"\S+\s*", answer)),
        {"type": "response.output_text.done", **location, "text": answer},
        {"type": "response.content_part.done", **location, "part": _output_text(answer)},
        {"type": "response.output_item.done", "output_index": 0, "item": _message_item(answer, item)},
        {"type": "response.completed", "response": _response_object(answer, identity, item)},
    )
    return tuple(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': number})}\n\n".encode()
        for number, event in enumerate(events)
    )


def _model(request: Request) -> Reply:
    body: Final = _ModelRequest.model_validate_json(request.body)
    answer: Final = _model_answer(body.prompt)
    match request.target:
        case "/v1/chat/completions" if body.stream:
            return Reply(content_type="text/event-stream", chunks=_chat_frames(answer))
        case "/v1/chat/completions":
            return Reply(body=json.dumps(_chat_completion(answer)).encode())
        case "/v1/responses" if body.stream:
            return Reply(content_type="text/event-stream", chunks=_response_frames(answer))
        case "/v1/responses":
            return Reply(
                body=json.dumps(
                    _response_object(answer, f"resp_{uuid.uuid4().hex}", f"msg_{uuid.uuid4().hex}")
                ).encode()
            )
        case _:
            raise AssertionError(request.target)


def _presidio_guardrail(
    name: str,
    analyzer: Wire,
    anonymizer: Wire,
    *,
    mode: str | list[str],
    scope: str | None,
    entities: dict[str, JsonValue] | None,
) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "presidio",
            "mode": mode,
            "default_on": False,
            "presidio_analyzer_api_base": analyzer.url,
            "presidio_anonymizer_api_base": anonymizer.url,
            **({} if scope is None else {"presidio_filter_scope": scope}),
            **({} if entities is None else {"pii_entities_config": entities}),
        },
    }


@dataclass(frozen=True, slots=True)
class Guardrails:
    input_scope: str
    logged: str
    output_scope: str
    default_scope: str
    card_output_scope: str


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    model: Wire
    guardrails: Guardrails


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("presidio-masking-endpoints")
    suffix: Final = uuid.uuid4().hex[:8]
    names: Final = Guardrails(
        input_scope=f"presidio-input-{suffix}",
        logged=f"presidio-logged-{suffix}",
        output_scope=f"presidio-output-{suffix}",
        default_scope=f"presidio-default-{suffix}",
        card_output_scope=f"presidio-card-output-{suffix}",
    )
    contact_entities: Final[dict[str, JsonValue]] = {"EMAIL_ADDRESS": "MASK", "PHONE_NUMBER": "MASK"}
    card_entities: Final[dict[str, JsonValue]] = {"CREDIT_CARD": "MASK"}
    with ExitStack() as stack:
        model: Final = stack.enter_context(wire_server(_model))
        analyzer: Final = stack.enter_context(wire_server(_analyze))
        anonymizer: Final = stack.enter_context(wire_server(_anonymize))
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"]["store_prompts_in_spend_logs"] = True
        config["guardrails"] = [
            _presidio_guardrail(names.input_scope, analyzer, anonymizer, mode="pre_call", scope="input", entities=None),
            _presidio_guardrail(
                names.logged, analyzer, anonymizer, mode="pre_call", scope="input", entities=contact_entities
            ),
            _presidio_guardrail(
                names.output_scope, analyzer, anonymizer, mode="post_call", scope="output", entities=contact_entities
            ),
            _presidio_guardrail(
                names.default_scope,
                analyzer,
                anonymizer,
                mode=["pre_call", "post_call"],
                scope=None,
                entities=card_entities,
            ),
            _presidio_guardrail(
                names.card_output_scope, analyzer, anonymizer, mode="post_call", scope="output", entities=card_entities
            ),
        ]
        path: Final = directory / "presidio-masking.yaml"
        path.write_text(yaml.safe_dump(config))
        shared: Final = stack.enter_context(gateway_from_environment())
        owned: Final = stack.enter_context(owned_proxy_process(shared, directory, {}, config=path))
        yield Rig(owned.gateway, model, names)


@pytest.fixture
def model(rig: Rig) -> Iterator[str]:
    with rig.gateway.scenario() as scenario:
        yield scenario.model(api_base=rig.model.url + "/v1")


def _content_strings(node: JsonValue, key: str | None = None) -> Iterator[str]:
    match node:
        case str() if key in {"content", "text"}:
            yield node
        case dict():
            for child_key, child in node.items():
                yield from _content_strings(child, child_key)
        case list():
            for item in node:
                yield from _content_strings(item, key)
        case _:
            return


def _json_text(raw: str) -> str:
    return "\n".join(_content_strings(JSON.validate_json(raw)))


def _data_events(raw: str) -> tuple[str, ...]:
    return tuple(
        line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"
    )


def _chat_stream_text(raw: str) -> str:
    return "".join(
        choice.delta.content
        for event in _data_events(raw)
        for choice in _ChatChunk.model_validate_json(event).choices[:1]
        if choice.delta.content
    )


def _messages_stream_text(raw: str) -> str:
    return "".join(
        event.delta.text
        for payload in _data_events(raw)
        for event in [_AnthropicEvent.model_validate_json(payload)]
        if event.type == "content_block_delta"
        and event.delta is not None
        and event.delta.type == "text_delta"
        and event.delta.text
    )


@dataclass(frozen=True, slots=True)
class Surface:
    path: str
    stream: bool
    takes_input: bool
    read: Callable[[str], str]


CHAT: Final = Surface("/v1/chat/completions", False, False, _json_text)
CHAT_STREAM: Final = Surface("/v1/chat/completions", True, False, _chat_stream_text)
MESSAGES: Final = Surface("/v1/messages", False, False, _json_text)
MESSAGES_STREAM: Final = Surface("/v1/messages", True, False, _messages_stream_text)
RESPONSES: Final = Surface("/v1/responses", False, True, _json_text)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    wire: str
    applied_guardrails: str
    sent_to_model: tuple[str, ...]


def _request_body(surface: Surface, model: str, guardrail: str, prompt: str) -> dict[str, JsonValue]:
    common: Final[dict[str, JsonValue]] = {"model": model, "guardrails": [guardrail], "stream": surface.stream}
    if surface.takes_input:
        return {**common, "input": prompt}
    return {**common, "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]}


def _ask(rig: Rig, model: str, surface: Surface, guardrail: str, prompt: str) -> Answer:
    rig.model.drain()
    with rig.gateway.client.stream(
        "POST",
        surface.path,
        json=_request_body(surface, model, guardrail, prompt),
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
    ) as response:
        wire: Final = response.read().decode()
        status: Final = response.status_code
        applied: Final = response.headers.get("x-litellm-applied-guardrails", "")
    assert status == 200, wire
    return Answer(
        surface.read(wire),
        wire,
        applied,
        tuple(_ModelRequest.model_validate_json(request.body).prompt for request in rig.model.drain()),
    )


def _spend_row(rig: Rig, model: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: rig.gateway.rows('SELECT response, metadata FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _marker() -> str:
    return uuid.uuid4().hex[:8]


def _repeat_prompt(marker: str, email: str) -> str:
    return f"{marker} {REPEAT_REQUEST}My email address is {email} and my phone number is {PHONE}."


def _three_line_prompt(marker: str, local_part: str) -> str:
    return f"{marker} My email address is {local_part}@example.com and my phone number is {PHONE}. {THREE_LINE_REQUEST}"


def _card_prompt(marker: str) -> str:
    return f"{marker} {CARD_REQUEST}"


@pytest.mark.parametrize("surface", (CHAT, MESSAGES), ids=("chat", "messages"))
def test_pre_call_masks_contact_details_before_the_model_sees_them(rig: Rig, model: str, surface: Surface) -> None:
    marker: Final = _marker()
    email: Final = f"jane.doe.{marker}@example.com"

    answer: Final = _ask(rig, model, surface, rig.guardrails.input_scope, _repeat_prompt(marker, email))

    masked_sentence: Final = f"My email address is {MASKED_EMAIL} and my phone number is {MASKED_PHONE}."
    assert answer.sent_to_model == (f"{marker} {REPEAT_REQUEST}{masked_sentence}",), answer.sent_to_model
    assert answer.text == masked_sentence, answer.wire


def test_post_call_masks_the_model_output_while_the_model_sees_the_raw_request(rig: Rig, model: str) -> None:
    marker: Final = _marker()
    local_part: Final = f"e2euser{marker}"
    prompt: Final = _three_line_prompt(marker, local_part)

    answer: Final = _ask(rig, model, CHAT, rig.guardrails.output_scope, prompt)

    assert answer.sent_to_model == (prompt,), answer.sent_to_model
    assert answer.text == f"{local_part}\n{MASKED_EMAIL}\n{MASKED_PHONE}", answer.wire


@pytest.mark.parametrize(
    "surface", (CHAT, CHAT_STREAM, MESSAGES_STREAM), ids=("chat", "chat_stream", "messages_stream")
)
def test_default_scope_masks_a_card_number_the_model_generates(rig: Rig, model: str, surface: Surface) -> None:
    answer: Final = _ask(rig, model, surface, rig.guardrails.default_scope, _card_prompt(_marker()))

    assert answer.text == MASKED_CARD, answer.wire
    assert "4111" not in answer.wire, answer.wire


@pytest.mark.parametrize(
    "surface",
    (CHAT, CHAT_STREAM, MESSAGES, MESSAGES_STREAM, RESPONSES),
    ids=("chat", "chat_stream", "messages", "messages_stream", "responses"),
)
def test_spend_log_stores_the_masked_output_the_caller_received(rig: Rig, model: str, surface: Surface) -> None:
    answer: Final = _ask(rig, model, surface, rig.guardrails.card_output_scope, _card_prompt(_marker()))
    assert answer.text == MASKED_CARD, answer.wire

    stored: Final = _spend_row(rig, model)["response"]

    assert "\n".join(_content_strings(stored)) == MASKED_CARD, stored
    assert "4111" not in json.dumps(stored), stored


def test_masking_run_is_recorded_on_the_spend_log_with_the_detected_entities(rig: Rig, model: str) -> None:
    marker: Final = _marker()
    email: Final = f"jane.doe.{marker}@example.com"
    prompt: Final = _repeat_prompt(marker, email)

    answer: Final = _ask(rig, model, CHAT, rig.guardrails.logged, prompt)

    assert rig.guardrails.logged in answer.applied_guardrails, answer.applied_guardrails
    metadata: Final = object_value(_spend_row(rig, model)["metadata"])
    records: Final = tuple(
        object_value(record)
        for record in list_value(metadata["guardrail_information"])
        if object_value(record)["guardrail_name"] == rig.guardrails.logged
    )
    assert len(records) == 1, metadata["guardrail_information"]
    record: Final = records[0]
    assert {
        key: record[key]
        for key in (
            "guardrail_mode",
            "guardrail_status",
            "guardrail_provider",
            "masked_entity_count",
        )
    } == {
        "guardrail_mode": "pre_call",
        "guardrail_status": "success",
        "guardrail_provider": "presidio",
        "masked_entity_count": {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1},
    }, record
    detected: Final = sorted(
        (_LoggedEntity.model_validate(entity) for entity in list_value(record["guardrail_response"])),
        key=lambda entity: entity.start,
    )
    email_start: Final = prompt.index(email)
    phone_start: Final = prompt.index(PHONE)
    assert [(entity.entity_type, entity.start, entity.end, entity.score) for entity in detected] == [
        ("EMAIL_ADDRESS", email_start, email_start + len(email), 1.0),
        ("PHONE_NUMBER", phone_start, phone_start + len(PHONE), 0.4),
    ], record["guardrail_response"]
