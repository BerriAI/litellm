"""Live e2e: the Presidio PII guardrail masks, per its configured hook point.

pre_call: the guardrail calls the Presidio analyzer/anonymizer on the request
messages BEFORE the model runs, so the model only ever sees placeholders like
<EMAIL_ADDRESS>. A prompt asking the model to repeat a fake email + phone back
must come back with the placeholders echoed and the raw PII absent, on
/chat/completions and on /v1/messages (Anthropic format).

post_call: the mirror hook. The request reaches the model unmasked and the
MODEL OUTPUT is what gets anonymized, so the caller never receives raw PII the
model repeated back. The two hooks are told apart behaviorally rather than by
configuration: the post_call prompt asks for a value derived from the raw email
(its local part, which is not itself an entity Presidio masks) alongside the
address itself, so the answer proves the model saw the raw address while the
address in the same response comes back as <EMAIL_ADDRESS>.

The analyzer/anonymizer endpoints come from PRESIDIO_ANALYZER_API_BASE /
PRESIDIO_ANONYMIZER_API_BASE; missing env is a hard failure, never a skip.
Each guardrail registers with an explicit presidio_filter_scope so only the
configured hook's callback exists (the default "both" registers input masking
AND a post_call output masker), and is deleted on teardown.

The spend-log audit test requires general_settings.store_prompts_in_spend_logs:
true in the proxy config, or STORE_PROMPTS_IN_SPEND_LOGS=true in the proxy
process environment before startup. Setting it only on pytest has no effect.
Without this opt-in, redacting guardrail_response is expected proxy behavior;
this suite deliberately requires the detected-entity details to remain visible.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from typing import Final, Literal

import pytest
from pydantic import BaseModel, TypeAdapter

from e2e_config import unique_marker
from e2e_http import Result, StreamingResponse, Success
from guardrails_client import GuardrailMode, GuardrailsClient, PiiAction, PiiEntity, PresidioParamsBody
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesResponse,
    ChatResponse,
    GuardrailEntityMatch,
    GuardrailRunRecord,
    SpendLogRow,
)

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"

# A guardrail created via POST /guardrails reaches the worker that served the
# create immediately, but every other worker only picks it up on its next
# periodic DB sync (~30s), so the first requests can be served unguarded.
GUARDRAIL_PROPAGATION_DEADLINE_SECONDS = 40.0
GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS = 5.0

# Presidio's anonymizer replaces a detected entity with its unnumbered type
# placeholder, e.g. <EMAIL_ADDRESS>. The pre_call assertions match on the bare
# token because the model is echoing the masked prompt and may not preserve the
# angle brackets; the logged payload keeps the placeholder verbatim.
MASKED_EMAIL_TOKEN = "EMAIL_ADDRESS"
MASKED_PHONE_TOKEN = "PHONE_NUMBER"
MASKED_CREDIT_CARD_TOKEN = "CREDIT_CARD"

# Fictional NANP 555 number; a standard format Presidio's phone recognizer detects.
FAKE_PHONE = "+1 415-555-0134"
FAKE_VISA_TEST_CARD = "4111 1111 1111 1111"

_CARD_DIGIT_RUN: Final = re.compile(r"(?:\d[ -]?){13,19}")


def _presidio_bases() -> tuple[str, str]:
    analyzer = os.environ.get("PRESIDIO_ANALYZER_API_BASE", "").strip()
    anonymizer = os.environ.get("PRESIDIO_ANONYMIZER_API_BASE", "").strip()
    if not analyzer or not anonymizer:
        pytest.fail(
            "Presidio e2e requires PRESIDIO_ANALYZER_API_BASE and PRESIDIO_ANONYMIZER_API_BASE "
            "(the running Presidio analyzer/anonymizer services); missing env is a hard failure, not a skip"
        )
    return analyzer, anonymizer


def _register_presidio(
    client: GuardrailsClient,
    resources: ResourceManager,
    *,
    name: str,
    mode: GuardrailMode | list[GuardrailMode] = "pre_call",
    filter_scope: Literal["input", "output", "both"] | None = "input",
    entities: dict[PiiEntity, PiiAction] | None = None,
) -> None:
    analyzer, anonymizer = _presidio_bases()
    guardrail_id = client.register(
        name,
        PresidioParamsBody(
            mode=mode,
            default_on=False,
            presidio_analyzer_api_base=analyzer,
            presidio_anonymizer_api_base=anonymizer,
            presidio_filter_scope=filter_scope,
            pii_entities_config=entities,
        ),
    )
    resources.defer(lambda: client.delete_guardrail(guardrail_id))


def _fake_email() -> str:
    return f"jane.doe.{unique_marker()}@example.com"


def _pii_prompt(marker: str, email: str) -> str:
    return (
        f"{marker} Repeat this sentence back to me exactly, word for word: "
        f"My email address is {email} and my phone number is {FAKE_PHONE}."
    )


def _first_content(response: ChatResponse) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    return (message.content if message else None) or ""


class _StreamDelta(BaseModel):
    content: str | None = None


class _StreamChoice(BaseModel):
    delta: _StreamDelta


class _StreamChunk(BaseModel):
    choices: tuple[_StreamChoice, ...] = ()


class _AnthropicStreamDelta(BaseModel):
    type: str | None = None
    text: str | None = None


class _AnthropicStreamEvent(BaseModel):
    type: str
    delta: _AnthropicStreamDelta | None = None


def _credit_card_prompt(marker: str) -> str:
    return (
        f"{marker} Reply with only the well known Visa sandbox test card number that starts with 4111, "
        "the 16 digits grouped in fours separated by spaces, and nothing else."
    )


def _passes_luhn(digits: str) -> bool:
    checksum = sum(
        digit if position % 2 == 0 else (digit * 2 - 9 if digit * 2 > 9 else digit * 2)
        for position, digit in enumerate(int(char) for char in reversed(digits))
    )
    return checksum % 10 == 0


def _contains_card_number(text: str) -> bool:
    """Presidio's CREDIT_CARD recognizer only reports Luhn-valid digit runs, so a
    Luhn-invalid number the model hallucinates is not something masking can catch."""
    return any(
        13 <= len(digits) <= 19 and _passes_luhn(digits)
        for digits in (re.sub(r"[ -]", "", match.group()) for match in _CARD_DIGIT_RUN.finditer(text))
    )


def _stream_content(result: StreamingResponse) -> str:
    return "".join(
        choice.delta.content
        for event in result.stream_events
        if event != "[DONE]"
        for choice in _StreamChunk.model_validate_json(event).choices[:1]
        if choice.delta.content
    )


def _anthropic_stream_content(result: StreamingResponse) -> str:
    return "".join(
        event.delta.text
        for payload in result.stream_events
        for event in [_AnthropicStreamEvent.model_validate_json(payload)]
        if event.type == "content_block_delta"
        and event.delta is not None
        and event.delta.type == "text_delta"
        and event.delta.text
    )


def _messages_text(response: AnthropicMessagesResponse) -> str:
    """The text of a /v1/messages answer, whichever shape the proxy produced
    (Anthropic-native content blocks or OpenAI-normalized choices)."""
    parts: list[str] = []
    for block in response.content or []:
        if block.text:
            parts.append(block.text)
    for choice in response.choices or []:
        if choice.message and choice.message.content:
            parts.append(choice.message.content)
    return "\n".join(parts)


def _assert_eventually_masked[R: BaseModel](
    fetch: Callable[[], Result[R]], extract: Callable[[R], str], *, email: str
) -> None:
    """Retry the call until the response comes back masked, to the propagation
    deadline. An unmasked early response is in-flight guardrail propagation, not
    a failure, and neither is a transient non-Success (a replica that has not
    reloaded the guardrail answers 404, the live model can rate-limit) - only a
    response that still carries the raw PII at the deadline is."""
    deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS
    last: str = "<no successful response yet>"
    while True:
        result = fetch()
        match result:
            case Success(data=data):
                content = extract(data)
                last = content
                masked = MASKED_EMAIL_TOKEN in content and MASKED_PHONE_TOKEN in content and email not in content
                if masked:
                    assert FAKE_PHONE not in content, (
                        f"the raw phone number must be masked before the model sees it, but the "
                        f"response echoed it: {content[:300]!r}"
                    )
                    return
            case _:
                last = f"<non-Success result: {result}>"
        if time.monotonic() >= deadline:
            pytest.fail(
                f"presidio pre_call guardrail never masked the PII within "
                f"{GUARDRAIL_PROPAGATION_DEADLINE_SECONDS}s; last observation: {last[:300]!r}"
            )
        time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)


class TestPresidioPreCallMasking:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.masks",
        exercised_on=["chat_completions"],
    )
    def test_pre_call_masks_pii_on_chat_completions(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-presidio-pre-chat-{unique_marker()}"
        _register_presidio(client, resources, name=name)

        email = _fake_email()
        prompt = _pii_prompt(unique_marker(), email)

        _assert_eventually_masked(
            lambda: client.chat(scoped_key, MODEL, prompt, guardrails=[name], max_tokens=128),
            _first_content,
            email=email,
        )

    @pytest.mark.covers(
        "guardrail.presidio.pre_call.masks",
        exercised_on=["messages"],
    )
    def test_pre_call_masks_pii_on_messages(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-presidio-pre-msg-{unique_marker()}"
        _register_presidio(client, resources, name=name)

        email = _fake_email()
        prompt = _pii_prompt(unique_marker(), email)

        _assert_eventually_masked(
            lambda: client.messages(scoped_key, MODEL, prompt, guardrails=[name], max_tokens=128),
            _messages_text,
            email=email,
        )


#: Room for the model's reasoning tokens plus the three-line answer; a lower cap
#: truncates the response before the address it is supposed to mask.
_POST_CALL_MAX_TOKENS = 512

#: The post_call scenario masks these two entities and nothing else. Left
#: unscoped, Presidio's broader recognizers claim the local part too (a random
#: marker reads as an NRP), which would erase the very token that tells output
#: masking apart from input masking.
_POST_CALL_ENTITIES: dict[PiiEntity, PiiAction] = {"EMAIL_ADDRESS": "MASK", "PHONE_NUMBER": "MASK"}


def _post_call_prompt(marker: str, local_part: str) -> str:
    """Ask for the local part and the full address in one answer. Presidio masks
    an EMAIL_ADDRESS entity and a bare local part is not one, so the two land
    differently in the same response and pin the hook point behaviorally."""
    return (
        f"{marker} My email address is {local_part}@example.com and my phone number is {FAKE_PHONE}. "
        "Reply with exactly three lines and nothing else. "
        "Line 1: the part of the email address before the @ sign. "
        "Line 2: the full email address. "
        "Line 3: the phone number."
    )


class TestPresidioPostCallMasking:
    @pytest.mark.covers(
        "guardrail.presidio.post_call.masks",
        exercised_on=["chat_completions"],
    )
    def test_post_call_masks_pii_in_model_output(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """A guardrail scoped to the output must anonymize the PII the model
        repeats back, so a caller (or a downstream log of the response) never
        receives it, while the request itself reaches the model untouched.

        Both facts are asserted from one response: the local part comes back raw,
        which is only possible if the model saw the real address, and the address
        itself comes back as <EMAIL_ADDRESS> in the same answer.
        """
        name = f"e2e-presidio-post-chat-{unique_marker()}"
        _register_presidio(
            client,
            resources,
            name=name,
            mode="post_call",
            filter_scope="output",
            entities=_POST_CALL_ENTITIES,
        )

        local_part = f"e2euser{unique_marker()}"
        email = f"{local_part}@example.com"
        prompt = _post_call_prompt(unique_marker(), local_part)

        deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS
        last = "<no successful response yet>"
        while True:
            result = client.chat(scoped_key, MODEL, prompt, guardrails=[name], max_tokens=_POST_CALL_MAX_TOKENS)
            match result:
                case Success(data=data):
                    last = _first_content(data)
                    if MASKED_EMAIL_TOKEN in last and email not in last:
                        assert local_part in last, (
                            "the model must have seen the RAW address (it is asked for the local "
                            "part, which Presidio does not mask); the local part is missing, so "
                            f"this response cannot tell post_call masking from pre_call: {last[:300]!r}"
                        )
                        assert MASKED_PHONE_TOKEN in last and FAKE_PHONE not in last, (
                            f"the phone number in the model's answer must be masked too, got: {last[:300]!r}"
                        )
                        return
                case _:
                    last = f"<non-Success result: {result}>"
            if time.monotonic() >= deadline:
                pytest.fail(
                    f"presidio post_call guardrail never masked the model's output within "
                    f"{GUARDRAIL_PROPAGATION_DEADLINE_SECONDS}s; last observation: {last[:300]!r}"
                )
            time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)


def _assert_eventually_masks_generated_card(fetch: Callable[[], str | None]) -> None:
    deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS
    last: str = "<no successful response yet>"
    while True:
        content = fetch()
        if content is not None:
            last = content
            if _contains_card_number(content):
                pytest.fail(
                    "the post_call output masking let a card number through: "
                    f"{content[:300]!r}"
                )
            if MASKED_CREDIT_CARD_TOKEN in content:
                return
        if time.monotonic() >= deadline:
            pytest.fail(
                "presidio post_call output masking never masked the generated card within "
                f"{GUARDRAIL_PROPAGATION_DEADLINE_SECONDS}s; last observation: {last[:300]!r}"
            )
        time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)


class TestPresidioCreditCardOutputMasking:
    """Proves the UI-default Presidio scope masks model-generated card output."""

    @pytest.mark.covers(
        "guardrail.presidio.post_call.masks_generated_output",
        exercised_on=["chat_completions"],
    )
    def test_ui_default_scope_masks_a_card_number_the_model_generates_on_chat_completions(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-presidio-card-chat-{unique_marker()}"
        _register_presidio(
            client,
            resources,
            name=name,
            mode=["pre_call", "post_call"],
            filter_scope=None,
            entities={"CREDIT_CARD": "MASK"},
        )
        prompt: Final = _credit_card_prompt(unique_marker())

        def fetch() -> str | None:
            result: Final = client.chat(scoped_key, MODEL, prompt, guardrails=[name], max_tokens=512)
            match result:
                case Success(data=data):
                    return _first_content(data)
                case _:
                    return None

        _assert_eventually_masks_generated_card(fetch)

    @pytest.mark.covers(
        "guardrail.presidio.post_call.masks_generated_output",
        exercised_on=["chat_completions_stream"],
    )
    def test_ui_default_scope_masks_a_card_number_the_model_generates_on_streaming_chat_completions(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-presidio-card-stream-{unique_marker()}"
        _register_presidio(
            client,
            resources,
            name=name,
            mode=["pre_call", "post_call"],
            filter_scope=None,
            entities={"CREDIT_CARD": "MASK"},
        )
        prompt: Final = _credit_card_prompt(unique_marker())

        def fetch() -> str | None:
            result: Final = client.chat_stream_raw(
                scoped_key,
                MODEL,
                prompt,
                guardrails=[name],
                max_tokens=512,
            )
            if not result.ok or result.stream_error:
                return None
            return _stream_content(result)

        _assert_eventually_masks_generated_card(fetch)

    @pytest.mark.covers(
        "guardrail.presidio.post_call.masks_generated_output",
        exercised_on=["anthropic_messages_stream"],
    )
    def test_ui_default_scope_masks_a_card_number_the_model_generates_on_streaming_anthropic_messages(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-presidio-card-messages-stream-{unique_marker()}"
        _register_presidio(
            client,
            resources,
            name=name,
            mode=["pre_call", "post_call"],
            filter_scope=None,
            entities={"CREDIT_CARD": "MASK"},
        )
        prompt: Final = _credit_card_prompt(unique_marker())

        def fetch() -> str | None:
            result: Final = client.messages_stream_raw(
                scoped_key,
                MODEL,
                prompt,
                guardrails=[name],
                max_tokens=512,
            )
            if not result.ok or result.stream_error:
                return None
            return _anthropic_stream_content(result)

        _assert_eventually_masks_generated_card(fetch)


def _wire_text(outcome: StreamingResponse) -> str:
    return "\n".join(outcome.stream_events) if outcome.is_streaming else outcome.body


def _poll_until_generated_card_masked(fetch: Callable[[], StreamingResponse]) -> StreamingResponse:
    """The raw HTTP outcome once the output masker is in effect on the serving
    worker: whichever wire shape the endpoint speaks, a masked body carries the
    CREDIT_CARD placeholder and no Luhn-valid card run. A raw card is a worker
    that has not loaded the guardrail yet, so it is polled through like any
    other unmasked answer."""
    deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS
    last: str = "<no successful response yet>"
    while True:
        outcome = fetch()
        if outcome.ok and not outcome.stream_error:
            wire = _wire_text(outcome)
            last = wire
            if MASKED_CREDIT_CARD_TOKEN in wire and not _contains_card_number(wire):
                return outcome
        if time.monotonic() >= deadline:
            pytest.fail(
                "presidio post_call output masking never masked the generated card within "
                f"{GUARDRAIL_PROPAGATION_DEADLINE_SECONDS}s; last observation: {last[:300]!r}"
            )
        time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)


def _spend_log_response_text(client: GuardrailsClient, key: str, call_id: str) -> str:
    rows = client.proxy.poll_logs_for_key(
        key,
        predicate=lambda logged: any(row.litellm_call_id == call_id for row in logged),
    )
    row = next((row for row in rows if row.litellm_call_id == call_id), None)
    assert row is not None, f"no spend log row ever appeared for x-litellm-call-id {call_id}"
    return json.dumps(row.response)


class TestPresidioSpendLogStoresMaskedOutput:
    """The spend log stores the response the caller received, on every endpoint
    and both stream modes, when an output-only post_call Presidio guardrail masks
    a card number the model generated."""

    _CELL: Final = "guardrail.presidio.post_call.spend_log_stores_masked_output"

    def _assert_spend_log_is_masked(
        self,
        client: GuardrailsClient,
        resources: ResourceManager,
        key: str,
        *,
        name: str,
        fetch: Callable[[str, str], StreamingResponse],
    ) -> None:
        _register_presidio(
            client,
            resources,
            name=name,
            mode="post_call",
            filter_scope="output",
            entities={"CREDIT_CARD": "MASK"},
        )
        prompt: Final = _credit_card_prompt(unique_marker())
        outcome = _poll_until_generated_card_masked(lambda: fetch(prompt, name))
        assert outcome.call_id, f"the served response must carry x-litellm-call-id: {dict(outcome.headers)}"

        logged = _spend_log_response_text(client, key, outcome.call_id)
        assert not _contains_card_number(logged), (
            "the caller got the masked response but the spend log stored the raw model output: "
            f"{logged[:400]!r}"
        )
        assert MASKED_CREDIT_CARD_TOKEN in logged, (
            f"the spend log response carries neither the card nor the placeholder: {logged[:400]!r}"
        )

    @pytest.mark.covers(_CELL, exercised_on=["chat_completions"])
    def test_spend_log_stores_masked_output_on_chat_completions(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        self._assert_spend_log_is_masked(
            client,
            resources,
            scoped_key,
            name=f"e2e-presidio-log-card-chat-{unique_marker()}",
            fetch=lambda prompt, guardrail: client.chat_raw(
                scoped_key, MODEL, prompt, guardrails=[guardrail], max_tokens=512
            ),
        )

    @pytest.mark.covers(_CELL, exercised_on=["chat_completions_stream"])
    def test_spend_log_stores_masked_output_on_streaming_chat_completions(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        self._assert_spend_log_is_masked(
            client,
            resources,
            scoped_key,
            name=f"e2e-presidio-log-card-chat-stream-{unique_marker()}",
            fetch=lambda prompt, guardrail: client.chat_stream_raw(
                scoped_key, MODEL, prompt, guardrails=[guardrail], max_tokens=512
            ),
        )

    @pytest.mark.covers(_CELL, exercised_on=["messages"])
    def test_spend_log_stores_masked_output_on_anthropic_messages(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        self._assert_spend_log_is_masked(
            client,
            resources,
            scoped_key,
            name=f"e2e-presidio-log-card-messages-{unique_marker()}",
            fetch=lambda prompt, guardrail: client.messages_raw(
                scoped_key, MODEL, prompt, guardrails=[guardrail], max_tokens=512
            ),
        )

    @pytest.mark.covers(_CELL, exercised_on=["anthropic_messages_stream"])
    def test_spend_log_stores_masked_output_on_streaming_anthropic_messages(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        self._assert_spend_log_is_masked(
            client,
            resources,
            scoped_key,
            name=f"e2e-presidio-log-card-messages-stream-{unique_marker()}",
            fetch=lambda prompt, guardrail: client.messages_stream_raw(
                scoped_key, MODEL, prompt, guardrails=[guardrail], max_tokens=512
            ),
        )

    @pytest.mark.covers(_CELL, exercised_on=["responses"])
    def test_spend_log_stores_masked_output_on_responses(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        self._assert_spend_log_is_masked(
            client,
            resources,
            scoped_key,
            name=f"e2e-presidio-log-card-responses-{unique_marker()}",
            fetch=lambda prompt, guardrail: client.responses(scoped_key, MODEL, prompt, guardrails=[guardrail]),
        )


_LOGGED_ENTITIES: dict[PiiEntity, PiiAction] = {"EMAIL_ADDRESS": "MASK", "PHONE_NUMBER": "MASK"}

_ENTITY_LIST_ADAPTER: Final = TypeAdapter(list[GuardrailEntityMatch])


def _applied_guardrails(outcome: StreamingResponse) -> str:
    return outcome.headers.get("x-litellm-applied-guardrails", "")


def _guardrail_records(row: SpendLogRow) -> tuple[GuardrailRunRecord, ...]:
    metadata = row.metadata
    if metadata is None:
        return ()
    return tuple(metadata.guardrail_information or ())


def _poll_until_guardrail_applied(
    client: GuardrailsClient, key: str, guardrail_name: str, prompt: str
) -> StreamingResponse:
    deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS
    last = client.chat_raw(key, MODEL, prompt, guardrails=[guardrail_name], max_tokens=128)
    while time.monotonic() < deadline:
        if last.ok and guardrail_name in _applied_guardrails(last):
            return last
        time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)
        last = client.chat_raw(key, MODEL, prompt, guardrails=[guardrail_name], max_tokens=128)
    return last


class TestPresidioSpendLogRecord:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.logs_masked_entities",
        exercised_on=["chat_completions"],
    )
    def test_masking_run_is_recorded_on_the_spend_log(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-presidio-log-{unique_marker()}"
        _register_presidio(client, resources, name=name, entities=_LOGGED_ENTITIES)

        email = _fake_email()
        prompt = _pii_prompt(unique_marker(), email)

        outcome = _poll_until_guardrail_applied(client, scoped_key, name, prompt)
        assert outcome.ok, f"the guarded call must be served, got {outcome.status_code}: {outcome.body[:400]}"
        assert name in _applied_guardrails(outcome), (
            "the response must carry x-litellm-applied-guardrails naming the guardrail; without it "
            f"the 200 only proves the guardrail never attached. Got {_applied_guardrails(outcome)!r}"
        )

        request_id = ChatResponse.model_validate_json(outcome.body).id
        assert request_id, f"the served response must carry an id to look the spend log up by: {outcome.body[:400]}"

        rows = client.proxy.poll_logs_for_request_id(
            request_id,
            predicate=lambda logged: bool(_guardrail_records(logged[0])),
        )
        assert rows, f"no spend log row ever appeared for request {request_id}"

        records = _guardrail_records(rows[0])
        assert records, (
            f"the spend log for {request_id} carries no guardrail_information, so the dashboard's "
            "guardrail panel would render nothing for a request the guardrail demonstrably ran on"
        )

        record = next(
            (entry for entry in records if entry.guardrail_name == name and entry.guardrail_mode == "pre_call"),
            None,
        )
        assert record is not None, (
            "guardrail_information carries no pre_call record for this guardrail, only "
            f"{[(entry.guardrail_name, entry.guardrail_mode) for entry in records]}"
        )
        assert record.guardrail_status == "success", (
            f"the recorded status must be success for a run that masked and served, got {record.guardrail_status!r}"
        )
        assert record.guardrail_provider == "presidio", (
            f"the record must attribute the run to presidio so the dashboard picks the right "
            f"renderer, got {record.guardrail_provider!r}"
        )

        counts = record.masked_entity_count or {}
        assert _LOGGED_ENTITIES.keys() <= counts.keys(), (
            f"every entity the guardrail was configured to mask must appear in masked_entity_count, got {counts}"
        )
        assert all(counts[entity] >= 1 for entity in _LOGGED_ENTITIES), (
            f"each masked entity must be counted at least once, got {counts}"
        )

        assert not isinstance(record.guardrail_response, str), (
            "This audit test requires general_settings.store_prompts_in_spend_logs: true in the proxy config "
            "or STORE_PROMPTS_IN_SPEND_LOGS=true in the proxy process environment before startup "
            "(not just the pytest environment). Default prompt redaction is valid proxy behavior, "
            f"but prevents entity-detail assertions; got guardrail_response={record.guardrail_response!r}"
        )
        entities = _ENTITY_LIST_ADAPTER.validate_python(record.guardrail_response)
        assert entities, (
            "guardrail_response must carry the detected entities; the dashboard's Detected Entities "
            "list and its per-entity scores are rendered from exactly this array"
        )
        assert {entity.entity_type for entity in entities} >= _LOGGED_ENTITIES.keys(), (
            f"the detected entities must cover what was masked, got "
            f"{sorted(entity.entity_type for entity in entities)}"
        )
        for entity in entities:
            assert 0.0 < entity.score <= 1.0, f"{entity.entity_type} carries an out-of-range score: {entity.score}"
            assert 0 <= entity.start < entity.end, (
                f"{entity.entity_type} carries a degenerate span: {entity.start}-{entity.end}"
            )
