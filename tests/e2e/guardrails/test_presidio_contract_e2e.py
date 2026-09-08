"""Live e2e: the contract between LiteLLM and a real Presidio deployment.

Whether Presidio finds an email address is Presidio's business, and this suite
does not test it. What it tests is our half of the exchange: that the entity
filter we were configured with is actually put on the /analyze request, that the
spans and entity types Presidio answers with are turned into the right
placeholders, blocks, and restored values, and that a Presidio we cannot reach
stops the request instead of forwarding raw PII to the model.

Every case is written to survive Presidio changing its mind about confidence
scores or adding a recognizer: the assertions turn on which entity types were
asked for and on what our code did with the answer, never on a score.

Most cases drive POST /guardrails/apply_guardrail, which runs the guardrail over
literal text and hands back what the model would have received. That puts the
assertion on the masked text itself rather than on a model's willingness to echo
it, and costs no provider spend. The two cases that must prove an effect on a
real call, blocking and restoring the masked value on the way out, go through
/chat/completions.

The analyzer and anonymizer endpoints come from the environment and are treated
as secrets; see presidio_env.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final, Literal

import pytest

from e2e_config import MASTER_KEY, POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import Result, Success, UnknownApiError
from guardrails_client import (
    ApplyGuardrailResponse,
    GuardrailMode,
    GuardrailsClient,
    PiiAction,
    PiiEntity,
    PresidioParamsBody,
)
from lifecycle import ResourceManager
from models import ChatResponse
from presidio_env import presidio_bases, scrub

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"

RAW_EMAIL = "dana.reyes@example.com"
RAW_PHONE = "+1 415-555-0134"
RAW_CARD = "4111 1111 1111 1111"
PII_SENTENCE = f"Reach Dana at {RAW_EMAIL} or on {RAW_PHONE} about card {RAW_CARD} today."

UNREACHABLE_BASE = "http://127.0.0.1:9/"

MAX_ECHO_TOKENS = 128


def _register(
    client: GuardrailsClient,
    resources: ResourceManager,
    *,
    name: str,
    entities: dict[PiiEntity, PiiAction],
    mode: GuardrailMode = "pre_call",
    filter_scope: Literal["input", "output", "both"] = "input",
    output_parse_pii: bool = False,
    analyzer_base: str | None = None,
    anonymizer_base: str | None = None,
) -> None:
    configured_analyzer, configured_anonymizer = presidio_bases()
    guardrail_id: Final = client.register(
        name,
        PresidioParamsBody(
            mode=mode,
            default_on=False,
            presidio_analyzer_api_base=analyzer_base or configured_analyzer,
            presidio_anonymizer_api_base=anonymizer_base or configured_anonymizer,
            presidio_filter_scope=filter_scope,
            output_parse_pii=output_parse_pii,
            pii_entities_config=entities,
        ),
    )
    resources.defer(lambda: client.delete_guardrail(guardrail_id))


def _apply(client: GuardrailsClient, name: str, text: str) -> str:
    """Run the guardrail over literal text and return what the model would see.

    /guardrails/apply_guardrail is a management route, so it is called with the
    master key like the rest of the apply surface. A replica that has not
    reloaded its config yet answers 404, which is in-flight propagation and
    worth retrying; every other refusal is a verdict and fails immediately
    rather than being waited out to the deadline.
    """
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    last: Result[ApplyGuardrailResponse] = client.apply_guardrail(MASTER_KEY, name=name, text=text)
    while not isinstance(last, Success):
        if not isinstance(last, UnknownApiError) or last.status_code != 404:
            pytest.fail(f"apply_guardrail refused to run {name!r}: {scrub(str(last))}")
        if time.monotonic() >= deadline:
            pytest.fail(f"apply_guardrail never ran {name!r} within {POLL_TIMEOUT}s: {scrub(str(last))}")
        time.sleep(POLL_INTERVAL)
        last = client.apply_guardrail(MASTER_KEY, name=name, text=text)
    return last.data.response_text


def _poll_until_refused(call: Callable[[], Result[ChatResponse]]) -> Result[ChatResponse]:
    """Retry a call a guardrail should refuse until it does, returning the last
    result. A call served right after the create ran on a worker that has not
    picked the guardrail up yet, which is propagation, not a guardrail that
    failed to refuse; one that is still served at the deadline is the failure."""
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    last: Result[ChatResponse] = call()
    while isinstance(last, Success) and time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        last = call()
    return last


def _first_content(response: ChatResponse) -> str:
    if not response.choices:
        return ""
    message: Final = response.choices[0].message
    return (message.content if message else None) or ""


class TestPresidioEntityFilterContract:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.masks_only_configured_entities",
        exercised_on=["chat_completions"],
    )
    def test_only_the_configured_entity_types_are_masked(
        self, client: GuardrailsClient, resources: ResourceManager
    ) -> None:
        """The configured entity filter must reach Presidio, not stay in our own
        bookkeeping.

        A guardrail configured for EMAIL_ADDRESS alone puts `entities` on the
        analyze request, so the phone number and card in the same sentence come
        back untouched. Stop sending that field and Presidio answers with
        everything it recognizes, which masks all three and fails here.
        """
        name: Final = f"e2e-presidio-filter-{unique_marker()}"
        _register(client, resources, name=name, entities={"EMAIL_ADDRESS": "MASK"})

        masked: Final = _apply(client, name, PII_SENTENCE)

        assert "<EMAIL_ADDRESS>" in masked, f"the configured entity must be masked, got {scrub(masked)!r}"
        assert RAW_EMAIL not in masked, f"the raw address must not survive masking, got {scrub(masked)!r}"
        assert RAW_PHONE in masked, (
            "PHONE_NUMBER was not configured, so it must reach the model untouched; the guardrail "
            f"masked more than it was asked to: {scrub(masked)!r}"
        )
        assert RAW_CARD in masked, (
            "CREDIT_CARD was not configured, so it must reach the model untouched; the guardrail "
            f"masked more than it was asked to: {scrub(masked)!r}"
        )

    @pytest.mark.covers(
        "guardrail.presidio.pre_call.masks_each_entity_in_place",
        exercised_on=["chat_completions"],
    )
    def test_every_detection_is_replaced_in_place_by_its_own_type_placeholder(
        self, client: GuardrailsClient, resources: ResourceManager
    ) -> None:
        """Presidio answers with one span per detection and we replace each one
        where it sits.

        Pinning the whole sentence is what makes this more than "something was
        redacted": spans applied at the wrong offsets would still remove the PII
        while eating the words around it, and a placeholder built from the wrong
        field would name the wrong type.
        """
        name: Final = f"e2e-presidio-placeholders-{unique_marker()}"
        _register(
            client,
            resources,
            name=name,
            entities={"EMAIL_ADDRESS": "MASK", "PHONE_NUMBER": "MASK", "CREDIT_CARD": "MASK"},
        )

        masked: Final = _apply(client, name, PII_SENTENCE)

        assert masked == "Reach Dana at <EMAIL_ADDRESS> or on <PHONE_NUMBER> about card <CREDIT_CARD> today.", (
            "each detection must be replaced by a placeholder naming its own entity type, in place, "
            f"leaving the rest of the sentence byte for byte; got {scrub(masked)!r}"
        )


class TestPresidioBlockContract:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_a_block_entity_refuses_the_request_while_clean_text_still_passes(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """BLOCK is decided by matching Presidio's entity_type against our config,
        so it proves we read the analyze response and not only its masked text.

        The clean prompt is the liveness half. Without it a guardrail that
        refused every request, or an analyzer erroring on everything, would look
        exactly like one that blocks the right thing.
        """
        name: Final = f"e2e-presidio-block-{unique_marker()}"
        _register(client, resources, name=name, entities={"CREDIT_CARD": "BLOCK"})

        marker: Final = unique_marker()
        blocked: Final = _poll_until_refused(
            lambda: client.chat(scoped_key, MODEL, f"{marker} Charge card {RAW_CARD}.", guardrails=[name])
        )
        assert not isinstance(blocked, Success), (
            f"a request carrying a CREDIT_CARD must be refused, but it was served: {scrub(str(blocked))}"
        )
        assert "CREDIT_CARD" in scrub(str(blocked)), (
            "the refusal must name the entity that caused it, so an operator can tell it from an "
            f"unrelated 4xx; got {scrub(str(blocked))}"
        )

        allowed: Final = client.chat(
            scoped_key,
            MODEL,
            f"{marker} Say the word hello and nothing else.",
            guardrails=[name],
            max_tokens=32,
        )
        assert isinstance(allowed, Success), (
            "text carrying no blocked entity must still be served by the same guardrail; it refused "
            f"everything, so the block above proves nothing: {scrub(str(allowed))}"
        )


class TestPresidioOutputParseContract:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.restores_masked_values",
        exercised_on=["chat_completions"],
    )
    def test_numbered_tokens_mask_the_prompt_and_are_restored_in_the_answer(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """With output_parse_pii each placeholder is numbered per detection and
        the caller gets the original values back.

        Both halves are ours and both are built from Presidio's spans: the
        numbering comes from sorting the analyze results left to right, and the
        restore puts back the text those offsets pointed at. The apply_guardrail
        call pins what the model receives, the chat call pins what the caller
        receives, and the caller only sees the real address if the
        token-to-value mapping survived the round trip.
        """
        name: Final = f"e2e-presidio-parse-{unique_marker()}"
        _register(
            client,
            resources,
            name=name,
            entities={"EMAIL_ADDRESS": "MASK", "PHONE_NUMBER": "MASK"},
            output_parse_pii=True,
        )

        to_the_model: Final = _apply(client, name, f"Dana is {RAW_EMAIL} on {RAW_PHONE}")
        assert to_the_model == "Dana is <EMAIL_ADDRESS_1> on <PHONE_NUMBER_2>", (
            "output_parse_pii numbers each placeholder in the order its entity appears, which is what "
            f"makes the restore reversible; got {scrub(to_the_model)!r}"
        )

        restored: Final = _poll_until_restored(client, scoped_key, name, _echo_prompt(unique_marker()))
        assert RAW_EMAIL in restored, (
            "the caller must get the real address back, not the placeholder the model saw; a token left "
            f"in the answer means the mapping was lost between the two hooks: {scrub(restored)!r}"
        )
        assert "EMAIL_ADDRESS_1" not in restored, (
            f"no numbered token may survive into the caller's response: {scrub(restored)!r}"
        )


def _echo_prompt(marker: str) -> str:
    """A transcription framing rather than "repeat this back".

    The model is handed the numbered placeholders. Asked to repeat them, a model
    may instead explain that it will not echo someone's contact details, which
    leaves the restore nothing to act on and turns a product assertion into a
    test of the model's mood. Framed as transcription, the line comes back
    verbatim.
    """
    return (
        f"{marker} You are a text transcription tool. Output the input text character for character, "
        f"with no commentary and no explanation. Input: Dana is {RAW_EMAIL} on {RAW_PHONE}"
    )


def _poll_until_restored(client: GuardrailsClient, key: str, name: str, prompt: str) -> str:
    """Retry until the guardrail has attached and the model has echoed the line.

    The applied-guardrails header is the liveness gate: without it a response
    carrying the raw address would be indistinguishable from one the guardrail
    never touched, and the restore assertion would pass vacuously.
    """
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    last = "<no successful response yet>"  # rebind-ok: last-observation accumulator for the failure message
    while True:
        outcome = client.chat_raw(key, MODEL, prompt, guardrails=[name], max_tokens=MAX_ECHO_TOKENS)
        if outcome.ok and name in outcome.headers.get("x-litellm-applied-guardrails", ""):
            last = _first_content(ChatResponse.model_validate_json(outcome.body))
            if RAW_EMAIL in last or "EMAIL_ADDRESS" in last:
                return last
        else:
            last = f"<guardrail not applied: HTTP {outcome.status_code}>"
        if time.monotonic() >= deadline:
            pytest.fail(
                f"the guarded call never came back with the model's echo within {POLL_TIMEOUT}s; "
                f"last observation: {scrub(last)[:300]!r}"
            )
        time.sleep(POLL_INTERVAL)


class TestPresidioUnreachableContract:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.fails_closed_when_unreachable",
        exercised_on=["chat_completions"],
    )
    def test_an_unreachable_analyzer_refuses_the_request_instead_of_forwarding_raw_pii(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """A Presidio outage must not degrade into "no masking today".

        The guardrail is pointed at a closed local port so the analyzer call
        cannot connect. With PII entities configured the guardrail fails closed
        and the prompt never reaches the model with the address still in it. The
        error goes straight back to the caller, so it is also asserted to carry
        neither the raw address nor the endpoint it failed to reach.
        """
        name: Final = f"e2e-presidio-unreachable-{unique_marker()}"
        _register(
            client,
            resources,
            name=name,
            entities={"EMAIL_ADDRESS": "MASK"},
            analyzer_base=UNREACHABLE_BASE,
            anonymizer_base=UNREACHABLE_BASE,
        )

        marker: Final = unique_marker()
        outcome: Final = _poll_until_refused(
            lambda: client.chat(scoped_key, MODEL, f"{marker} Repeat: {RAW_EMAIL}", guardrails=[name])
        )

        assert not isinstance(outcome, Success), (
            "with the analyzer unreachable the guardrail cannot know whether the prompt holds PII, so "
            f"the request must be refused rather than forwarded unmasked: {scrub(str(outcome))}"
        )
        body: Final = scrub(str(outcome))
        assert RAW_EMAIL not in body, f"the failure must not echo the prompt's PII back to the caller: {body}"
        assert "Presidio" in body, (
            f"the refusal must say the guardrail could not run, or an operator cannot tell an outage "
            f"from a model error: {body}"
        )
