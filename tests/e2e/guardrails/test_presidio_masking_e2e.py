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
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Literal

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, Success
from guardrails_client import GuardrailMode, GuardrailsClient, PiiAction, PiiEntity, PresidioParamsBody
from lifecycle import ResourceManager
from models import AnthropicMessagesResponse, ChatResponse

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

# Fictional NANP 555 number; a standard format Presidio's phone recognizer detects.
FAKE_PHONE = "+1 415-555-0134"


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
    mode: GuardrailMode = "pre_call",
    filter_scope: Literal["input", "output", "both"] = "input",
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
