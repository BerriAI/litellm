"""Live e2e: the Presidio PII guardrail masks, per its configured hook point.

pre_call: the guardrail calls the Presidio analyzer/anonymizer on the request
messages BEFORE the model runs, so the model only ever sees placeholders like
<EMAIL_ADDRESS>. A prompt asking the model to repeat a fake email + phone back
must come back with the placeholders echoed and the raw PII absent, on
/chat/completions and on /v1/messages (Anthropic format).

The analyzer/anonymizer endpoints come from PRESIDIO_ANALYZER_API_BASE /
PRESIDIO_ANONYMIZER_API_BASE; missing env is a hard failure, never a skip.
Each guardrail registers with presidio_filter_scope="input" so only the
configured hook's callback exists (the default "both" adds a second post_call
output masker), and is deleted on teardown.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, Success
from guardrails_client import GuardrailsClient, PresidioParamsBody
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
) -> None:
    analyzer, anonymizer = _presidio_bases()
    guardrail_id = client.register(
        name,
        PresidioParamsBody(
            mode="pre_call",
            default_on=False,
            presidio_analyzer_api_base=analyzer,
            presidio_anonymizer_api_base=anonymizer,
            presidio_filter_scope="input",
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
