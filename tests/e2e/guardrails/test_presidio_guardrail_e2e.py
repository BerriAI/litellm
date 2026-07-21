"""Live e2e: the built-in Presidio PII guardrail masks PII on the request and on
the model output.

Presidio replaces detected PII with `<ENTITY_TYPE>` placeholders (e.g.
`<EMAIL_ADDRESS>`) via a real analyzer + anonymizer. Two modes are checked
independently, each opted into per request (default_on=False) so it never touches
unrelated traffic:

- pre_call: the prompt is anonymized before it reaches the model, so a
  repeat-verbatim request comes back with the placeholder, never the raw email
- post_call (apply_to_output): PII the model itself emits is masked on the way
  out, so the caller never receives the raw value the model produced

Analyzer/anonymizer bases come from PRESIDIO_ANALYZER_API_BASE /
PRESIDIO_ANONYMIZER_API_BASE (compose provides the in-network hosts; point them at
locally published container ports for a host run). The chat backend is a gemini
deployment created for the test; masking is independent of it.

Not covered here: the third registry cell guardrail.presidio.logging_only.masks.
Its contract (mask what is logged, leave the response untouched) is not
observable through the harness - /spend/logs returns an empty `messages` object,
so the masked prompt cannot be read back, and a logging_only run was observed to
also mask the response, which contradicts the contract. Needs product/registry
clarification (candidate Linear ticket) before a truthful test can be written.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import unwrap
from guardrails_client import GuardrailMode, GuardrailsClient, PresidioParamsBody
from lifecycle import ResourceManager
from models import ChatResponse

pytestmark = pytest.mark.e2e

RAW_EMAIL = "alice.example.person@example.com"
PLACEHOLDER = "<EMAIL_ADDRESS>"

ECHO_REQUEST = f"Repeat the following text back exactly, verbatim, with no changes: My email is {RAW_EMAIL}"
EMIT_REQUEST = f"Output exactly this one line and nothing else: Please contact {RAW_EMAIL} today"


def _content(response: ChatResponse) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    return (message.content if message else None) or ""


def _presidio_params(mode: GuardrailMode, *, apply_to_output: bool = False) -> PresidioParamsBody:
    analyzer, anonymizer = require_env(
        "PRESIDIO_ANALYZER_API_BASE", "PRESIDIO_ANONYMIZER_API_BASE"
    )
    return PresidioParamsBody(
        mode=mode,
        default_on=False,
        presidio_analyzer_api_base=analyzer,
        presidio_anonymizer_api_base=anonymizer,
        apply_to_output=apply_to_output,
    )


class TestPresidioGuardrail:
    @pytest.mark.covers(
        "guardrail.presidio.pre_call.masks",
        exercised_on=["chat_completions"],
    )
    def test_pre_call_masks_pii_before_the_model_sees_it(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        require_env("GEMINI_API_KEY")
        model = client.create_backend_model(resources, prefix="e2e-presidio-pre")
        name = f"e2e-presidio-pre-{unique_marker()}"
        guardrail_id = client.register(name, _presidio_params("pre_call"))
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        echoed = _content(
            unwrap(client.chat(scoped_key, model, ECHO_REQUEST, guardrails=[name], max_tokens=128))
        )
        assert RAW_EMAIL not in echoed, (
            "pre_call masking must strip the raw email before the model sees it, but the "
            f"model echoed it back: {echoed[:300]!r}"
        )
        assert PLACEHOLDER in echoed, (
            "the model should have echoed the masked placeholder the guardrail substituted, "
            f"got: {echoed[:300]!r}"
        )

    @pytest.mark.covers(
        "guardrail.presidio.post_call.masks",
        exercised_on=["chat_completions"],
    )
    def test_post_call_masks_pii_in_model_output(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        require_env("GEMINI_API_KEY")
        model = client.create_backend_model(resources, prefix="e2e-presidio-post")
        name = f"e2e-presidio-post-{unique_marker()}"
        guardrail_id = client.register(name, _presidio_params("post_call", apply_to_output=True))
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        out = _content(
            unwrap(client.chat(scoped_key, model, EMIT_REQUEST, guardrails=[name], max_tokens=128))
        )
        assert RAW_EMAIL not in out, (
            "post_call masking must strip PII the model emitted, but the raw email reached the "
            f"caller: {out[:300]!r}"
        )
        assert PLACEHOLDER in out, (
            f"the masked placeholder should replace the model's PII output, got: {out[:300]!r}"
        )
