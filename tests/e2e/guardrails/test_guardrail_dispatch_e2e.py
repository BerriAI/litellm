"""Live e2e: the per-request `guardrails` selector must fail closed.

A request that names a guardrail is a caller asking for protection. When the
proxy does not serve that name (a typo, a deleted guardrail, or a worker that
never loaded it), answering 200 silently drops the protection the caller asked
for; the contract this test pins is a 4xx naming the unknown guardrail.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import UnknownApiError, ValidationError
from guardrails_client import GuardrailsClient

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"


@pytest.mark.skip(
    reason=(
        "stage red: product gap, a request naming a guardrail the proxy does not "
        "serve is silently served unguarded (200) instead of failing closed"
    )
)
@pytest.mark.covers(
    "guardrail.dispatch.pre_call.rejects_unknown_name",
    exercised_on=["chat_completions"],
)
def test_request_naming_an_unknown_guardrail_fails_closed(client: GuardrailsClient, scoped_key: str) -> None:
    result = client.chat(scoped_key, MODEL, "say hi", guardrails=[f"e2e-no-such-guardrail-{unique_marker()}"])

    match result:
        case UnknownApiError(status_code=status, body=body):
            assert status == 400, f"expected a 400 for an unknown guardrail name, got {status}: {body[:400]}"
            assert "guardrail" in body.lower(), f"the rejection should name the guardrail; got: {body[:400]}"
        case ValidationError(message=message):
            assert "guardrail" in message.lower(), f"the rejection should name the guardrail; got: {message[:400]}"
        case _:
            pytest.fail(f"a request naming an unknown guardrail must fail closed with a 4xx; got {result}")
