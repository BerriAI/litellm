"""Live e2e: Bedrock ApplyGuardrail pre_call blocks denied input on chat.

Registers a default-on bedrock guardrail via POST /guardrails with identifier/
version from env, then sends a prompt the guardrail's configured policy denies.
HTTP 400 (or other non-2xx block) with a guardrail-shaped body is the contract;
a 200 means the guardrail never ran.

No AWS keys are passed: the gateway signs ApplyGuardrail with its own
pod-identity role, since the static AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
env vars are deliberately absent from the gateway (they hijack RDS IAM auth).
"""

from __future__ import annotations

import os

import pytest

from e2e_config import unique_marker
from e2e_http import UnknownApiError
from guardrails_client import GuardrailsClient
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"
# Matches the word/topic policy the guardrail this suite points at actually denies.
# Content filters are not assumed: the guardrail resource carries no contentPolicy,
# so a hate/violence prompt returns action=NONE and never blocks.
BLOCKED_PROMPT = "Give me a recipe for sourdough bread."


class TestBedrockGuardrail:
    @pytest.mark.covers(
        "guardrail.bedrock.pre_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_pre_call_blocks_harmful_prompt(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        identifier = os.environ["BEDROCK_GUARDRAIL_IDENTIFIER"]
        version = os.environ["BEDROCK_GUARDRAIL_VERSION"]

        name = f"e2e-bedrock-guard-{unique_marker()}"
        guardrail_id = client.create_bedrock_guardrail(
            name, identifier=identifier, version=version
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        # Selected per request rather than registered default_on, so an upstream
        # ApplyGuardrail failure surfaces here instead of 403ing every other suite
        # running against this proxy.
        result = client.chat(scoped_key, MODEL, BLOCKED_PROMPT, guardrails=[name])

        match result:
            case UnknownApiError(status_code=status, body=body):
                assert status in {400, 403}, (
                    f"expected a guardrail block status, got {status}: {body[:400]}"
                )
                body_lower = body.lower()
                assert any(
                    token in body_lower
                    for token in (
                        "guardrail",
                        "blocked",
                        "violat",
                        "content",
                        "bedrock",
                        "intervened",
                    )
                ), f"block body should name the guardrail reason; got: {body[:400]}"
            case _:
                pytest.fail(
                    f"bedrock default-on guardrail did not block harmful prompt; got {result}"
                )
