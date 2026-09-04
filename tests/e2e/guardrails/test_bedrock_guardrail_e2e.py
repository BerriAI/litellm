"""Live e2e: Bedrock ApplyGuardrail blocks on chat, pre_call and post_call.

pre_call registers a bedrock guardrail via POST /guardrails with identifier/
version from env, then sends a prompt the guardrail's configured policy denies.
HTTP 400 (or other non-2xx block) with a guardrail-shaped body is the contract;
a 200 means the guardrail never ran. post_call scans the MODEL OUTPUT only, so
its test makes the model echo the word the guardrail's word policy denies
(BEDROCK_GUARDRAIL_BLOCKED_WORD, default FORBIDDENWORD) and the block must
arrive without leaking the model's text.

No AWS keys are passed: the gateway signs ApplyGuardrail with its own
pod-identity role, since the static AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
env vars are deliberately absent from the gateway (they hijack RDS IAM auth).
"""

from __future__ import annotations

import json
import os
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import UnknownApiError
from guardrails_client import (
    BedrockGuardrailParamsBody,
    GuardrailsClient,
    poll_until_blocked,
)
from lifecycle import ResourceManager
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.e2e

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _without_assessments(value: JsonValue) -> JsonValue:
    """The assessments echo guardrail CONFIG, not content: the stage guardrail's
    topic policy is itself named after the denied word, so its label lands in
    every assessment listing and would trip a leak check aimed at model output."""
    if isinstance(value, dict):
        return {key: _without_assessments(child) for key, child in value.items() if key != "assessments"}
    if isinstance(value, list):
        return [_without_assessments(item) for item in value]
    return value


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
        guardrail_id = client.create_bedrock_guardrail(name, identifier=identifier, version=version)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        # Selected per request rather than registered default_on, so an upstream
        # ApplyGuardrail failure surfaces here instead of 403ing every other suite
        # running against this proxy.
        result = poll_until_blocked(lambda: client.chat(scoped_key, MODEL, BLOCKED_PROMPT, guardrails=[name]))

        match result:
            case UnknownApiError(status_code=status, body=body):
                assert status in {400, 403}, f"expected a guardrail block status, got {status}: {body[:400]}"
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
                pytest.fail(f"bedrock default-on guardrail did not block harmful prompt; got {result}")

    @pytest.mark.covers(
        "guardrail.bedrock.post_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_post_call_blocks_denied_model_output(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        identifier = os.environ["BEDROCK_GUARDRAIL_IDENTIFIER"]
        version = os.environ["BEDROCK_GUARDRAIL_VERSION"]
        blocked_word = os.environ.get("BEDROCK_GUARDRAIL_BLOCKED_WORD", "FORBIDDENWORD")

        name = f"e2e-bedrock-post-{unique_marker()}"
        guardrail_id = client.register(
            name,
            BedrockGuardrailParamsBody(
                mode="post_call",
                default_on=False,
                guardrailIdentifier=identifier,
                guardrailVersion=version,
            ),
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        # post_call scans OUTPUT only, so the denied word has to come out of the
        # model: ask it to echo the word verbatim. The word in the prompt itself
        # is not scanned in this mode.
        prompt = f"Reply with exactly this one word and nothing else: {blocked_word}"
        result = poll_until_blocked(lambda: client.chat(scoped_key, MODEL, prompt, guardrails=[name], max_tokens=128))

        match result:
            case UnknownApiError(status_code=status, body=body):
                # A policy block is a 400 naming the verdict; a failed
                # ApplyGuardrail call surfaces as 403 "guardrail request
                # failed", which must not count as a block.
                assert status == 400, f"expected the guardrail block status 400, got {status}: {body[:400]}"
                body_lower = body.lower()
                assert any(token in body_lower for token in ("violated", "blocked", "intervened")), (
                    f"block body should name the guardrail verdict; got: {body[:400]}"
                )
                assert blocked_word not in json.dumps(_without_assessments(_JSON.validate_json(body))), (
                    f"the blocked model output must not leak into the error body; got: {body[:400]}"
                )
            case _:
                pytest.fail(f"bedrock post_call guardrail did not block denied model output; got {result}")
