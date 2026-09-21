"""Live e2e: a Bedrock guardrail in during_call mode blocks a streamed chat.

during_call runs the Bedrock ApplyGuardrail INPUT scan in an asyncio.gather
alongside the LLM call (common_request_processing.py); when the scan flags the
prompt, the raised block cancels the LLM task before the stream ever starts, so
the client sees a non-2xx JSON error - not an SSE stream, not an in-stream
error frame - and zero content chunks are delivered.

The prompt deliberately contains the exact word the guardrail's word policy
denies (BEDROCK_GUARDRAIL_BLOCKED_WORD, default FORBIDDENWORD), so the INPUT
scan intervenes deterministically. Identifier/version come from
BEDROCK_GUARDRAIL_IDENTIFIER / BEDROCK_GUARDRAIL_VERSION like the rest of the
bedrock suite; no AWS keys are passed (the gateway signs with pod identity).
The guardrail registers default_on=False and is selected per request, so an
upstream ApplyGuardrail failure surfaces here instead of 403ing other suites.
"""

from __future__ import annotations

import os

import pytest

from e2e_config import unique_marker
from guardrails_client import (
    BedrockGuardrailParamsBody,
    GuardrailsClient,
    poll_until_blocked_stream,
)
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"


class TestBedrockDuringCallStreaming:
    @pytest.mark.covers(
        "guardrail.bedrock.during.blocks",
        exercised_on=["chat_completions"],
    )
    def test_during_call_blocks_stream_before_first_chunk(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        identifier = os.environ["BEDROCK_GUARDRAIL_IDENTIFIER"]
        version = os.environ["BEDROCK_GUARDRAIL_VERSION"]
        blocked_word = os.environ.get("BEDROCK_GUARDRAIL_BLOCKED_WORD", "FORBIDDENWORD")

        name = f"e2e-bedrock-during-{unique_marker()}"
        guardrail_id = client.register(
            name,
            BedrockGuardrailParamsBody(
                mode="during_call",
                default_on=False,
                guardrailIdentifier=identifier,
                guardrailVersion=version,
            ),
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        # The denied word sits in the INPUT: during_call scans the request
        # messages while the model call runs, and the flag must win the race
        # by cancelling the stream outright.
        prompt = f"Please use the word {blocked_word} in a sentence."
        result = poll_until_blocked_stream(
            lambda: client.chat_stream_raw(scoped_key, MODEL, prompt, guardrails=[name], max_tokens=64)
        )

        assert not result.ok, (
            f"the during_call guardrail never blocked the streamed request; got a "
            f"{result.status_code} with {result.chunks} chunks"
        )
        assert result.status_code == 400, (
            f"a during_call block surfaces as HTTP 400 before the stream starts, got "
            f"{result.status_code}: {result.body[:400]}"
        )
        assert result.chunks == 0 and not result.stream_events, (
            f"no content chunk may be delivered on a during_call block, but "
            f"{result.chunks} chunks arrived: {result.stream_events[:3]}"
        )
        assert "text/event-stream" not in (result.content_type or ""), (
            f"the block must be a JSON error response, not an SSE stream; got content-type {result.content_type!r}"
        )
        body_lower = result.body.lower()
        assert any(token in body_lower for token in ("guardrail", "violated", "blocked", "bedrock", "intervened")), (
            f"block body should name the guardrail reason; got: {result.body[:400]}"
        )
