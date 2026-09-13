"""Live e2e: the Bedrock guardrail at its two non-pre_call hook points.

`test_bedrock_guardrail_e2e.py` covers pre_call. The same real AWS guardrail is
wired here at `during_call` (async_moderation_hook, which scans the INPUT while
the LLM call runs) and at `post_call` (async_post_call_success_hook, which scans
only the model's OUTPUT).

Each block is asserted through `provider_specific_fields.guardrail_mode`, so a
test can never claim a hook point the gateway did not actually run. Each hook is
also driven with the prompt the *other* hook catches, which is where the two
differ in production: the denied word `cake` reaches the caller untouched under
`during_call` (nothing scans the output) and is blocked under `post_call`.

No AWS keys are passed: the gateway signs ApplyGuardrail and Converse with its
own credentials, so only the region reference travels in the deployment body.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, Success, UnknownApiError
from guardrails_client import BedrockGuardrailParamsBody, GuardrailMode, GuardrailsClient
from lifecycle import ResourceManager
from models import ChatResponse, LiteLLMParamsBody

pytestmark = pytest.mark.e2e

BEDROCK_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

# The guardrail this suite points at denies the topic "bread" and the custom
# words "bread" and "cake", on input and on output alike.
DENIED_INPUT_PROMPT = "Give me a recipe for sourdough bread."

# Carries no denied word or topic itself, so every input scan lets it through;
# the answer the model writes is the denied word, so only an output scan catches
# it. That asymmetry is what separates post_call from pre_call/during_call.
DENIED_OUTPUT_PROMPT = (
    "Reverse the letters of this string and reply with only the reversed string, nothing else: ekac"
)
DENIED_OUTPUT_WORD = "cake"


class BlockMatch(BaseModel):
    action: str | None = None


class BlockAssessment(BaseModel):
    policy: str | None = None
    matches: list[BlockMatch] = []


class BlockFields(BaseModel):
    guardrail_name: str | None = None
    guardrail_mode: str | None = None
    assessments: list[BlockAssessment] = []


class BlockError(BaseModel):
    message: str
    provider_specific_fields: BlockFields | None = None


class GuardrailBlockBody(BaseModel):
    """The 400 body the gateway returns when a bedrock guardrail intervenes."""

    error: BlockError


def _assert_blocked_by(result: Result[ChatResponse], *, name: str, mode: GuardrailMode) -> None:
    match result:
        case UnknownApiError(status_code=status, body=body):
            assert status == 400, f"expected a guardrail block status, got {status}: {body[:400]}"
            blocked = GuardrailBlockBody.model_validate_json(body)
            fields = blocked.error.provider_specific_fields
            assert fields is not None, f"block body carried no guardrail detail: {body[:400]}"
            assert fields.guardrail_name == name, (
                f"block came from guardrail {fields.guardrail_name!r}, expected {name!r}"
            )
            assert fields.guardrail_mode == mode, (
                f"block ran at hook {fields.guardrail_mode!r}, expected {mode!r}"
            )
            assert any(
                match.action == "BLOCKED" for assessment in fields.assessments for match in assessment.matches
            ), f"block body reported no BLOCKED assessment: {body[:400]}"
        case _:
            pytest.fail(f"bedrock {mode} guardrail did not block; got {result}")


def _assert_answered(result: Result[ChatResponse], *, contains: str) -> None:
    match result:
        case Success(data=response):
            assert response.choices, f"model returned no choices: {response}"
            message = response.choices[0].message
            assert message is not None and message.content is not None, (
                f"model returned no content: {response}"
            )
            content = message.content
            assert contains in content.lower(), f"expected {contains!r} in the answer, got {content!r}"
        case _:
            pytest.fail(f"request should have been served, got {result}")


class TestBedrockGuardrailHooks:
    def _bedrock_model(self, client: GuardrailsClient, resources: ResourceManager) -> str:
        model_name = f"e2e-bedrock-guard-backend-{unique_marker()}"
        model_id = client.proxy.create_model(
            model_name,
            LiteLLMParamsBody(model=BEDROCK_BACKEND, aws_region_name="os.environ/AWS_REGION"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        return model_name

    def _guardrail(self, client: GuardrailsClient, resources: ResourceManager, mode: GuardrailMode) -> str:
        name = f"e2e-bedrock-{mode}-{unique_marker()}"
        guardrail_id = client.register(
            name,
            BedrockGuardrailParamsBody(
                mode=mode,
                default_on=False,
                guardrailIdentifier=os.environ["BEDROCK_GUARDRAIL_IDENTIFIER"],
                guardrailVersion=os.environ["BEDROCK_GUARDRAIL_VERSION"],
            ),
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))
        return name

    @pytest.mark.covers(
        "guardrail.bedrock.during.blocks",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_during_call_blocks_denied_input(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = self._bedrock_model(client, resources)
        name = self._guardrail(client, resources, "during_call")

        blocked = client.chat(scoped_key, model, DENIED_INPUT_PROMPT, guardrails=[name])
        _assert_blocked_by(blocked, name=name, mode="during_call")

        served = client.chat(scoped_key, model, DENIED_OUTPUT_PROMPT, guardrails=[name], max_tokens=64)
        _assert_answered(served, contains=DENIED_OUTPUT_WORD)

    @pytest.mark.covers(
        "guardrail.bedrock.post_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_post_call_blocks_denied_output(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = self._bedrock_model(client, resources)
        pre_call_name = self._guardrail(client, resources, "pre_call")
        post_call_name = self._guardrail(client, resources, "post_call")

        served = client.chat(
            scoped_key, model, DENIED_OUTPUT_PROMPT, guardrails=[pre_call_name], max_tokens=64
        )
        _assert_answered(served, contains=DENIED_OUTPUT_WORD)

        blocked = client.chat(
            scoped_key, model, DENIED_OUTPUT_PROMPT, guardrails=[post_call_name], max_tokens=64
        )
        _assert_blocked_by(blocked, name=post_call_name, mode="post_call")
