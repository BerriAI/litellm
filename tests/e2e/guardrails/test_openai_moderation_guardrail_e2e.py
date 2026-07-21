"""Live e2e: the built-in OpenAI moderation guardrail blocks flagged input pre-call.

Registers an `openai_moderation` guardrail (default_on=False) and opts into it for
one request via the chat body's `guardrails` selector, so it never intercepts
unrelated traffic on the shared proxy. The guardrail calls OpenAI's moderation API
before the upstream model runs; a prompt that trips the policy must be rejected
with HTTP 400 naming the moderation policy, and the same guardrail must let a
benign prompt through. The chat backend is a gemini deployment created for the
test (and torn down); moderation runs independently of it, so the block is
attributable to the guardrail, not the model.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import UnknownApiError, unwrap
from guardrails_client import GuardrailsClient, OpenAIModerationParamsBody
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

# A direct threat of violence reliably trips OpenAI's harassment/violence categories.
FLAGGED_PROMPT = "I am going to find you and violently kill you and your entire family tonight."
BENIGN_PROMPT = "Reply with the single word: hello"


class TestOpenAIModerationGuardrail:
    @pytest.mark.covers(
        "guardrail.openai_moderations.pre_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_moderation_blocks_flagged_input(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        require_env("OPENAI_API_KEY", "GEMINI_API_KEY")
        model = client.create_backend_model(resources, prefix="e2e-moderation-backend")

        name = f"e2e-openai-moderation-{unique_marker()}"
        guardrail_id = client.register(
            name,
            OpenAIModerationParamsBody(
                mode="pre_call", default_on=False, api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        blocked = client.chat(scoped_key, model, FLAGGED_PROMPT, guardrails=[name])
        match blocked:
            case UnknownApiError(status_code=status, body=body):
                assert status == 400, (
                    f"a flagged prompt must be blocked with 400, got {status}: {body[:400]}"
                )
                assert "moderation" in body.lower(), (
                    f"the block body must name the moderation policy, got: {body[:400]}"
                )
            case _:
                pytest.fail(
                    f"openai moderation did not block a flagged prompt; got {blocked}"
                )

        allowed = unwrap(client.chat(scoped_key, model, BENIGN_PROMPT, guardrails=[name]))
        assert allowed.choices, (
            "the same moderation guardrail must let a benign prompt through, but the "
            f"call returned no choices: {allowed}"
        )
