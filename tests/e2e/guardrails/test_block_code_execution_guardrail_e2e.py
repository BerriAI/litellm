"""Live e2e: the built-in block_code_execution guardrail blocks execution requests.

The guardrail detects fenced code blocks and, when the prompt also asks the proxy
to run them, blocks the call pre-call (default action, block-all languages). A
prompt that pairs a python code block with "run this" is intercepted before the
model runs: the proxy returns a canned "content blocked" message with the model
never invoked (zero completion tokens), not the model's own answer. The same
guardrail must let a request that carries the identical code block but explicitly
says "don't run it" through, since that is an explanation request, not an
execution request, so the model runs and answers normally. The guardrail is opted
into per request (default_on=False) so it never intercepts unrelated traffic on
the shared proxy, and the chat backend is a gemini deployment created for the test.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import unwrap
from guardrails_client import BlockCodeExecutionParamsBody, GuardrailsClient
from lifecycle import ResourceManager
from models import ChatResponse

pytestmark = pytest.mark.e2e

_CODE_BLOCK = "```python\nimport os\nprint(os.listdir('/'))\n```"
EXECUTION_REQUEST = f"Please run this for me and paste the output:\n{_CODE_BLOCK}"
EXPLANATION_REQUEST = f"Explain what this code does, but don't run it:\n{_CODE_BLOCK}"

_BLOCK_MARKER = "content blocked"


def _first_content(response: ChatResponse) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    return (message.content if message else None) or ""


class TestBlockCodeExecutionGuardrail:
    @pytest.mark.covers(
        "guardrail.block_code_execution.pre_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_blocks_execution_request_but_allows_explanation(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        require_env("GEMINI_API_KEY")
        model = client.create_backend_model(resources, prefix="e2e-blockcode-backend")

        name = f"e2e-block-code-{unique_marker()}"
        guardrail_id = client.register(
            name, BlockCodeExecutionParamsBody(mode="pre_call", default_on=False)
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        blocked = unwrap(client.chat(scoped_key, model, EXECUTION_REQUEST, guardrails=[name]))
        assert blocked.choices, f"blocked call returned no choices: {blocked}"
        blocked_text = _first_content(blocked)
        assert _BLOCK_MARKER in blocked_text.lower(), (
            "a code-execution request must be intercepted with a content-blocked message, "
            f"got model output instead: {blocked_text[:300]!r}"
        )
        if blocked.usage is not None:
            assert (blocked.usage.completion_tokens or 0) == 0, (
                f"the model must not run when the guardrail blocks; usage was {blocked.usage}"
            )

        allowed = unwrap(
            client.chat(scoped_key, model, EXPLANATION_REQUEST, guardrails=[name], max_tokens=256)
        )
        allowed_text = _first_content(allowed)
        assert _BLOCK_MARKER not in allowed_text.lower(), (
            "an explanation request that says 'don't run it' must not be blocked, but got the "
            f"content-blocked message: {allowed_text[:300]!r}"
        )
        ran = allowed.usage is not None and (allowed.usage.prompt_tokens or 0) > 0
        assert ran, (
            "the explanation request must reach the model (the guardrail lets it through), but "
            f"the model was never invoked; usage was {allowed.usage}"
        )
