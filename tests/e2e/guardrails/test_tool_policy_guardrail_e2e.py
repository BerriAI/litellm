"""Live e2e: the tool_policy guardrail enforces a virtual key's blocked-tool list.

Where tool_permission carries its own allow-list, tool_policy reads the tool
registry: an admin marks a tool blocked for one virtual key with POST
/v1/tool/policy, and the guardrail resolves that key's effective policy pre-call.
A chat completion from that key carrying the blocked tool must be rejected before
the upstream model runs, with the response naming the offending tool. A sibling
tool left at the default `untrusted` policy, sent by the same key through the same
guardrail, must still reach the model; otherwise the rejection would prove nothing
beyond the guardrail refusing all tool use.

The override is scoped to this test's own key and deleted on teardown, and the
guardrail is opted into per request (`default_on=False`), so nothing here reaches
unrelated traffic on the shared proxy.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from guardrails_client import GuardrailsClient
from lifecycle import ResourceManager
from native_guardrails import (
    ToolPolicyParamsBody,
    block_tool_for_key,
    called_tool_names,
    chat,
    function_tool,
    register_guardrail,
    unblock_tool_for_key,
)

pytestmark = pytest.mark.e2e


class TestToolPolicyGuardrail:
    @pytest.mark.covers(
        "guardrail.tool_policy.pre_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_blocked_tool_is_rejected_while_a_sibling_tool_passes(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker = unique_marker()
        blocked_tool = f"wire_transfer_{marker}"
        allowed_tool = f"get_weather_{marker}"

        name = f"e2e-tool-policy-{marker}"
        guardrail_id = register_guardrail(
            client, name, ToolPolicyParamsBody(mode="pre_call", default_on=False)
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        block_tool_for_key(client, tool_name=blocked_tool, key=scoped_key)
        resources.defer(lambda: unblock_tool_for_key(client, tool_name=blocked_tool, key=scoped_key))

        blocked = chat(
            client,
            scoped_key,
            prompt="Send $10,000 to account 12345.",
            tools=[function_tool(blocked_tool)],
            guardrail_name=name,
        )
        match blocked:
            case UnknownApiError(status_code=400, body=body):
                assert blocked_tool in body, (
                    "the rejection must name the tool the policy blocked, so the caller knows "
                    f"which tool to drop; got: {body[:400]}"
                )
                assert "Violated tool policy" in body, (
                    f"the rejection must come from the tool policy, got: {body[:400]}"
                )
            case UnknownApiError(status_code=status, body=body):
                pytest.fail(f"expected a 400 tool-policy block, got {status}: {body[:400]}")
            case _:
                pytest.fail(
                    "tool_policy let a request through carrying a tool blocked for this key; "
                    f"got {blocked}"
                )

        allowed = unwrap(
            chat(
                client,
                scoped_key,
                prompt="What is the weather in Paris right now?",
                tools=[function_tool(allowed_tool)],
                guardrail_name=name,
                force_tool_call=True,
            )
        )
        assert called_tool_names(allowed) == [allowed_tool], (
            "a tool the key has no block on must still reach the model through the same "
            f"guardrail, but the response called {called_tool_names(allowed)}"
        )
