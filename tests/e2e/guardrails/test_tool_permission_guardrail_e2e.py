"""Live e2e: the tool_permission guardrail gates which tools a request may declare.

The guardrail is registered `mode="pre_call"` with `default_action="deny"`, so its
rules are an allow-list applied to the tools the CALLER declares, before the model
runs. Two halves of one product promise:

- blocks: a request declaring a tool outside the allow-list is rejected with a 400
  naming the denied tool, and never reaches the model
- allows: a request declaring only the permitted tool is served normally, comes
  back with a real tool call for that tool, and carries an
  `x-litellm-applied-guardrails` header naming the guardrail, which is what
  separates "the guardrail ran and allowed it" from "the guardrail was never
  attached". `tool_choice="required"` keeps the model from answering directly and
  making the outcome depend on its mood

No vendor API is involved: `tool_permission` is a built-in guardrail, so the
verdict comes from the proxy itself.
"""

from __future__ import annotations

from typing import Final

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse, UnknownApiError
from guardrails_client import (
    GuardrailsClient,
    ToolPermissionParamsBody,
    ToolPermissionRuleBody,
    poll_until_blocked,
)
from lifecycle import ResourceManager
from models import ChatResponse, ChatTool, ChatToolFunction

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"

#: The one tool the guardrail permits, and one it does not. Both are declared by
#: the caller in the request body; the guardrail reads them there.
ALLOWED_TOOL: Final = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
)
DENIED_TOOL: Final = ChatTool(
    function=ChatToolFunction(
        name="delete_customer_database",
        description="Permanently delete the customer database",
        parameters={"type": "object", "properties": {}},
    )
)

TOOL_PROMPT: Final = "What is the weather in Paris right now?"


def _register_tool_permission(client: GuardrailsClient, resources: ResourceManager, *, name: str) -> None:
    """Allow-list exactly one tool: everything else falls to `default_action=deny`
    and, with `on_disallowed_action=block`, is rejected outright."""
    guardrail_id = client.register(
        name,
        ToolPermissionParamsBody(
            mode="pre_call",
            default_on=False,
            default_action="deny",
            on_disallowed_action="block",
            rules=[
                ToolPermissionRuleBody(
                    id="allow-get-weather",
                    tool_name=ALLOWED_TOOL.function.name,
                    decision="allow",
                )
            ],
        ),
    )
    resources.defer(lambda: client.delete_guardrail(guardrail_id))


def _applied_guardrails(outcome: StreamingResponse) -> str:
    return outcome.headers.get("x-litellm-applied-guardrails", "")


def _tool_call_names(response: ChatResponse) -> tuple[str, ...]:
    return tuple(
        call.function.name
        for choice in response.choices
        if choice.message
        for call in choice.message.tool_calls or ()
        if call.function.name
    )


class TestToolPermissionPreCall:
    @pytest.mark.covers("guardrail.tool_permission.pre_call.blocks", exercised_on=["chat_completions"])
    def test_pre_call_blocks_tool_outside_the_allow_list(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """A request declaring a tool the guardrail does not permit must be
        rejected with a 400 that names the denied tool. An unauthorized tool that
        merely reaches the model is the whole failure mode this guardrail exists
        to prevent, so a 200 here is a hard failure."""
        name = f"e2e-toolperm-block-{unique_marker()}"
        _register_tool_permission(client, resources, name=name)

        result = poll_until_blocked(
            lambda: client.chat(
                scoped_key,
                MODEL,
                TOOL_PROMPT,
                guardrails=[name],
                max_tokens=128,
                tools=[DENIED_TOOL],
            )
        )

        match result:
            case UnknownApiError(status_code=status, body=body):
                assert status == 400, f"expected the guardrail block status 400, got {status}: {body[:400]}"
                assert DENIED_TOOL.function.name in body, (
                    f"the block must name the denied tool so the caller can fix the request; got: {body[:400]}"
                )
                assert "guardrail" in body.lower(), (
                    f"the block body should identify itself as a guardrail verdict; got: {body[:400]}"
                )
            case _:
                pytest.fail(f"tool_permission let a tool outside the allow-list through; got {result}")

    @pytest.mark.covers("guardrail.tool_permission.pre_call.allows", exercised_on=["chat_completions"])
    def test_pre_call_allows_permitted_tool(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """The mirror half: a request declaring only the permitted tool is served
        and the model calls it. Without the header check a guardrail that never
        attached would pass this test for the wrong reason, so the 200 alone is
        not the contract."""
        name = f"e2e-toolperm-allow-{unique_marker()}"
        _register_tool_permission(client, resources, name=name)

        outcome = client.chat_raw(
            scoped_key,
            MODEL,
            TOOL_PROMPT,
            guardrails=[name],
            max_tokens=128,
            tools=[ALLOWED_TOOL],
            tool_choice="required",
        )

        assert outcome.ok, f"the permitted tool must be served, got {outcome.status_code}: {outcome.body[:400]}"
        applied = _applied_guardrails(outcome)
        assert name in applied, (
            "the allowed call must carry x-litellm-applied-guardrails naming the guardrail; "
            f"without it the 200 only proves the guardrail never ran. Got {applied!r}"
        )

        called = _tool_call_names(ChatResponse.model_validate_json(outcome.body))
        assert called == (ALLOWED_TOOL.function.name,), (
            f"the served call must carry one tool call for the permitted tool, got {called!r}: {outcome.body[:400]}"
        )
