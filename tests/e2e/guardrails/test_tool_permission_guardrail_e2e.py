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

from typing import Final, Literal

import pytest
from e2e_config import unique_marker
from e2e_http import StreamingResponse, UnknownApiError
from e2e_metadata import Capability, Domain, Mode, Provider, Subject, meta
from guardrails_client import (
    GuardrailsClient,
    ToolPermissionParamsBody,
    ToolPermissionRuleBody,
    poll_until_blocked,
    poll_until_blocked_stream,
    poll_until_guardrail_applied,
)
from lifecycle import ResourceManager
from models import (
    AnthropicCustomTool,
    AnthropicMessagesBody,
    AnthropicMessagesResponse,
    AnthropicToolChoice,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    JsonSchemaProperty,
    ResponsesFunctionTool,
    ResponsesToolBody,
    ResponsesToolOutput,
    ToolInputSchema,
)

pytestmark: Final = pytest.mark.e2e

MODEL: Final = "gemini-2.5-flash"

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

type FlatToolEndpoint = Literal["messages", "responses"]
ALLOWED_TOOL_SCHEMA: Final = ToolInputSchema(properties={"city": JsonSchemaProperty(type="string")}, required=["city"])
DENIED_TOOL_SCHEMA: Final = ToolInputSchema()


def _register_tool_permission(client: GuardrailsClient, resources: ResourceManager, *, name: str) -> None:
    """Allow-list exactly one tool: everything else falls to `default_action=deny`
    and, with `on_disallowed_action=block`, is rejected outright."""
    guardrail_id: Final = client.register(
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


def _applied_guardrails(outcome: StreamingResponse) -> tuple[str, ...]:
    return tuple(name.strip() for name in outcome.headers.get("x-litellm-applied-guardrails", "").split(","))


def _tool_call_names(response: ChatResponse) -> tuple[str, ...]:
    return tuple(
        call.function.name
        for choice in response.choices
        if choice.message
        for call in choice.message.tool_calls or ()
        if call.function.name
    )


def _send_flat_tool(
    client: GuardrailsClient,
    key: str,
    endpoint: FlatToolEndpoint,
    *,
    guardrail: str,
    tool: ChatTool,
    schema: ToolInputSchema,
    require_tool: bool = False,
) -> StreamingResponse:
    name: Final = tool.function.name
    match endpoint:
        case "messages":
            return client.proxy.transport.send(
                "/v1/messages",
                headers=client.proxy.transport.bearer(key),
                json=AnthropicMessagesBody(
                    model=MODEL,
                    messages=[ChatMessage(role="user", content=TOOL_PROMPT)],
                    max_tokens=128,
                    tools=[
                        AnthropicCustomTool(name=name, description=tool.function.description or "", input_schema=schema)
                    ],
                    tool_choice=AnthropicToolChoice(type="any") if require_tool else None,
                    guardrails=[guardrail],
                ),
            )
        case "responses":
            return client.proxy.transport.send(
                "/v1/responses",
                headers=client.proxy.transport.bearer(key),
                json=ResponsesToolBody(
                    model=MODEL,
                    input=TOOL_PROMPT,
                    tools=[ResponsesFunctionTool(name=name, description=tool.function.description, parameters=schema)],
                    tool_choice="required" if require_tool else None,
                    guardrails=[guardrail],
                ),
            )


def _flat_tool_call_names(endpoint: FlatToolEndpoint, body: str) -> tuple[str, ...]:
    match endpoint:
        case "messages":
            content: Final = AnthropicMessagesResponse.model_validate_json(body).content or []
            return tuple(block.name for block in content if block.type == "tool_use" and block.name)
        case "responses":
            output: Final = ResponsesToolOutput.model_validate_json(body).output
            return tuple(item.name for item in output if item.type == "function_call" and item.name)


class TestToolPermissionPreCall:
    @pytest.mark.covers("guardrail.tool_permission.pre_call.blocks", exercised_on=["chat_completions"])
    @meta(
        Subject(
            domain=Domain.GUARDRAILS,
            providers=(Provider.GEMINI,),
            models=(MODEL,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_pre_call_blocks_tool_outside_the_allow_list(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """A request declaring a tool the guardrail does not permit must be
        rejected with a 400 that names the denied tool. An unauthorized tool that
        merely reaches the model is the whole failure mode this guardrail exists
        to prevent, so a 200 here is a hard failure."""
        name: Final = f"e2e-toolperm-block-{unique_marker()}"
        _register_tool_permission(client, resources, name=name)

        result: Final = poll_until_blocked(
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
    @meta(
        Subject(
            domain=Domain.GUARDRAILS,
            providers=(Provider.GEMINI,),
            models=(MODEL,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_pre_call_allows_permitted_tool(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """The mirror half: a request declaring only the permitted tool is served
        and the model calls it. Without the header check a guardrail that never
        attached would pass this test for the wrong reason, so the 200 alone is
        not the contract."""
        name: Final = f"e2e-toolperm-allow-{unique_marker()}"
        _register_tool_permission(client, resources, name=name)

        outcome: Final = poll_until_guardrail_applied(
            lambda: client.chat_raw(
                scoped_key,
                MODEL,
                TOOL_PROMPT,
                guardrails=[name],
                max_tokens=128,
                tools=[ALLOWED_TOOL],
                tool_choice="required",
            ),
            name,
        )

        assert outcome.ok, f"the permitted tool must be served, got {outcome.status_code}: {outcome.body[:400]}"
        applied: Final = _applied_guardrails(outcome)
        assert name in applied, (
            "the allowed call must carry x-litellm-applied-guardrails naming the guardrail; "
            f"without it the 200 only proves the guardrail never ran. Got {applied!r}"
        )

        called: Final = _tool_call_names(ChatResponse.model_validate_json(outcome.body))
        assert called == (ALLOWED_TOOL.function.name,), (
            f"the served call must carry one tool call for the permitted tool, got {called!r}: {outcome.body[:400]}"
        )

    @pytest.mark.parametrize(
        "endpoint",
        [
            pytest.param(
                "messages",
                marks=pytest.mark.covers("guardrail.tool_permission.pre_call.blocks", exercised_on=["messages"]),
            ),
            pytest.param(
                "responses",
                marks=pytest.mark.covers("guardrail.tool_permission.pre_call.blocks", exercised_on=["responses"]),
            ),
        ],
    )
    @meta(
        Subject(
            domain=Domain.GUARDRAILS,
            providers=(Provider.GEMINI,),
            models=(MODEL,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_pre_call_blocks_flat_tool_outside_the_allow_list(
        self,
        client: GuardrailsClient,
        resources: ResourceManager,
        scoped_key: str,
        endpoint: FlatToolEndpoint,
    ) -> None:
        name: Final = f"e2e-toolperm-block-{endpoint}-{unique_marker()}"
        _register_tool_permission(client, resources, name=name)

        outcome: Final = poll_until_blocked_stream(
            lambda: _send_flat_tool(
                client, scoped_key, endpoint, guardrail=name, tool=DENIED_TOOL, schema=DENIED_TOOL_SCHEMA
            )
        )

        assert outcome.status_code == 400, (
            f"tool_permission let a tool outside the allow-list through /{endpoint}; "
            f"got {outcome.status_code}: {outcome.body[:400]}"
        )
        assert DENIED_TOOL.function.name in outcome.body, (
            f"the block must name the denied tool so the caller can fix the request; got: {outcome.body[:400]}"
        )
        assert "Violated guardrail policy" in outcome.body, (
            f"the block must be a guardrail verdict, got: {outcome.body[:400]}"
        )

    @pytest.mark.parametrize(
        "endpoint",
        [
            pytest.param(
                "messages",
                marks=pytest.mark.covers("guardrail.tool_permission.pre_call.allows", exercised_on=["messages"]),
            ),
            pytest.param(
                "responses",
                marks=pytest.mark.covers("guardrail.tool_permission.pre_call.allows", exercised_on=["responses"]),
            ),
        ],
    )
    @meta(
        Subject(
            domain=Domain.GUARDRAILS,
            providers=(Provider.GEMINI,),
            models=(MODEL,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_pre_call_allows_permitted_flat_tool(
        self,
        client: GuardrailsClient,
        resources: ResourceManager,
        scoped_key: str,
        endpoint: FlatToolEndpoint,
    ) -> None:
        name: Final = f"e2e-toolperm-allow-{endpoint}-{unique_marker()}"
        _register_tool_permission(client, resources, name=name)

        outcome: Final = poll_until_guardrail_applied(
            lambda: _send_flat_tool(
                client,
                scoped_key,
                endpoint,
                guardrail=name,
                tool=ALLOWED_TOOL,
                schema=ALLOWED_TOOL_SCHEMA,
                require_tool=True,
            ),
            name,
        )

        assert outcome.ok, f"the permitted tool must be served, got {outcome.status_code}: {outcome.body[:400]}"
        applied: Final = _applied_guardrails(outcome)
        assert name in applied, (
            f"the allowed call must carry x-litellm-applied-guardrails naming the guardrail; got {applied!r}"
        )
        called: Final = _flat_tool_call_names(endpoint, outcome.body)
        assert called == (ALLOWED_TOOL.function.name,), (
            f"the served call must carry one tool call for the permitted tool, got {called!r}: {outcome.body[:400]}"
        )
