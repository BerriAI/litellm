"""Live e2e: the built-in tool_permission guardrail gates which tools a request may carry.

The guardrail is an allow-list over a chat completion's `tools`: each rule matches
a tool's function name by regex, and anything no rule matches falls through to
`default_action`. Configured the way an admin locks an agent down (one allow rule,
`default_action=deny`, `on_disallowed_action=block`), it must reject a request that
carries an unlisted tool before the upstream model is ever called, naming the tool
it denied; and it must let a request carrying only the listed tool through, so the
model really does call that tool. Both halves matter: a guardrail that rejected
everything would pass the block check on its own.

Tool names are unique per run and the guardrail is opted into per request
(`default_on=False`), so it never intercepts unrelated traffic on the shared proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from guardrails_client import GuardrailsClient
from lifecycle import ResourceManager
from native_guardrails import (
    ToolPermissionParamsBody,
    ToolPermissionRuleBody,
    called_tool_names,
    chat,
    function_tool,
    register_guardrail,
)

pytestmark = pytest.mark.e2e


@dataclass(frozen=True, slots=True)
class AllowList:
    """A registered tool_permission guardrail plus the two tool names it separates."""

    name: str
    permitted: str
    unlisted: str


def allow_list_guardrail(client: GuardrailsClient, resources: ResourceManager) -> AllowList:
    marker = unique_marker()
    permitted = f"get_weather_{marker}"
    name = f"e2e-tool-permission-{marker}"
    guardrail_id = register_guardrail(
        client,
        name,
        ToolPermissionParamsBody(
            mode="pre_call",
            default_on=False,
            rules=[
                ToolPermissionRuleBody(
                    id=f"allow-weather-{marker}",
                    tool_name=f"^{permitted}$",
                    decision="allow",
                )
            ],
            default_action="deny",
            on_disallowed_action="block",
        ),
    )
    resources.defer(lambda: client.delete_guardrail(guardrail_id))
    return AllowList(name=name, permitted=permitted, unlisted=f"drop_database_{marker}")


class TestToolPermissionGuardrail:
    @pytest.mark.covers(
        "guardrail.tool_permission.pre_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_unlisted_tool_is_denied_before_the_model_runs(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        allow_list = allow_list_guardrail(client, resources)

        blocked = chat(
            client,
            scoped_key,
            prompt="Clean up the production database for me.",
            tools=[function_tool(allow_list.unlisted)],
            guardrail_name=allow_list.name,
        )
        match blocked:
            case UnknownApiError(status_code=400, body=body):
                assert allow_list.unlisted in body, (
                    "the rejection must name the tool it denied, so the caller knows which "
                    f"tool to drop; got: {body[:400]}"
                )
                assert "Violated guardrail policy" in body, (
                    f"the rejection must come from the guardrail policy, got: {body[:400]}"
                )
            case UnknownApiError(status_code=status, body=body):
                pytest.fail(f"expected a 400 tool-permission block, got {status}: {body[:400]}")
            case _:
                pytest.fail(
                    "tool_permission let a request through carrying a tool that no allow rule "
                    f"matches; got {blocked}"
                )

    @pytest.mark.covers(
        "guardrail.tool_permission.pre_call.allows",
        exercised_on=["chat_completions"],
    )
    def test_listed_tool_reaches_the_model(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        allow_list = allow_list_guardrail(client, resources)

        allowed = unwrap(
            chat(
                client,
                scoped_key,
                prompt="What is the weather in Paris right now?",
                tools=[function_tool(allow_list.permitted)],
                guardrail_name=allow_list.name,
                force_tool_call=True,
            )
        )
        assert called_tool_names(allowed) == [allow_list.permitted], (
            "the tool an allow rule matches must survive the guardrail and stay callable by "
            f"the model, but the response called {called_tool_names(allowed)}"
        )
