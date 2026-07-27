"""Typed bodies and helpers for the guardrails litellm implements itself.

`tool_permission` and `tool_policy` decide, before the upstream model runs,
whether the tools a request carries may be used at all: the first from a regex
allow-list configured on the guardrail, the second from the per-key blocked-tool
overrides admins set through POST /v1/tool/policy. `llm_as_a_judge` runs after
the model instead, scoring the response against weighted criteria with a second
LLM. All three register through the same POST /guardrails route the rest of the
suite uses and only their params differ, so the params compose onto the shared
`GuardrailParamsBase` and the create body is written once here.
"""

from __future__ import annotations

import time
from typing import Literal, Sequence

from pydantic import BaseModel

from e2e_config import CHEAP_OPENAI_MODEL, POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import NoBody, Result, Success, unwrap
from guardrails_client import (
    GuardrailCreateResponse,
    GuardrailParamsBase,
    GuardrailsClient,
)
from models import ChatBody, ChatMessage, ChatResponse, ChatTool, ChatToolFunction

ToolInputPolicy = Literal["blocked", "trusted", "untrusted"]


class ToolPermissionRuleBody(BaseModel):
    """One allow/deny rule: `tool_name` is a regex matched against the tool's
    function name."""

    id: str
    tool_name: str
    decision: Literal["allow", "deny"]


class ToolPermissionParamsBody(GuardrailParamsBase):
    guardrail: Literal["tool_permission"] = "tool_permission"
    rules: list[ToolPermissionRuleBody]
    default_action: Literal["allow", "deny"]
    on_disallowed_action: Literal["block", "rewrite"]


class ToolPolicyParamsBody(GuardrailParamsBase):
    """tool_policy reads its decisions from the tool registry and the caller's
    blocked-tool overrides, so the guardrail itself carries no rules."""

    guardrail: Literal["tool_policy"] = "tool_policy"


class JudgeCriterionBody(BaseModel):
    """One scoring criterion. Weights across a guardrail's criteria must sum to
    100 or the proxy rejects the registration."""

    name: str
    weight: int
    description: str


class LLMAsAJudgeParamsBody(GuardrailParamsBase):
    guardrail: Literal["llm_as_a_judge"] = "llm_as_a_judge"
    judge_model: str
    criteria: list[JudgeCriterionBody]
    overall_threshold: float
    on_failure: Literal["block", "log"]


NativeGuardrailParamsBody = (
    ToolPermissionParamsBody | ToolPolicyParamsBody | LLMAsAJudgeParamsBody
)


class NativeGuardrailSpecBody(BaseModel):
    guardrail_name: str
    litellm_params: NativeGuardrailParamsBody


class NativeGuardrailCreateBody(BaseModel):
    guardrail: NativeGuardrailSpecBody


class ToolPolicyOverrideBody(BaseModel):
    """POST /v1/tool/policy scoped to one virtual key: sets that key's policy for
    a single tool without touching the global registry entry."""

    tool_name: str
    input_policy: ToolInputPolicy
    key_hash: str


class ToolPolicyUpdateResponse(BaseModel):
    tool_name: str
    updated: bool


class ToolOverrideParams(BaseModel):
    key_hash: str


class ToolOverrideDeleteResponse(BaseModel):
    deleted: bool
    tool_name: str


class GuardrailUsageLogEntry(BaseModel):
    """One row of GET /guardrails/usage/logs. `action` is the guardrail's own
    verdict on that request: `passed` when it ran and approved, `blocked` when it
    intervened, and `flagged` when it errored and let the request through anyway.
    `id` is the completion id for a request the guardrail approved."""

    id: str
    action: Literal["passed", "blocked", "flagged"]


class GuardrailUsageLogs(BaseModel):
    logs: list[GuardrailUsageLogEntry]
    total: int


class GuardrailUsageLogsParams(BaseModel):
    guardrail_id: str


def register_guardrail(
    client: GuardrailsClient, name: str, params: NativeGuardrailParamsBody
) -> str:
    """Register a native guardrail and return its id. Registered with
    `default_on=False` and opted into per request, so it never intercepts
    unrelated traffic on the shared proxy."""
    return unwrap(
        client.proxy.transport.post(
            "/guardrails",
            headers=client.proxy.transport.master,
            json=NativeGuardrailCreateBody(
                guardrail=NativeGuardrailSpecBody(guardrail_name=name, litellm_params=params)
            ),
            response_type=GuardrailCreateResponse,
        )
    ).guardrail_id


def block_tool_for_key(client: GuardrailsClient, *, tool_name: str, key: str) -> None:
    """Mark one tool blocked for one virtual key. The proxy resolves the raw key
    to its object permission (creating one when the key has none) and resyncs the
    in-memory tool-policy registry before answering."""
    _ = unwrap(
        client.proxy.transport.post(
            "/v1/tool/policy",
            headers=client.proxy.transport.master,
            json=ToolPolicyOverrideBody(
                tool_name=tool_name, input_policy="blocked", key_hash=key
            ),
            response_type=ToolPolicyUpdateResponse,
        )
    )


def unblock_tool_for_key(client: GuardrailsClient, *, tool_name: str, key: str) -> None:
    _ = client.proxy.transport.delete(
        f"/v1/tool/{tool_name}/overrides",
        headers=client.proxy.transport.master,
        json=NoBody(),
        params=ToolOverrideParams(key_hash=key),
        response_type=ToolOverrideDeleteResponse,
    )


def function_tool(name: str) -> ChatTool:
    return ChatTool(
        function=ChatToolFunction(
            name=name,
            description="Look up the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    )


def chat(
    client: GuardrailsClient,
    key: str,
    *,
    prompt: str,
    guardrail_name: str,
    tools: Sequence[ChatTool] = (),
    force_tool_call: bool = False,
) -> Result[ChatResponse]:
    """Drive a real chat completion, opting into one guardrail for this request
    only. `tools` carries the function definitions a tool-gating guardrail reads;
    `force_tool_call` sets tool_choice=required so an allowed tool is provably
    reachable rather than left to the model's mood. No max_tokens is sent: gpt-5.5
    spends tokens on reasoning before it writes anything, and a truncated empty
    response would leave a post-call guardrail with nothing to judge."""
    return client.proxy.chat(
        key,
        ChatBody(
            model=CHEAP_OPENAI_MODEL,
            messages=[ChatMessage(role="user", content=prompt)],
            tools=list(tools) or None,
            tool_choice="required" if force_tool_call else None,
            guardrails=[guardrail_name],
        ),
    )


def called_tool_names(response: ChatResponse) -> list[str]:
    if not response.choices:
        return []
    message = response.choices[0].message
    calls = (message.tool_calls if message else None) or []
    return [call.function.name for call in calls if call.function.name]


def poll_guardrail_usage_logs(
    client: GuardrailsClient, guardrail_id: str, *, min_rows: int
) -> list[GuardrailUsageLogEntry]:
    """Poll the guardrail's own run log until it has recorded `min_rows` requests.

    This is the only caller-visible record of what a guardrail decided. It matters
    for a guardrail that adjudicates with a second LLM, because that call fails
    open on any internal error: the request then returns a normal 200 that looks
    exactly like an approval. `action` separates the two, so a test can assert the
    guardrail actually ran rather than that the response merely came back. Rows are
    fetched by guardrail_id, and each test registers its own guardrail, so the log
    holds that test's requests and nothing else."""
    deadline = time.monotonic() + POLL_TIMEOUT
    last: Result[GuardrailUsageLogs] | None = None
    while time.monotonic() < deadline:
        last = client.proxy.transport.get(
            "/guardrails/usage/logs",
            headers=client.proxy.transport.master,
            params=GuardrailUsageLogsParams(guardrail_id=guardrail_id),
            response_type=GuardrailUsageLogs,
        )
        if isinstance(last, Success) and len(last.data.logs) >= min_rows:
            return last.data.logs
        time.sleep(POLL_INTERVAL)
    raise AssertionError(
        f"guardrail {guardrail_id!r} never recorded {min_rows} run(s) in "
        f"/guardrails/usage/logs; last response was {last}"
    )


def response_text(response: ChatResponse) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    return (message.content if message else None) or ""
