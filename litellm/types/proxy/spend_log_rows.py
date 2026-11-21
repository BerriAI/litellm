"""What ``_native.RustSpendLogRowBuilder`` hands back to Python."""

from collections.abc import Mapping
from typing import Literal

from typing_extensions import ReadOnly, TypedDict


class NativeSpendLogRow(TypedDict):
    """``SpendLogRow`` in ``callbacks-spend-log/src/row.rs``, field for field and in its order.

    The timestamps are the UTC ``isoformat()`` text the builder writes; ``spend`` is the request's
    ``response_cost``, ``None`` when it had none, as Python's builder passed it through.
    """

    request_id: ReadOnly[str]
    call_type: ReadOnly[str]
    api_key: ReadOnly[str]
    cache_hit: ReadOnly[str]
    startTime: ReadOnly[str]
    endTime: ReadOnly[str]
    completionStartTime: ReadOnly[str | None]
    model: ReadOnly[str]
    user: ReadOnly[str]
    team_id: ReadOnly[str]
    organization_id: ReadOnly[str]
    metadata: ReadOnly[str]
    cache_key: ReadOnly[str]
    spend: ReadOnly[float | None]
    total_tokens: ReadOnly[int]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    request_tags: ReadOnly[str]
    end_user: ReadOnly[str]
    api_base: ReadOnly[str]
    model_group: ReadOnly[str]
    model_id: ReadOnly[str]
    mcp_namespaced_tool_name: ReadOnly[str | None]
    agent_id: ReadOnly[str | None]
    billing_agent_id: ReadOnly[str | None]
    requester_ip_address: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str]
    messages: ReadOnly[str]
    response: ReadOnly[str]
    proxy_server_request: ReadOnly[str]
    session_id: ReadOnly[str | None]
    request_duration_ms: ReadOnly[int | None]
    status: ReadOnly[Literal["success", "failure"]]
    litellm_call_id: ReadOnly[str | None]


class LedgerColumns(TypedDict):
    """The three ledger columns ``RustSpendLogRowBuilder.ledger_columns`` computes for a row."""

    spend: ReadOnly[float]
    accounting_delivery_id: ReadOnly[str | None]
    accounting_payload: ReadOnly[Mapping[str, object]]
