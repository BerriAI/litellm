"""Intercepted web searches are billed under the LLM request's session.

The websearch_interception callback turns an Anthropic ``web_search`` server tool
into a ``litellm.asearch`` call against a configured search tool, so each search is
its own spend row (call_type ``asearch``) next to the ``anthropic_messages`` row for
the turn that asked for it. Claude Code and the Admin UI group spend by
``session_id``, so the search row has to carry the same session as the turn that
triggered it; before the fix it landed under a session of its own and the session
view under-counted both requests and spend (LIT-8063).

Needs a proxy booted with the callback and a real search backend, which
``gateway/stage_mirror_ci_config.yml`` carries as the ``e2e-search`` Perplexity tool.
"""

from typing import Final, Literal

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import (
    AnthropicContentBlock,
    AnthropicMessagesBody,
    AnthropicWebSearchTool,
    ChatMessage,
    LiteLLMParamsBody,
    SpendLogRow,
)
from proxy_client import ProxyClient
from pydantic import BaseModel, ValidationError

pytestmark = pytest.mark.e2e

BEDROCK_INVOKE_BACKEND: Final = "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0"
SEARCH_CALL_TYPE: Final = "asearch"


def _has_search_row(rows: list[SpendLogRow]) -> bool:
    return any(row.call_type == SEARCH_CALL_TYPE for row in rows)


class _SearchResultError(BaseModel):
    type: Literal["web_search_tool_result_error"]
    error_code: str


def _search_error_code(block: AnthropicContentBlock) -> str | None:
    try:
        return _SearchResultError.model_validate((block.model_extra or {}).get("content")).error_code
    except ValidationError:
        return None


class TestWebSearchInterceptionSession:
    @pytest.mark.covers(
        "quota_management.spend_tracking.websearch_interception.bills_under_request_session",
        exercised_on=("messages",),
    )
    def test_intercepted_search_is_billed_under_the_request_session(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        """One /v1/messages turn that runs an intercepted web search must produce an
        ``asearch`` spend row in the same session as its ``anthropic_messages`` row,
        billed separately and with its own request id."""
        marker: Final = unique_marker()
        model: Final = f"e2e-websearch-session-{marker}"
        model_id: Final = proxy.create_model(
            model, LiteLLMParamsBody(model=BEDROCK_INVOKE_BACKEND, aws_region_name="us-east-1")
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key: Final = resources.key(models=[model])
        session_id: Final = f"e2e-websearch-session-{marker}"

        response: Final = unwrap(
            proxy.messages(
                key,
                AnthropicMessagesBody(
                    model=model,
                    max_tokens=512,
                    tools=[AnthropicWebSearchTool(type="web_search_20250305", name="web_search", max_uses=1)],
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Use web search to find one recent news headline about Anthropic ({marker}).",
                        )
                    ],
                ),
                session_id=session_id,
            )
        )
        block_types: Final = tuple(block.type for block in response.content or ())
        assert "web_search_tool_result" in block_types, (
            f"precondition: the turn never ran an intercepted search, so there is no search row to attribute. "
            f"blocks={block_types}"
        )
        search_errors: Final = tuple(
            code
            for block in response.content or ()
            if block.type == "web_search_tool_result" and (code := _search_error_code(block)) is not None
        )
        assert not search_errors, (
            f"precondition: the e2e-search tool failed upstream ({search_errors}), so no {SEARCH_CALL_TYPE} row is "
            "billed at all; check the proxy's search tool credentials before reading this as a session bug"
        )

        rows: Final = proxy.poll_logs_for_session(session_id, min_rows=2, predicate=_has_search_row)
        by_call_type: Final = {row.call_type or "" for row in rows}
        assert SEARCH_CALL_TYPE in by_call_type, (
            f"session {session_id} has no {SEARCH_CALL_TYPE} row, so the intercepted search was billed under a "
            f"different session and the session view misses its cost. call_types={sorted(by_call_type)} "
            f"rows={[(row.call_type, row.request_id, row.spend) for row in rows]}"
        )
        search_rows: Final = tuple(row for row in rows if row.call_type == SEARCH_CALL_TYPE)
        turn_rows: Final = tuple(row for row in rows if row.call_type != SEARCH_CALL_TYPE)
        assert turn_rows, f"session {session_id} carries only search rows: {rows!r}"
        assert all(row.session_id == session_id for row in rows), (
            f"session view returned rows outside {session_id}: {[row.session_id for row in rows]}"
        )
        assert all((row.spend or 0.0) > 0 for row in search_rows), (
            f"an intercepted search must stay a separately billed row: {[row.spend for row in search_rows]}"
        )
        assert {row.request_id for row in search_rows}.isdisjoint({row.request_id for row in turn_rows}), (
            "a search row reused its parent turn's request_id instead of keeping its own: "
            f"{[(row.call_type, row.request_id) for row in rows]}"
        )
