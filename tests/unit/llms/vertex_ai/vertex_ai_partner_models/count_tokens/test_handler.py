from typing import Final

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import count_tokens_headers

_HISTORY: Final = ({"role": "user", "content": "Hello"},)


@pytest.mark.parametrize(
    "request_data",
    [
        {"messages": [{"role": "assistant", "content": [{"type": "compaction", "content": "so far", "signature": "sig"}]}]},
        {"messages": list(_HISTORY), "compaction": {"type": "summarize"}},
    ],
    ids=["signed_compaction_block", "compaction_param"],
)
def test_count_tokens_on_a_compaction_request_sends_the_compact_beta(request_data: dict[str, object]) -> None:
    assert count_tokens_headers("token", request_data) == {
        "Authorization": "Bearer token",
        "anthropic-beta": "compact-2026-09-04",
    }


@pytest.mark.parametrize(
    "request_data",
    [
        {"messages": list(_HISTORY)},
        {"messages": [{"role": "assistant", "content": [{"type": "compaction", "content": "so far", "signature": ""}]}]},
    ],
    ids=["plain", "unsigned_compaction_block"],
)
def test_count_tokens_without_compaction_sends_no_beta(request_data: dict[str, object]) -> None:
    assert count_tokens_headers("token", request_data) == {"Authorization": "Bearer token"}
