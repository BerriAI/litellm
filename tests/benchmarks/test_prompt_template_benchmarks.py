from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt, anthropic_messages_pt

LONG_RUN_OF_EMPTY_CACHED_USER_MESSAGES: Final[tuple[Mapping[str, object], ...]] = (
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    *({"role": "user", "content": [], "cache_control": {"type": "ephemeral"}} for _ in range(1_000)),
)


@pytest.mark.benchmark
def test_anthropic_messages_pt_long_run_of_empty_cached_user_messages() -> None:
    result: Final = anthropic_messages_pt(
        messages=list(LONG_RUN_OF_EMPTY_CACHED_USER_MESSAGES), model="claude-opus-5", llm_provider="vertex_ai"
    )

    assert len(result[-1]["content"]) == 1_000


CHAINED_DEFS_LENGTH: Final = 2_000


@pytest.mark.benchmark
def test_bedrock_tools_pt_claude_tool_with_a_long_chain_of_defs() -> None:
    chained_defs: Final = {
        f"Step{i}": {"type": "object", "properties": {"next": {"$ref": f"#/$defs/Step{i + 1}"}}}
        for i in range(CHAINED_DEFS_LENGTH)
    }
    defs: Final = {**chained_defs, f"Step{CHAINED_DEFS_LENGTH}": {"type": "string"}}
    parameters: Final = {"type": "object", "properties": {"start": {"$ref": "#/$defs/Step0"}}, "$defs": defs}

    sent: Final = _bedrock_tools_pt(
        [{"type": "function", "function": {"name": "walk", "parameters": parameters}}],
        model="us.anthropic.claude-sonnet-5-5",
    )[0]["toolSpec"]["inputSchema"]["json"]

    assert sent["$defs"] == defs
