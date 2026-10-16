from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt

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
