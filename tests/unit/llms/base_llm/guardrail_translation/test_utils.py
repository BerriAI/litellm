from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail


@pytest.mark.parametrize(
    ("per_guardrail", "global_default", "expected"),
    [
        (True, False, True),
        (False, True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_a_per_guardrail_skip_flag_wins_over_the_global_setting(
    monkeypatch: pytest.MonkeyPatch, per_guardrail: bool | None, global_default: bool, expected: bool
):
    monkeypatch.setattr(litellm, "skip_system_message_in_guardrail", global_default)
    monkeypatch.setattr(litellm, "skip_tool_message_in_guardrail", global_default)
    guardrail: Final = LakeraAIGuardrail(
        api_key="lakera-test-key",
        skip_system_message_in_guardrail=per_guardrail,
        skip_tool_message_in_guardrail=per_guardrail,
    )

    assert effective_skip_system_message_for_guardrail(guardrail) is expected
    assert effective_skip_tool_message_for_guardrail(guardrail) is expected
