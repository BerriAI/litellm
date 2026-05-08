from __future__ import annotations

from typing import Any, List

from litellm.types.llms.openai import AllMessageValues


def effective_skip_system_message_for_guardrail(guardrail_to_apply: Any) -> bool:
    per = getattr(guardrail_to_apply, "skip_system_message_in_guardrail", None)
    if per is not None:
        return bool(per)
    import litellm

    return bool(getattr(litellm, "skip_system_message_in_guardrail", False))


def openai_messages_without_system(
    messages: List[AllMessageValues],
) -> List[AllMessageValues]:
    return [m for m in messages if str((m or {}).get("role") or "").lower() != "system"]
