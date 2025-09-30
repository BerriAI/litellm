"""Helpers that mimic OpenAI responses for mini-agent local debugging."""

from __future__ import annotations

import os
import time
from typing import Any, Dict

# Stable default expected by tests; override for manual probes if needed
SHIM_REPLY = os.environ.get("MINI_AGENT_OPENAI_SHIM_REPLY", "shim ok")


def build_shim_completion(model: str, content: str | None = None) -> Dict[str, Any]:
    now = int(time.time())
    resolved_content = content if content is not None else SHIM_REPLY
    resolved_model = model or "mini-agent-openai-shim"
    return {
        "id": f"shim-{now}",
        "object": "chat.completion",
        "created": now,
        "model": resolved_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": resolved_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def chat_completions(request: Any) -> Dict[str, Any]:
    """Minimal async helper for contexts that call into the shim directly."""
    model = getattr(request, "model", None) or "mini-agent-openai-shim"
    return build_shim_completion(model)
