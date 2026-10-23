"""`/v1/chat/completions` at 2 rps: `gpt-6-luna` is served by the OpenAI mock's chat completions
route, `gpt-6-luna-responses` (`openai/responses/gpt-6-luna`) is bridged to its Responses route."""

from __future__ import annotations

from typing import Final

from .direct_vs_bridge import direct_vs_bridge

CASE: Final = direct_vs_bridge(
    "chat_completions_2rps",
    "chat_completions",
    direct="gpt-6-luna",
    bridge="gpt-6-luna-responses",
)
