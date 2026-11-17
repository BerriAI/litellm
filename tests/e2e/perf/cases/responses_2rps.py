"""`/v1/responses` at 2 rps: `gpt-6-luna` is served by the OpenAI mock's Responses route,
`claude-sonnet-5` is bridged to the Anthropic mock's Messages route."""

from __future__ import annotations

from typing import Final

from .direct_vs_bridge import direct_vs_bridge

CASE: Final = direct_vs_bridge(
    "responses_2rps",
    "responses",
    direct="gpt-6-luna",
    bridge="claude-sonnet-5",
)
