"""`/v1/messages` at 2 rps: `claude-sonnet-5` is served by the Anthropic mock's Messages route,
`claude-sonnet-5-bedrock-converse` (`bedrock/converse/anthropic.claude-sonnet-5`) is bridged to the
Bedrock mock's Converse route."""

from __future__ import annotations

from typing import Final

from .direct_vs_bridge import direct_vs_bridge

CASE: Final = direct_vs_bridge(
    "messages_2rps",
    "messages",
    direct="claude-sonnet-5",
    bridge="claude-sonnet-5-bedrock-converse",
)
