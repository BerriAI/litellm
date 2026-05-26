"""``PolyfillResult`` — the shape returned by the context-management dispatcher.

Threaded from the dispatcher through ``async_anthropic_messages_handler`` into
the adapter so it can prepend the ``compaction`` block to the response and
attach ``iterations`` to ``usage``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from litellm.types.llms.anthropic import (
    AppliedEdit,
    CompactionBlock,
    UsageIteration,
)


@dataclass
class PolyfillResult:
    messages: List[Dict[str, Any]]
    system: Optional[Union[str, List[Dict[str, Any]]]]
    applied_edits: List[AppliedEdit] = field(default_factory=list)
    compaction_block: Optional[CompactionBlock] = None
    iterations_usage: Optional[List[UsageIteration]] = None
