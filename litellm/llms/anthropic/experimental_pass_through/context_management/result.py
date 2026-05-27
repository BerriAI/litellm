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

from .constants import COMPACT_EDIT_TYPE


@dataclass
class PolyfillResult:
    messages: List[Dict[str, Any]]
    system: Optional[Union[str, List[Dict[str, Any]]]]
    applied_edits: List[AppliedEdit] = field(default_factory=list)
    compaction_block: Optional[CompactionBlock] = None
    iterations_usage: Optional[List[UsageIteration]] = None

    def applied_edits_for_response(self) -> Optional[List[AppliedEdit]]:
        """``applied_edits`` to attach on the client-visible response.

        ``compact_20260112`` is included only when a new compaction block was
        synthesized (slice-only / under-threshold paths omit it). Other edit
        types are included when the editor returned an ``AppliedEdit``.
        """
        visible: List[AppliedEdit] = []
        for edit in self.applied_edits:
            if edit.get("type") == COMPACT_EDIT_TYPE:
                if self.compaction_block is not None:
                    visible.append(edit)
            else:
                visible.append(edit)
        return visible or None
