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

        ``compact_20260112`` is included when a new compaction block was
        synthesized (success), when the edit carries an ``error`` field
        (``summary_model_not_configured``, ``summary_call_failed``,
        ``summary_extraction_failed``), or when the edit carries
        ``warnings`` (e.g. ``unsupported_trigger_type_X_using_input_tokens``,
        ``pause_after_compaction_ignored``) — operators and clients need to
        see why compaction was requested but not applied as expected.
        Slice-only / under-threshold paths that produced no edit at all
        (no block, no error, no warnings) are omitted. Other edit types are
        included when the editor returned an ``AppliedEdit``.
        """
        visible: List[AppliedEdit] = []
        for edit in self.applied_edits:
            if edit.get("type") == COMPACT_EDIT_TYPE:
                if self.compaction_block is not None or edit.get("error") or edit.get("warnings"):
                    visible.append(edit)
            else:
                visible.append(edit)
        return visible or None
