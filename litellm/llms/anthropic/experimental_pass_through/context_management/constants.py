"""Constants for the in-gateway context-management polyfill."""

CLEAR_TOOL_USES_EDIT_TYPE = "clear_tool_uses_20250919"

DEFAULT_INPUT_TOKENS_TRIGGER = 100_000
DEFAULT_KEEP_TOOL_USES = 3

CLEARED_TOOL_RESULT_PLACEHOLDER = "[Cleared by context management]"

# compact_20260112
COMPACT_EDIT_TYPE = "compact_20260112"
COMPACT_DEFAULT_TRIGGER_TOKENS = 150_000
COMPACT_MIN_TRIGGER_TOKENS = 50_000
# Default ``max_tokens`` for the summary call. Required by providers like
# Anthropic that reject requests without it; safely accepted by providers that
# don't strictly require it. Chosen to comfortably fit a long structured
# summary. Operators can override via
# ``general_settings.context_management_summary_max_tokens``.
COMPACT_SUMMARY_MAX_TOKENS = 4096
COMPACT_SUMMARY_MAX_TOKENS_SETTING_KEY = "context_management_summary_max_tokens"
# Wall-clock bound for the summary sub-call. Without this a slow or
# unresponsive summary model would hang the parent ``/v1/messages`` request
# with no escape hatch; on timeout the editor falls into the standard
# ``summary_call_failed`` path and forwards the request without compaction.
COMPACT_SUMMARY_TIMEOUT_SECONDS = 60.0
COMPACT_SUMMARY_MODEL_SETTING_KEY = "context_management_summary_model"
COMPACT_SUMMARY_SYSTEM_PREFIX = "Previous conversation summary: "

# Default summarization prompt from the Anthropic spec.
COMPACT_DEFAULT_INSTRUCTIONS = (
    "You have written a partial transcript for the initial task above. Please "
    "write a summary of the transcript. The purpose of this summary is to "
    "provide continuity so you can continue to make progress towards solving "
    "the task in a future context, where the raw history above may not be "
    "accessible and will be replaced with this summary. Write down anything "
    "that would be helpful, including the state, next steps, learnings etc. "
    "You must wrap your summary in a <summary></summary> block."
)

# Appended to the default prompt when ``tools`` are present and the caller
# did not supply custom ``instructions``. Matches the guidance in the
# Anthropic docs under "Compaction might fail when tools are defined".
COMPACT_NO_TOOL_CALLS_SUFFIX = (
    " Do not call any tools while writing this summary; respond with text only."
)
