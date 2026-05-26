"""``compact_20260112`` polyfill (server-side context compaction).

Mirrors Anthropic's native ``compact_20260112`` for non-Anthropic providers:

- Scans the message history for an existing ``compaction`` block; everything
  before it is dropped (slice).
- If still over the configured trigger, calls a separately-configured
  summarization model and synthesizes a fresh ``compaction`` block.
- The summary is injected as a system-message prefix on the downstream call
  (the user/assistant log carries no ``compaction`` block downstream).
- The synthesized ``compaction`` block is returned via ``PolyfillResult`` so
  the response adapter can prepend it to the response ``content`` array.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import (
    AppliedEdit,
    CompactionBlock,
    UsageIteration,
)

from ..constants import (
    COMPACT_DEFAULT_INSTRUCTIONS,
    COMPACT_DEFAULT_TRIGGER_TOKENS,
    COMPACT_EDIT_TYPE,
    COMPACT_MIN_TRIGGER_TOKENS,
    COMPACT_NO_TOOL_CALLS_SUFFIX,
    COMPACT_SUMMARY_MODEL_SETTING_KEY,
    COMPACT_SUMMARY_SYSTEM_PREFIX,
)
from ..errors import AnthropicContextManagementError
from ..result import PolyfillResult

# Auth metadata fields propagated from the parent request to the summary call
# so the summary's spend is attributed to the same team/key. The list mirrors
# the fields populated by
# ``LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata``.
_PROPAGATED_METADATA_KEYS = (
    "user_api_key",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_org_id",
    "litellm_call_id",
    "litellm_parent_otel_span",
)

_SUMMARY_TAG_RE = re.compile(r"<summary>(.*?)</summary>", re.IGNORECASE | re.DOTALL)


def _read_summary_model_setting() -> Optional[str]:
    """Look up the configured summarization model from proxy general_settings."""
    try:
        from litellm.proxy.proxy_server import general_settings
    except Exception:
        return None
    value = general_settings.get(COMPACT_SUMMARY_MODEL_SETTING_KEY)
    return value if isinstance(value, str) and value else None


def _find_latest_compaction_index(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int]]:
    """Return (message_index, block_index) of the most recent compaction block.

    ``None, None`` if no compaction block is present. Iterates from the end so
    only the latest one is considered.
    """
    for msg_idx in range(len(messages) - 1, -1, -1):
        content = messages[msg_idx].get("content")
        if not isinstance(content, list):
            continue
        for blk_idx in range(len(content) - 1, -1, -1):
            block = content[blk_idx]
            if isinstance(block, dict) and block.get("type") == "compaction":
                return msg_idx, blk_idx
    return None, None


def _slice_around_compaction_block(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Apply Anthropic's "drop everything before the compaction block" rule.

    Returns ``(sliced_messages_with_compaction_block, compaction_block_dict)``
    if a block was found, else ``(original_messages, None)``. The sliced result
    keeps the compaction block in the assistant turn that originally carried
    it (in practice it's the only block in that turn) so callers can still
    extract the summary text from it.
    """
    msg_idx, blk_idx = _find_latest_compaction_index(messages)
    if msg_idx is None or blk_idx is None:
        return messages, None

    original_msg = messages[msg_idx]
    original_content = original_msg["content"]
    compaction_block = cast(Dict[str, Any], original_content[blk_idx])

    # Per Anthropic's contract everything before the compaction block is
    # dropped, including earlier blocks within the same assistant message.
    sliced_content = list(original_content[blk_idx:])
    sliced_first_msg = {**original_msg, "content": sliced_content}

    sliced_messages: List[Dict[str, Any]] = [sliced_first_msg]
    sliced_messages.extend(messages[msg_idx + 1 :])
    return sliced_messages, compaction_block


def _strip_compaction_blocks(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop any ``compaction`` content blocks from messages.

    Used to build the downstream-bound message list — the adapter has no
    concept of a compaction block, so it must not see one.
    """
    cleaned: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        filtered = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "compaction")
        ]
        if not filtered:
            # The compaction block was the only content; drop the whole turn.
            continue
        cleaned.append({**msg, "content": filtered})
    return cleaned


def _augment_system_with_summary(
    system: Optional[Union[str, List[Dict[str, Any]]]],
    summary_text: str,
) -> Union[str, List[Dict[str, Any]]]:
    """Prepend a "Previous conversation summary: ..." block to ``system``."""
    prefix = f"{COMPACT_SUMMARY_SYSTEM_PREFIX}{summary_text}\n\n"
    if system is None:
        return prefix.rstrip()
    if isinstance(system, str):
        return f"{prefix}{system}"
    # List of content blocks: prepend the prefix to the first text block,
    # otherwise insert a new text block at the head.
    for idx, block in enumerate(system):
        if isinstance(block, dict) and block.get("type") == "text":
            existing = block.get("text", "") or ""
            new_block = {**block, "text": f"{prefix}{existing}"}
            return [*system[:idx], new_block, *system[idx + 1 :]]
    return [{"type": "text", "text": prefix.rstrip()}, *system]


def _resolve_trigger_tokens(edit_spec: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Validate and resolve ``trigger.value``.

    Raises ``AnthropicContextManagementError`` if the explicitly-supplied value
    is below the 50k minimum. Unknown ``trigger.type`` values fall back to
    ``input_tokens`` with a warning.
    """
    warnings: List[str] = []
    trigger = edit_spec.get("trigger") or {}
    if not isinstance(trigger, dict):
        warnings.append("trigger_not_a_dict_using_default")
        return COMPACT_DEFAULT_TRIGGER_TOKENS, warnings

    trigger_type = trigger.get("type", "input_tokens")
    if trigger_type != "input_tokens":
        warnings.append(f"unsupported_trigger_type_{trigger_type}_using_input_tokens")

    value = trigger.get("value")
    if value is None:
        return COMPACT_DEFAULT_TRIGGER_TOKENS, warnings
    if not isinstance(value, int):
        warnings.append("trigger_value_not_int_using_default")
        return COMPACT_DEFAULT_TRIGGER_TOKENS, warnings
    if value < COMPACT_MIN_TRIGGER_TOKENS:
        raise AnthropicContextManagementError(
            status_code=400,
            message=(
                f"context_management.compact_20260112.trigger.value must be at "
                f"least {COMPACT_MIN_TRIGGER_TOKENS} tokens"
            ),
        )
    return value, warnings


def _build_summary_prompt(
    edit_spec: Dict[str, Any], tools: Optional[List[Dict[str, Any]]]
) -> str:
    custom = edit_spec.get("instructions")
    if isinstance(custom, str) and custom.strip():
        return custom
    prompt = COMPACT_DEFAULT_INSTRUCTIONS
    if tools:
        prompt = f"{prompt}{COMPACT_NO_TOOL_CALLS_SUFFIX}"
    return prompt


def _propagate_metadata(parent_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not parent_metadata:
        return {}
    propagated: Dict[str, Any] = {}
    for key in _PROPAGATED_METADATA_KEYS:
        if key in parent_metadata:
            propagated[key] = parent_metadata[key]
    return propagated


def _count_effective_tokens(
    model: str,
    effective_messages: List[Dict[str, Any]],
    compaction_block: Optional[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> int:
    """Token-count the conversation as it will appear downstream.

    The compaction block (if any) becomes a system prefix on the downstream
    call, so its content still counts even though it isn't in ``messages``.
    """
    # Local import to avoid pulling the adapter at module load time.
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    messages_without_compaction = _strip_compaction_blocks(effective_messages)
    try:
        openai_shape = (
            LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
                messages=cast(Any, messages_without_compaction)
            )
        )
    except Exception as e:
        verbose_logger.debug(
            "compact_20260112: anthropic→openai translation failed during token "
            "count, falling back to raw messages: %s",
            e,
        )
        openai_shape = cast(Any, messages_without_compaction)

    total = litellm.token_counter(
        model=model,
        messages=cast(Any, openai_shape),
        tools=cast(Any, tools),
    )
    if compaction_block is not None:
        content = compaction_block.get("content") or ""
        if content:
            total += litellm.token_counter(model=model, text=content)
    return total


def _is_tool_result_only_user_turn(msg: Dict[str, Any]) -> bool:
    """Return True if ``msg`` is a ``role=user`` turn that carries only
    ``tool_result`` blocks (i.e. an Anthropic tool-use response).

    Such turns are not real "user question" turns and must not be used as
    the sole downstream message after a full compaction: the adapter
    translates them to OpenAI ``tool``-role messages, which require a
    preceding assistant ``tool_calls`` turn that no longer exists once the
    history has been summarized away.
    """
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    for block in content:
        if not isinstance(block, dict):
            return False
        if block.get("type") != "tool_result":
            return False
    return True


def _select_last_user_question(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pick the most recent ``user`` turn that is a real question.

    Returns a one-element message list, or a synthetic continuation prompt
    if no eligible turn exists (e.g. the conversation only ever contained
    ``tool_result`` turns, or contained no user turns at all). The
    downstream call always needs a non-empty user message.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user" and not _is_tool_result_only_user_turn(msg):
            return [msg]
    return [
        {
            "role": "user",
            "content": "Please continue based on the conversation summary above.",
        }
    ]


def _extract_summary_text(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    match = _SUMMARY_TAG_RE.search(raw)
    if match is None:
        return None
    summary = match.group(1).strip()
    return summary or None


def _build_summary_messages(
    effective_messages: List[Dict[str, Any]],
    prompt: str,
) -> List[Dict[str, Any]]:
    """Build the OpenAI-shape message list for the summary call.

    The conversation history is translated to OpenAI shape; the
    summarization prompt is appended as a final user turn.
    """
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    stripped = _strip_compaction_blocks(effective_messages)
    try:
        openai_messages = (
            LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
                messages=cast(Any, stripped)
            )
        )
    except Exception as e:
        verbose_logger.warning(
            "compact_20260112: anthropic→openai translation failed when "
            "building summary call; falling back to raw shape: %s",
            e,
        )
        openai_messages = cast(Any, stripped)

    return [*openai_messages, {"role": "user", "content": prompt}]


async def _call_summary_model(
    *,
    summary_model: str,
    summary_messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    llm_router: Any,
) -> Any:
    """Invoke the configured summary model.

    Prefers ``llm_router.acompletion`` so the model alias resolves against the
    proxy's ``model_list``; falls back to ``litellm.acompletion`` if no router
    is available (e.g. SDK usage outside the proxy).
    """
    call_kwargs: Dict[str, Any] = {
        "model": summary_model,
        "messages": summary_messages,
        "metadata": metadata,
    }
    if llm_router is not None and hasattr(llm_router, "acompletion"):
        return await llm_router.acompletion(**call_kwargs)
    return await litellm.acompletion(**call_kwargs)


def _extract_response_text(response: Any) -> Optional[str]:
    try:
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        # Some providers return a list of content parts.
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(text_parts) or None
    except (AttributeError, IndexError, KeyError):
        return None
    return None


def _extract_usage(response: Any) -> Tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


async def apply_compact_20260112(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    system: Optional[Union[str, List[Dict[str, Any]]]],
    edit_spec: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    llm_router: Any = None,
) -> PolyfillResult:
    """Apply ``compact_20260112``; return a ``PolyfillResult``.

    See module docstring for the algorithm. Errors are best-effort: when the
    summary call fails or the response is malformed, the editor returns the
    pre-summary state (with ``applied_edits[0].error`` populated) so the
    original request still proceeds.
    """
    # Validation runs first. Raising AnthropicContextManagementError here is
    # the only path on which the polyfill aborts the request.
    trigger_tokens, warnings = _resolve_trigger_tokens(edit_spec)
    if edit_spec.get("pause_after_compaction"):
        warnings.append("pause_after_compaction_ignored")

    applied: AppliedEdit = {"type": COMPACT_EDIT_TYPE}
    if warnings:
        applied["warnings"] = warnings

    # Opt-in gate: no summary model configured → no-op.
    summary_model = _read_summary_model_setting()
    if summary_model is None:
        applied["error"] = "summary_model_not_configured"
        return PolyfillResult(
            messages=messages,
            system=system,
            applied_edits=[applied],
        )

    # Phase A: slice around any existing compaction block.
    effective_messages, prior_compaction_block = _slice_around_compaction_block(
        messages
    )
    prior_summary_text = (
        prior_compaction_block.get("content") if prior_compaction_block else None
    )
    augmented_system: Union[str, List[Dict[str, Any]], None] = system
    if isinstance(prior_summary_text, str) and prior_summary_text:
        augmented_system = _augment_system_with_summary(system, prior_summary_text)

    downstream_messages = _strip_compaction_blocks(effective_messages)

    # Phase B: threshold check.
    try:
        current_tokens = _count_effective_tokens(
            model=model,
            effective_messages=effective_messages,
            compaction_block=prior_compaction_block,
            tools=tools,
        )
    except Exception as e:
        verbose_logger.warning(
            "compact_20260112: token_counter failed; assuming under threshold: %s", e
        )
        current_tokens = 0

    verbose_logger.debug(
        "compact_20260112: current_tokens=%s trigger=%s", current_tokens, trigger_tokens
    )

    if current_tokens <= trigger_tokens:
        # Slice-only path. If Phase A fired we still slice + system-prefix.
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    # Phase C: summarize.
    prompt = _build_summary_prompt(edit_spec, tools)
    summary_messages = _build_summary_messages(effective_messages, prompt)
    propagated_metadata = _propagate_metadata(metadata)

    try:
        response = await _call_summary_model(
            summary_model=summary_model,
            summary_messages=summary_messages,
            metadata=propagated_metadata,
            llm_router=llm_router,
        )
    except Exception as e:
        verbose_logger.warning("compact_20260112: summary call failed: %s", e)
        applied["error"] = "summary_call_failed"
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    summary_text = _extract_summary_text(_extract_response_text(response))
    if summary_text is None:
        applied["error"] = "summary_extraction_failed"
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    summary_input_tokens, summary_output_tokens = _extract_usage(response)
    applied["summary_input_tokens"] = summary_input_tokens
    applied["summary_output_tokens"] = summary_output_tokens

    compaction_block: CompactionBlock = {
        "type": "compaction",
        "content": summary_text,
    }
    iterations_usage: List[UsageIteration] = [
        {
            "type": "compaction",
            "input_tokens": summary_input_tokens,
            "output_tokens": summary_output_tokens,
        }
    ]

    # Per Anthropic's contract, everything before the compaction block is
    # dropped. Phase D: the user/assistant log goes empty; the summary lives
    # on the system message instead. Anthropic requires a non-empty messages
    # array, so keep the most recent original user *question* turn so the
    # model has something to answer. Skip ``tool_result``-only user turns:
    # in Anthropic's format those are role=user but represent the response
    # from a tool, and surfacing one as the sole downstream message would
    # produce an orphaned ``tool``-role message on non-Anthropic providers
    # with no matching ``tool_calls`` in the prior assistant history. If no
    # eligible turn exists, fall back to a synthetic continuation prompt so
    # the downstream call still has a non-empty user message.
    summarized_system = _augment_system_with_summary(system, summary_text)
    downstream_messages_after_summary = _select_last_user_question(messages)

    return PolyfillResult(
        messages=downstream_messages_after_summary,
        system=summarized_system,
        applied_edits=[applied],
        compaction_block=compaction_block,
        iterations_usage=iterations_usage,
    )
