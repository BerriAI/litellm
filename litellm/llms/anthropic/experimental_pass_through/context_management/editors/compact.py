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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

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
    COMPACT_SUMMARY_MAX_TOKENS,
    COMPACT_SUMMARY_MAX_TOKENS_SETTING_KEY,
    COMPACT_SUMMARY_MODEL_SETTING_KEY,
    COMPACT_SUMMARY_SYSTEM_PREFIX,
    COMPACT_SUMMARY_TIMEOUT_SECONDS,
)
from ..errors import AnthropicContextManagementError
from ..result import PolyfillResult

# Auth metadata fields propagated from the parent request to the summary call
# so the summary's spend is attributed to the same scopes. The list mirrors the
# fields populated by
# ``LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata``.
# ``user_api_key_model_max_budget`` / ``user_api_key_end_user_model_max_budget``
# are what ``_PROXY_VirtualKeyModelMaxBudgetLimiter`` reads post-call to update
# the per-model spend caches, so without them the summary spend would never
# count against the caller's model budget. ``user_api_key_end_user_id`` /
# ``user_api_key_project_id`` are the scope identifiers the post-call spend hook
# and rate limiter key their counters on, and ``user_api_end_user_max_budget``
# is the end-user budget the cost callback enforces — without these the summary
# tokens escape the caller's end-user/project budgets and counters.
_PROPAGATED_METADATA_KEYS = (
    "user_api_key",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_org_id",
    "user_api_key_project_id",
    "user_api_key_end_user_id",
    "user_api_end_user_max_budget",
    "user_api_key_model_max_budget",
    "user_api_key_end_user_model_max_budget",
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


def _read_summary_max_tokens_setting() -> int:
    """Look up the configured summary ``max_tokens`` from proxy general_settings.

    Falls back to :data:`COMPACT_SUMMARY_MAX_TOKENS` when the setting is
    missing or invalid (non-positive int, wrong type). Operators tune this
    when the default doesn't fit their chosen summary model's output budget.
    """
    try:
        from litellm.proxy.proxy_server import general_settings
    except Exception:
        return COMPACT_SUMMARY_MAX_TOKENS
    value = general_settings.get(COMPACT_SUMMARY_MAX_TOKENS_SETTING_KEY)
    if isinstance(value, int) and value > 0:
        return value
    return COMPACT_SUMMARY_MAX_TOKENS


async def _check_summary_model_access(  # noqa: PLR0915
    user_api_key_auth: Any,
    summary_model: str,
    llm_router: Any,
) -> bool:
    """Return True when every model-allowlist scope on the parent request is
    satisfied for ``summary_model``.

    The summary subrequest does not pass through ``user_api_key_auth`` again,
    so without this gate a caller whose configured scope at any of these
    levels excludes ``context_management_summary_model`` could still get the
    proxy to invoke that model and return its ``<summary>`` output as a
    compaction block. Mirrors the model-scope enforcement that
    ``litellm.proxy.auth.common_checks`` runs for the client-requested model:
    key, team, user (personal), project, and team-member allowlists.

    Returns True (allow) when ``user_api_key_auth`` is not present — SDK
    callers and tests run outside the proxy, where no key/team policy exists.
    Returns False when any of the active allowlists denies the summary model
    (``ProxyException`` from ``_can_object_call_model`` / ``can_*_model``).
    Unexpected errors during an access check fail closed but are logged
    separately so operators can distinguish them from a real access-denied
    response. DB-lookup failures (object missing from cache or DB) skip the
    corresponding scope — matching ``common_checks``, which only enforces a
    scope when its backing object can be loaded.
    """
    if user_api_key_auth is None:
        return True
    try:
        from litellm.proxy._types import ProxyException
        from litellm.proxy.auth.auth_checks import (
            _can_object_call_model,
            can_project_access_model,
            can_user_call_model,
            get_project_object,
            get_team_membership,
            get_user_object,
        )
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )
    except Exception:
        return True

    key_models = list(getattr(user_api_key_auth, "models", None) or [])
    team_id = getattr(user_api_key_auth, "team_id", None)
    team_model_aliases = getattr(user_api_key_auth, "team_model_aliases", None)
    team_models = list(getattr(user_api_key_auth, "team_models", None) or [])
    user_id = getattr(user_api_key_auth, "user_id", None)
    project_id = getattr(user_api_key_auth, "project_id", None)

    checks: Tuple[Tuple[Literal["key", "team"], List[str]], ...] = (
        ("key", key_models),
        ("team", team_models),
    )
    for object_type, models in checks:
        if not models:
            continue
        try:
            _can_object_call_model(
                model=summary_model,
                llm_router=llm_router,
                models=models,
                team_model_aliases=team_model_aliases,
                team_id=team_id,
                object_type=object_type,
            )
        except ProxyException:
            return False
        except Exception as e:
            verbose_logger.warning(
                "compact_20260112: unexpected error during %s-level access "
                "check for summary_model=%s; denying access: %s",
                object_type,
                summary_model,
                e,
            )
            return False

    if user_id is not None and prisma_client is not None:
        try:
            user_obj = await get_user_object(
                user_id=user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.debug(
                "compact_20260112: user object lookup failed for "
                "summary_model=%s access check; skipping user-level scope: %s",
                summary_model,
                e,
            )
            user_obj = None
        if user_obj is not None:
            try:
                await can_user_call_model(
                    model=summary_model,
                    llm_router=llm_router,
                    user_object=user_obj,
                )
            except ProxyException:
                return False
            except Exception as e:
                verbose_logger.warning(
                    "compact_20260112: unexpected error during user-level "
                    "access check for summary_model=%s; denying access: %s",
                    summary_model,
                    e,
                )
                return False

    if project_id is not None and prisma_client is not None:
        try:
            project_obj = await get_project_object(
                project_id=project_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.debug(
                "compact_20260112: project object lookup failed for "
                "summary_model=%s access check; skipping project-level scope: %s",
                summary_model,
                e,
            )
            project_obj = None
        if project_obj is not None and project_obj.models:
            try:
                can_project_access_model(
                    model=summary_model,
                    project_object=project_obj,
                    llm_router=llm_router,
                )
            except ProxyException:
                return False
            except Exception as e:
                verbose_logger.warning(
                    "compact_20260112: unexpected error during project-level "
                    "access check for summary_model=%s; denying access: %s",
                    summary_model,
                    e,
                )
                return False

    if user_id is not None and team_id is not None and prisma_client is not None:
        try:
            team_membership = await get_team_membership(
                user_id=user_id,
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.debug(
                "compact_20260112: team membership lookup failed for "
                "summary_model=%s access check; skipping member-level scope: %s",
                summary_model,
                e,
            )
            team_membership = None
        member_allowed_models = (
            team_membership.litellm_budget_table.allowed_models
            if team_membership is not None
            and team_membership.litellm_budget_table is not None
            else None
        )
        if member_allowed_models:
            try:
                _can_object_call_model(
                    model=summary_model,
                    llm_router=llm_router,
                    models=list(member_allowed_models),
                    team_model_aliases=team_model_aliases,
                    team_id=team_id,
                    object_type="team",
                )
            except ProxyException:
                return False
            except Exception as e:
                verbose_logger.warning(
                    "compact_20260112: unexpected error during member-level "
                    "access check for summary_model=%s; denying access: %s",
                    summary_model,
                    e,
                )
                return False

    return True


async def _check_summary_model_budget(
    user_api_key_auth: Any,
    summary_model: str,
) -> bool:
    """Return True when the caller is within their per-model budget for
    ``summary_model``.

    The summary subrequest never passes back through ``user_api_key_auth``, so
    without this gate a caller whose ``model_max_budget`` for
    ``context_management_summary_model`` is exhausted could keep consuming that
    model via compaction. Mirrors the ``model_max_budget`` /
    ``end_user_model_max_budget`` enforcement that ``user_api_key_auth`` runs for
    the client-requested model. Returns True outside the proxy or when no
    per-model budget is configured.
    """
    if user_api_key_auth is None:
        return True
    try:
        from litellm.proxy.proxy_server import model_max_budget_limiter
    except Exception:
        return True

    model_max_budget = getattr(user_api_key_auth, "model_max_budget", None)
    token = getattr(user_api_key_auth, "token", None)
    if isinstance(model_max_budget, dict) and model_max_budget and token is not None:
        try:
            await model_max_budget_limiter.is_key_within_model_budget(
                user_api_key_dict=user_api_key_auth,
                model=summary_model,
            )
        except litellm.BudgetExceededError:
            return False
        except Exception as e:
            verbose_logger.warning(
                "compact_20260112: unexpected error during key model-budget "
                "check for summary_model=%s; denying: %s",
                summary_model,
                e,
            )
            return False

    end_user_model_max_budget = getattr(
        user_api_key_auth, "end_user_model_max_budget", None
    )
    end_user_id = getattr(user_api_key_auth, "end_user_id", None)
    if (
        isinstance(end_user_model_max_budget, dict)
        and end_user_model_max_budget
        and end_user_id is not None
    ):
        try:
            await model_max_budget_limiter.is_end_user_within_model_budget(
                end_user_id=end_user_id,
                end_user_model_max_budget=end_user_model_max_budget,
                model=summary_model,
            )
        except litellm.BudgetExceededError:
            return False
        except Exception as e:
            verbose_logger.warning(
                "compact_20260112: unexpected error during end-user model-budget "
                "check for summary_model=%s; denying: %s",
                summary_model,
                e,
            )
            return False

    return True


async def _check_summary_model_rate_limit(
    user_api_key_auth: Any,
    summary_model: str,
) -> bool:
    """Return True when the caller is within their configured RPM/TPM limits
    for ``summary_model``.

    The summary subrequest never passes back through the proxy's pre-call
    rate limiter, so without this gate a caller already at their key / team /
    user RPM or TPM could still drive an extra summary-model completion per
    allowed ``/v1/messages`` request. This mirrors the read side of
    ``_PROXY_MaxParallelRequestsHandler_v3.async_pre_call_hook`` for the
    summary model: it builds the same descriptor set and runs the check in
    ``read_only`` mode so no counter is reserved or incremented — the summary
    call's actual usage is still charged exactly once by the limiter's
    post-call success hook (via the propagated ``litellm_metadata``).

    Returns True (allow) outside the proxy, when the active limiter does not
    expose the read-only descriptor check (legacy limiter), or when the
    descriptor set cannot be built — the only deny signal is a definitive
    ``OVER_LIMIT`` response, so an internal error here forwards the request
    uncompacted rather than blocking every summary.
    """
    if user_api_key_auth is None:
        return True
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj
    except Exception:
        return True

    limiter = getattr(proxy_logging_obj, "max_parallel_request_limiter", None)
    if (
        limiter is None
        or not hasattr(limiter, "should_rate_limit")
        or not hasattr(limiter, "_create_rate_limit_descriptors")
    ):
        return True

    try:
        metadata = getattr(user_api_key_auth, "metadata", None) or {}
        data = {"model": summary_model}
        descriptors = limiter._create_rate_limit_descriptors(
            user_api_key_dict=user_api_key_auth,
            data=data,
            rpm_limit_type=metadata.get("rpm_limit_type"),
            tpm_limit_type=metadata.get("tpm_limit_type"),
            model_has_failures=False,
        )
        limiter._add_team_model_rate_limit_descriptor_from_metadata(
            user_api_key_dict=user_api_key_auth,
            requested_model=summary_model,
            descriptors=descriptors,
        )
        limiter._add_project_model_rate_limit_descriptor_from_metadata(
            user_api_key_dict=user_api_key_auth,
            requested_model=summary_model,
            descriptors=descriptors,
        )
        descriptors.extend(
            limiter.create_organization_rate_limit_descriptor(
                user_api_key_auth, summary_model
            )
        )
        if not descriptors:
            return True
        response = await limiter.should_rate_limit(
            descriptors=descriptors,
            parent_otel_span=getattr(user_api_key_auth, "parent_otel_span", None),
            read_only=True,
        )
    except Exception as e:
        verbose_logger.warning(
            "compact_20260112: unexpected error during rate-limit check for "
            "summary_model=%s; allowing: %s",
            summary_model,
            e,
        )
        return True
    return response.get("overall_code") != "OVER_LIMIT"


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


def _propagate_metadata(
    parent_litellm_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract the parent request's auth/spend-attribution fields for the summary subcall.

    The proxy attaches ``user_api_key``, ``user_api_key_team_id`` etc. to
    ``data["litellm_metadata"]`` (see
    ``LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata``).
    Without these on the summary subrequest, the router's post-call hooks
    cannot attribute summary tokens to the caller's key/team budget.
    """
    if not parent_litellm_metadata:
        return {}
    propagated: Dict[str, Any] = {}
    for key in _PROPAGATED_METADATA_KEYS:
        if key in parent_litellm_metadata:
            propagated[key] = parent_litellm_metadata[key]
    return propagated


def _count_effective_tokens(
    model: str,
    effective_messages: List[Dict[str, Any]],
    compaction_block: Optional[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
) -> int:
    """Token-count the conversation as it will appear downstream.

    The compaction block (if any) becomes a system prefix on the downstream
    call, so its content still counts even though it isn't in ``messages``.
    The system prompt (which may already include a prior compaction summary
    prepended via ``_augment_system_with_summary``) is also counted so the
    threshold check matches the downstream ``input_tokens`` metric.
    """
    # Local import to avoid pulling the adapter at module load time.
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    messages_without_compaction = _strip_compaction_blocks(effective_messages)
    adapter = LiteLLMAnthropicMessagesAdapter()
    try:
        openai_shape = adapter.translate_anthropic_messages_to_openai(
            messages=cast(Any, messages_without_compaction)
        )
    except Exception as e:
        verbose_logger.debug(
            "compact_20260112: anthropic→openai translation failed during token "
            "count, falling back to raw messages: %s",
            e,
        )
        openai_shape = cast(Any, messages_without_compaction)

    # Translate Anthropic-shaped tools (``input_schema``) to OpenAI-shaped
    # tools (``{"type": "function", "function": {...}}``) so ``token_counter``
    # gets a consistent format regardless of which counting path it uses.
    # An inaccurate tool token count here could cause the polyfill to skip
    # needed compaction or trigger unnecessary summarization.
    openai_tools: Optional[List[Dict[str, Any]]] = None
    if tools:
        try:
            translated_tools, _ = adapter.translate_anthropic_tools_to_openai(
                tools=cast(Any, tools)
            )
            openai_tools = cast(List[Dict[str, Any]], translated_tools)
        except Exception as e:
            verbose_logger.debug(
                "compact_20260112: anthropic→openai tools translation failed "
                "during token count, falling back to raw tools: %s",
                e,
            )
            openai_tools = tools

    total = litellm.token_counter(
        model=model,
        messages=cast(Any, openai_shape),
        tools=cast(Any, openai_tools),
    )
    if compaction_block is not None:
        content = compaction_block.get("content") or ""
        if content:
            total += litellm.token_counter(model=model, text=content)
    system_text = _system_to_text(system)
    if system_text:
        total += litellm.token_counter(model=model, text=system_text)
    return total


def _system_to_text(
    system: Optional[Union[str, List[Dict[str, Any]]]],
) -> str:
    """Flatten an Anthropic-style ``system`` value into a single string for
    token counting. Returns ``""`` when ``system`` carries no text."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts: List[str] = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _select_last_user_question(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pick the most recent ``user`` turn that is a real question.

    Returns a one-element message list with any ``tool_result`` blocks
    stripped: after compaction the paired ``tool_use`` assistant turn no
    longer exists in the downstream context, so forwarding ``tool_result``
    blocks would translate to orphaned ``role=tool`` messages on
    non-Anthropic providers (OpenAI, Gemini, …) and cause a 400 error.

    Falls back to a synthetic continuation prompt if no eligible turn
    exists (e.g. the conversation only ever contained ``tool_result``
    turns, or contained no user turns at all). The downstream call always
    needs a non-empty user message.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            filtered = [
                blk
                for blk in content
                if not (isinstance(blk, dict) and blk.get("type") == "tool_result")
            ]
            if not filtered:
                # Purely tool_result — skip and look for an earlier turn.
                continue
            if len(filtered) < len(content):
                return [{**msg, "content": filtered}]
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


def _system_to_openai_message(
    system: Optional[Union[str, List[Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    """Translate Anthropic-shaped ``system`` to an OpenAI system message.

    Accepts a bare string or a list of Anthropic content blocks; returns
    ``None`` if no usable text is present. Only ``type=="text"`` blocks are
    carried over — the summary model has no use for ``cache_control`` or
    other non-text metadata.
    """
    if isinstance(system, str):
        return {"role": "system", "content": system} if system else None
    if isinstance(system, list):
        parts = [
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "\n\n".join(part for part in parts if part)
        return {"role": "system", "content": joined} if joined else None
    return None


def _build_summary_messages(
    effective_messages: List[Dict[str, Any]],
    prompt: str,
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Build the OpenAI-shape message list for the summary call.

    The caller's ``system`` prompt is prepended (the default summarization
    instructions reference "the initial task above", which lives in that
    system prompt); the conversation history is translated to OpenAI shape;
    the summarization prompt is appended as a final user turn.
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

    summary_messages: List[Dict[str, Any]] = []
    system_message = _system_to_openai_message(system)
    if system_message is not None:
        summary_messages.append(system_message)
    summary_messages.extend(openai_messages)
    # If the last turn is already a user message, merge the summarization
    # prompt into it. Some providers (and strict OpenAI-compatible endpoints)
    # reject two consecutive ``role=user`` messages, which would otherwise
    # silently fall into the ``summary_call_failed`` error path.
    if summary_messages and _is_user_message(summary_messages[-1]):
        last_msg = summary_messages[-1]
        summary_messages[-1] = {
            **last_msg,
            "content": _append_text_to_content(last_msg.get("content"), prompt),
        }
    else:
        summary_messages.append({"role": "user", "content": prompt})
    return summary_messages


def _is_user_message(msg: Any) -> bool:
    return isinstance(msg, dict) and msg.get("role") == "user"


def _append_text_to_content(content: Any, extra_text: str) -> Any:
    """Append ``extra_text`` to an OpenAI-shape message ``content`` field.

    Handles the two common shapes: ``str`` and ``list`` of content parts.
    For unexpected/empty shapes, fall back so the caller gets a usable value.
    """
    if content is None or content == "":
        return extra_text
    if isinstance(content, str):
        return f"{content}\n\n{extra_text}"
    if isinstance(content, list):
        return [*content, {"type": "text", "text": extra_text}]
    return [content, {"type": "text", "text": extra_text}]


async def _call_summary_model(
    *,
    summary_model: str,
    summary_messages: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    llm_router: Any,
    allowed_model_region: Optional[str] = None,
    max_tokens: int = COMPACT_SUMMARY_MAX_TOKENS,
) -> Any:
    """Invoke the configured summary model.

    Prefers ``llm_router.acompletion`` so the model alias resolves against the
    proxy's ``model_list``; falls back to ``litellm.acompletion`` if no router
    is available (e.g. SDK usage outside the proxy).
    """
    # ``max_tokens`` is required by providers like Anthropic and silently
    # accepted by providers that don't strictly require it (OpenAI etc.).
    # Setting a sensible default here means the feature works regardless of
    # which model an admin configures as ``context_management_summary_model``;
    # operators can override via ``context_management_summary_max_tokens`` in
    # ``general_settings`` when the default doesn't fit the chosen model's
    # output budget.
    # The propagated proxy auth/spend-attribution fields (``user_api_key`` etc.)
    # must travel as ``litellm_metadata`` — that is the parameter the proxy's
    # post-call spend hooks read for budget attribution. The provider-level
    # ``metadata`` kwarg corresponds to the upstream API request body and would
    # not flow into spend tracking.
    # ``allowed_model_region`` must travel as a top-level kwarg because the
    # router enforces region restrictions by reading ``request_kwargs`` directly
    # (see ``Router._common_checks_available_deployment``); without this the
    # summary subrequest could be routed to a deployment outside the caller's
    # permitted region.
    # ``timeout`` bounds how long a slow/unresponsive summary model can stall
    # the parent ``/v1/messages`` request. On timeout the caller catches the
    # exception and surfaces ``applied_edits[0].error = "summary_call_failed"``,
    # forwarding the request without compaction rather than hanging.
    call_kwargs: Dict[str, Any] = {
        "model": summary_model,
        "messages": summary_messages,
        "max_tokens": max_tokens,
        "timeout": COMPACT_SUMMARY_TIMEOUT_SECONDS,
        "litellm_metadata": metadata,
    }
    # The end-user id must also travel as the top-level ``user`` kwarg: legacy
    # limiter hooks and prometheus end-user tracking read it from there rather
    # than from ``litellm_metadata``, so without it the summary tokens would not
    # debit the caller's end-user counters.
    end_user_id = metadata.get("user_api_key_end_user_id")
    if end_user_id:
        call_kwargs["user"] = end_user_id
    if allowed_model_region is not None:
        call_kwargs["allowed_model_region"] = allowed_model_region
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


def apply_client_compaction_block_history(
    *,
    messages: List[Dict[str, Any]],
    system: Optional[Union[str, List[Dict[str, Any]]]],
) -> Optional[PolyfillResult]:
    """Honor client-sent compaction blocks without a ``compact_20260112`` edit.

    When the request omits ``context_management`` but the message history already
    contains a ``compaction`` content block (e.g. Claude Code client-side
    compaction), apply the same slice-only forwarding as the under-threshold
    path: the prior summary is prepended to ``system`` and the post-compaction
    tail is forwarded unchanged (with compaction blocks stripped) so recent
    turns the summary does not cover are preserved.
    """
    effective_messages, prior_compaction_block = _slice_around_compaction_block(
        messages
    )
    if prior_compaction_block is None:
        return None

    verbose_logger.info(
        "compact_20260112: client compaction block in message history; "
        "applying slice-only forwarding (no context_management edit)"
    )

    prior_summary_text = prior_compaction_block.get("content") or ""
    augmented_system: Union[str, List[Dict[str, Any]], None] = system
    if isinstance(prior_summary_text, str) and prior_summary_text:
        augmented_system = _augment_system_with_summary(system, prior_summary_text)
        verbose_logger.info(
            "compact_20260112: compaction summary added to main call system prefix (%s chars)",
            len(prior_summary_text),
        )

    # Post-compaction turns are recent context the prior summary does not cover,
    # so forward them unchanged. Only fall back to the last user question if the
    # strip leaves the downstream call with nothing to answer.
    downstream_messages = _strip_compaction_blocks(effective_messages)
    if not downstream_messages:
        downstream_messages = _select_last_user_question(effective_messages)

    return PolyfillResult(
        messages=downstream_messages,
        system=augmented_system,
        applied_edits=[],
    )


async def apply_compact_20260112(  # noqa: PLR0915
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    system: Optional[Union[str, List[Dict[str, Any]]]],
    edit_spec: Dict[str, Any],
    litellm_metadata: Optional[Dict[str, Any]] = None,
    llm_router: Any = None,
    user_api_key_auth: Any = None,
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
    verbose_logger.info(
        "compact_20260112: request has compaction trigger (input_tokens threshold=%s)",
        trigger_tokens,
    )
    if edit_spec.get("pause_after_compaction"):
        warnings.append("pause_after_compaction_ignored")

    applied: AppliedEdit = {"type": COMPACT_EDIT_TYPE}
    if warnings:
        applied["warnings"] = warnings

    # Phase A: slice around any existing compaction block. Runs before the
    # opt-in gate below so that even when summarization is disabled we still
    # strip Anthropic-only ``compaction`` blocks from messages going to
    # non-Anthropic backends (which would reject them).
    effective_messages, prior_compaction_block = _slice_around_compaction_block(
        messages
    )
    prior_summary_text = (
        prior_compaction_block.get("content") if prior_compaction_block else None
    )
    augmented_system: Union[str, List[Dict[str, Any]], None] = system
    if isinstance(prior_summary_text, str) and prior_summary_text:
        augmented_system = _augment_system_with_summary(system, prior_summary_text)
        verbose_logger.info(
            "compact_20260112: compaction summary added to main call system prefix (%s chars)",
            len(prior_summary_text),
        )

    downstream_messages = _strip_compaction_blocks(effective_messages)

    # Opt-in gate: no summary model configured → no-op (but still return the
    # Phase A-sliced/stripped messages so compaction blocks don't leak).
    summary_model = _read_summary_model_setting()
    if summary_model is None:
        applied["error"] = "summary_model_not_configured"
        # Slice-only forwarding: ``augmented_system`` already carries any prior
        # compaction summary, and the post-compaction tail in
        # ``downstream_messages`` is recent context the summary does not cover,
        # so forward it unchanged. Only fall back to the last user question when
        # the strip leaves nothing for the downstream call to answer.
        if not downstream_messages:
            downstream_messages = _select_last_user_question(effective_messages)
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    # Phase B: threshold check.
    try:
        current_tokens = _count_effective_tokens(
            model=model,
            effective_messages=effective_messages,
            # ``augmented_system`` already carries the prior compaction summary
            # (prepended via ``_augment_system_with_summary``); pass ``None``
            # here so we don't double-count the summary text.
            compaction_block=None,
            tools=tools,
            system=augmented_system,
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
        # Slice-only path: the prior compaction summary already lives in
        # ``augmented_system``. Post-compaction turns are recent context the
        # summary does not cover, so forward ``downstream_messages`` (the
        # post-compaction tail with compaction blocks stripped) unchanged.
        # Only fall back to the last user question when the strip leaves
        # nothing for the downstream call to answer.
        if not downstream_messages:
            downstream_messages = _select_last_user_question(effective_messages)
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    # Phase C: summarize. ``augmented_system`` carries any prior compaction
    # summary so multi-round compaction does not lose accumulated history —
    # ``effective_messages`` only contains turns since the last compaction.
    if not await _check_summary_model_access(
        user_api_key_auth=user_api_key_auth,
        summary_model=summary_model,
        llm_router=llm_router,
    ):
        verbose_logger.warning(
            "compact_20260112: caller not authorized for summary_model=%s; "
            "skipping summary call",
            summary_model,
        )
        applied["error"] = "summary_model_access_denied"
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    if not await _check_summary_model_budget(
        user_api_key_auth=user_api_key_auth,
        summary_model=summary_model,
    ):
        verbose_logger.warning(
            "compact_20260112: caller over model budget for summary_model=%s; "
            "skipping summary call",
            summary_model,
        )
        applied["error"] = "summary_model_budget_exceeded"
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    if not await _check_summary_model_rate_limit(
        user_api_key_auth=user_api_key_auth,
        summary_model=summary_model,
    ):
        verbose_logger.warning(
            "compact_20260112: caller over rate limit for summary_model=%s; "
            "skipping summary call",
            summary_model,
        )
        applied["error"] = "summary_model_rate_limit_exceeded"
        return PolyfillResult(
            messages=downstream_messages,
            system=augmented_system,
            applied_edits=[applied],
        )

    prompt = _build_summary_prompt(edit_spec, tools)
    summary_messages = _build_summary_messages(
        effective_messages, prompt, system=augmented_system
    )
    propagated_metadata = _propagate_metadata(litellm_metadata)
    allowed_model_region = getattr(user_api_key_auth, "allowed_model_region", None)

    try:
        response = await _call_summary_model(
            summary_model=summary_model,
            summary_messages=summary_messages,
            metadata=propagated_metadata,
            llm_router=llm_router,
            allowed_model_region=allowed_model_region,
            max_tokens=_read_summary_max_tokens_setting(),
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
    verbose_logger.info(
        "compact_20260112: compaction summary added to main call system prefix (%s chars)",
        len(summary_text),
    )
    downstream_messages_after_summary = _select_last_user_question(effective_messages)

    return PolyfillResult(
        messages=downstream_messages_after_summary,
        system=summarized_system,
        applied_edits=[applied],
        compaction_block=compaction_block,
        iterations_usage=iterations_usage,
    )
