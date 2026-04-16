"""
Advisor Orchestration Handler

Implements the advisor tool loop for providers that don't support
advisor_20260301 natively (i.e. everything except Anthropic direct for now).

How it works:
1. Detects advisor_20260301 in tools + non-native provider → intercepts.
2. Translates the advisor tool to a regular function tool the provider understands.
3. Calls the executor model (non-streaming).
4. If the executor makes a tool_use call named "advisor", runs the advisor model
   and injects the result as a tool_result before re-calling the executor.
5. Repeats until the executor produces a final text response or max_uses is hit.
6. Wraps in FakeAnthropicMessagesStreamIterator if the caller requested streaming.
"""

import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import litellm.constants as _c
from litellm._logging import verbose_logger
from litellm.llms.anthropic.common_utils import strip_advisor_blocks_from_messages
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE
from litellm.utils import (
    resolve_proxy_model_alias_to_litellm_model,
    supports_native_advisor_tool,
)

ADVISOR_MAX_USES: int = _c.ADVISOR_MAX_USES
ADVISOR_TOOL_DESCRIPTION: str = _c.ADVISOR_TOOL_DESCRIPTION

from .base import MessagesInterceptor


class AdvisorMaxIterationsError(Exception):
    """Raised when the advisor loop exceeds max_uses."""


class AdvisorOrchestrationHandler(MessagesInterceptor):
    """Orchestrates the advisor tool loop for /v1/messages requests."""

    def can_handle(
        self,
        tools: Optional[List[Dict]],
        custom_llm_provider: Optional[str],
    ) -> bool:
        if not tools:
            return False
        has_advisor = any(t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE for t in tools)
        if not has_advisor:
            return False
        # Keep Anthropic-native advisor behavior for Claude Opus 4.6.
        if _should_use_native_anthropic_advisor(tools, custom_llm_provider):
            return False
        return True

    async def handle(
        self,
        *,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: Optional[bool],
        max_tokens: int,
        custom_llm_provider: Optional[str],
        **kwargs,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
            FakeAnthropicMessagesStreamIterator,
        )

        # Extract advisor tool config.
        advisor_tool = next(
            (t for t in (tools or []) if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE),
            None,
        )
        if advisor_tool is None:
            raise ValueError(
                f"handle() called but no {ANTHROPIC_ADVISOR_TOOL_TYPE} tool found in tools list"
            )
        advisor_model: str = advisor_tool.get("model") or ""
        if not advisor_model:
            advisor_model = _resolve_default_advisor_model()
        if not advisor_model:
            raise ValueError(
                "No advisor model specified. Either:\n"
                "  1. Set 'default_advisor_model' in advisor_interception_params in your proxy config YAML, or\n"
                "  2. Include a 'model' field in the advisor tool definition."
            )
        _raw_max_uses = advisor_tool.get("max_uses")
        max_uses: int = ADVISOR_MAX_USES if _raw_max_uses is None else int(_raw_max_uses)
        # Optional routing overrides for the advisor sub-call (e.g. proxy routing).
        # If not set in the tool definition, litellm resolves from env vars.
        advisor_api_key: Optional[str] = advisor_tool.get("api_key")
        advisor_api_base: Optional[str] = advisor_tool.get("api_base")

        # Build the synthetic tool definition the provider will receive.
        synthetic_advisor_tool = _make_synthetic_advisor_tool()

        # Executor tools = all original tools with advisor replaced by the synthetic one.
        executor_tools: List[Dict] = [
            (
                synthetic_advisor_tool
                if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE
                else t
            )
            for t in (tools or [])
        ]

        # Strip prior advisor blocks from history, preserving advice text as context.
        current_messages: List[Dict] = strip_advisor_blocks_from_messages(
            [dict(m) for m in messages], replace_with_text=True
        )

        parent_request_id: str = str(
            kwargs.pop("litellm_call_id", None) or uuid.uuid4()
        )
        metadata_base: Dict = dict(kwargs.pop("metadata", None) or {})
        iteration = 0
        advisor_interactions: List[Dict] = []

        while True:
            # --- Executor call (always non-streaming) ---
            executor_response: AnthropicMessagesResponse = await _call_messages_handler(
                model=model,
                messages=current_messages,
                tools=executor_tools,
                stream=False,
                max_tokens=max_tokens,
                custom_llm_provider=custom_llm_provider,
                metadata={
                    **metadata_base,
                    "advisor_sub_call": False,
                    "parent_request_id": parent_request_id,
                },
                **kwargs,
            )

            advisor_use_block = _find_advisor_tool_use(executor_response)

            if advisor_use_block is None:
                # No more advisor calls — this is the final response.
                # Inject advisor_tool_result blocks to match Anthropic native format.
                _inject_advisor_blocks_into_response(
                    executor_response, advisor_interactions
                )
                if stream:
                    return FakeAnthropicMessagesStreamIterator(executor_response)
                return executor_response

            iteration += 1
            if iteration > max_uses:
                raise AdvisorMaxIterationsError(
                    f"Advisor orchestration loop exceeded max_uses={max_uses}. "
                    "Increase max_uses in the advisor tool definition or cap the request."
                )

            # --- Build advisor context ---
            advisor_messages = _build_advisor_context(
                current_messages, executor_response, advisor_use_block
            )

            # --- Advisor sub-call (always non-streaming, no tools) ---
            advisor_response: AnthropicMessagesResponse = await _call_advisor_with_router(
                model=advisor_model,
                messages=advisor_messages,
                max_tokens=max_tokens,
                metadata={
                    **metadata_base,
                    "advisor_sub_call": True,
                    "parent_request_id": parent_request_id,
                },
                api_key=advisor_api_key,
                api_base=advisor_api_base,
            )

            advisor_text = _extract_response_text(advisor_response)

            # Record the interaction for later injection into the final response.
            advisor_interactions.append({
                "tool_use_id": advisor_use_block.get("id", f"srvtoolu_{uuid.uuid4().hex[:24]}"),
                "advisor_text": advisor_text,
            })

            # --- Inject advisor result and continue loop ---
            current_messages = _inject_advisor_turn(
                current_messages,
                executor_response,
                advisor_use_block,
                advisor_text,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_default_advisor_model() -> str:
    """Resolve the default advisor model from proxy config / litellm settings."""
    import litellm

    params = getattr(litellm, "advisor_interception_params", None) or {}
    return params.get("default_advisor_model", "") or ""


def _should_use_native_anthropic_advisor(
    tools: List[Dict], custom_llm_provider: Optional[str]
) -> bool:
    """
    Use Anthropic's native advisor path only when:
    - executor provider is Anthropic, and
    - advisor model supports the native advisor capability.
    """
    if custom_llm_provider != "anthropic":
        return False

    advisor_tool = next(
        (t for t in tools if t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE),
        None,
    )
    if advisor_tool is None:
        return False

    advisor_model = (advisor_tool.get("model") or _resolve_default_advisor_model() or "").strip()
    if not advisor_model:
        return False

    # Proxy requests commonly pass advisor model as a model_name alias.
    resolved_proxy_model = resolve_proxy_model_alias_to_litellm_model(advisor_model)
    model_to_check = resolved_proxy_model or advisor_model
    if supports_native_advisor_tool(
        model=model_to_check, custom_llm_provider="anthropic"
    ):
        return True

    try:
        import litellm

        resolved_model, advisor_provider, _, _ = litellm.get_llm_provider(
            model=advisor_model
        )
        return advisor_provider == "anthropic" and supports_native_advisor_tool(
            model=resolved_model, custom_llm_provider=advisor_provider
        )
    except Exception:
        return False


_SYNTHETIC_ADVISOR_TOOL_NAME = "consult_advisor"


def _make_synthetic_advisor_tool() -> Dict:
    """Build a regular tool definition the executor provider can understand.

    Uses a name that does NOT collide with ``_ADVISOR_TOOL_NAMES`` in the
    chat-completions interception handler so pre-request hooks won't
    double-convert it.
    """
    return {
        "name": _SYNTHETIC_ADVISOR_TOOL_NAME,
        "description": ADVISOR_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or challenge you want guidance on.",
                }
            },
            "required": ["question"],
        },
    }


def _find_advisor_tool_use(response: Any) -> Optional[Dict]:
    """Return the first tool_use block whose name matches our synthetic advisor."""
    content = response.get("content") if isinstance(response, dict) else []
    if not isinstance(content, list):
        return None
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == _SYNTHETIC_ADVISOR_TOOL_NAME
        ):
            return block
    return None


def _openai_response_to_anthropic_dict(response: Any) -> Dict:
    """Convert an OpenAI ChatCompletion response to a minimal Anthropic Messages dict.

    Only the fields used by ``_extract_response_text`` are needed.
    """
    try:
        choices = response.choices if hasattr(response, "choices") else []
        content_blocks: List[Dict] = []
        for choice in choices:
            msg = choice.message if hasattr(choice, "message") else choice.get("message", {})
            text = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if text:
                content_blocks.append({"type": "text", "text": text})
        return {
            "id": getattr(response, "id", ""),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": "end_turn",
        }
    except Exception:
        return {"content": [], "stop_reason": "end_turn"}


def _inject_advisor_blocks_into_response(
    response: Any, advisor_interactions: List[Dict]
) -> None:
    """
    Mutate *response* in place so its ``content`` array includes
    ``advisor_tool_result`` blocks that mirror Anthropic's native advisor
    response format.

    Each advisor interaction produces two blocks appended after existing
    content:

    * ``server_tool_use``  – records the executor's call to the advisor
    * ``advisor_tool_result`` – carries the advisor's answer

    This ensures callers see an identical structure regardless of whether the
    advisor ran natively or via LiteLLM interception.
    """
    if not advisor_interactions:
        return

    content = response.get("content") if isinstance(response, dict) else None
    if not isinstance(content, list):
        return

    for interaction in advisor_interactions:
        tool_use_id = interaction["tool_use_id"]
        advisor_text = interaction["advisor_text"]

        content.append({
            "type": "server_tool_use",
            "id": tool_use_id,
            "name": "advisor",
        })
        content.append({
            "type": "advisor_tool_result",
            "tool_use_id": tool_use_id,
            "content": {
                "type": "advisor_result",
                "text": advisor_text,
            },
        })


def _extract_response_text(response: Any) -> str:
    """Extract concatenated text from all text blocks in a response."""
    content = response.get("content") if isinstance(response, dict) else []
    if not isinstance(content, list):
        return ""
    parts = [
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "\n".join(parts).strip()


_PROVIDER_SPECIFIC_KEYS = frozenset({"provider_specific_fields"})


def _build_advisor_context(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
) -> List[Dict]:
    """
    Build the message list for the advisor sub-call.

    Passes the full conversation + any text the executor produced so far, then
    poses the advisor question as the last user turn.

    tool_use blocks are excluded because Anthropic requires tool_use to be
    immediately followed by tool_result — not the advisor question.
    """
    question = (advisor_use_block.get("input") or {}).get("question") or (
        "Please provide guidance on the current task."
    )
    raw_content = (
        executor_response.get("content") if isinstance(executor_response, dict) else []
    ) or []
    # Keep only text blocks — strip tool_use and provider-specific fields.
    executor_text_blocks = [
        {k: v for k, v in block.items() if k not in _PROVIDER_SPECIFIC_KEYS}
        for block in raw_content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    result = list(messages)
    if executor_text_blocks:
        result.append({"role": "assistant", "content": executor_text_blocks})
    result.append({"role": "user", "content": question})
    return result


def _inject_advisor_turn(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
    advisor_text: str,
) -> List[Dict]:
    """
    Append the executor's response (as an assistant turn) and the advisor
    result (as a user tool_result turn) so the executor can continue.
    """
    executor_content = (
        executor_response.get("content") if isinstance(executor_response, dict) else []
    ) or []
    tool_use_id = advisor_use_block.get("id", "")
    return [
        *messages,
        {"role": "assistant", "content": executor_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": advisor_text,
                }
            ],
        },
    ]


def _inject_max_uses_error(
    messages: List[Dict],
    executor_response: Any,
    advisor_use_block: Dict,
) -> List[Dict]:
    """
    Inject a max_uses_exceeded error tool_result so the executor continues
    without further advisor calls (mirrors Anthropic's server-side behaviour).
    """
    executor_content = (
        executor_response.get("content") if isinstance(executor_response, dict) else []
    ) or []
    tool_use_id = advisor_use_block.get("id", "")
    return [
        *messages,
        {"role": "assistant", "content": executor_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "Advisor unavailable: max_uses limit reached. Continue without advisor guidance.",
                }
            ],
        },
    ]


async def _call_messages_handler(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    stream: bool,
    max_tokens: int,
    custom_llm_provider: Optional[str],
    **kwargs,
) -> Any:
    """
    Call anthropic_messages() — the public async /messages entry point — for
    orchestration sub-calls (executor or advisor).

    Using the public function (decorated with @client) ensures logging, retries,
    and provider resolution all work correctly, identical to a direct user call.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    return await anthropic_messages(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        max_tokens=max_tokens,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


async def _call_advisor_with_router(
    model: str,
    messages: List[Dict],
    max_tokens: int,
    metadata: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Any:
    """
    Call the advisor model via ``llm_router.acompletion()`` (proxy) or
    ``litellm.acompletion()`` (SDK-only).

    Returns a dict in Anthropic Messages format so the orchestration loop
    can process it uniformly.
    """
    import litellm as _litellm

    llm_router = None
    try:
        from litellm.proxy.proxy_server import llm_router as _router

        llm_router = _router
    except ImportError:
        pass

    kwargs: Dict[str, Any] = {}
    if metadata is not None:
        kwargs["metadata"] = metadata
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base is not None:
        kwargs["api_base"] = api_base

    openai_response = None
    if llm_router is not None:
        try:
            openai_response = await llm_router.acompletion(
                model=model,
                messages=messages,
                tools=None,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception:
            verbose_logger.debug(
                "AdvisorOrchestration: Router call for advisor model '%s' failed, "
                "falling back to direct litellm.acompletion()",
                model,
            )

    if openai_response is None:
        openai_response = await _litellm.acompletion(
            model=model,
            messages=messages,
            tools=None,
            max_tokens=max_tokens,
            **kwargs,
        )

    return _openai_response_to_anthropic_dict(openai_response)
