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
from litellm.llms.anthropic.common_utils import strip_advisor_blocks_from_messages
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE

ADVISOR_MAX_USES: int = _c.ADVISOR_MAX_USES
ADVISOR_NATIVE_PROVIDERS: frozenset = _c.ADVISOR_NATIVE_PROVIDERS
ADVISOR_TOOL_DESCRIPTION: str = _c.ADVISOR_TOOL_DESCRIPTION

from .base import MessagesInterceptor


class AdvisorMaxIterationsError(Exception):
    """Raised when the advisor loop exceeds max_uses."""


class AdvisorOrchestrationHandler(MessagesInterceptor):
    """Orchestrates the advisor tool loop for non-native providers."""

    def can_handle(
        self,
        tools: Optional[List[Dict]],
        custom_llm_provider: Optional[str],
    ) -> bool:
        if not tools:
            return False
        has_advisor = any(t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE for t in tools)
        is_non_native = custom_llm_provider not in ADVISOR_NATIVE_PROVIDERS
        return has_advisor and is_non_native

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
            raise ValueError(
                "advisor tool definition must include a 'model' field specifying the advisor model"
            )
        _raw_max_uses = advisor_tool.get("max_uses")
        max_uses: int = (
            ADVISOR_MAX_USES if _raw_max_uses is None else int(_raw_max_uses)
        )
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
            advisor_response: AnthropicMessagesResponse = await _call_messages_handler(
                model=advisor_model,
                messages=advisor_messages,
                tools=None,
                stream=False,
                max_tokens=max_tokens,
                custom_llm_provider=None,  # let litellm resolve from model name
                metadata={
                    **metadata_base,
                    "advisor_sub_call": True,
                    "parent_request_id": parent_request_id,
                },
                api_key=advisor_api_key,
                api_base=advisor_api_base,
            )

            advisor_text = _extract_response_text(advisor_response)

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


def _make_synthetic_advisor_tool() -> Dict:
    """Build a regular tool definition the executor provider can understand."""
    return {
        "name": "advisor",
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
    """Return the first tool_use block with name='advisor', or None."""
    content = response.get("content") if isinstance(response, dict) else []
    if not isinstance(content, list):
        return None
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == "advisor"
        ):
            return block
    return None


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
