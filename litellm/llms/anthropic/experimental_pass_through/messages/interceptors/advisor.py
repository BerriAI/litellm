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
6. Surfaces each advisor exchange to the caller as native server_tool_use /
   advisor_tool_result blocks so clients render the advisor activity; when
   streaming, the call block is flushed before the advisor runs so it renders as
   in-progress, then resolves when the result arrives.
"""

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

import litellm.constants as _c
from litellm.llms.anthropic.common_utils import strip_advisor_blocks_from_messages
from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    _sse,
    build_content_block_chunks,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE

ADVISOR_MAX_USES: int = _c.ADVISOR_MAX_USES
ADVISOR_NATIVE_PROVIDERS: frozenset = _c.ADVISOR_NATIVE_PROVIDERS
ADVISOR_TOOL_DESCRIPTION: str = _c.ADVISOR_TOOL_DESCRIPTION
ADVISOR_SYSTEM_PROMPT: str = _c.ADVISOR_SYSTEM_PROMPT

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
        advisor_api_key: Optional[str] = advisor_tool.get("api_key")
        advisor_api_base: Optional[str] = advisor_tool.get("api_base")

        await _check_advisor_model_access(advisor_model, kwargs)

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

        # Carry only the proxy's budget/auth/session context (litellm_metadata)
        # into the advisor leg so its spend is attributed to the caller's key and
        # groups under the same session. Deliberately do NOT inherit the executor's
        # generation params (system, tool_choice, thinking, sampling): the advisor
        # is a fresh consultation, and feeding it the executor's agent system prompt
        # makes it mimic the executor and echo the advisor call instead of
        # answering. api_key/api_base come from the advisor tool definition only —
        # the executor's would be None under the router and override the advisor
        # deployment's resolved credentials.
        advisor_kwargs: Dict = {}
        if "litellm_metadata" in kwargs:
            advisor_kwargs["litellm_metadata"] = kwargs["litellm_metadata"]
        if advisor_api_key is not None:
            advisor_kwargs["api_key"] = advisor_api_key
        if advisor_api_base is not None:
            advisor_kwargs["api_base"] = advisor_api_base

        loop = self._run_loop(
            model=model,
            current_messages=current_messages,
            executor_tools=executor_tools,
            advisor_model=advisor_model,
            max_uses=max_uses,
            max_tokens=max_tokens,
            custom_llm_provider=custom_llm_provider,
            parent_request_id=parent_request_id,
            metadata_base=metadata_base,
            advisor_kwargs=advisor_kwargs,
            kwargs=kwargs,
        )

        if stream:
            return self._stream(loop, model=model, request_id=parent_request_id)
        return await self._collect(loop)

    async def _run_loop(
        self,
        *,
        model: str,
        current_messages: List[Dict],
        executor_tools: List[Dict],
        advisor_model: str,
        max_uses: int,
        max_tokens: int,
        custom_llm_provider: Optional[str],
        parent_request_id: str,
        metadata_base: Dict,
        advisor_kwargs: Dict,
        kwargs: Dict,
    ) -> AsyncIterator[tuple]:
        """Drive the executor/advisor loop, yielding semantic events so streaming
        and non-streaming consumers share one orchestration path. The advisor
        sub-call runs between the "advisor_call" and "advice" events, so a
        streaming consumer that flushes "advisor_call" first surfaces the advisor
        as in-progress for the duration of its real latency. The advisor legs are
        accumulated into a usage.iterations[] list (carried on the "final" event)
        so clients attribute the advisor model's tokens to its own cost line."""
        iteration = 0
        iterations: List[Dict] = []
        while True:
            executor_response = await _call_messages_handler(
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
                iterations.append(_iteration_entry("message", model, executor_response))
                yield ("final", executor_response, iterations)
                return

            iteration += 1
            if iteration > max_uses:
                raise AdvisorMaxIterationsError(
                    f"Advisor orchestration loop exceeded max_uses={max_uses}. "
                    "Increase max_uses in the advisor tool definition or cap the request."
                )

            yield ("advisor_call", executor_response, advisor_use_block)

            advisor_messages = _build_advisor_context(
                current_messages, executor_response, advisor_use_block
            )
            advisor_response = await _call_messages_handler(
                model=advisor_model,
                messages=advisor_messages,
                tools=None,
                stream=False,
                max_tokens=max_tokens,
                custom_llm_provider=None,
                system=ADVISOR_SYSTEM_PROMPT,
                metadata={
                    **metadata_base,
                    "advisor_sub_call": True,
                    "parent_request_id": parent_request_id,
                },
                **advisor_kwargs,
            )
            advisor_text = _extract_response_text(advisor_response)
            iterations.append(
                _iteration_entry("advisor_message", advisor_model, advisor_response)
            )

            yield ("advice", advisor_use_block, advisor_text)

            current_messages = _inject_advisor_turn(
                current_messages,
                executor_response,
                advisor_use_block,
                advisor_text,
            )

    async def _collect(self, loop: AsyncIterator[tuple]) -> AnthropicMessagesResponse:
        """Accumulate the loop's advisor exchanges and executor text into a single
        non-streaming response."""
        output_content: List[Dict] = []
        async for kind, a, b in loop:
            if kind == "advisor_call":
                output_content.extend(_executor_text_blocks(a))
                output_content.append(_server_tool_use_block(b))
            elif kind == "advice":
                output_content.append(_advisor_result_block(a, b))
            elif kind == "final":
                return {
                    **a,
                    "content": output_content + list(a.get("content") or []),
                    "usage": cast(
                        AnthropicUsage,
                        {**(a.get("usage") or {}), "iterations": b},
                    ),
                }
        raise AdvisorMaxIterationsError("Advisor loop ended without a final response")

    async def _stream(
        self,
        loop: AsyncIterator[tuple],
        *,
        model: str,
        request_id: str,
    ) -> AsyncIterator[bytes]:
        """Emit the advisor exchanges as Anthropic SSE as the loop runs. The
        server_tool_use block is flushed before the advisor sub-call is awaited,
        so the client renders the advisor as running until its result arrives."""
        index = 0
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

        async for kind, a, b in loop:
            if kind == "advisor_call":
                for block in _executor_text_blocks(a):
                    for chunk in build_content_block_chunks(block, index):
                        yield chunk
                    index += 1
                for chunk in build_content_block_chunks(
                    _server_tool_use_block(b), index
                ):
                    yield chunk
                index += 1
            elif kind == "advice":
                for chunk in build_content_block_chunks(
                    _advisor_result_block(a, b), index
                ):
                    yield chunk
                index += 1
            elif kind == "final":
                for block in a.get("content") or []:
                    for chunk in build_content_block_chunks(block, index):
                        yield chunk
                    index += 1
                for chunk in _final_message_events(a, b):
                    yield chunk


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


def _executor_text_blocks(executor_response: Any) -> List[Dict]:
    """Executor's text blocks, stripped of provider-specific fields."""
    raw_content = (
        executor_response.get("content") if isinstance(executor_response, dict) else []
    ) or []
    return [
        {k: v for k, v in block.items() if k not in _PROVIDER_SPECIFIC_KEYS}
        for block in raw_content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _server_tool_use_block(advisor_use_block: Dict) -> Dict:
    """The advisor call, in Anthropic's native server-side advisor shape, so clients
    render the advisor activity instead of seeing a flattened response."""
    return {
        "type": "server_tool_use",
        "id": advisor_use_block.get("id", ""),
        "name": "advisor",
        "input": {},
    }


def _advisor_result_block(advisor_use_block: Dict, advisor_text: str) -> Dict:
    """The advisor reply that resolves the matching server_tool_use call."""
    return {
        "type": "advisor_tool_result",
        "tool_use_id": advisor_use_block.get("id", ""),
        "content": {"type": "advisor_result", "text": advisor_text},
    }



_USAGE_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _iteration_entry(entry_type: str, model: str, response: Any) -> Dict:
    """A usage.iterations[] entry. Clients attribute an advisor_message entry's
    tokens to its own model, so the advisor leg gets its own cost line instead of
    folding into the executor's."""
    usage = (response.get("usage") if isinstance(response, dict) else None) or {}
    entry: Dict[str, Any] = {"type": entry_type, "model": model}
    for key in _USAGE_TOKEN_KEYS:
        entry[key] = usage.get(key, 0)
    return entry


def _final_message_events(
    executor_response: Any, iterations: List[Dict]
) -> List[bytes]:
    """The closing message_delta (stop reason + usage, including the advisor
    iterations) and message_stop."""
    usage = (
        executor_response.get("usage") if isinstance(executor_response, dict) else None
    ) or {}
    delta_usage: Dict[str, Any] = {"output_tokens": usage.get("output_tokens", 0)}
    for key in (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        if usage.get(key) is not None:
            delta_usage[key] = usage[key]
    delta_usage["iterations"] = iterations
    return [
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": executor_response.get("stop_reason"),
                    "stop_sequence": executor_response.get("stop_sequence"),
                },
                "usage": delta_usage,
            },
        ),
        _sse(
            "message_stop",
            {"type": "message_stop", "usage": {**usage, "iterations": iterations}},
        ),
    ]


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
    result = list(messages)
    executor_text = _executor_text_blocks(executor_response)
    if executor_text:
        result.append({"role": "assistant", "content": executor_text})
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


def _get_llm_router():
    try:
        from litellm.proxy.proxy_server import llm_router

        return llm_router
    except ImportError:
        return None


async def _check_advisor_model_access(advisor_model: str, kwargs: Dict) -> None:
    """Validate that the caller's API key is allowed to invoke the advisor model.

    Only runs when a proxy router is available (i.e. inside the proxy); standalone
    SDK usage has no auth layer and skips the check."""
    router = _get_llm_router()
    if router is None:
        return

    litellm_metadata = kwargs.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        return

    user_api_key_auth = litellm_metadata.get("user_api_key_auth")
    if user_api_key_auth is None:
        return

    import litellm
    from litellm.proxy.auth.auth_checks import can_key_call_model

    await can_key_call_model(
        model=advisor_model,
        llm_model_list=getattr(litellm, "model_list", None),
        valid_token=user_api_key_auth,
        llm_router=router,
    )


async def _call_messages_handler(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    stream: bool,
    max_tokens: int,
    custom_llm_provider: Optional[str],
    **kwargs,
) -> Any:
    """Route through the proxy's llm_router when available; fall back to
    direct anthropic_messages() for standalone SDK usage."""
    router = _get_llm_router()
    if router is not None:
        return await router.anthropic_messages(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            max_tokens=max_tokens,
            **kwargs,
        )

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
