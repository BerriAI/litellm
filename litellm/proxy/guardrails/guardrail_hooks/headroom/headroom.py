from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypeGuard

import httpx
from fastapi import HTTPException
from httpx import Response as HttpxResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.compression.compress import get_protected_indices
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    get_attribute_or_key,
    get_tool_calls_from_response,
    group_tool_exchanges,
    has_tool_with_name,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.content_text import (
    assistant_text_from_response,
    content_to_text,
    is_all_text_parts,
    merge_rewritten_text_parts,
)
from litellm.proxy.spend_tracking.compression_savings import HEADROOM_GUARDRAIL_PROVIDER
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

BYPASS_HEADER: Final = "x-headroom-bypass"
HEADROOM_RETRIEVE_TOOL_NAME: Final = "headroom_retrieve"
_HASH_PATTERN: Final = re.compile(r"hash=([a-f0-9]{24})")
_HASH_CACHE_TTL_SECONDS: Final = 15 * 60


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


def _flatten_messages_for_compression(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse all-text list-of-parts content to plain strings for /v1/compress.

    The compression service's transforms only rewrite string content and skip
    the OpenAI list-of-parts shape, which is what every Anthropic-format
    request translates to. Only rows whose parts are ALL text are flattened:
    cache_control breakpoints are positional (each caches the prefix ending
    at its part), so merging text across a non-text part would move a later
    breakpoint to the other side of it. Rows with non-text parts are sent
    unchanged and pass through the service untouched.
    """
    flattened: Final[list[dict[str, object]]] = []
    for msg in messages:
        content = msg.get("content")
        if is_all_text_parts(content):
            text = content_to_text(content)
            if text:
                flattened.append({**msg, "content": text})
                continue
        flattened.append(msg)
    return flattened


def _restore_content_shapes(
    originals: list[dict[str, object]], returned: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Write compressed text back into each original row's content shape.

    Rows are matched positionally; the pairing is only trusted when the
    service kept the row count and every role lines up. If it restructured
    the conversation (e.g. dropped rows), its output is adopted as-is, which
    is the pre-flattening behavior.
    """
    if len(returned) != len(originals):
        return returned
    for orig, ret in zip(originals, returned):
        if orig.get("role") != ret.get("role"):
            return returned
    restored: Final[list[dict[str, object]]] = []
    for orig, ret in zip(originals, returned):
        orig_content = orig.get("content")
        ret_content = ret.get("content")
        if isinstance(orig_content, list) and isinstance(ret_content, str):
            if ret_content == content_to_text(orig_content):
                # Untouched row: keep the exact original parts, including
                # per-part fields like cache_control on later text parts.
                restored.append({**ret, "content": orig_content})
            else:
                restored.append({**ret, "content": merge_rewritten_text_parts(orig_content, ret_content)})
        else:
            restored.append(ret)
    return restored


def _protected_indices(messages: Sequence[Mapping[str, object]]) -> frozenset[int]:
    """Indices headroom must not send to the compression service.

    ``get_protected_indices`` is litellm's own compression policy: the system
    rows, the last user row, the last assistant row. It is expanded over whole
    tool exchanges the way ``compress()`` expands it, so a protected assistant
    tool call cannot end up answered by a marker standing in for the result the
    model just asked for.
    """
    protected: Final = frozenset(get_protected_indices(messages))
    return protected | frozenset(
        index
        for group in group_tool_exchanges(messages)
        if any(member in protected for member in group)
        for index in group
    )


def _restore_protected_messages(
    messages: Sequence[dict[str, object]],
    compressed: Sequence[dict[str, object]],
    protected_indices: frozenset[int],
) -> Sequence[dict[str, object]]:
    """Put the rows that were held back from compression at their original positions.

    Requires one returned row per row actually sent, which ``_call_compress``
    enforces; a service that changed the row count is treated as a failure
    there, because a reshaped conversation cannot be re-interleaved.
    """
    sent_positions: Final = tuple(index for index in range(len(messages)) if index not in protected_indices)
    compressed_by_index: Final = dict(zip(sent_positions, compressed))
    return [
        messages[index] if index in protected_indices else compressed_by_index[index] for index in range(len(messages))
    ]


def _build_compress_failure_detail(status_code: int, body: str) -> dict[str, object]:
    """Build error details for failed /v1/compress responses.

    Adds troubleshooting hints for known deployment-related errors while
    preserving the upstream status code and response body.
    """
    if status_code == 404:
        return {
            "status_code": status_code,
            "body": body,
            "hint": (
                "The Headroom compression endpoint returned HTTP 404. "
                "Verify that the configured Headroom endpoint is correct and that "
                "the compression endpoint is available. If you are using a "
                "self-hosted deployment, some deployments require enabling remote "
                "compression (for example, HEADROOM_COMPRESS_ALLOW_REMOTE=1)."
            ),
        }
    return {"status_code": status_code, "body": body}


def extract_hashes_from_messages(messages: list[dict[str, object]]) -> list[str]:
    hashes: Final[list[str]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            hashes.extend(_HASH_PATTERN.findall(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        hashes.extend(_HASH_PATTERN.findall(text))
    return hashes


def _build_headroom_retrieve_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": HEADROOM_RETRIEVE_TOOL_NAME,
            "description": (
                "Retrieve original content that was compressed by Headroom. "
                "Call this when you encounter a compression marker containing a hash."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "The 24-character hex hash from the compression marker.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query for BM25-ranked retrieval.",
                    },
                },
                "required": ["hash"],
            },
        },
    }


def _resolve_call_id(logging_obj: object, request_state: dict[str, object]) -> str | None:
    """Resolve the litellm_call_id shared by a request's pre-call hook and its
    agentic-loop hooks, so CCR hash validation can be scoped per call instead
    of trusting any hash-shaped string that shows up in message text."""
    logging_call_id: Final = getattr(logging_obj, "litellm_call_id", None)
    if isinstance(logging_call_id, str) and logging_call_id:
        return logging_call_id
    kwargs_call_id: Final = request_state.get("litellm_call_id")
    return kwargs_call_id if isinstance(kwargs_call_id, str) else None


def has_headroom_retrieve_tool(tools: object) -> bool:
    return has_tool_with_name(tools, HEADROOM_RETRIEVE_TOOL_NAME)


def _extract_headroom_tool_calls(response: object) -> list[dict[str, object]]:
    return [
        {"id": tc["id"], "type": "function", "name": tc["name"], "arguments": tc["arguments"]}
        for tc in get_tool_calls_from_response(response)
        if tc["name"] == HEADROOM_RETRIEVE_TOOL_NAME
    ]


def _build_assistant_message_from_response(
    response: object,
    retrieved: Sequence[tuple[dict[str, object], str]],
) -> dict[str, object]:
    """Rebuild the chat-completions assistant turn for the retrieval follow-up.

    Only the ``headroom_retrieve`` calls are echoed, each answered by a tool
    result below. Other tool calls made in the same turn are omitted on purpose:
    the follow-up re-runs the model with the recovered content so it re-plans
    them. Echoing them would leave tool_calls with no matching tool result and
    the provider would reject the request.
    """
    return {
        "role": "assistant",
        "content": assistant_text_from_response(response),
        "tool_calls": [
            {
                "id": tool_call.get("id"),
                "type": "function",
                "function": {
                    "name": tool_call.get("name"),
                    "arguments": json.dumps(tool_call.get("arguments", {})),
                },
            }
            for tool_call, _ in retrieved
        ],
    }


def _is_responses_api_response(response: object) -> bool:
    # Real response objects can be plain dicts at runtime (e.g. TypedDict-based
    # response types), so getattr alone would silently miss the key -- use the
    # same dict-or-object accessor as the tool-call extractors.
    return isinstance(get_attribute_or_key(response, "output", None), list)


def _is_anthropic_messages_response(response: object) -> bool:
    return isinstance(get_attribute_or_key(response, "content", None), list)


def _build_anthropic_followup_messages(
    response: object,
    retrieved: list[tuple[dict[str, object], str]],
) -> list[dict[str, object]]:
    """Build Anthropic Messages API follow-up messages for a tool round-trip.

    Anthropic requires the tool_use block to be echoed back in an assistant
    message, paired with a tool_result block in a user message keyed by the
    same tool_use_id -- it does not accept chat-style tool-role messages. Any
    text the model wrote alongside the tool call is preserved, so its reasoning
    survives into the follow-up turn.
    """
    text: Final = assistant_text_from_response(response)
    assistant_message: Final[dict[str, object]] = {
        "role": "assistant",
        "content": ([{"type": "text", "text": text}] if text else [])
        + [
            {
                "type": "tool_use",
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "input": tool_call.get("arguments", {}),
            }
            for tool_call, _ in retrieved
        ],
    }
    user_message: Final[dict[str, object]] = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_call.get("id"), "content": content}
            for tool_call, content in retrieved
        ],
    }
    return [assistant_message, user_message]


def _build_responses_followup_items(
    response: object,
    retrieved: list[tuple[dict[str, object], str]],
) -> list[dict[str, object]]:
    """Build Responses API input items for a tool round-trip.

    The Responses API does not accept chat-style assistant/tool messages as
    follow-up input; it requires the model's function_call to be echoed back
    paired with a function_call_output keyed by the same call_id. Any text the
    model wrote alongside the tool call is preserved.
    """
    text: Final = assistant_text_from_response(response)
    items: Final[list[dict[str, object]]] = [{"role": "assistant", "content": text}] if text else []
    for tool_call, content in retrieved:
        call_id = tool_call.get("id")
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("arguments", {})),
            }
        )
        items.append({"type": "function_call_output", "call_id": call_id, "output": content})
    return items


class HeadroomGuardrail(CustomGuardrail):
    records_own_guardrail_information: ClassVar[bool] = True
    server_fulfilled_tool_names: ClassVar[frozenset[str]] = frozenset({HEADROOM_RETRIEVE_TOOL_NAME})

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        unreachable_fallback: str | None = None,
    ):
        self.headroom_api_base = (api_base or get_secret_str("HEADROOM_API_BASE") or "").rstrip("/")
        if not self.headroom_api_base:
            raise ValueError(
                "Headroom guardrail requires an API base URL. "
                "Set `api_base` in the guardrail config or HEADROOM_API_BASE env var."
            )
        self.headroom_api_key = api_key or get_secret_str("HEADROOM_API_KEY")
        self.headroom_model = model
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self._issued_hashes_by_call_id: dict[str, tuple[frozenset[str], float]] = {}
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=list(self.get_supported_event_hooks()),
        )

    def _should_bypass(self, request_data: dict) -> bool:
        psr: Final = request_data.get("proxy_server_request")
        if not _is_str_object_dict(psr):
            return False
        headers: Final = psr.get("headers")
        if not _is_str_object_dict(headers):
            return False
        value: Final = headers.get(BYPASS_HEADER)
        return str(value).lower() == "true"

    def _request_headers(self) -> dict[str, str]:
        headers: Final[dict[str, str]] = {"Content-Type": "application/json"}
        if self.headroom_api_key:
            headers["Authorization"] = f"Bearer {self.headroom_api_key}"
        return headers

    def _prune_expired_hashes(self) -> None:
        now: Final = time.monotonic()
        self._issued_hashes_by_call_id = {
            call_id: (hashes, expiry)
            for call_id, (hashes, expiry) in self._issued_hashes_by_call_id.items()
            if expiry > now
        }

    def _handle_compress_failure(
        self,
        messages: list[dict[str, object]],
        error: str,
        detail: dict[str, object],
    ) -> list[dict[str, object]]:
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "Headroom: %s; fail_open configured, forwarding request uncompressed. detail=%s",
                error,
                detail,
            )
            return messages
        raise HTTPException(status_code=502, detail={"error": error, **detail})

    async def _call_compress(
        self,
        messages: list[dict[str, object]],
        model: str | None,
    ) -> tuple[list[dict[str, object]], bool, dict[str, object]]:
        payload: Final[dict[str, object]] = {"messages": messages}
        if model:
            payload["model"] = model

        try:
            raw_response: HttpxResponse | None = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]
                url=f"{self.headroom_api_base}/v1/compress",
                json=payload,
                headers=self._request_headers(),
            )
        except httpx.HTTPStatusError as e:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned an error",
                    _build_compress_failure_detail(e.response.status_code, e.response.text),
                ),
                False,
                {},
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError, litellm.Timeout) as e:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service unreachable",
                    {"detail": str(e)},
                ),
                False,
                {},
            )
        if raw_response is None:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned no response",
                    {},
                ),
                False,
                {},
            )
        response: Final[HttpxResponse] = raw_response

        if response.status_code != 200:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned an error",
                    _build_compress_failure_detail(response.status_code, response.text),
                ),
                False,
                {},
            )

        try:
            body: Final[object] = response.json()
        except ValueError:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned non-JSON response",
                    {"body": response.text[:500]},
                ),
                False,
                {},
            )
        if not _is_str_object_dict(body):
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned unexpected response shape",
                    {"body": response.text[:500]},
                ),
                False,
                {},
            )

        compressed_messages: Final = body.get("messages")
        if not _is_object_list(compressed_messages):
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service response missing 'messages'",
                    {"body": response.text},
                ),
                False,
                {},
            )

        filtered: Final = [item for item in compressed_messages if _is_str_object_dict(item)]
        if not filtered:
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service returned empty message list",
                    {"body": response.text},
                ),
                False,
                {},
            )

        if len(filtered) != len(messages):
            # Rows are matched positionally when the never-compressed messages
            # are put back, so a reshaped conversation cannot be applied at all.
            return (
                self._handle_compress_failure(
                    messages,
                    "Headroom compression service changed the message count",
                    {"sent": len(messages), "returned": len(filtered)},
                ),
                False,
                {},
            )

        verbose_proxy_logger.debug(
            "Headroom: compressed %s tokens -> %s tokens (ratio %.2f)",
            body.get("tokens_before", "?"),
            body.get("tokens_after", "?"),
            body.get("compression_ratio", 0),
        )

        stats: Final = {
            key: body[key]
            for key in (
                "tokens_before",
                "tokens_after",
                "tokens_saved",
                "compression_ratio",
                "transforms_applied",
            )
            if key in body
        }
        tokens_before: Final = stats.get("tokens_before")
        tokens_after: Final = stats.get("tokens_after")
        if (
            "tokens_saved" not in stats
            and isinstance(tokens_before, (int, float))
            and not isinstance(tokens_before, bool)
            and isinstance(tokens_after, (int, float))
            and not isinstance(tokens_after, bool)
        ):
            # Spend tracking (extract_compression_saved_tokens) reads only
            # tokens_saved, which the live compression service omits; derive it
            # so savings are counted, but let a service-sent value win.
            stats["tokens_saved"] = tokens_before - tokens_after
        return filtered, True, stats

    async def _call_retrieve(self, hash_value: str, query: str | None = None) -> str:
        params: Final[dict[str, str]] = {}
        if query:
            params["query"] = query

        try:
            raw_response: HttpxResponse | None = await self.async_handler.get(  # pyright: ignore[reportUnknownMemberType]
                url=f"{self.headroom_api_base}/v1/retrieve/{hash_value}",
                params=params,
                headers=self._request_headers(),
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError, litellm.Timeout) as e:
            verbose_proxy_logger.warning("Headroom: retrieve failed for hash=%s: %s", hash_value, e)
            return f"[Headroom: retrieval failed for hash={hash_value}]"

        if raw_response is None or raw_response.status_code == 404:
            return f"[Headroom: hash={hash_value} not found or expired]"

        if raw_response.status_code != 200:
            verbose_proxy_logger.warning(
                "Headroom: retrieve returned %s for hash=%s",
                raw_response.status_code,
                hash_value,
            )
            return f"[Headroom: retrieval error {raw_response.status_code} for hash={hash_value}]"

        try:
            body: Final[object] = raw_response.json()
        except ValueError:
            return raw_response.text

        if _is_str_object_dict(body):
            original_content: Final = body.get("original_content")
            if isinstance(original_content, str):
                return original_content

        return str(body)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        if self._should_bypass(request_data):
            verbose_proxy_logger.debug("Headroom: %s header set; skipping compression", BYPASS_HEADER)
            return inputs

        structured_messages: Final = inputs.get("structured_messages")
        if not _is_object_list(structured_messages) or not structured_messages:
            return inputs

        messages: Final = [m for m in structured_messages if _is_str_object_dict(m)]
        if not messages:
            return inputs

        # The last user message is the instruction the model is being asked to
        # act on, so replacing it with a marker means the model answers a
        # retrieval result instead of the request. Protected rows are held back
        # from the payload rather than pinned after the fact, so their tokens
        # are not counted as savings we never apply; the Anthropic write-back
        # discards a compressed system prompt outright. Keep it that way unless
        # /v1/compress grows a field for sending the live turn as the retrieval
        # query without compressing it: query-aware compression reads the newest
        # user message, so it is withheld here at some cost to history ranking.
        protected_indices: Final = _protected_indices(messages)
        compressible: Final = [m for i, m in enumerate(messages) if i not in protected_indices]
        if not compressible:
            return inputs

        model: Final = self.headroom_model or request_data.get("model")
        start_time: Final = time.time()
        returned, compression_succeeded, stats = await self._call_compress(
            messages=_flatten_messages_for_compression(compressible),
            model=model if isinstance(model, str) else None,
        )
        end_time: Final = time.time()

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        if not compression_succeeded:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={"error": "headroom compression unavailable; request forwarded uncompressed"},
                request_data=request_data,
                guardrail_status="guardrail_failed_to_respond",
                guardrail_provider=HEADROOM_GUARDRAIL_PROVIDER,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
            )
            add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)
            # Hand back the caller's own inputs object. Translation handlers
            # detect "the guardrail rewrote the messages" by identity, so
            # returning a rebuilt copy sends an unchanged request through the
            # write-back and restructures it for nothing.
            return inputs

        compressed: Final = _restore_protected_messages(
            messages=messages,
            compressed=_restore_content_shapes(originals=compressible, returned=returned),
            protected_indices=protected_indices,
        )

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=stats,
            request_data=request_data,
            guardrail_status="success",
            guardrail_provider=HEADROOM_GUARDRAIL_PROVIDER,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
        add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)

        hashes: Final = extract_hashes_from_messages(compressed)
        if not hashes:
            return {**inputs, "structured_messages": compressed}  # pyright: ignore[reportReturnType]

        self._prune_expired_hashes()
        call_id = _resolve_call_id(logging_obj, request_data)
        if not call_id:
            call_id = str(uuid.uuid4())
            request_data["litellm_call_id"] = call_id
        self._issued_hashes_by_call_id[call_id] = (frozenset(hashes), time.monotonic() + _HASH_CACHE_TTL_SECONDS)

        existing_tools: Final = inputs.get("tools")
        retrieve_tool: Final = _build_headroom_retrieve_tool()
        if isinstance(existing_tools, list) and not has_headroom_retrieve_tool(existing_tools):
            merged_tools: list[object] = list(existing_tools) + [retrieve_tool]
        elif existing_tools is None:
            merged_tools = [retrieve_tool]
        else:
            merged_tools = list(existing_tools) if isinstance(existing_tools, list) else [retrieve_tool]

        return {**inputs, "structured_messages": compressed, "tools": merged_tools}  # pyright: ignore[reportReturnType]

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        if not has_headroom_retrieve_tool(tools):
            return False, {}

        tool_calls: Final = _extract_headroom_tool_calls(response)
        if not tool_calls:
            return False, {}

        return True, {"tool_calls": tool_calls}

    async def async_build_agentic_loop_plan(
        self,
        tools: dict,
        model: str,
        messages: list[dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: dict,
        logging_obj: Any,
        stream: bool,
        kwargs: dict,
    ) -> AgenticLoopPlan:
        tool_calls: Final[list[dict[str, object]]] = tools.get("tool_calls", [])

        self._prune_expired_hashes()
        call_id: Final = _resolve_call_id(logging_obj, kwargs)
        valid_hashes = self._issued_hashes_by_call_id.get(call_id, (frozenset(), 0.0))[0] if call_id else frozenset()

        retrieved: Final[list[tuple[dict[str, object], str]]] = []
        for tc in tool_calls:
            arguments = tc.get("arguments", {})
            hash_value = arguments.get("hash", "") if isinstance(arguments, dict) else ""
            query = arguments.get("query") if isinstance(arguments, dict) else None
            # A hash is only honored if it was issued by *this request's own*
            # Headroom /v1/compress call, scoped by litellm_call_id. Scoping by
            # message text alone is forgeable -- an attacker can plant a
            # hash-shaped string in their own prompt, and a hash issued for one
            # request would validate for any other request that echoes it back.
            if str(hash_value) not in valid_hashes:
                verbose_proxy_logger.warning(
                    "Headroom CCR: rejecting hash=%s not produced by current request compression",
                    hash_value,
                )
                content = f"[Headroom: hash={hash_value} was not produced by the current request]"
            else:
                content = await self._call_retrieve(
                    hash_value=str(hash_value),
                    query=str(query) if query else None,
                )
            verbose_proxy_logger.debug("Headroom CCR: retrieved hash=%s (%d chars)", hash_value, len(content))
            retrieved.append((tc, content))

        if _is_responses_api_response(response):
            follow_up_messages = list(messages) + _build_responses_followup_items(response, retrieved)
        elif _is_anthropic_messages_response(response):
            follow_up_messages = list(messages) + _build_anthropic_followup_messages(response, retrieved)
        else:
            assistant_message: Final = _build_assistant_message_from_response(response, retrieved)
            tool_results: Final = [
                {"role": "tool", "tool_call_id": tc.get("id"), "content": content} for tc, content in retrieved
            ]
            follow_up_messages = list(messages) + [assistant_message] + tool_results

        max_tokens: Final[int | None] = anthropic_messages_optional_request_params.get("max_tokens") or kwargs.get(
            "max_tokens"
        )
        optional_params_without_max_tokens: Final = {
            k: v for k, v in anthropic_messages_optional_request_params.items() if k != "max_tokens"
        }

        full_model_name = model
        if logging_obj is not None:
            agentic_params: Final = getattr(logging_obj, "model_call_details", {}).get("agentic_loop_params", {})
            candidate: Final = agentic_params.get("model", model)
            if isinstance(candidate, str) and candidate:
                full_model_name = candidate

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                model=full_model_name,
                messages=follow_up_messages,
                max_tokens=max_tokens,
                optional_params=optional_params_without_max_tokens,
                kwargs={
                    k: v for k, v in kwargs.items() if not k.startswith("_headroom") and k != "litellm_logging_obj"
                },
            ),
            metadata={"tool_type": "headroom_ccr"},
        )

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[object]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.headroom import (
            HeadroomGuardrailConfigModel,
        )

        return HeadroomGuardrailConfigModel
