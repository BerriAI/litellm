"""
Post-call hook for the adaptive router.

On each successful or failed completion, build a Turn from the request/response
and push it through `AdaptiveRouter.record_turn`. The router then updates the
in-memory bandit cell + session state and queues writes for the proxy flusher.

All work happens after the response has been returned to the caller. Any
exception is swallowed — signal recording must never break a request.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, attribute_or_none, iter_items, require_attributes
from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
from litellm.router_strategy.adaptive_router.classifier import classify_prompt
from litellm.router_strategy.adaptive_router.config import (
    ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY,
    ADAPTIVE_ROUTER_RESPONSE_HEADER,
    SIGNAL_GATE_MIN_MESSAGES,
)
from litellm.router_strategy.adaptive_router.signals import Turn

if TYPE_CHECKING:
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

# Identity fields hashed into a derived session key so the same conversation
# from the same caller produces a stable key, while different keys/teams/users
# stay segregated even if they happen to send identical first messages.
_IDENTITY_FIELDS: Final = (
    "user_api_key_hash",
    "user_api_key_team_id",
    "user_api_key_user_id",
    "user_api_key_end_user_id",
)


class _LoggedCallKwargs(TypedDict, total=False):
    litellm_params: ReadOnly[Mapping[str, object]]
    messages: ReadOnly[Sequence[Mapping[str, object]] | None]


def _resolve_session_key(kwargs: _LoggedCallKwargs) -> str | None:
    """Pick a stable per-conversation key for owner-cache attribution.

    Order:
      1. Honor a client-supplied session id (`litellm_session_id` on either
         `litellm_params` or `litellm_params.metadata`, or `session_id` on
         metadata) — backward compat for callers already wired up.
      2. Otherwise derive a sha256 over (identity fields, first
         SIGNAL_GATE_MIN_MESSAGES messages) so the key is stable across turns
         and only materialises once there is enough context for the bandit to
         act on (matching the gate in the signal-processing path).

    Returns None if the conversation is shorter than SIGNAL_GATE_MIN_MESSAGES.
    """
    litellm_params: Final = kwargs.get("litellm_params") or {}
    sid = litellm_params.get("litellm_session_id")
    if sid:
        return str(sid)
    metadata: Final = litellm_params.get("metadata") or {}
    if isinstance(metadata, dict):
        sid = metadata.get("session_id") or metadata.get("litellm_session_id")
        if sid:
            return str(sid)

    messages: Final = kwargs.get("messages") or []
    if len(messages) < SIGNAL_GATE_MIN_MESSAGES:
        # Don't attribute until we have enough turns to match the signal gate —
        # ensures the hash is stable (same N messages every time) and avoids
        # crediting the bandit for conversations that are too short to signal.
        return None

    identity = ":".join(str(metadata.get(f) or "") if isinstance(metadata, dict) else "" for f in _IDENTITY_FIELDS)
    anchor: Final = messages[:SIGNAL_GATE_MIN_MESSAGES]
    payload: Final = (
        identity
        + "|"
        + json.dumps(
            [{"role": m.get("role"), "content": m.get("content")} for m in anchor],
            sort_keys=True,
            default=str,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _last_user_content(messages: Sequence[Mapping[str, object]] | None) -> str | None:
    if not messages:
        return None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # OpenAI vision-style content: pick first text part.
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text")
            return None
    return None


def _recent_tool_results(
    messages: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Extract the current turn's tool result payloads from the request messages.

    Tool results are `role == "tool"` messages that sit at the tail of the
    conversation — i.e. after the most recent assistant message with
    `tool_calls`, waiting for the model to produce a user-facing reply. Walk
    backwards from the end and collect the contiguous run of tool messages;
    stop at the first non-tool message.

    Each result is normalized to `{content, is_error}` — the only fields
    `signals._detect_failure` / `_detect_exhaustion` actually read.
    """
    if not messages:
        return []
    results: Final[list[dict[str, object]]] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            break
        if msg.get("role") != "tool":
            break
        content = msg.get("content")
        # Some providers (Anthropic-style) carry an explicit error flag; OpenAI
        # tool results don't, so fall back to an empty/missing content heuristic
        # inside `_detect_failure`.
        is_error = bool(msg.get("is_error"))
        results.append({"content": content, "is_error": is_error})
    results.reverse()
    return results


@runtime_checkable
class _SupportsGetItem(Protocol):
    def __getitem__(self, index: int, /) -> object: ...


class _SupportsGet(Protocol):
    def get(self, key: str, /) -> object: ...


class _SupportsModelDump(Protocol):
    def model_dump(self) -> Mapping[str, object]: ...


def _first_item(value: object) -> object:
    if isinstance(value, _SupportsGetItem):
        return value[0]
    raise TypeError(f"'{type(value).__name__}' object is not subscriptable")


def _dict_fields(value: object) -> Mapping[str, object] | None:
    fields: Final = as_str_mapping(value)
    return fields if isinstance(value, dict) else None


def _attribute_or_dict_field(source: object, name: str) -> object:
    attribute: Final = attribute_or_none(source, name)
    fields: Final = _dict_fields(source)
    return fields.get(name) if attribute is None and fields is not None else attribute


def _tool_call_fields(tool_call: object) -> Mapping[str, object]:
    fields: Final = _dict_fields(tool_call)
    if fields is not None:
        return fields
    try:
        return require_attributes(tool_call, _SupportsModelDump, "model_dump").model_dump()
    except Exception:
        return {"name": getattr(tool_call, "name", ""), "arguments": ""}


def _assistant_content_and_tool_calls(response_obj: object) -> tuple[object, Sequence[Mapping[str, object]]]:
    """Return (assistant_text, tool_calls_list) extracted from a ModelResponse-ish object."""
    if response_obj is None:
        return None, []
    try:
        choices: Final = attribute_or_none(response_obj, "choices") or require_attributes(
            response_obj, _SupportsGet, "get"
        ).get("choices")
    except Exception:
        return None, []
    if not choices:
        return None, []

    first_choice: Final = _first_item(choices)
    choice_fields: Final = _dict_fields(first_choice)
    msg: Final = attribute_or_none(first_choice, "message") or (
        choice_fields.get("message") if choice_fields is not None else None
    )
    if msg is None:
        return None, []

    content: Final = _attribute_or_dict_field(msg, "content")
    raw_tool_calls: Final = _attribute_or_dict_field(msg, "tool_calls")
    tool_calls: Final = [_tool_call_fields(tc) for tc in iter_items(raw_tool_calls or [])]
    return content, tool_calls


class AdaptiveRouterResponseHeaderHook(CustomLogger):
    async def async_post_call_response_headers_hook(
        self,
        data: Mapping[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> dict[str, str] | None:
        """
        Surface the chosen logical model as the `x-litellm-adaptive-router-model`
        response header for both streaming and non-streaming responses.

        `async_post_call_success_hook` fires after the stream is fully consumed,
        so writing to `_hidden_params["additional_headers"]` there is too late for
        streaming — the StreamingResponse headers are already frozen. This hook is
        called during header construction (before StreamingResponse is built), so
        the header is included for both paths.
        """
        metadata: Final = data.get("metadata") or {}
        chosen: Final = metadata.get(ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY) if isinstance(metadata, dict) else None
        if not chosen:
            return None
        return {ADAPTIVE_ROUTER_RESPONSE_HEADER: chosen}


class AdaptiveRouterPostCallHook(CustomLogger):
    """One hook instance per AdaptiveRouter. Registered into litellm.callbacks."""

    def __init__(self, adaptive_router: AdaptiveRouter) -> None:
        self.adaptive_router = adaptive_router

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._record(kwargs, response_obj, response_status=200)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        status = kwargs.get("response_status")
        if status is None:
            exc: Final = kwargs.get("exception")
            status = getattr(exc, "status_code", 500) if exc is not None else 500
        await self._record(kwargs, response_obj, response_status=int(status))

    async def _record(
        self,
        kwargs: _LoggedCallKwargs,
        response_obj: object,
        response_status: int,
    ) -> None:
        try:
            messages: Final = kwargs.get("messages") or []
            session_key: Final = _resolve_session_key(kwargs)
            if not session_key:
                return

            # The bandit cells are keyed by the *logical* model name from
            # `available_models` (e.g. "smart"/"fast"). `kwargs["model"]` at
            # post-call time is the physical upstream model
            # (e.g. "anthropic/claude-opus-4-7"), so it cannot be used directly.
            # The pre-routing hook stashes the logical pick under this key.
            litellm_params: Final = kwargs.get("litellm_params") or {}
            metadata: Final = litellm_params.get("metadata") or {}
            current_model = metadata.get(ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY) if isinstance(metadata, dict) else None
            if not current_model:
                return

            user_text: Final = _last_user_content(messages)
            assistant_text, tool_calls = _assistant_content_and_tool_calls(response_obj)
            tool_results: Final = _recent_tool_results(messages)

            request_type: Final = classify_prompt(user_text or "")
            turn: Final = Turn(
                user_content=user_text,
                assistant_content=(assistant_text if isinstance(assistant_text, str) else None),
                tool_calls=tool_calls,
                tool_results=tool_results,
                response_status=response_status,
            )
            await self.adaptive_router.record_turn(
                session_id=session_key,
                model_name=current_model,
                request_type=request_type,
                turn=turn,
            )
        except Exception as e:
            verbose_router_logger.exception("AdaptiveRouterPostCallHook: failed to record turn: %s", e)
