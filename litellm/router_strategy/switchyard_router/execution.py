from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import Final

import anyio
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._uuid import uuid
from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.litellm_core_utils.internal_call_metadata import forwarded_internal_call_metadata
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_streaming_response_async,
)
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_RESPONSE: Final = TypeAdapter(dict[str, object])


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING.validate_python(value)
    except ValidationError:
        return None


ACTIVE_SWITCHYARD: Final[ContextVar[bool]] = ContextVar("active_switchyard", default=False)
_EXECUTION_CONTEXT_KEYS: Final = frozenset(
    {
        "litellm_trace_id",
        "litellm_session_id",
        "timeout",
        "request_timeout",
        "num_retries",
        "disable_fallbacks",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "tags",
        "enable_tag_filtering",
        "proxy_server_request",
        "turn_off_message_logging",
        "user",
        "guardrails",
    }
)
_CALL_LOCAL_KEYS: Final = frozenset(
    {
        "model",
        "original_function",
        "litellm_call_id",
        "litellm_logging_obj",
        "metadata",
        "litellm_metadata",
    }
)


@contextmanager
def switchyard_execution() -> Generator[None]:
    token: Final = ACTIVE_SWITCHYARD.set(True)
    try:
        yield
    finally:
        ACTIVE_SWITCHYARD.reset(token)


def child_call_kwargs(parent: Mapping[str, object], payload: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = _optional_mapping(parent.get("metadata"))
    litellm_metadata: Final = _optional_mapping(parent.get("litellm_metadata"))
    return {
        **{key: value for key, value in parent.items() if key in _EXECUTION_CONTEXT_KEYS},
        **{key: value for key, value in payload.items() if key not in _CALL_LOCAL_KEYS},
        "metadata": forwarded_internal_call_metadata(metadata, "switchyard_router"),
        **(
            {"litellm_metadata": forwarded_internal_call_metadata(litellm_metadata, "switchyard_router")}
            if litellm_metadata is not None
            else {}
        ),
        "litellm_call_id": str(uuid.uuid4()),
    }


def scoped_switchyard_headers(
    request: Mapping[str, object], message_hash_fallback: bool = False, sdk_namespace: str | None = None
) -> Mapping[str, str]:
    proxy_request: Final = _optional_mapping(request.get("proxy_server_request"))
    raw_headers: Final = proxy_request.get("headers") if proxy_request is not None else None
    headers: Final = _optional_mapping(raw_headers) or {}
    metadata: Final = request.get("litellm_metadata") or request.get("metadata")
    caller: Final = _optional_mapping(metadata) or {}
    caller_key: Final = caller.get("user_api_key_hash")
    sdk_session: Final = request.get("litellm_session_id")
    sdk_context: Final = (
        sdk_namespace is not None
        and proxy_request is None
        and "user_api_key_auth" not in caller
        and isinstance(sdk_session, str)
        and bool(sdk_session)
    )
    caller_scope: Final = (
        (caller_key, caller.get("user_api_key_team_id"))
        if isinstance(caller_key, str) and caller_key
        else (sdk_namespace,)
        if sdk_context
        else (str(uuid.uuid4()),)
    )
    scope: Final = json.dumps(caller_scope, sort_keys=True)
    normalized: Final = {key.lower(): value for key, value in headers.items()}
    aliases: Final = (
        "x-switchyard-session-id",
        "x-claude-code-session-id",
        "x-nemo-relay-session-id",
        "x-session-id",
        "session-id",
    )
    session: Final = (
        sdk_session
        if sdk_context
        else next((normalized[key] for key in aliases if isinstance(normalized.get(key), str)), None)
    )
    fallback: Final = (
        json.dumps(request.get("messages"), sort_keys=True, default=str) if message_hash_fallback else str(uuid.uuid4())
    )
    return {"x-switchyard-session-id": sha256(f"{scope}:{session or fallback}".encode()).hexdigest()}


class SwitchyardReplay:
    def __init__(self, response: ModelResponse, include_usage: bool) -> None:
        self._hidden_params = get_hidden_params_dict(response)
        self.response: Final = response
        self.include_usage: Final = include_usage

    def __aiter__(self) -> AsyncIterator[ModelResponseStream]:
        return self._frames()

    async def _frames(self) -> AsyncIterator[ModelResponseStream]:
        payload: Final = _RESPONSE.validate_json(
            BaseModel.model_dump_json(self.response, exclude={"usage"}, exclude_unset=True)
        )
        async for frame in convert_to_streaming_response_async(payload):
            yield frame
        if self.include_usage:
            usage_frame: Final = ModelResponseStream(
                id=self.response.id,
                created=self.response.created,
                model=self.response.model,
                choices=[],
                usage=Usage.model_validate(
                    _RESPONSE.validate_json(BaseModel.model_dump_json(self.response)).get("usage")
                ),
            )
            yield usage_frame


class FinalizingStream:
    def __init__(
        self,
        stream: AsyncIterator[ModelResponseStream],
        finalize: Callable[[], Awaitable[None]],
        close: Callable[[], Awaitable[None]],
        hidden_params: object,
        logging_obj: object,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.stream: Final = stream
        self.finalize: Final = finalize
        self.close: Final = close
        self._hidden_params = hidden_params
        self.logging_obj = logging_obj
        self.closed = False
        self.completed = False
        self.on_disconnect: Final = on_disconnect

    def __aiter__(self) -> FinalizingStream:
        return self

    async def __anext__(self) -> ModelResponseStream:
        if self.closed:
            raise StopAsyncIteration
        try:
            with switchyard_execution():
                return await self.stream.__anext__()
        except Exception:
            self.completed = True
            await self.aclose()
            raise
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        with anyio.CancelScope(shield=True):
            try:
                if not self.completed and self.on_disconnect is not None:
                    await self.on_disconnect()
            finally:
                try:
                    await self.close()
                finally:
                    await self.finalize()
