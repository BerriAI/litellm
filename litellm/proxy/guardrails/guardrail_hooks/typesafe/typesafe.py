"""TypeSafe (Jev) relevance-based compaction guardrail.

Instead of summarizing tool output, the guardrail asks TypeSafe's Jev model
one yes/no question per completed tool exchange ("is this result still needed
for the current task?") over ``POST {api_base}/v1/systemone`` and blanks the
tool results Jev judges no longer relevant. The assistant tool-call rows stay
intact, so the conversation remains well-formed while the dead context stops
consuming input tokens.

Exchanges follow litellm's own compression protection policy: system rows, the
last user row, and the last assistant row (which, expanded over its tool
exchange, covers the most recent exchange) are never evaluated or rewritten.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeGuard, cast

import httpx
from fastapi import HTTPException
from httpx import Response as HttpxResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.compression.compress import get_protected_indices
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped in custom_guardrail
)
from litellm.litellm_core_utils.prompt_templates.factory import group_tool_exchanges
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper is untyped in http_handler
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.content_text import content_to_text
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.typesafe import (
        TypeSafeGuardrailConfigModel,
    )

DEFAULT_API_BASE: Final = "https://api.typesafe.ai"
DEFAULT_MODEL: Final = "jev-latest"
DEFAULT_RELEVANCE_THRESHOLD: Final = 0.2
DEFAULT_MIN_CHARS_TO_EVALUATE: Final = 200
DEFAULT_MAX_RESULT_CHARS_IN_STATE: Final = 4000
_MAX_EXCHANGES_EVALUATED: Final = 200
# The shared GuardrailCallback client carries no per-call bound; an on-request
# guardrail must not hold the caller's request for the client's pooled timeout.
_JEV_TIMEOUT_SECONDS: Final = 30.0
DROPPED_RESULT_TEXT: Final = (
    "[Tool result removed by TypeSafe compaction: judged no longer relevant to the current task]"
)


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


def _safe_response_text(response: object, limit: int = 500) -> str:
    try:
        text: Final = getattr(response, "text", "")
    except httpx.DecodingError:
        return "<undecodable response body>"
    return (text or "")[:limit]


class _JevNoulAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    type: Literal["noul"]
    noul: Annotated[float, Field(ge=0.0, le=1.0)]


class _JevSystemOneResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answers: Mapping[str, _JevNoulAnswer]


_JEV_RESPONSE_ADAPTER: Final = TypeAdapter(_JevSystemOneResponse)


def _question_instructions(question_id: str) -> str:
    return (
        f"Is tool exchange `{question_id}` in `tool_exchanges` still needed by the assistant to "
        "complete `task`? Answer yes if its result contains information the assistant has not yet "
        "fully used or will need again; answer no if it is off-topic, superseded, or already "
        "incorporated into later messages."
    )


def _tool_call_entries(assistant_message: Mapping[str, object]) -> list[dict[str, object]]:
    tool_calls: Final = assistant_message.get("tool_calls")
    if not _is_object_list(tool_calls):
        return []
    entries: Final[list[dict[str, object]]] = []
    for tool_call in tool_calls:
        if not _is_str_object_dict(tool_call):
            continue
        function = tool_call.get("function")
        fn = function if _is_str_object_dict(function) else tool_call
        entries.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    return entries


def _protected_indices(messages: Sequence[Mapping[str, object]]) -> frozenset[int]:
    """Rows typesafe must not rewrite, expanded over whole tool exchanges.

    ``get_protected_indices`` covers system rows, the last user row, the last
    assistant row, and cache_control prefixes. Expanding over exchanges keeps an
    exchange atomic: the last assistant row protects its own tool results too,
    so the most recent exchange is never evaluated.
    """
    protected: Final = frozenset(get_protected_indices(messages))
    return protected | frozenset(
        index
        for group in group_tool_exchanges(messages)
        if any(member in protected for member in group)
        for index in group
    )


class TypeSafeGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        relevance_threshold: float | None = None,
        min_chars_to_evaluate: int | None = None,
        max_result_chars_in_state: int | None = None,
        unreachable_fallback: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        async_handler: AsyncHTTPHandler | None = None,
    ):
        raw_api_base: Final = (api_base or get_secret_str("TYPESAFE_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        self.typesafe_api_base = raw_api_base
        self.typesafe_api_key = api_key or get_secret_str("TYPESAFE_API_KEY")
        if not self.typesafe_api_key:
            raise ValueError(
                "TypeSafe guardrail requires an API key. Set `api_key` in the "
                "guardrail config or the TYPESAFE_API_KEY env var."
            )
        self.jev_model = model or DEFAULT_MODEL
        self.relevance_threshold = DEFAULT_RELEVANCE_THRESHOLD if relevance_threshold is None else relevance_threshold
        self.min_chars_to_evaluate = (
            DEFAULT_MIN_CHARS_TO_EVALUATE if min_chars_to_evaluate is None else min_chars_to_evaluate
        )
        self.max_result_chars_in_state = (
            DEFAULT_MAX_RESULT_CHARS_IN_STATE if max_result_chars_in_state is None else max_result_chars_in_state
        )
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_closed" if unreachable_fallback == "fail_closed" else "fail_open"
        )
        self.async_handler: AsyncHTTPHandler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ is untyped
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )

    def _handle_failure(self, error: str, log_detail: dict[str, object]) -> None:
        """fail_open logs and the caller forwards uncompacted; fail_closed raises.
        Upstream bodies go to server logs only; the raised HTTPException is generic."""
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.warning(
                "TypeSafe: %s; fail_open configured, forwarding request uncompacted. detail=%s",
                error,
                log_detail,
            )
            return
        verbose_proxy_logger.error("TypeSafe: %s. detail=%s", error, log_detail)
        raise HTTPException(status_code=500, detail={"error": error})

    def _candidate_exchanges(self, messages: list[dict[str, object]]) -> list[tuple[int, ...]]:
        """Message-index groups eligible for relevance evaluation, oldest first.

        A candidate is a completed tool exchange: an assistant row that made
        tool calls plus at least one ``tool``/``function`` row answering it,
        with no member protected, and enough combined tool-result text to be
        worth an evaluation call.
        """
        protected: Final = _protected_indices(messages)
        candidates: Final[list[tuple[int, ...]]] = []
        for group in group_tool_exchanges(messages):
            if len(group) < 2:
                continue
            if messages[group[0]].get("role") != "assistant":
                continue
            if any(member in protected for member in group):
                continue
            tool_text = "".join(
                content_to_text(messages[index].get("content"))
                for index in group[1:]
                if messages[index].get("role") in ("tool", "function")
            )
            if not tool_text or len(tool_text) < self.min_chars_to_evaluate:
                continue
            candidates.append(group)
        return candidates[-_MAX_EXCHANGES_EVALUATED:]

    def _build_state(self, messages: list[dict[str, object]], candidates: list[tuple[int, ...]]) -> dict[str, object]:
        task: Final = next(
            (
                content_to_text(messages[index].get("content"))
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            "",
        )
        system: Final = "\n\n".join(
            content_to_text(message.get("content")) for message in messages if message.get("role") == "system"
        )
        tool_exchanges: Final[dict[str, object]] = {}
        for ordinal, group in enumerate(candidates):
            result_text = "".join(
                content_to_text(messages[index].get("content"))
                for index in group[1:]
                if messages[index].get("role") in ("tool", "function")
            )
            tool_exchanges[f"e{ordinal}"] = {
                "tool_calls": _tool_call_entries(messages[group[0]]),
                "result": result_text[: self.max_result_chars_in_state],
            }
        return {"task": task, "system": system, "tool_exchanges": tool_exchanges}

    async def _call_systemone(self, state: dict[str, object], question_ids: list[str]) -> _JevSystemOneResponse | None:
        """Evaluate each exchange. Returns the response, or None when the service
        failed and fail_open applies."""
        payload: Final[dict[str, object]] = {
            "model": self.jev_model,
            "state": state,
            "questions": {
                question_id: {"type": "noul", "instructions": _question_instructions(question_id)}
                for question_id in question_ids
            },
        }
        try:
            raw_response: HttpxResponse = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post is untyped
                url=f"{self.typesafe_api_base}/v1/systemone",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.typesafe_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=_JEV_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            resp: Final = getattr(e, "response", None)
            self._handle_failure(
                "TypeSafe evaluation service returned an error",
                {"status_code": getattr(resp, "status_code", None), "body": _safe_response_text(resp)},
            )
            return None
        except (httpx.RequestError, litellm.Timeout) as e:
            self._handle_failure("TypeSafe evaluation service request failed", {"detail": str(e)})
            return None
        except Exception as e:
            self._handle_failure("TypeSafe evaluation service request failed", {"detail": str(e)})
            return None
        if not 200 <= raw_response.status_code < 300:
            self._handle_failure(
                "TypeSafe evaluation service returned an error",
                {"status_code": raw_response.status_code, "body": _safe_response_text(raw_response)},
            )
            return None
        try:
            body: Final = cast(object, raw_response.json())
        except (ValueError, httpx.DecodingError, RecursionError):
            self._handle_failure(
                "TypeSafe evaluation service returned an unreadable response",
                {"body": _safe_response_text(raw_response)},
            )
            return None
        try:
            return _JEV_RESPONSE_ADAPTER.validate_python(body)
        except ValidationError:
            self._handle_failure(
                "TypeSafe evaluation service returned unexpected response shape",
                {"body": _safe_response_text(raw_response)},
            )
            return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        structured_messages: Final = inputs.get("structured_messages")
        if not _is_object_list(structured_messages) or not structured_messages:
            return inputs
        messages: Final = [m for m in structured_messages if _is_str_object_dict(m)]
        if len(messages) != len(structured_messages):
            return inputs

        candidates: Final = self._candidate_exchanges(messages)
        if not candidates:
            verbose_proxy_logger.debug("TypeSafe: no completed tool exchanges eligible for evaluation")
            return inputs

        question_ids: Final = [f"e{ordinal}" for ordinal in range(len(candidates))]
        state: Final = self._build_state(messages, candidates)

        start_time: Final = time.monotonic()
        response: Final = await self._call_systemone(state, question_ids)
        end_time: Final = time.monotonic()
        if response is None:
            return inputs

        dropped_ordinals: Final = frozenset(
            ordinal
            for ordinal in range(len(candidates))
            if (answer := response.answers.get(f"e{ordinal}")) is not None and answer.noul < self.relevance_threshold
        )
        dropped_tool_indices: Final[frozenset[int]] = frozenset(
            index
            for ordinal in dropped_ordinals
            for index in candidates[ordinal][1:]
            if messages[index].get("role") in ("tool", "function")
        )
        if not dropped_tool_indices:
            verbose_proxy_logger.debug("TypeSafe: all evaluated exchanges still relevant; request unchanged")
            return inputs

        compacted_messages: Final = [
            {**message, "content": DROPPED_RESULT_TEXT} if index in dropped_tool_indices else message
            for index, message in enumerate(messages)
        ]
        chars_removed: Final = sum(
            len(content_to_text(messages[index].get("content"))) - len(DROPPED_RESULT_TEXT)
            for index in dropped_tool_indices
        )
        exchanges_dropped: Final = len(dropped_ordinals)
        verbose_proxy_logger.info(
            "TypeSafe: evaluated %s tool exchange(s), dropped %s, ~%s chars removed",
            len(candidates),
            exchanges_dropped,
            chars_removed,
        )
        self.add_standard_logging_guardrail_information_to_request_data(  # pyright: ignore[reportUnknownMemberType]  # untyped base helper
            guardrail_json_response={
                "exchanges_evaluated": len(candidates),
                "exchanges_dropped": exchanges_dropped,
                "chars_removed": chars_removed,
                "model": self.jev_model,
            },
            request_data=request_data,
            guardrail_status="success",
            guardrail_provider="typesafe",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
        return {**inputs, "structured_messages": compacted_messages}  # pyright: ignore[reportReturnType]  # plain dicts satisfy AllMessageValues at runtime

    @staticmethod
    def get_config_model() -> type[TypeSafeGuardrailConfigModel] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.typesafe import (
            TypeSafeGuardrailConfigModel,
        )

        return TypeSafeGuardrailConfigModel
