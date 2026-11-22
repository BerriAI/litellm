"""Decision-model guardrail: a decisions-API model (e.g. Jev) scores the request or
response against predicate checks and blocks when any check crosses its threshold."""

from __future__ import annotations

import asyncio
import hashlib
import json
from asyncio import AbstractEventLoop, Semaphore
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import chain
from time import time
from typing import TYPE_CHECKING, ClassVar, Final, Literal, TypeAlias

from fastapi import HTTPException
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped upstream
)
from litellm.litellm_core_utils.llm_judge import attached_auth, caller_team_for, default_router_provider, judge_target
from litellm.types.decisions import MAX_DECISION_QUESTIONS, OpenAIDecisionResponse, OpenAIPredicateAnswer
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.llms.openai import ChatCompletionAssistantToolCall, ChatCompletionToolCallChunk
from litellm.types.proxy.guardrails.guardrail_hooks.decision_model import (
    DecisionModelCheck,
    DecisionModelGuardrailConfigModel,
)
from litellm.types.utils import ChatCompletionMessageToolCall, GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm import Router
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


class _TeamScope(TypedDict, total=False):
    litellm_metadata: ReadOnly[Mapping[str, str]]


def _team_scope(team_id: str | None, team_alias: str | None) -> _TeamScope:
    if team_id is None:
        return {}
    alias: Final = {} if team_alias is None else {"user_api_key_team_alias": team_alias}
    return {"litellm_metadata": {"user_api_key_team_id": team_id, **alias}}


class _CheckVerdict(TypedDict):
    name: ReadOnly[str]
    probability: ReadOnly[float | None]
    threshold: ReadOnly[float]
    action: ReadOnly[Literal["block", "log"]]
    flagged: ReadOnly[bool]
    reason: ReadOnly[str | None]


def _resolve_checks(checks: tuple[DecisionModelCheck, ...]) -> tuple[DecisionModelCheck, ...]:
    """Enforce the check invariants (non-empty list, unique names, bounded count)
    that decide what a run sends, wherever the guardrail is built."""
    if not checks:
        raise ValueError("decision_model guardrail requires at least one check")
    if len(checks) > MAX_DECISION_QUESTIONS:
        raise ValueError(
            f"decision_model guardrail supports at most {MAX_DECISION_QUESTIONS} checks, got {len(checks)}"
        )
    names: Final = [check.name for check in checks]
    if len(set(names)) != len(names):
        raise ValueError("decision_model guardrail check names must be unique")
    return checks


_STR_KEY_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_OBJECT_SEQ_ADAPTER: Final = TypeAdapter(tuple[object, ...])

_ToolCall: TypeAlias = ChatCompletionToolCallChunk | ChatCompletionAssistantToolCall | ChatCompletionMessageToolCall


def _message_field(message: object, key: str) -> object | None:
    if isinstance(message, Mapping):
        return _STR_KEY_DICT_ADAPTER.validate_python(message).get(key)
    return getattr(message, key, None)


def _tool_call_text(
    tool_call: _ToolCall,
) -> str | None:
    if isinstance(tool_call, dict):
        function: Final = tool_call.get("function")
        raw_arguments: Final = _message_field(function, "arguments")
        if not isinstance(raw_arguments, str) or not raw_arguments.strip():
            return None
        raw_name: Final = _message_field(function, "name")
        dict_name: Final = raw_name if isinstance(raw_name, str) else ""
        return f"{dict_name}({raw_arguments})"
    arguments_attr: Final = tool_call.function.arguments
    if not arguments_attr or not arguments_attr.strip():
        return None
    return f"{tool_call.function.name or ''}({arguments_attr})"


def _tool_call_id(tool_call: object) -> str | None:
    call_id: Final = _message_field(tool_call, "id")
    return str(call_id) if isinstance(call_id, (str, int)) else None


def _structured_tool_calls(inputs: GenericGuardrailAPIInputs) -> Iterator[ChatCompletionAssistantToolCall]:
    for message in inputs.get("structured_messages") or ():
        if "role" in message and message["role"] == "assistant":
            yield from message.get("tool_calls") or ()


def _tool_call_texts(inputs: GenericGuardrailAPIInputs) -> tuple[str, ...]:
    tool_calls: Final = chain(inputs.get("tool_calls") or (), _structured_tool_calls(inputs))
    return tuple(dict.fromkeys(text for text in map(_tool_call_text, tool_calls) if text is not None))


def _chunk_text(text: str, max_chars: int) -> tuple[str, ...]:
    """Split text into chunks of at most max_chars that overlap by
    min(2000, max_chars // 4) and together cover the whole text."""
    if len(text) <= max_chars:
        return (text,)
    step: Final = max_chars - min(2000, max_chars // 4)
    return tuple(text[start : start + max_chars] for start in range(0, len(text) - max_chars + step, step))


def _message_role(message: object) -> str | None:
    role: Final = _message_field(message, "role")
    return role if isinstance(role, str) else None


def _last_index_where(
    sequence: Sequence[object],
    predicate: Callable[[object], bool],
) -> int | None:
    return next((idx for idx in range(len(sequence) - 1, -1, -1) if predicate(sequence[idx])), None)


def _new_messages(
    messages: Sequence[object],
) -> tuple[object, ...]:
    if not messages:
        return ()
    last_non_assistant: Final = _last_index_where(messages, lambda m: _message_role(m) != "assistant")
    if last_non_assistant is None:
        return tuple(messages)
    prior_assistant: Final = _last_index_where(messages[:last_non_assistant], lambda m: _message_role(m) == "assistant")
    if prior_assistant is None:
        return tuple(messages)
    return tuple(messages[prior_assistant + 1 :])


def _part_text(part: object) -> str | None:
    val: Final = _message_field(part, "text")
    return val if isinstance(val, str) else None


def _message_content_texts(message: object) -> Iterator[str]:
    content: Final = _message_field(message, "content")
    if isinstance(content, str):
        if content.strip():
            yield content
    elif isinstance(content, (list, tuple)):
        part_list: Final = _OBJECT_SEQ_ADAPTER.validate_python(content)
        for part in part_list:
            val = _part_text(part)
            if val is not None and val.strip():
                yield val


def _message_tool_calls(message: object) -> Iterator[_ToolCall]:
    role: Final = _message_role(message)
    if role != "assistant":
        return
    tool_calls: Final = _message_field(message, "tool_calls")
    if isinstance(tool_calls, (list, tuple)):
        call_list: Final = _OBJECT_SEQ_ADAPTER.validate_python(tool_calls)
        for call in call_list:
            if isinstance(call, (ChatCompletionMessageToolCall, dict)):
                yield call  # pyright: ignore[reportReturnType]  # dict matches TypedDict
    provider_specific: Final = _message_field(message, "provider_specific_fields")
    if provider_specific is not None:
        server_calls: Final = _message_field(provider_specific, "server_tool_calls")
        if isinstance(server_calls, (list, tuple)):
            server_list: Final = _OBJECT_SEQ_ADAPTER.validate_python(server_calls)
            for call in server_list:
                if isinstance(call, (ChatCompletionMessageToolCall, dict)):
                    yield call  # pyright: ignore[reportReturnType]  # dict matches TypedDict
    function_call: Final = _message_field(message, "function_call")
    if function_call is not None and not isinstance(function_call, (list, tuple)):
        yield {"type": "function", "function": function_call}  # pyright: ignore[reportReturnType]  # dict matches TypedDict


def _policy_digest(decision_model: str, checks: Sequence[DecisionModelCheck]) -> str:
    sorted_checks: Final = sorted(checks, key=lambda c: c.name)
    payload: Final = {
        "decision_model": decision_model,
        "checks": [
            {
                "name": c.name,
                "instructions": c.instructions,
                "threshold": c.threshold,
                "action": c.action,
            }
            for c in sorted_checks
        ],
    }
    encoded: Final = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


_MAX_SCREENED_PREFIXES: Final = 10000


def _non_assistant_messages(
    messages: Sequence[object],
) -> tuple[object, ...]:
    return tuple(m for m in messages if _message_role(m) != "assistant")


def _message_fingerprint_structure(message: object) -> Mapping[str, object]:
    role: Final = _message_role(message)
    name: Final = _message_field(message, "name")
    tool_call_id: Final = _message_field(message, "tool_call_id")
    contents: Final = tuple(_message_content_texts(message))
    tool_calls: Final = tuple(text for text in map(_tool_call_text, _message_tool_calls(message)) if text is not None)
    return {
        "role": str(role) if role is not None else None,
        "name": str(name) if name is not None else None,
        "tool_call_id": str(tool_call_id) if tool_call_id is not None else None,
        "contents": contents,
        "tool_calls": tool_calls,
    }


def _messages_fingerprint(messages: Sequence[object]) -> str:
    structured: Final = tuple(map(_message_fingerprint_structure, messages))
    encoded: Final = json.dumps(structured, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DecisionModelGuardrail(CustomGuardrail):
    """Runs predicate checks on request/response text via the Decisions API and blocks on flagged checks."""

    records_own_guardrail_information: ClassVar[bool] = True

    def __init__(
        self,
        guardrail_name: str,
        decision_model: str,
        checks: Sequence[DecisionModelCheck],
        event_hook: GuardrailEventHooks | tuple[GuardrailEventHooks, ...] | Mode | None = None,
        default_on: bool = False,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        max_input_chars: int = 24000,
        router_provider: Callable[[], Router | None] | None = None,
        timeout: float | None = None,
        max_concurrent_decision_calls: int = 8,
        streaming_end_of_stream_only: bool = False,
        streaming_sampling_rate: int = 5,
        streaming_buffer_until_moderated: bool | None = None,
        screen_scope: Literal["all", "new_messages", "latest_turn"] = "all",
        cache: DualCache | None = None,
    ) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # base init takes untyped **kwargs
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=list(event_hook)
            if isinstance(event_hook, tuple)
            else (event_hook or GuardrailEventHooks.pre_call),
            default_on=default_on,
        )
        self.decision_model = decision_model
        self.checks = _resolve_checks(tuple(checks))
        self._policy_digest: Final = _policy_digest(self.decision_model, self.checks)
        self.unreachable_fallback = unreachable_fallback
        self.max_input_chars = max_input_chars
        self._router_provider = router_provider or default_router_provider
        self.timeout = timeout
        self._max_concurrent_decision_calls = max_concurrent_decision_calls
        self.streaming_end_of_stream_only = streaming_end_of_stream_only
        self.streaming_sampling_rate = streaming_sampling_rate
        if streaming_buffer_until_moderated is not None:
            self.streaming_buffer_until_moderated = streaming_buffer_until_moderated
        self.screen_scope: Final[Literal["all", "new_messages", "latest_turn"]] = screen_scope
        self._cache: Final[DualCache] = cache if cache is not None else self._incremental_scan_cache()
        self._screened_prefixes: Final[OrderedDict[str, None]] = OrderedDict()  # mutable-ok: in-memory cache
        self._semaphores_by_loop: dict[AbstractEventLoop, Semaphore] = {}  # mutable-ok: per-loop semaphore registry

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: base signature returns list
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call, GuardrailEventHooks.post_call]

    @staticmethod
    def get_config_model() -> type[DecisionModelGuardrailConfigModel] | None:
        return DecisionModelGuardrailConfigModel

    async def _call_decisions(
        self, text: str, questions: Sequence[Mapping[str, object]], auth: UserAPIKeyAuth | None = None
    ) -> object:
        router: Final = self._router_provider()
        team_id: Final = caller_team_for(router, self.decision_model, auth)
        if judge_target(router, self.decision_model, team_id).via == "router" and router is not None:
            return await router.adecisions(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]  # factory_function-generated member
                model=self.decision_model,
                input=text,
                questions=questions,
                num_retries=0,
                fallbacks=[],
                timeout=self.timeout,
                **_team_scope(team_id, auth.team_alias if auth is not None else None),
            )
        return await litellm.adecisions(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]  # SDK dispatch returns a broad union
            model=self.decision_model,
            input=text,
            questions=questions,
            num_retries=0,
            timeout=self.timeout,
        )

    def _get_decision_semaphore(self) -> asyncio.Semaphore:
        """Per-event-loop semaphore shared by every decisions call on this instance."""
        loop: Final = asyncio.get_running_loop()
        existing: Final = self._semaphores_by_loop.get(loop)
        if existing is not None:
            return existing
        created: Final = asyncio.Semaphore(self._max_concurrent_decision_calls)
        self._semaphores_by_loop[loop] = created
        return created

    async def _run_checks(
        self,
        text: str,
        questions: Sequence[Mapping[str, object]],
        semaphore: asyncio.Semaphore,
        auth: UserAPIKeyAuth | None = None,
    ) -> OpenAIDecisionResponse:
        async with semaphore:
            raw_response: Final = await self._call_decisions(text, questions, auth)
        if not isinstance(raw_response, OpenAIDecisionResponse):
            raise TypeError(f"decisions call returned {type(raw_response).__name__}, expected OpenAI format")
        return raw_response

    @staticmethod
    def _incremental_scan_cache() -> DualCache:
        from litellm.integrations.custom_guardrail import dc as fallback_cache

        try:
            from litellm.proxy import proxy_server

            logging_obj: Final[object] = getattr(proxy_server, "proxy_logging_obj", None)
            if logging_obj is not None:
                internal_cache: Final[object] = getattr(logging_obj, "internal_usage_cache", None)
                if internal_cache is not None:
                    dc: Final[object] = getattr(internal_cache, "dual_cache", None)
                    if isinstance(dc, DualCache):
                        return dc
        except Exception:  # noqa: BLE001  # proxy not importable outside server context
            return fallback_cache
        return fallback_cache

    def _screened_cache_key(self, fingerprint: str) -> str:
        return f"decision_model_screened:{self.guardrail_name}:{self._policy_digest}:{fingerprint}"

    def _record_screened_in_memory(self, fingerprint: str) -> None:
        memory_key: Final = f"{self._policy_digest}:{fingerprint}"
        self._screened_prefixes[memory_key] = None
        self._screened_prefixes.move_to_end(memory_key)
        if len(self._screened_prefixes) > _MAX_SCREENED_PREFIXES:
            self._screened_prefixes.popitem(last=False)

    async def _is_prefix_screened(self, non_assistant_messages: Sequence[object]) -> bool:
        if not non_assistant_messages:
            return False
        fingerprint: Final = _messages_fingerprint(non_assistant_messages)
        memory_key: Final = f"{self._policy_digest}:{fingerprint}"
        if memory_key in self._screened_prefixes:
            self._screened_prefixes.move_to_end(memory_key)
            return True
        try:
            key: Final = self._screened_cache_key(fingerprint)
            cached: Final[object] = (  # pyright: ignore[reportUnknownVariableType]  # DualCache
                await self._cache.async_get_cache(key=key)  # pyright: ignore[reportUnknownMemberType]  # method
            )
            if cached is not None:
                self._record_screened_in_memory(fingerprint)
                return True
        except Exception as e:  # noqa: BLE001  # safe fallback: if cache read fails, screen prefix
            verbose_logger.warning(
                "decision_model guardrail %s: failed to read screened prefix cache: %s", self.guardrail_name, e
            )
        return False

    async def _record_screened_prefix(self, non_assistant_messages: Sequence[object]) -> None:
        if not non_assistant_messages:
            return
        fingerprint: Final = _messages_fingerprint(non_assistant_messages)
        self._record_screened_in_memory(fingerprint)
        try:
            await self._cache.async_set_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache method
                key=self._screened_cache_key(fingerprint), value="1", ttl=86400
            )
        except Exception as e:  # noqa: BLE001  # best-effort cache write
            verbose_logger.warning(
                "decision_model guardrail %s: failed to write screened prefix cache: %s", self.guardrail_name, e
            )

    @staticmethod
    def _extract_messages_texts(
        messages: Sequence[object],
    ) -> tuple[str, ...]:
        content_texts: Final = tuple(chain.from_iterable(map(_message_content_texts, messages)))
        assistant_tool_calls: Final = tuple(chain.from_iterable(map(_message_tool_calls, messages)))
        tool_texts: Final = tuple(
            dict.fromkeys(text for text in map(_tool_call_text, assistant_tool_calls) if text is not None)
        )
        return tuple(text for text in content_texts + tool_texts if text and text.strip())

    async def _scoped_request_texts(
        self,
        inputs: GenericGuardrailAPIInputs,
    ) -> tuple[str, ...]:
        if self.screen_scope == "all":
            return tuple(text for text in inputs.get("texts") or [] if text and text.strip()) + _tool_call_texts(inputs)
        raw_messages: Final = inputs.get("structured_messages")
        if not isinstance(raw_messages, Sequence) or not raw_messages:
            return tuple(text for text in inputs.get("texts") or [] if text and text.strip()) + _tool_call_texts(inputs)

        last_non_assistant: Final = _last_index_where(raw_messages, lambda m: _message_role(m) != "assistant")
        if last_non_assistant is None:
            return tuple(dict.fromkeys(self._extract_messages_texts(raw_messages) + _tool_call_texts(inputs)))
        prior_assistant: Final = _last_index_where(
            raw_messages[:last_non_assistant], lambda m: _message_role(m) == "assistant"
        )
        if prior_assistant is None:
            return tuple(dict.fromkeys(self._extract_messages_texts(raw_messages) + _tool_call_texts(inputs)))

        prefix: Final = raw_messages[: prior_assistant + 1]
        prefix_non_assistant: Final = _non_assistant_messages(prefix)
        is_verified: Final = await self._is_prefix_screened(prefix_non_assistant)
        if not is_verified:
            verbose_logger.warning(
                "decision_model guardrail %s: unverified assistant turn prefix; screening full conversation",
                self.guardrail_name,
            )
            return tuple(dict.fromkeys(self._extract_messages_texts(raw_messages) + _tool_call_texts(inputs)))

        new_msgs: Final = _new_messages(raw_messages)
        new_assistant_tool_ids: Final = frozenset(
            _tool_call_id(c)
            for c in chain.from_iterable(map(_message_tool_calls, new_msgs))
            if _tool_call_id(c) is not None
        )
        matched_input_calls: Final = tuple(
            c
            for c in (inputs.get("tool_calls") or ())
            if _tool_call_id(c) is not None and _tool_call_id(c) in new_assistant_tool_ids
        )
        matched_call_texts: Final = tuple(
            text for text in map(_tool_call_text, matched_input_calls) if text is not None
        )
        return tuple(dict.fromkeys(self._extract_messages_texts(new_msgs) + matched_call_texts))

    def _check_verdict(
        self, check: DecisionModelCheck, answers_by_call: Sequence[Mapping[str | None, object]]
    ) -> _CheckVerdict:
        """Aggregate one check across calls: probability is the max over predicate answers;
        a call with no predicate answer counts as unanswered, which blocks a Block check under
        fail_closed and only warns under fail_open or for log-only checks."""
        answers: Final = tuple(call_answers.get(check.name) for call_answers in answers_by_call)
        probabilities: Final = tuple(
            answer.probability for answer in answers if isinstance(answer, OpenAIPredicateAnswer)
        )
        missing: Final = len(probabilities) < len(answers)
        if missing:
            verbose_logger.warning(
                "decision_model guardrail %s: check '%s' got no predicate answer in %d of %d decisions calls",
                self.guardrail_name,
                check.name,
                len(answers) - len(probabilities),
                len(answers),
            )
        probability: Final = max(probabilities) if probabilities else None
        unanswered_block: Final = missing and check.action == "block" and self.unreachable_fallback == "fail_closed"
        return {
            "name": check.name,
            "probability": probability,
            "threshold": check.threshold,
            "action": check.action,
            "flagged": unanswered_block or (probability is not None and probability >= check.threshold),
            "reason": "no_answer" if missing else None,
        }

    def _verdicts(self, responses: Sequence[OpenAIDecisionResponse]) -> tuple[_CheckVerdict, ...]:
        answers_by_call: Final = tuple({answer.name: answer for answer in response.answers} for response in responses)
        return tuple(self._check_verdict(check, answers_by_call) for check in self.checks)

    def _handle_call_failure(self, error: Exception) -> None:
        """fail_open logs and returns; fail_closed raises a generic 502 (upstream detail stays in server logs)."""
        if self.unreachable_fallback == "fail_open":
            verbose_logger.warning(
                "decision_model guardrail %s: decisions call failed; fail_open configured, allowing the request. Error: %s",
                self.guardrail_name,
                error,
            )
            return
        verbose_logger.error(
            "decision_model guardrail %s: decisions call failed. Error: %s", self.guardrail_name, error
        )
        raise HTTPException(
            status_code=502,
            detail={"error": "Decision model guardrail failed to respond"},
        )

    def _log(
        self,
        request_data: dict[str, object],  # mutable-ok: base helper writes into the live request dict
        verdicts: tuple[_CheckVerdict, ...],
        status: GuardrailStatus,
        start_time: float,
    ) -> None:
        self.add_standard_logging_guardrail_information_to_request_data(  # pyright: ignore[reportUnknownMemberType]  # untyped base helper
            guardrail_provider="decision_model",
            guardrail_json_response={
                "model": self.decision_model,
                "checks": [
                    {
                        "name": v["name"],
                        "probability": v["probability"],
                        "threshold": v["threshold"],
                        "action": v["action"],
                        "flagged": v["flagged"],
                        "reason": v["reason"],
                    }
                    for v in verdicts
                ],
            },
            request_data=request_data,
            guardrail_status=status,
            start_time=start_time,
            end_time=time(),
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: base signature passes the live request dict
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = (
            await self._scoped_request_texts(inputs)
            if input_type == "request"
            else tuple(text for text in inputs.get("texts") or [] if text and text.strip()) + _tool_call_texts(inputs)
        )
        if not texts:
            return inputs
        start_time: Final = time()

        chunks: Final = tuple(chain.from_iterable(_chunk_text(text, self.max_input_chars) for text in texts))
        if not chunks:
            return inputs

        questions: Final = [
            {"type": "predicate", "name": check.name, "instructions": check.instructions or ""} for check in self.checks
        ]
        semaphore: Final = self._get_decision_semaphore()
        auth: Final = attached_auth(request_data)

        batch_size: Final = max(1, self._max_concurrent_decision_calls)
        responses_acc: tuple[OpenAIDecisionResponse, ...] = ()  # rebind-ok: accumulate responses across chunk batches
        errors_acc: tuple[Exception, ...] = ()  # rebind-ok: accumulate errors across chunk batches
        for start_idx in range(0, len(chunks), batch_size):
            batch = chunks[start_idx : start_idx + batch_size]
            results = await asyncio.gather(
                *(self._run_checks(chunk, questions, semaphore, auth) for chunk in batch),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, Exception):
                    raise r
            responses_acc = responses_acc + tuple(r for r in results if isinstance(r, OpenAIDecisionResponse))
            errors_acc = errors_acc + tuple(r for r in results if isinstance(r, Exception))
            batch_verdicts = self._verdicts(responses_acc)
            if any(v["action"] == "block" for v in batch_verdicts if v["flagged"]):
                break

        if errors_acc and not responses_acc:
            self._log(request_data, (), "guardrail_failed_to_respond", start_time)
            self._handle_call_failure(errors_acc[0])
            return inputs

        final_verdicts: Final = self._verdicts(responses_acc)
        flagged: Final = tuple(v for v in final_verdicts if v["flagged"])
        intervened: Final = any(v["action"] == "block" for v in flagged)
        if errors_acc and not intervened:
            self._log(request_data, final_verdicts, "guardrail_failed_to_respond", start_time)
            self._handle_call_failure(errors_acc[0])
            return inputs

        status: Final = "guardrail_intervened" if intervened else ("guardrail_flagged" if flagged else "success")
        self._log(request_data, final_verdicts, status, start_time)
        if intervened:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated decision model guardrail policy",
                    "guardrail_name": self.guardrail_name,
                    "flagged_checks": [v["name"] for v in flagged if v["action"] == "block"],
                },
            )
        if input_type == "request":
            raw_messages: Final = inputs.get("structured_messages")
            if isinstance(raw_messages, Sequence) and raw_messages:
                await self._record_screened_prefix(_non_assistant_messages(raw_messages))
        return inputs


__all__ = ["DecisionModelGuardrail"]
