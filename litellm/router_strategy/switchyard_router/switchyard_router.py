from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.internal_call_metadata import (
    effective_turn_off_message_logging,
    forwarded_internal_call_metadata,
    parent_session_kwargs,
)
from litellm.router_strategy.switchyard_router.config import (
    DecisionClassifierConfig,
    LlmClassifierConfig,
    SwitchyardRouterConfig,
)
from litellm.router_strategy.switchyard_router.translation import (
    chat_messages,
    decision_request,
    decision_response,
    judge_request,
    judge_response,
    mapping,
    request_to_switchyard,
)
from litellm.types.decisions import OpenAIDecisionResponse
from litellm.types.router import PreRoutingHookResponse
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN, ModelResponse, StandardLoggingRoutingDecision

if TYPE_CHECKING:
    from switchyard_rust.libsy import Algorithm, DecisionCall, ModelCall, RoutingOutcome

JudgeCompletion: TypeAlias = Callable[[Mapping[str, object]], Awaitable[ModelResponse]]
DecisionCompletion: TypeAlias = Callable[[Mapping[str, object]], Awaitable[OpenAIDecisionResponse]]
FallbackReason: TypeAlias = Literal[
    "unsupported_request_type",
    "unsupported_content",
    "libsy_error",
    "unexpected_judge_call",
    "unexpected_decision_call",
    "target_outside_config",
    "no_decision",
]
_T = TypeVar("_T")
_SESSION_HEADER: Final = "x-switchyard-session-id"
_FORWARDED_HEADER_PREFIXES: Final = ("x-switchyard-", "x-codex-", "x-nemo-relay-", "x-dynamo-")
_FORWARDED_HEADERS: Final = frozenset(
    {"x-session-id", "session-id", "thread-id", "x-task-id", "x-request-id", "x-client-request-id"}
)


async def _resolved(value: _T) -> _T:
    return value


_SESSION_ALIASES: Final = frozenset(
    {_SESSION_HEADER, "x-claude-code-session-id", "x-nemo-relay-session-id", "x-session-id", "session-id"}
)
_CODEX_TURN_METADATA_HEADER: Final = "x-codex-turn-metadata"


def _forwarded_header(name: str, session_scoped: bool) -> bool:
    lowered: Final = name.lower()
    if lowered in _SESSION_ALIASES or (lowered == _CODEX_TURN_METADATA_HEADER and not session_scoped):
        return False
    return lowered in _FORWARDED_HEADERS or lowered.startswith(_FORWARDED_HEADER_PREFIXES)


def _caller_key_hash(request_kwargs: Mapping[str, object]) -> str:
    internal: Final = mapping(request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs)) or {})
    key: Final = internal.get("user_api_key_hash")
    return str(key) if key is not None else ""


def _scoped_session(request_kwargs: Mapping[str, object]) -> str | None:
    session: Final = parent_session_kwargs(request_kwargs).get("litellm_session_id")
    if not session:
        return None
    return hashlib.sha256(f"{_caller_key_hash(request_kwargs)}\0{session}".encode()).hexdigest()


def switchyard_headers(request_kwargs: Mapping[str, object]) -> Mapping[str, str]:
    proxy_request: Final = mapping(request_kwargs.get("proxy_server_request") or {})
    client: Final = mapping(proxy_request.get("headers") or {})
    session: Final = _scoped_session(request_kwargs)
    forwarded: Final = {
        name.lower(): value
        for name, value in client.items()
        if isinstance(value, str) and _forwarded_header(name, session is not None)
    }
    return {**forwarded, **({_SESSION_HEADER: session} if session else {})}


class SwitchyardRouter:
    def __init__(
        self,
        model_name: str,
        config: SwitchyardRouterConfig,
        algorithm: Algorithm,
        completion: JudgeCompletion,
        decide: DecisionCompletion,
    ) -> None:
        self.model_name = model_name
        self.config = config
        self.algorithm = algorithm
        self.completion = completion
        self.decide = decide

    def _fallback(
        self, reason: FallbackReason, messages: Sequence[Mapping[str, object]] | None
    ) -> PreRoutingHookResponse:
        return PreRoutingHookResponse(
            model=self.config.models.capable,
            messages=[dict(message) for message in messages] if messages is not None else None,
            routing_decision={
                "router_type": "switchyard",
                "router_model_name": self.model_name,
                "routed_model": self.config.models.capable,
                "switchyard_algorithm": self.config.algorithm,
                "switchyard_fallback_reason": reason,
            },
        )

    def _internal_call(
        self,
        request: Mapping[str, object],
        caller: Mapping[str, object],
        bucket: Literal["metadata", "litellm_metadata"],
    ) -> Mapping[str, object]:
        caller_metadata: Final = mapping(caller.get(get_metadata_variable_name_from_kwargs(caller)) or {})
        return {
            **request,
            bucket: {
                **forwarded_internal_call_metadata(caller_metadata, AUTOROUTER_CLASSIFIER_CALL_ORIGIN),
                INTERNAL_CALL_ORIGIN_METADATA_KEY: AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
            },
            "proxy_server_request": {"body": request},
            "num_retries": 0,
            "turn_off_message_logging": effective_turn_off_message_logging(caller),
            **parent_session_kwargs(caller),
        }

    def _decision(
        self, pick: str, evidence: object, messages: Sequence[Mapping[str, object]] | None
    ) -> PreRoutingHookResponse:
        decision: Final[StandardLoggingRoutingDecision] = {
            "router_type": "switchyard",
            "router_model_name": self.model_name,
            "routed_model": pick,
            "switchyard_algorithm": self.config.algorithm,
            "switchyard_evidence": evidence,
        }
        return PreRoutingHookResponse(
            model=pick,
            messages=[dict(message) for message in messages] if messages is not None else None,
            routing_decision=decision,
        )

    async def _answer_judge(self, call: ModelCall, caller: Mapping[str, object]) -> None:
        try:
            converted: Final = judge_response(
                await self.completion(
                    self._internal_call(judge_request(call.request, call.models[0]), caller, "metadata")
                )
            )
        except Exception as error:  # noqa: BLE001  # judge/provider failures are inputs to Switchyard's fall-open decision
            await call.fail(error)
        else:
            await call.respond(_resolved(converted))

    async def _answer_decision(self, call: DecisionCall, caller: Mapping[str, object]) -> None:
        try:
            answer: Final = decision_response(
                await self.decide(
                    self._internal_call(decision_request(call.request, call.model), caller, "litellm_metadata")
                ),
                call.request,
            )
        except Exception as error:  # noqa: BLE001  # decision/provider failures are inputs to Switchyard's fall-open decision
            await call.fail(error)
        else:
            await call.respond(_resolved(answer))

    def _outcome(
        self, outcome: RoutingOutcome, messages: Sequence[Mapping[str, object]] | None
    ) -> PreRoutingHookResponse:
        pick: Final = outcome.selected_model_ids[0]
        if pick not in (self.config.models.capable, self.config.models.efficient):
            return self._fallback("target_outside_config", messages)
        metadata: Final = outcome.metadata
        evidence: Final[object] = metadata.evidence if metadata is not None else None
        return self._decision(pick, evidence, messages)

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Mapping[str, object],
        messages: Sequence[Mapping[str, object]] | None = None,
        input: object = None,
        specific_deployment: bool | None = False,
        call_type: str | None = None,
    ) -> PreRoutingHookResponse | None:
        from switchyard_rust.libsy import LibsyError, Step

        if model != self.model_name:
            return None
        try:
            chat: Final = chat_messages(call_type, model, messages, input, request_kwargs)
            request: Final = request_to_switchyard(chat) if chat is not None else None
        except (ValueError, KeyError, TypeError):
            return self._fallback("unsupported_content", messages)
        if request is None:
            return self._fallback("unsupported_request_type", messages)
        judge: Final = self.config.models.judge
        try:
            async for step in self.algorithm.run_stream(
                request, self.config.runtime_models(), headers=switchyard_headers(request_kwargs)
            ):
                match step:
                    case Step.CallModel(call) if isinstance(
                        self.config.classifier, LlmClassifierConfig
                    ) and call.models[:1] == [judge]:
                        await self._answer_judge(call, request_kwargs)
                    case Step.CallDecision(call) if (
                        isinstance(self.config.classifier, DecisionClassifierConfig) and call.model == judge
                    ):
                        await self._answer_decision(call, request_kwargs)
                    case Step.CallModel():
                        return self._fallback("unexpected_judge_call", messages)
                    case Step.CallDecision():
                        return self._fallback("unexpected_decision_call", messages)
                    case Step.Done(outcome):
                        return self._outcome(outcome, messages)
        except LibsyError:
            return self._fallback("libsy_error", messages)
        return self._fallback("no_decision", messages)
