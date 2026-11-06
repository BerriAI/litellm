from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar

from pydantic import TypeAdapter

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
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
    from switchyard.libsy import Algorithm

JudgeCompletion: TypeAlias = Callable[[Mapping[str, object]], Awaitable[ModelResponse]]
DecisionCompletion: TypeAlias = Callable[[Mapping[str, object]], Awaitable[OpenAIDecisionResponse]]
FallbackReason: TypeAlias = Literal[
    "unsupported_content",
    "libsy_error",
    "unexpected_judge_call",
    "unexpected_decision_call",
    "target_outside_config",
    "no_decision",
]
_T = TypeVar("_T")
_OBJECT: Final = TypeAdapter(object)
_SESSION_HEADER: Final = "x-switchyard-session-id"
_FORWARDED_HEADER_PREFIXES: Final = ("x-switchyard-", "x-codex-", "x-nemo-relay-", "x-dynamo-")
_FORWARDED_HEADERS: Final = frozenset(
    {"x-session-id", "session-id", "thread-id", "x-task-id", "x-request-id", "x-client-request-id"}
)


async def _resolved(value: _T) -> _T:
    return value


def _forwarded_header(name: str) -> bool:
    lowered: Final = name.lower()
    return lowered in _FORWARDED_HEADERS or lowered.startswith(_FORWARDED_HEADER_PREFIXES)


def switchyard_headers(request_kwargs: Mapping[str, object]) -> Mapping[str, str]:
    proxy_request: Final = mapping(request_kwargs.get("proxy_server_request") or {})
    client: Final = mapping(proxy_request.get("headers") or {})
    session: Final = parent_session_kwargs(request_kwargs).get("litellm_session_id")
    return {
        **{name.lower(): value for name, value in client.items() if isinstance(value, str) and _forwarded_header(name)},
        **({_SESSION_HEADER: session} if session else {}),
    }


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

    def _internal_call(self, request: Mapping[str, object], caller: Mapping[str, object]) -> Mapping[str, object]:
        return {
            **request,
            "metadata": {
                **forwarded_internal_call_metadata(
                    mapping(caller.get("metadata") or {}), AUTOROUTER_CLASSIFIER_CALL_ORIGIN
                ),
                INTERNAL_CALL_ORIGIN_METADATA_KEY: AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
            },
            "litellm_metadata": {
                **forwarded_internal_call_metadata(
                    mapping(caller.get("litellm_metadata") or {}), AUTOROUTER_CLASSIFIER_CALL_ORIGIN
                ),
                INTERNAL_CALL_ORIGIN_METADATA_KEY: AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
            },
            "proxy_server_request": {"body": request},
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
            **({"switchyard_evidence": evidence} if evidence is not None else {}),
        }
        return PreRoutingHookResponse(
            model=pick,
            messages=[dict(message) for message in messages] if messages is not None else None,
            routing_decision=decision,
        )

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Mapping[str, object],
        messages: Sequence[Mapping[str, object]] | None = None,
        input: object = None,
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse | None:
        from switchyard.libsy import LibsyError, Step

        if model != self.model_name:
            return None
        try:
            request: Final = request_to_switchyard(messages, request_kwargs)
        except (ValueError, KeyError, TypeError):
            return self._fallback("unsupported_content", messages)
        judge: Final = self.config.models.judge
        try:
            async for step in self.algorithm.run_stream(
                request, self.config.runtime_models(), headers=switchyard_headers(request_kwargs)
            ):
                match step:
                    case Step.CallModel(call):
                        if not isinstance(self.config.classifier, LlmClassifierConfig) or call.models[:1] != [judge]:
                            return self._fallback("unexpected_judge_call", messages)
                        try:
                            converted: Final = judge_response(
                                await self.completion(
                                    self._internal_call(judge_request(call.request, call.models[0]), request_kwargs)
                                )
                            )
                        except Exception as error:  # noqa: BLE001  # judge/provider failures are inputs to Switchyard's fall-open decision
                            await call.fail(error)
                        else:
                            await call.respond(_resolved(converted))
                    case Step.CallDecision(call):
                        if not isinstance(self.config.classifier, DecisionClassifierConfig) or call.model != judge:
                            return self._fallback("unexpected_decision_call", messages)
                        try:
                            answer: Final = decision_response(
                                await self.decide(
                                    self._internal_call(decision_request(call.request, call.model), request_kwargs)
                                ),
                                call.request,
                            )
                        except Exception as error:  # noqa: BLE001  # decision/provider failures are inputs to Switchyard's fall-open decision
                            await call.fail(error)
                        else:
                            await call.respond(_resolved(answer))
                    case Step.Done(outcome):
                        pick: Final = outcome.selected_model_ids[0]
                        if pick not in (self.config.models.capable, self.config.models.efficient):
                            return self._fallback("target_outside_config", messages)
                        return self._decision(
                            pick,
                            _OBJECT.validate_python(outcome.metadata.evidence)
                            if outcome.metadata is not None
                            else None,
                            messages,
                        )
        except LibsyError:
            return self._fallback("libsy_error", messages)
        return self._fallback("no_decision", messages)
