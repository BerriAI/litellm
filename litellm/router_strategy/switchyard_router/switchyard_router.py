from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, TypeAlias

from pydantic import TypeAdapter

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.internal_call_metadata import (
    effective_turn_off_message_logging,
    forwarded_internal_call_metadata,
    parent_session_kwargs,
)
from litellm.router_strategy.switchyard_router.config import SwitchyardRouterConfig, build_algorithm
from litellm.router_strategy.switchyard_router.translation import (
    judge_request,
    judge_response,
    mapping,
    request_to_switchyard,
)
from litellm.types.router import PreRoutingHookResponse
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN, ModelResponse, StandardLoggingRoutingDecision

JudgeCompletion: TypeAlias = Callable[[Mapping[str, object]], Awaitable[ModelResponse]]
_OBJECT: Final = TypeAdapter(object)


class SwitchyardRouter:
    def __init__(
        self,
        model_name: str,
        config: SwitchyardRouterConfig,
        completion: JudgeCompletion,
    ) -> None:
        self.model_name = model_name
        self.config = config
        self.algorithm = build_algorithm(config)
        self.completion = completion

    def _fallback(self, reason: str, messages: Sequence[Mapping[str, object]] | None) -> PreRoutingHookResponse:
        return PreRoutingHookResponse(
            model=self.config.default_model,
            messages=[dict(message) for message in messages] if messages is not None else None,
            routing_decision={
                "router_type": "switchyard",
                "router_model_name": self.model_name,
                "routed_model": self.config.default_model,
                "switchyard_algorithm": self.config.algorithm,
                "switchyard_fallback_reason": reason,
            },
        )

    def _session_headers(self, request: Mapping[str, object]) -> Mapping[str, str] | None:
        if self.config.classifier is None or not self.config.classifier.session_affinity:
            return None
        session: Final = parent_session_kwargs(request).get("litellm_session_id")
        return {"x-switchyard-session-id": session} if session else None

    async def _judge(self, request: Mapping[str, object], caller: Mapping[str, object]) -> ModelResponse:
        return await self.completion(
            {
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
        try:
            async for step in self.algorithm.run_stream(
                request,
                self.config.runtime_models(),
                headers=self._session_headers(request_kwargs),
            ):
                match step:
                    case Step.CallModel(call):
                        if (
                            self.config.classifier is None
                            or not call.models
                            or call.models[0] != self.config.models.judge
                        ):
                            raise RuntimeError("Switchyard returned an unexpected judge call")
                        try:
                            response: Final = await self._judge(
                                judge_request(call.request, call.models[0]), request_kwargs
                            )
                            converted: Final = judge_response(response)
                        except Exception as error:  # noqa: BLE001  # judge/provider failures are inputs to Switchyard's fall-open decision
                            call.fail(error)
                        else:
                            call.respond(converted)
                    case Step.Done(outcome):
                        pick: Final = outcome.selected_model_ids[0]
                        if pick not in (self.config.models.capable, self.config.models.efficient):
                            raise RuntimeError("Switchyard selected a model outside the configured targets")
                        evidence: Final = (
                            _OBJECT.validate_python(outcome.metadata.evidence) if outcome.metadata is not None else None
                        )
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
        except LibsyError:
            return self._fallback("libsy_error", messages)
        raise RuntimeError("Switchyard ended without a routing decision")
