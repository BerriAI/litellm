from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast  # noqa: TID251  # cost helpers type raw frames as a TypedDict list

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.cost_calculator import RealtimeAPITokenUsageProcessor, handle_realtime_stream_cost_calculation
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.openai import OpenAIRealtimeStreamList, ResponseAPIUsage
from litellm.types.passthrough_endpoints.pass_through_endpoints import WebsocketUpstreamFrameKind
from litellm.types.utils import LlmProviders, ModelResponse, Usage

WebsocketMessages: TypeAlias = Sequence[Mapping[str, object]]

OPENAI_WEBSOCKET_PROVIDER: Final = LlmProviders.OPENAI.value

_SESSION_CREATED: Final = "session.created"
_RESPONSE_CREATED: Final = "response.created"
_REALTIME_RESPONSE_DONE: Final = "response.done"
_RESPONSES_TURN_ENDS: Final = frozenset({"response.completed", "response.incomplete", "response.failed"})
_TURN_ENDS: Final = _RESPONSES_TURN_ENDS | {_REALTIME_RESPONSE_DONE}
_MODEL_NAMING_EVENTS: Final = _TURN_ENDS | {_SESSION_CREATED, _RESPONSE_CREATED}
_RESPONSE_COST_HEADER: Final = "llm_provider-x-litellm-response-cost"


class _NamedModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str | None = None


class _ModelNamingEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    session: _NamedModel | None = None
    response: _NamedModel | None = None


class _EndedResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str | None = None
    usage: ResponseAPIUsage | None = None


class _ResponsesTurnEndEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["response.completed", "response.incomplete", "response.failed"]
    response: _EndedResponse


_MODEL_NAMING_EVENT: Final = TypeAdapter(_ModelNamingEvent)
_RESPONSES_TURN_END_EVENT: Final = TypeAdapter(_ResponsesTurnEndEvent)


@dataclass(frozen=True, slots=True)
class _Billing:
    usage: Usage
    cost: float


@dataclass(frozen=True, slots=True)
class _CompletedTurn:
    model: str
    usage: Usage


def _event_model(event: Mapping[str, object]) -> str | None:
    if event.get("type") not in _MODEL_NAMING_EVENTS:
        return None
    try:
        naming: Final = _MODEL_NAMING_EVENT.validate_python(event)
    except ValidationError:
        return None
    return next(
        (
            container.model
            for container in (naming.session, naming.response)
            if container is not None and container.model
        ),
        None,
    )


def observed_websocket_model(messages: WebsocketMessages) -> str | None:
    return next((model for model in map(_event_model, messages) if model is not None), None)


def openai_websocket_frame_kind(frame: Mapping[str, object]) -> WebsocketUpstreamFrameKind:
    frame_type: Final = frame.get("type")
    if frame_type in _TURN_ENDS:
        return "ends_turn"
    if frame_type == _SESSION_CREATED:
        return "opens_session"
    return "relayed"


def _realtime_billing(messages: WebsocketMessages, model: str, logging_obj: LiteLLMLoggingObj) -> _Billing | None:
    if not any(event.get("type") == _REALTIME_RESPONSE_DONE for event in messages):
        return None
    results: Final = cast(  # cast-ok: the realtime cost helpers read raw provider frames through their TypedDict view
        OpenAIRealtimeStreamList, tuple(event for event in messages if "type" in event)
    )
    usage: Final = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(results)
    cost: Final = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=usage,
        custom_llm_provider=OPENAI_WEBSOCKET_PROVIDER,
        litellm_model_name=model,
        litellm_logging_obj=logging_obj,
    )
    return _Billing(usage=usage, cost=cost)


def _responses_turn(event: Mapping[str, object], session_model: str) -> _CompletedTurn | None:
    if event.get("type") not in _RESPONSES_TURN_ENDS:
        return None
    try:
        ended: Final = _RESPONSES_TURN_END_EVENT.validate_python(event)
    except ValidationError:
        verbose_proxy_logger.warning(
            "OpenAI websocket passthrough: a %s frame carried no readable usage, so it is not billed", event.get("type")
        )
        return None
    if ended.response.usage is None:
        return None
    return _CompletedTurn(
        model=ended.response.model or session_model,
        usage=ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(  # pyright: ignore[reportPrivateUsage]  # the one shared Responses usage converter
            ended.response.usage
        ),
    )


def _responses_billing(messages: WebsocketMessages, model: str) -> _Billing | None:
    turns: Final = tuple(turn for turn in (_responses_turn(event, model) for event in messages) if turn is not None)
    if not turns:
        return None
    usage: Final = RealtimeAPITokenUsageProcessor.combine_usage_objects([turn.usage for turn in turns])
    return _Billing(usage=usage, cost=sum(_responses_cost(turn.usage, turn.model) for turn in turns))


def _responses_cost(usage: Usage, model: str) -> float:
    try:
        return litellm.completion_cost(
            completion_response=ModelResponse(model=model, usage=usage),
            model=model,
            custom_llm_provider=OPENAI_WEBSOCKET_PROVIDER,
        )
    except Exception:  # noqa: BLE001  # completion_cost raises a bare Exception for an unpriced model, and the turn must still log
        verbose_proxy_logger.warning(
            "OpenAI websocket passthrough: could not price a Responses turn on %s, logging it at zero", model
        )
        return 0.0


def _priced_result(model: str, billing: _Billing, start_time: datetime) -> ModelResponse:
    result: Final = ModelResponse(
        id=f"openai-websocket-{start_time.timestamp()}",
        object="chat.completion",
        created=int(start_time.timestamp()),
        model=model,
        usage=billing.usage,
    )
    hidden: Final = {"additional_headers": MappingProxyType({_RESPONSE_COST_HEADER: billing.cost})}
    result._hidden_params = hidden  # pyright: ignore[reportPrivateUsage]  # the one settled-cost channel
    return result


class OpenAIWebsocketPassthroughLoggingHandler:
    def openai_websocket_passthrough_handler(
        self,
        websocket_messages: WebsocketMessages,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        start_time: datetime,
        kwargs: Mapping[str, object],
    ) -> PassThroughEndpointLoggingTypedDict:
        model: Final = observed_websocket_model(websocket_messages)
        if model is None:
            verbose_proxy_logger.debug("OpenAI websocket passthrough (%s): no model named, skipping cost", url_route)
            unpriced: Final[PassThroughEndpointLoggingTypedDict] = {"result": None, "kwargs": {**kwargs}}
            return unpriced

        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details.update(
            MappingProxyType({"model": model, "custom_llm_provider": OPENAI_WEBSOCKET_PROVIDER})
        )

        billing: Final = _realtime_billing(websocket_messages, model, logging_obj) or _responses_billing(
            websocket_messages, model
        )
        if billing is None:
            verbose_proxy_logger.debug("OpenAI websocket passthrough (%s): no usage reported, skipping cost", url_route)
            unbilled: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": None,
                "kwargs": {**kwargs, "model": model, "custom_llm_provider": OPENAI_WEBSOCKET_PROVIDER},
            }
            return unbilled

        logging_obj.model_call_details.update(MappingProxyType({"response_cost": billing.cost}))
        priced: Final[PassThroughEndpointLoggingTypedDict] = {
            "result": _priced_result(model, billing, start_time),
            "kwargs": {
                **kwargs,
                "model": model,
                "custom_llm_provider": OPENAI_WEBSOCKET_PROVIDER,
                "response_cost": billing.cost,
            },
        }
        return priced
