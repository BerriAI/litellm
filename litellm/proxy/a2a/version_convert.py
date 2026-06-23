"""
Normalize A2A JSON-RPC payloads to the protocol version LiteLLM serves for an agent.

LiteLLM fronts upstream agents and lets an admin pin the protocol version it speaks
to clients (``0.3`` or ``1.0``) per agent. Upstream responses may arrive in either
wire shape, so every response, stream event, forwarded request and extended card is
converted to the served version here. Conversion is shape-detecting (we infer the
payload's current version rather than trusting a stored one) and best-effort: any
failure falls back to returning the input unchanged so a conversion bug can never
break an otherwise-valid response.

The two wire shapes:

- ``0.3``: JSON dump of the compat pydantic types, discriminated by a ``kind`` field
  (``message`` / ``task`` / ``status-update`` / ``artifact-update``). A send result is
  the bare object.
- ``1.0``: protobuf JSON (``MessageToDict``), a oneof envelope keyed by
  ``message`` / ``task`` / ``statusUpdate`` / ``artifactUpdate`` with no ``kind``. A
  ``Task`` result is a bare object without ``kind``.
"""

from types import ModuleType
from typing import Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger

A2AVersion = Literal["0.3", "1.0"]
RequestId = Optional[Union[str, int]]
JsonDict = Dict[str, object]

_V1_SEND_ENVELOPE_KEYS = frozenset({"message", "task"})
_V1_STREAM_ENVELOPE_KEYS = frozenset(
    {"message", "task", "statusUpdate", "artifactUpdate"}
)


def _dump_03(model: BaseModel) -> JsonDict:
    """Dump a compat (0.3) pydantic model to its camelCase wire dict."""
    return model.model_dump(by_alias=True, exclude_none=True, mode="json")


def _best_effort(
    convert: Callable[[], JsonDict], fallback: JsonDict, *, label: str
) -> JsonDict:
    """Run a conversion, returning ``fallback`` unchanged if it raises."""
    try:
        return convert()
    except Exception as e:  # noqa: BLE001 - best-effort passthrough
        verbose_proxy_logger.debug("A2A %s conversion failed: %s", label, e)
        return fallback


def normalize_jsonrpc_response(
    content: JsonDict, target: A2AVersion, *, method: str
) -> JsonDict:
    """Convert a JSON-RPC response's ``result`` to ``target``.

    Errors and non-dict results pass through untouched.
    """
    if content.get("error") is not None:
        return content
    result = content.get("result")
    if not isinstance(result, dict):
        return content

    converted = _convert_result(
        result, target, method=method, request_id=_as_request_id(content.get("id"))
    )
    if converted is result:
        return content
    return {**content, "result": converted}


def normalize_stream_event(
    event: JsonDict, target: A2AVersion, *, request_id: RequestId
) -> JsonDict:
    """Convert a single streamed JSON-RPC event's ``result`` to ``target``."""
    if event.get("error") is not None:
        return event
    result = event.get("result")
    if not isinstance(result, dict):
        return event

    converted = _convert_stream_result(result, target, request_id=request_id)
    if converted is result:
        return event
    return {**event, "result": converted}


def normalize_request_params(
    params: JsonDict, served: A2AVersion, *, method: str
) -> JsonDict:
    """Down-convert forwarded request ``params`` from the served version to 0.3.

    Upstream agents in this proxy pivot on 0.3 wire format, so when LiteLLM serves
    1.0 the inbound params must be lowered before forwarding. A no-op when the served
    version is already 0.3.
    """
    if served == "0.3":
        return params
    return _best_effort(
        lambda: _lower_request_params(params, method=method),
        params,
        label=f"request params ({method})",
    )


def normalize_agent_card(card: JsonDict, target: A2AVersion) -> JsonDict:
    """Convert an extended agent card to ``target``.

    When lowering to 0.3, ``additionalInterfaces`` is stripped so the conversion never
    re-exposes upstream backend URLs that the LiteLLM-fronting merge deliberately drops.
    """
    if not isinstance(card, dict):
        return card

    current: A2AVersion = "1.0" if "supportedInterfaces" in card else "0.3"
    if current == target:
        return card
    return _best_effort(
        lambda: _convert_agent_card(card, target), card, label="agent card"
    )


def _as_request_id(value: object) -> RequestId:
    return value if isinstance(value, (str, int)) else None


def _convert_result(
    result: JsonDict,
    target: A2AVersion,
    *,
    method: str,
    request_id: RequestId,
) -> JsonDict:
    if method == "message/send":
        return _convert_send_result(result, target, request_id=request_id)
    if method in ("tasks/get", "tasks/cancel"):
        return _convert_task(result, target)
    return result


def _detect_send_version(result: JsonDict) -> Optional[A2AVersion]:
    if "kind" in result:
        return "0.3"
    if result.keys() & _V1_SEND_ENVELOPE_KEYS:
        return "1.0"
    return None


def _convert_send_result(
    result: JsonDict, target: A2AVersion, *, request_id: RequestId
) -> JsonDict:
    current = _detect_send_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _send_result_to(result, target, request_id),
        result,
        label="send result",
    )


def _send_result_to(
    result: JsonDict, target: A2AVersion, request_id: RequestId
) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_send_message_response,
        to_core_send_message_response,
        types_v03,
    )

    if target == "1.0":
        compat_result = _validate_message_or_task(result, types_v03)
        response = types_v03.SendMessageResponse(
            root=types_v03.SendMessageSuccessResponse(
                id=str(request_id or ""), result=compat_result
            )
        )
        return MessageToDict(
            to_core_send_message_response(response),
            preserving_proto_field_name=False,
        )

    pb = pb2_v10.SendMessageResponse()
    ParseDict(result, pb)
    return _dump_03(to_compat_send_message_response(pb, request_id).root.result)


def _convert_task(result: JsonDict, target: A2AVersion) -> JsonDict:
    current: A2AVersion = "0.3" if "kind" in result else "1.0"
    if current == target:
        return result
    return _best_effort(lambda: _task_to(result, target), result, label="task")


def _task_to(result: JsonDict, target: A2AVersion) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_task,
        to_core_task,
        types_v03,
    )

    if target == "1.0":
        core = to_core_task(types_v03.Task.model_validate(result))
        return MessageToDict(core, preserving_proto_field_name=False)

    pb = pb2_v10.Task()
    ParseDict(result, pb)
    return _dump_03(to_compat_task(pb))


def _detect_stream_version(result: JsonDict) -> Optional[A2AVersion]:
    if "kind" in result:
        return "0.3"
    if result.keys() & _V1_STREAM_ENVELOPE_KEYS:
        return "1.0"
    return None


def _convert_stream_result(
    result: JsonDict, target: A2AVersion, *, request_id: RequestId
) -> JsonDict:
    current = _detect_stream_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _stream_result_to(result, target, request_id),
        result,
        label="stream event",
    )


def _stream_result_to(
    result: JsonDict, target: A2AVersion, request_id: RequestId
) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_stream_response,
        to_core_stream_response,
        types_v03,
    )

    if target == "1.0":
        event = _validate_stream_event(result, types_v03)
        wrapper = types_v03.SendStreamingMessageSuccessResponse(
            id=str(request_id or ""), result=event
        )
        return MessageToDict(
            to_core_stream_response(wrapper), preserving_proto_field_name=False
        )

    pb = pb2_v10.StreamResponse()
    ParseDict(result, pb)
    return _dump_03(to_compat_stream_response(pb, request_id).result)


def _convert_agent_card(card: JsonDict, target: A2AVersion) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_agent_card,
        to_core_agent_card,
        types_v03,
    )

    if target == "0.3":
        pb = pb2_v10.AgentCard()
        ParseDict(card, pb, ignore_unknown_fields=True)
        lowered = _dump_03(to_compat_agent_card(pb))
        lowered.pop("additionalInterfaces", None)
        return lowered

    core = to_core_agent_card(types_v03.AgentCard.model_validate(card))
    return MessageToDict(core, preserving_proto_field_name=False)


def _validate_message_or_task(result: JsonDict, types_v03: ModuleType) -> BaseModel:
    if result.get("kind") == "task":
        return types_v03.Task.model_validate(result)
    return types_v03.Message.model_validate(result)


def _validate_stream_event(result: JsonDict, types_v03: ModuleType) -> BaseModel:
    kind = result.get("kind")
    if kind == "task":
        return types_v03.Task.model_validate(result)
    if kind == "status-update":
        return types_v03.TaskStatusUpdateEvent.model_validate(result)
    if kind == "artifact-update":
        return types_v03.TaskArtifactUpdateEvent.model_validate(result)
    return types_v03.Message.model_validate(result)


def _lower_request_params(params: JsonDict, *, method: str) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        ParseDict,
        pb2_v10,
        to_compat_cancel_task_request,
        to_compat_create_task_push_notification_config_request,
        to_compat_delete_task_push_notification_config_request,
        to_compat_get_task_push_notification_config_request,
        to_compat_get_task_request,
        to_compat_list_task_push_notification_config_request,
        to_compat_subscribe_to_task_request,
    )

    lowerings: Dict[str, Callable[[JsonDict], BaseModel]] = {
        "tasks/get": lambda p: to_compat_get_task_request(
            _parse(ParseDict, p, pb2_v10.GetTaskRequest()), ""
        ).params,
        "tasks/cancel": lambda p: to_compat_cancel_task_request(
            _parse(ParseDict, p, pb2_v10.CancelTaskRequest()), ""
        ).params,
        "tasks/resubscribe": lambda p: to_compat_subscribe_to_task_request(
            _parse(ParseDict, p, pb2_v10.SubscribeToTaskRequest()), ""
        ).params,
        "tasks/pushNotificationConfig/set": lambda p: to_compat_create_task_push_notification_config_request(
            _parse(ParseDict, p, pb2_v10.TaskPushNotificationConfig()), ""
        ).params,
        "tasks/pushNotificationConfig/get": lambda p: to_compat_get_task_push_notification_config_request(
            _parse(ParseDict, p, pb2_v10.GetTaskPushNotificationConfigRequest()), ""
        ).params,
        "tasks/pushNotificationConfig/list": lambda p: to_compat_list_task_push_notification_config_request(
            _parse(ParseDict, p, pb2_v10.ListTaskPushNotificationConfigsRequest()), ""
        ).params,
        "tasks/pushNotificationConfig/delete": lambda p: to_compat_delete_task_push_notification_config_request(
            _parse(ParseDict, p, pb2_v10.DeleteTaskPushNotificationConfigRequest()), ""
        ).params,
    }
    lower = lowerings.get(method)
    if lower is None:
        return params
    return _dump_03(lower(params))


def _parse(
    parse_dict: Callable[..., object], data: JsonDict, message: object
) -> object:
    parse_dict(data, message)
    return message
