# pyright: reportUnknownArgumentType=false
# a2a-sdk's compat conversions (pb2_v10, ParseDict, MessageToDict, to_compat_*)
# are protobuf-generated/untyped, so every conversion call here takes Unknown-typed
# arguments. This module is the A2A 0.3<->1.0 boundary; the rule is off file-wide
# rather than scattering per-line ignores across every SDK call.
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

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.proxy.a2a.agent_card import normalize_protocol_version

A2AVersion = Literal["0.3", "1.0"]
RequestId = str | int | None
JsonDict = dict[str, object]

_V1_SEND_ENVELOPE_KEYS: Final = frozenset({"message", "task"})
_V1_STREAM_ENVELOPE_KEYS: Final = frozenset({"message", "task", "statusUpdate", "artifactUpdate"})
_V03_TYPE_TO_V1_SECURITY_SCHEME_KEY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "apiKey": "apiKeySecurityScheme",
        "http": "httpAuthSecurityScheme",
        "oauth2": "oauth2SecurityScheme",
        "openIdConnect": "openIdConnectSecurityScheme",
        "mutualTLS": "mtlsSecurityScheme",
    }
)
_V1_SECURITY_SCHEME_KEY_TO_V03_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {v1_key: v03_type for v03_type, v1_key in _V03_TYPE_TO_V1_SECURITY_SCHEME_KEY.items()}
)


def _convert_security_scheme_to_v1(scheme: Mapping[str, object]) -> JsonDict:
    scheme_type: Final = scheme.get("type")
    if not isinstance(scheme_type, str):
        return dict(scheme)
    wrapper_key: Final = _V03_TYPE_TO_V1_SECURITY_SCHEME_KEY.get(scheme_type)
    if wrapper_key is None:
        return dict(scheme)
    has_location: Final = "location" in scheme
    inner: Final[JsonDict] = {
        ("location" if scheme_type == "apiKey" and key == "in" and not has_location else key): value
        for key, value in scheme.items()
        if key != "type"
    }
    return {wrapper_key: inner}


def _convert_security_scheme_to_v03(scheme: Mapping[str, object]) -> JsonDict:
    if "type" in scheme or len(scheme) != 1:
        return dict(scheme)
    wrapper_key, raw_inner = next(iter(scheme.items()))
    v03_type: Final = _V1_SECURITY_SCHEME_KEY_TO_V03_TYPE.get(wrapper_key)
    inner: Final = as_str_mapping(raw_inner)
    if v03_type is None or inner is None:
        return dict(scheme)
    has_in: Final = "in" in inner
    flat: Final[JsonDict] = {
        "type": v03_type,
        **{
            ("in" if v03_type == "apiKey" and key == "location" and not has_in else key): value
            for key, value in inner.items()
        },
    }
    return flat


def _extract_v1_scopes(raw_entry: object) -> tuple[object, ...]:
    entry: Final = as_str_mapping(raw_entry)
    if entry is None:
        return ()
    raw_list: Final = as_object_list(entry.get("list"))
    return tuple(raw_list) if raw_list is not None else ()


def _convert_security_requirement_to_v1(item: object) -> object:
    req: Final = as_str_mapping(item)
    if req is None:
        return item
    if len(req) == 1 and as_str_mapping(req.get("schemes")) is not None:
        return dict(req)
    schemes: Final[JsonDict] = {
        scheme_name: {"list": list(scopes) if (scopes := as_object_list(raw_scopes)) is not None else []}
        for scheme_name, raw_scopes in req.items()
    }
    return {"schemes": schemes}


def _convert_security_requirement_to_v03(item: object) -> object:
    req: Final = as_str_mapping(item)
    if req is None:
        return item
    raw_schemes: Final = as_str_mapping(req.get("schemes"))
    if len(req) != 1 or raw_schemes is None:
        return dict(req)
    return {scheme_name: list(_extract_v1_scopes(raw_entry)) for scheme_name, raw_entry in raw_schemes.items()}


def _is_v03_security_scheme(scheme: object) -> bool:
    mapping: Final = as_str_mapping(scheme)
    if mapping is None:
        return False
    scheme_type: Final = mapping.get("type")
    return isinstance(scheme_type, str) and scheme_type in _V03_TYPE_TO_V1_SECURITY_SCHEME_KEY


def _is_v1_security_scheme(scheme: object) -> bool:
    mapping: Final = as_str_mapping(scheme)
    if mapping is None or "type" in mapping or len(mapping) != 1:
        return False
    wrapper_key, raw_inner = next(iter(mapping.items()))
    return wrapper_key in _V1_SECURITY_SCHEME_KEY_TO_V03_TYPE and as_str_mapping(raw_inner) is not None


def _has_mismatched_security(card: Mapping[str, object], target: A2AVersion) -> bool:
    source_key: Final = "security" if target == "1.0" else "securityRequirements"
    if source_key in card:
        return True
    scheme_check: Final = _is_v03_security_scheme if target == "1.0" else _is_v1_security_scheme
    schemes: Final = as_str_mapping(card.get("securitySchemes"))
    if schemes is not None and any(scheme_check(s) for s in schemes.values()):
        return True
    skills: Final = as_object_list(card.get("skills"))
    if skills is not None:
        return any((skill_map := as_str_mapping(skill)) is not None and source_key in skill_map for skill in skills)
    return False


def _normalize_security_container(
    container: Mapping[str, object],
    *,
    source_key: str,
    target_key: str,
    convert_req: Callable[[object], object],
) -> JsonDict:
    normalized: Final[JsonDict] = {k: v for k, v in container.items() if k != source_key}
    raw_reqs: Final = as_object_list(container.get(source_key))
    if target_key not in normalized and raw_reqs is not None:
        normalized[target_key] = [convert_req(item) for item in raw_reqs]
    return normalized


def _normalize_skill_security(
    skill: object,
    *,
    source_key: str,
    target_key: str,
    convert_req: Callable[[object], object],
) -> object:
    skill_map: Final = as_str_mapping(skill)
    if skill_map is None or source_key not in skill_map:
        return skill
    return _normalize_security_container(
        skill_map,
        source_key=source_key,
        target_key=target_key,
        convert_req=convert_req,
    )


def _convert_security_schemes(schemes: Mapping[str, object], target: A2AVersion) -> JsonDict:
    converter: Final = _convert_security_scheme_to_v1 if target == "1.0" else _convert_security_scheme_to_v03
    converted: Final[JsonDict] = {}
    for name, raw_scheme in schemes.items():
        scheme_map = as_str_mapping(raw_scheme)
        converted[name] = converter(scheme_map) if scheme_map is not None else raw_scheme
    return converted


def _normalize_card_security(card: JsonDict, target: A2AVersion) -> JsonDict:
    if not _has_mismatched_security(card, target):
        return card
    source_key: Final = "security" if target == "1.0" else "securityRequirements"
    target_key: Final = "securityRequirements" if target == "1.0" else "security"
    convert_req: Final = (
        _convert_security_requirement_to_v1 if target == "1.0" else _convert_security_requirement_to_v03
    )
    normalized: Final = _normalize_security_container(
        card,
        source_key=source_key,
        target_key=target_key,
        convert_req=convert_req,
    )
    schemes: Final = as_str_mapping(card.get("securitySchemes"))
    if schemes is not None:
        normalized["securitySchemes"] = _convert_security_schemes(schemes, target)
    skills: Final = as_object_list(card.get("skills"))
    if skills is not None:
        normalized["skills"] = [
            _normalize_skill_security(
                skill,
                source_key=source_key,
                target_key=target_key,
                convert_req=convert_req,
            )
            for skill in skills
        ]
    return normalized


def _dump_03(model: BaseModel) -> JsonDict:
    """Dump a compat (0.3) pydantic model to its camelCase wire dict."""
    return model.model_dump(by_alias=True, exclude_none=True, mode="json")


def _best_effort(convert: Callable[[], JsonDict], fallback: JsonDict, *, label: str) -> JsonDict:
    """Run a conversion, returning ``fallback`` unchanged if it raises."""
    try:
        return convert()
    except Exception as e:  # noqa: BLE001 - best-effort passthrough
        verbose_proxy_logger.debug("A2A %s conversion failed: %s", label, e)
        return fallback


def normalize_jsonrpc_response(content: JsonDict, target: A2AVersion, *, method: str) -> JsonDict:
    """Convert a JSON-RPC response's ``result`` to ``target``.

    Errors and non-dict results pass through untouched.
    """
    if content.get("error") is not None:
        return content
    result: Final = content.get("result")
    if not isinstance(result, dict):
        return content

    converted: Final = _convert_result(result, target, method=method, request_id=_as_request_id(content.get("id")))
    if converted is result:
        return content
    return {**content, "result": converted}


def normalize_stream_event(event: JsonDict, target: A2AVersion, *, request_id: RequestId) -> JsonDict:
    """Convert a single streamed JSON-RPC event's ``result`` to ``target``."""
    if event.get("error") is not None:
        return event
    result: Final = event.get("result")
    if not isinstance(result, dict):
        return event

    converted: Final = _convert_stream_result(result, target, request_id=request_id)
    if converted is result:
        return event
    return {**event, "result": converted}


def normalize_request_params(params: JsonDict, served: A2AVersion, *, method: str) -> JsonDict:
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


def _detect_card_version(card: JsonDict) -> A2AVersion:
    """Infer the wire version of an agent card dict.

    ``protocolVersion`` is the authoritative indicator; semver values normalize to
    their major.minor (``"0.3.0"`` -> ``"0.3"``). Fall back to presence of
    ``supportedInterfaces`` (a 1.0-only field) only when the explicit field is
    absent or unrecognized; cards carrying neither signal are treated as 0.3.
    """
    normalized: Final = normalize_protocol_version(card.get("protocolVersion"))
    if normalized is not None:
        return normalized
    return "1.0" if "supportedInterfaces" in card else "0.3"


def normalize_agent_card(card: JsonDict, target: A2AVersion) -> JsonDict:
    """Convert an extended agent card to ``target``.

    When lowering to 0.3, ``additionalInterfaces`` is stripped so the conversion never
    re-exposes upstream backend URLs that the LiteLLM-fronting merge deliberately drops.
    """
    if not isinstance(card, dict):
        return card

    current: Final = _detect_card_version(card)
    if current == target and not (target == "0.3" and "supportedInterfaces" in card):
        return _normalize_card_security(card, target)
    return _best_effort(lambda: _convert_agent_card(card, target), card, label="agent card")


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
    if method == "tasks/list":
        return _convert_list_tasks_result(result, target)
    return result


def _detect_send_version(result: JsonDict) -> A2AVersion | None:
    if "kind" in result:
        return "0.3"
    if result.keys() & _V1_SEND_ENVELOPE_KEYS:
        return "1.0"
    return None


def _convert_send_result(result: JsonDict, target: A2AVersion, *, request_id: RequestId) -> JsonDict:
    current: Final = _detect_send_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _send_result_to(result, target, request_id),
        result,
        label="send result",
    )


def _send_result_to(result: JsonDict, target: A2AVersion, request_id: RequestId) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_send_message_response,
        to_core_send_message_response,
        types_v03,
    )

    if target == "1.0":
        compat_result: Final = _validate_message_or_task(result)
        response: Final = types_v03.SendMessageResponse(
            root=types_v03.SendMessageSuccessResponse(
                id=str(request_id) if request_id is not None else "",
                result=compat_result,  # pyright: ignore[reportArgumentType]
            )
        )
        return MessageToDict(
            to_core_send_message_response(response),
            preserving_proto_field_name=False,
        )

    pb: Final = pb2_v10.SendMessageResponse()
    ParseDict(result, pb, ignore_unknown_fields=True)
    return _dump_03(to_compat_send_message_response(pb, request_id).root.result)


def _convert_task(result: JsonDict, target: A2AVersion) -> JsonDict:
    current: Final[A2AVersion] = "0.3" if "kind" in result else "1.0"
    if current == target:
        return result
    return _best_effort(lambda: _task_to(result, target), result, label="task")


def _detect_list_tasks_version(result: JsonDict) -> A2AVersion | None:
    tasks: Final = result.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None
    first: Final = tasks[0]
    if not isinstance(first, dict):
        return None
    return "0.3" if "kind" in first else "1.0"


def _convert_list_tasks_result(result: JsonDict, target: A2AVersion) -> JsonDict:
    current: Final = _detect_list_tasks_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _list_tasks_result_to(result, target),
        result,
        label="list tasks result",
    )


def _list_tasks_result_to(result: JsonDict, target: A2AVersion) -> JsonDict:
    tasks: Final = result.get("tasks")
    if not isinstance(tasks, list):
        return result
    return {
        **result,
        "tasks": [_task_to(item, target) if isinstance(item, dict) else item for item in tasks],
    }


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
        core: Final = to_core_task(types_v03.Task.model_validate(result))
        return MessageToDict(core, preserving_proto_field_name=False)

    pb: Final = pb2_v10.Task()
    ParseDict(result, pb, ignore_unknown_fields=True)
    return _dump_03(to_compat_task(pb))


def _detect_stream_version(result: JsonDict) -> A2AVersion | None:
    if "kind" in result:
        return "0.3"
    if result.keys() & _V1_STREAM_ENVELOPE_KEYS:
        return "1.0"
    return None


def _convert_stream_result(result: JsonDict, target: A2AVersion, *, request_id: RequestId) -> JsonDict:
    current: Final = _detect_stream_version(result)
    if current is None or current == target:
        return result
    return _best_effort(
        lambda: _stream_result_to(result, target, request_id),
        result,
        label="stream event",
    )


def _stream_result_to(result: JsonDict, target: A2AVersion, request_id: RequestId) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        to_compat_stream_response,
        to_core_stream_response,
        types_v03,
    )

    if target == "1.0":
        event: Final = _validate_stream_event(result)
        wrapper: Final = types_v03.SendStreamingMessageSuccessResponse(
            id=str(request_id) if request_id is not None else "",
            result=event,  # pyright: ignore[reportArgumentType]
        )
        return MessageToDict(to_core_stream_response(wrapper), preserving_proto_field_name=False)

    pb: Final = pb2_v10.StreamResponse()
    ParseDict(result, pb, ignore_unknown_fields=True)
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
        prepared_v1: Final = _normalize_card_security(card, "1.0")
        pb: Final = pb2_v10.AgentCard()
        ParseDict(prepared_v1, pb, ignore_unknown_fields=True)
        lowered: Final = _dump_03(to_compat_agent_card(pb))
        lowered.pop("additionalInterfaces", None)
        return lowered

    prepared_v03: Final = _normalize_card_security(card, "0.3")
    core: Final = to_core_agent_card(types_v03.AgentCard.model_validate(prepared_v03))
    return MessageToDict(core, preserving_proto_field_name=False)


def _validate_message_or_task(result: JsonDict) -> BaseModel:
    from a2a.compat.v0_3.conversions import types_v03

    if result.get("kind") == "task":
        return types_v03.Task.model_validate(result)
    return types_v03.Message.model_validate(result)


def _validate_stream_event(result: JsonDict) -> BaseModel:
    from a2a.compat.v0_3.conversions import types_v03

    kind: Final = result.get("kind")
    if kind == "task":
        return types_v03.Task.model_validate(result)
    if kind == "status-update":
        return types_v03.TaskStatusUpdateEvent.model_validate(result)
    if kind == "artifact-update":
        return types_v03.TaskArtifactUpdateEvent.model_validate(result)
    return types_v03.Message.model_validate(result)


def _lower_request_params(params: JsonDict, *, method: str) -> JsonDict:
    if method == "tasks/list":
        return _lower_list_tasks_params(params)

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

    lowerings: Final[dict[str, Callable[[JsonDict], BaseModel]]] = {
        "tasks/get": lambda p: to_compat_get_task_request(_parse(ParseDict, p, pb2_v10.GetTaskRequest()), "").params,
        "tasks/cancel": lambda p: (
            to_compat_cancel_task_request(_parse(ParseDict, p, pb2_v10.CancelTaskRequest()), "").params
        ),
        "tasks/resubscribe": lambda p: (
            to_compat_subscribe_to_task_request(_parse(ParseDict, p, pb2_v10.SubscribeToTaskRequest()), "").params
        ),
        "tasks/pushNotificationConfig/set": lambda p: (
            to_compat_create_task_push_notification_config_request(
                _parse(
                    ParseDict,
                    _flatten_create_push_notification_params(p),
                    pb2_v10.TaskPushNotificationConfig(),
                ),
                "",
            ).params
        ),
        "tasks/pushNotificationConfig/get": lambda p: (
            to_compat_get_task_push_notification_config_request(
                _parse(ParseDict, p, pb2_v10.GetTaskPushNotificationConfigRequest()), ""
            ).params
        ),
        "tasks/pushNotificationConfig/list": lambda p: (
            to_compat_list_task_push_notification_config_request(
                _parse(ParseDict, p, pb2_v10.ListTaskPushNotificationConfigsRequest()), ""
            ).params
        ),
        "tasks/pushNotificationConfig/delete": lambda p: (
            to_compat_delete_task_push_notification_config_request(
                _parse(ParseDict, p, pb2_v10.DeleteTaskPushNotificationConfigRequest()), ""
            ).params
        ),
    }
    lower: Final = lowerings.get(method)
    if lower is None:
        return params
    return _dump_03(lower(params))


def _lower_list_tasks_params(params: JsonDict) -> JsonDict:
    from a2a.compat.v0_3.conversions import (
        MessageToDict,
        ParseDict,
        pb2_v10,
        types_v03,
    )

    proto: Final = pb2_v10.ListTasksRequest()
    _parse(ParseDict, params, proto)
    lowered: Final = MessageToDict(proto, preserving_proto_field_name=False)
    status_name: Final = str(pb2_v10.TaskState.Name(proto.status))
    valid_0_3_values: Final = frozenset(str(member.value) for member in types_v03.TaskState)
    compat_status: Final = _proto_task_state_name_to_0_3(status_name, valid_0_3_values)
    if compat_status is None:
        lowered.pop("status", None)
    else:
        lowered["status"] = compat_status
    return lowered


def _proto_task_state_name_to_0_3(name: str, valid_0_3_values: frozenset[str]) -> str | None:
    """Map a 1.0 protobuf ``TaskState`` enum name to its 0.3 wire string.

    The ``TASK_STATE_<NAME>`` enum names line up with the 0.3 wire values once the
    prefix is dropped and underscores become dashes, so no private SDK mapping is
    needed. The result is validated against the 0.3 enum's own values; an unspecified
    or unrecognized state yields ``None`` so the status filter is dropped.
    """
    base: Final = name.removeprefix("TASK_STATE_")
    if base == "UNSPECIFIED":
        return None
    candidate: Final = base.lower().replace("_", "-")
    return candidate if candidate in valid_0_3_values else None


def _flatten_create_push_notification_params(params: JsonDict) -> JsonDict:
    """Merge 1.x create envelope fields (parent/configId/config) into flat pb fields."""
    flat: Final = dict(params)
    config: Final = flat.pop("config", None)
    push_config: Final = flat.pop("pushNotificationConfig", None)
    nested: Final = config if config is not None else push_config
    if not isinstance(nested, dict):
        return params
    parent: Final = flat.pop("parent", None)
    if isinstance(parent, str) and parent.startswith("tasks/") and "taskId" not in flat:
        flat["taskId"] = parent.removeprefix("tasks/").split("/")[0]
    if (config_id := flat.pop("configId", None)) and "id" not in nested:
        nested["id"] = config_id
    flat.update(nested)
    return flat


def _parse(parse_dict: Callable[..., object], data: JsonDict, message: object) -> object:
    parse_dict(data, message, ignore_unknown_fields=True)
    return message
