"""Access checks for the models and vector stores a GPT-Live session delegates to."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    can_key_call_resolved_model,
    check_tools_allowlist,  # pyright: ignore[reportUnknownVariableType]  # its request_body is declared as a bare dict
    vector_store_access_check_for_caller,
)

_DELEGATION_KEY: Final = "delegation"
_JSON_ESCAPE: Final = "\\u"
_DELEGATED_TOOLS_ROUTE: Final = "/v1/responses"
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class LiveDelegation:
    models: tuple[str, ...]
    vector_store_ids: tuple[str, ...]


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        yield from filter(lambda item: isinstance(item, str), value)  # pyright: ignore[reportReturnType]  # filter keeps only str items


def _named(value: JsonValue, key: str) -> Iterator[str]:
    if isinstance(value, dict):
        for name, nested in value.items():
            yield from _strings(nested) if name == key else _named(nested, key)
    elif isinstance(value, list):
        for nested in value:
            yield from _named(nested, key)


def _tool_requests(value: JsonValue) -> Iterator[Mapping[str, JsonValue]]:
    if isinstance(value, dict):
        if "tools" in value:
            yield value
        for nested in value.values():
            yield from _tool_requests(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _tool_requests(nested)


def _delegated_tool_requests(session: object) -> tuple[Mapping[str, JsonValue], ...]:
    try:
        parsed: Final = _JSON.validate_python(session)
    except ValidationError:
        return ()
    return tuple(_tool_requests(parsed.get(_DELEGATION_KEY))) if isinstance(parsed, dict) else ()


def live_delegation(session: object) -> LiveDelegation:
    try:
        parsed: Final = _JSON.validate_python(session)
    except ValidationError:
        return LiveDelegation(models=(), vector_store_ids=())
    delegation: Final = parsed.get(_DELEGATION_KEY) if isinstance(parsed, dict) else None
    return LiveDelegation(
        models=tuple(dict.fromkeys(_named(delegation, "model"))),
        vector_store_ids=tuple(dict.fromkeys(_named(delegation, "vector_store_ids"))),
    )


def _proxy_model_list() -> Sequence[object] | None:
    from litellm.proxy import proxy_server

    return proxy_server.llm_model_list  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # proxy_server declares a bare list


async def authorize_live_delegation(session: object, caller: UserAPIKeyAuth) -> None:
    from litellm.proxy import proxy_server

    delegation: Final = live_delegation(session)
    for model in delegation.models:
        await can_key_call_resolved_model(
            model=model, valid_token=caller, llm_model_list=_proxy_model_list(), llm_router=proxy_server.llm_router
        )
    if delegation.vector_store_ids:
        await vector_store_access_check_for_caller(delegation.vector_store_ids, caller)
    for tool_request in _delegated_tool_requests(session):
        await check_tools_allowlist(
            request_body=dict(tool_request), valid_token=caller, team_object=None, route=_DELEGATED_TOOLS_ROUTE
        )


async def live_delegation_denial(client_frame: str, caller: UserAPIKeyAuth) -> str | None:
    if _DELEGATION_KEY not in client_frame and _JSON_ESCAPE not in client_frame:
        return None
    try:
        frame: Final = _JSON.validate_json(client_frame)
    except ValidationError:
        return None
    try:
        await authorize_live_delegation(frame.get("session") if isinstance(frame, dict) else None, caller)
    except ProxyException as error:
        return str(error.message)
    except HTTPException as error:
        return str(error.detail)
    return None
