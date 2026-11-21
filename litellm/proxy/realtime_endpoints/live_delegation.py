"""Access checks for the models and vector stores a GPT-Live session delegates to."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model, vector_store_access_check_for_caller

_DELEGATION_KEY: Final = "delegation"
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


async def live_delegation_denial(client_frame: str, caller: UserAPIKeyAuth) -> str | None:
    if _DELEGATION_KEY not in client_frame:
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
