from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.mcp_server.events import EventOccurrence, JsonObject

_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class Screened:
    data: JsonObject


@dataclass(frozen=True, slots=True)
class Blocked:
    guardrail: str


ScreenOutcome: TypeAlias = Screened | Blocked


def _registered_guardrails() -> Sequence[object]:
    import litellm

    return tuple(litellm.callbacks)


def _screens_events(callback: object, request_data: Mapping[str, object]) -> bool:
    from litellm.proxy.utils import routes_through_apply_guardrail

    return (
        isinstance(callback, CustomGuardrail)
        and routes_through_apply_guardrail(callback)
        and callback.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.mcp_event) is True
    )


async def screen_event(
    occurrence: EventOccurrence,
    *,
    server_name: str,
    gateway_name: str,
    subscription_id: str | None,
    caller: UserAPIKeyAuth | None,
    callbacks: Callable[[], Sequence[object]] = _registered_guardrails,
) -> ScreenOutcome:
    from mcp.types import CallToolResult

    from litellm.proxy.mcp_server.guardrail_translation.handler import MCPGuardrailTranslationHandler

    request_data: Final[dict[str, object]] = {
        "mcp_event": {
            "server": server_name,
            "event_name": gateway_name,
            "subscription_id": subscription_id,
            "event_id": occurrence.event_id,
        },
        "metadata": dict(caller.metadata or {}) if caller is not None else {},
    }
    guardrails: Final = tuple(callback for callback in callbacks() if _screens_events(callback, request_data))
    if not guardrails:
        return Screened(occurrence.data)
    result: Final = CallToolResult(content=[], structured_content=dict(occurrence.data))
    for guardrail in guardrails:
        try:
            await MCPGuardrailTranslationHandler().process_output_response(
                response=result, guardrail_to_apply=guardrail, user_api_key_dict=caller, request_data=request_data
            )
        except Exception:  # noqa: BLE001  # any guardrail rejection drops the event; the subscription stays
            return Blocked(guardrail.guardrail_name or type(guardrail).__name__)
    try:
        return Screened(_JSON_OBJECT.validate_python(result.structured_content or {}))
    except ValidationError:
        return Blocked("payload")
