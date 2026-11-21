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


def _screens_events(callback: CustomGuardrail, request_data: Mapping[str, object]) -> bool:
    from litellm.proxy.utils import routes_through_apply_guardrail

    return routes_through_apply_guardrail(callback) and (
        callback.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.mcp_event) is True  # pyright: ignore[reportUnknownMemberType]  # legacy untyped data parameter
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
    from litellm.proxy.mcp_server.utils import mcp_tool_result_structured_content

    request_data: Final[Mapping[str, object]] = {
        "mcp_event": {
            "server": server_name,
            "event_name": gateway_name,
            "subscription_id": subscription_id,
            "event_id": occurrence.event_id,
        },
    }
    guardrails: Final = tuple(
        callback for callback in _guardrails(callbacks()) if _screens_events(callback, request_data)
    )
    if not guardrails:
        return Screened(occurrence.data)
    result: Final = CallToolResult(content=[], structured_content=dict(occurrence.data))
    for guardrail in guardrails:
        try:
            await MCPGuardrailTranslationHandler().process_output_response(  # pyright: ignore[reportUnknownMemberType]  # legacy untyped request_data parameter
                response=result, guardrail_to_apply=guardrail, user_api_key_dict=caller, request_data=dict(request_data)
            )
        except Exception:  # noqa: BLE001  # any guardrail rejection drops the event; the subscription stays
            return Blocked(guardrail.guardrail_name or type(guardrail).__name__)
    try:
        return Screened(_JSON_OBJECT.validate_python(mcp_tool_result_structured_content(result) or {}))
    except ValidationError:
        return Blocked("payload")


def _guardrails(callbacks: Sequence[object]) -> tuple[CustomGuardrail, ...]:
    return tuple(callback for callback in callbacks if isinstance(callback, CustomGuardrail))
