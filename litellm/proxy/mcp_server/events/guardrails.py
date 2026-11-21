from collections.abc import Awaitable, Callable, Mapping, Sequence
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
_RequestData: TypeAlias = dict[str, object]


def _registered_guardrails() -> Sequence[object]:
    import litellm

    return tuple(litellm.callbacks)


def _screens_events(callback: CustomGuardrail, request_data: Mapping[str, object]) -> bool:
    from litellm.proxy.utils import routes_through_apply_guardrail

    return routes_through_apply_guardrail(callback) and (
        callback.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.mcp_event) is True  # pyright: ignore[reportUnknownMemberType]  # legacy untyped data parameter
    )


def caller_metadata(caller: UserAPIKeyAuth) -> _RequestData:
    """The admin metadata and inherited tags a request from this caller carries, so opt-outs and tag modes apply."""
    from litellm.proxy.common_utils.callback_utils import strip_callback_config
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    return {
        "user_api_key_metadata": strip_callback_config(caller.metadata),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # UserAPIKeyAuth.metadata is a bare dict
        "user_api_key_team_metadata": strip_callback_config(caller.team_metadata),
        "tags": list(LiteLLMProxyRequestSetup.get_inherited_tags(caller)),
    }


async def _assigned_guardrails(request_data: _RequestData, caller: UserAPIKeyAuth) -> None:
    """The key, team and policy guardrails a request from this caller would carry, as the tool path assigns them."""
    from litellm.proxy.litellm_pre_call_utils import (
        move_guardrails_to_metadata,  # pyright: ignore[reportUnknownVariableType]  # legacy untyped request-data helper
    )

    await move_guardrails_to_metadata(data=request_data, _metadata_variable_name="metadata", user_api_key_dict=caller)


async def screen_event(
    occurrence: EventOccurrence,
    *,
    server_name: str,
    gateway_name: str,
    subscription_id: str | None,
    caller: UserAPIKeyAuth | None,
    callbacks: Callable[[], Sequence[object]] = _registered_guardrails,
    assign: Callable[[_RequestData, UserAPIKeyAuth], Awaitable[None]] = _assigned_guardrails,
) -> ScreenOutcome:
    from mcp.types import CallToolResult

    from litellm.proxy.mcp_server.guardrail_translation.handler import MCPGuardrailTranslationHandler
    from litellm.proxy.mcp_server.utils import mcp_tool_result_structured_content

    request_data: Final[_RequestData] = {
        "metadata": caller_metadata(caller) if caller is not None else {},
        "mcp_event": {
            "server": server_name,
            "event_name": gateway_name,
            "subscription_id": subscription_id,
            "event_id": occurrence.event_id,
        },
    }
    if caller is not None:
        try:
            await assign(request_data, caller)
        except Exception:  # noqa: BLE001  # an assigned guardrail that cannot be resolved must not be skipped
            return Blocked("guardrail-assignment")
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
