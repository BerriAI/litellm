from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.models.fallback import STORED_FALLBACKS, FallbackRulesEdit, FallbackRulesEditor
from litellm.repositories.config_repository import ConfigRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

ROUTER_SETTINGS_PARAM: Final = "router_settings"
_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_JSON_RULES: Final = TypeAdapter(list[JsonValue])
_ROUTER_SETTINGS: Final = TypeAdapter(dict[str, JsonValue])


def _stored_router_settings(param_value: object) -> Mapping[str, JsonValue]:
    try:
        decoded: Final[object] = _JSON_VALUE.validate_json(param_value) if isinstance(param_value, str) else param_value
    except ValidationError:
        return {}
    if not isinstance(decoded, Mapping):
        return {}
    return _ROUTER_SETTINGS.validate_python(decoded)


def _edited_router_settings(
    param_value: object, fallback_key: str, edit: FallbackRulesEditor
) -> tuple[Mapping[str, JsonValue], FallbackRulesEdit]:
    router_settings: Final = _stored_router_settings(param_value)
    edited: Final = edit(STORED_FALLBACKS.validate_python(router_settings.get(fallback_key) or []))
    return {**router_settings, fallback_key: _JSON_RULES.validate_python(edited.rules)}, edited


class RouterFallbackRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._config: Final = ConfigRepository(prisma_client)

    async def edit_rules(self, fallback_key: str, edit: FallbackRulesEditor) -> FallbackRulesEdit:
        return await self._config.update_param_locked(
            ROUTER_SETTINGS_PARAM, lambda param_value: _edited_router_settings(param_value, fallback_key, edit)
        )
