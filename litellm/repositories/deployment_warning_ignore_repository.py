import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.models.deployment_warning import AdminIgnore
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # resolved only from the quoted base-class subscript below

DEPLOYMENT_WARNING_IGNORES_ROW_ID: Final = "deployment_warning_ignores"

_IGNORES_ADAPTER: Final[TypeAdapter[Mapping[str, AdminIgnore]]] = TypeAdapter(Mapping[str, AdminIgnore])


def parse_ignores(raw: object) -> Mapping[str, AdminIgnore]:
    if raw is None:
        return MappingProxyType({})
    try:
        return MappingProxyType(_IGNORES_ADAPTER.validate_python(json.loads(raw) if isinstance(raw, str) else raw))
    except (json.JSONDecodeError, ValidationError):
        verbose_proxy_logger.warning("Ignoring unreadable %s row", DEPLOYMENT_WARNING_IGNORES_ROW_ID)
        return MappingProxyType({})


class DeploymentWarningIgnoreRepository(PrismaTableRepository["prisma_models.LiteLLM_UISettings"]):
    table_name = "litellm_uisettings"

    async def get_ignores(self) -> Mapping[str, AdminIgnore]:
        db_record: Final = await self.table.find_unique(where={"id": DEPLOYMENT_WARNING_IGNORES_ROW_ID})
        return parse_ignores(db_record.ui_settings if db_record is not None else None)

    async def save_ignores(self, ignores: Mapping[str, AdminIgnore]) -> None:
        payload: Final = _IGNORES_ADAPTER.dump_json(dict(ignores)).decode()
        row: Final = {"id": DEPLOYMENT_WARNING_IGNORES_ROW_ID, "ui_settings": payload}
        await self.table.upsert(
            where={"id": DEPLOYMENT_WARNING_IGNORES_ROW_ID},
            data={"create": row, "update": {"ui_settings": payload}},
        )
