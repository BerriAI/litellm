from dataclasses import dataclass
from typing import Final

from opentelemetry.trace import Span

from litellm._auth.failure import NativeAuthFailure
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_BudgetTable
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, get_management_object_ttl
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.budget_repository import BudgetRepository


@dataclass(frozen=True, slots=True)
class EndUserIO:
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    span: Span | None = None
    logging: ProxyLogging | None = None
    route: str = ""


async def read_default_budget(budget_id: str, io: EndUserIO) -> LiteLLM_BudgetTable | None:
    from litellm.proxy.auth import auth_checks

    assert io.prisma is not None
    key: Final = f"default_end_user_budget:{budget_id}"
    cached: Final = await io.cache.async_get_cache(key=key, model_type=LiteLLM_BudgetTable)
    if cached is not None:
        return cached
    try:
        table: Final = auth_checks._dictable_table(BudgetRepository(io.prisma), "budget")  # pyright: ignore[reportPrivateUsage]  # shared raw repository projection
        record: Final = await table.find_unique(where={"budget_id": budget_id})
        if record is None:
            verbose_proxy_logger.warning(
                "Default end user budget not found in database: %s", budget_id.replace("\r", "").replace("\n", "")
            )
            return None
        row: Final = LiteLLM_BudgetTable.model_validate(record.dict())
        await io.cache.async_set_cache(
            key=key, value=row, model_type=LiteLLM_BudgetTable, ttl=get_management_object_ttl(io.cache)
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # preserve the default budget DB/cache IO failure contract
        verbose_proxy_logger.error("Error fetching default end user budget: %s", error)
        return None

    else:
        return row
