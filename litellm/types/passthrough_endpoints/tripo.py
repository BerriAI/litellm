from decimal import Decimal
from typing import Final
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

TRIPO_TERMINAL_STATUSES: Final = frozenset({'success', 'failed', 'cancelled', 'canceled', 'banned', 'expired'})
TRIPO_POLLING_INTERVAL: Final = 10.0
TRIPO_MAX_POLLING_ATTEMPTS: Final = 180
TRIPO_USD_PER_CREDIT: Final = 0.01
TRIPO_TASK_FAMILIES: Final = frozenset({'generation', 'models', 'mesh', 'animations'})


class TripoTaskData(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str | None = None
    status: str | None = None
    credits_consumed: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)


class TripoResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: int
    data: TripoTaskData | None = None


def tripo_region_from_url(url: str) -> str | None:
    match urlsplit(url).hostname:
        case 'openapi.tripo3d.com':
            return 'cn'
        case 'openapi.tripo3d.ai':
            return 'global'
        case _:
            return None
