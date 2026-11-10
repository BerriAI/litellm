"""Types for gateway request counts (SGR), recorded at the ASGI edge."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import ConfigDict

from litellm.types.llms.base import LiteLLMBaseModel


@dataclass(frozen=True, slots=True)
class GatewayRequestKey:
    date: str
    category: str
    route: str
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class GatewayRequestCounts:
    successful_requests: int
    failed_requests: int

    def plus(self, *, succeeded: bool) -> "GatewayRequestCounts":
        return GatewayRequestCounts(
            successful_requests=self.successful_requests + (1 if succeeded else 0),
            failed_requests=self.failed_requests + (0 if succeeded else 1),
        )


GatewayRequestSnapshot: TypeAlias = Mapping[GatewayRequestKey, GatewayRequestCounts]


class GatewayRequestBreakdownEntry(LiteLLMBaseModel):
    category: str
    route: str
    successful_requests: int = 0
    failed_requests: int = 0


class GatewayRequestDailyEntry(LiteLLMBaseModel):
    date: str
    successful_requests: int = 0
    failed_requests: int = 0


class GatewayRequestStatusCodeEntry(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    status_code: int
    failed_requests: int = 0


class GatewayRequestActivityResponse(LiteLLMBaseModel):
    """Response for GET /gateway/daily/activity."""

    total_successful_requests: int = 0
    total_failed_requests: int = 0
    by_date: tuple[GatewayRequestDailyEntry, ...] = ()
    by_route: tuple[GatewayRequestBreakdownEntry, ...] = ()
    by_status_code: tuple[GatewayRequestStatusCodeEntry, ...] = ()
