from typing import Literal

from typing_extensions import TypedDict


class IntegrationHealthCheckStatus(TypedDict):
    status: Literal["healthy", "unhealthy"]
    error_message: str | None
