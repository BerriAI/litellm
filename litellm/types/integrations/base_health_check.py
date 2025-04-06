from typing import Literal, Optional, TypedDict


class IntegrationHealthCheckStatus(TypedDict):
    status: Literal["healthy", "unhealthy"]
    error_message: Optional[str]
