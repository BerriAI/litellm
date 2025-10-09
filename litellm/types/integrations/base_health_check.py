from typing import Literal, Optional

from typing_extensions import TypedDict


class IntegrationHealthCheckStatus(TypedDict):
    status: Literal["healthy", "unhealthy"]
    error_message: Optional[str]
