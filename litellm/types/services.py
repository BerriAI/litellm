import uuid, enum
from pydantic import BaseModel, Field
from typing import Optional


class ServiceTypes(str, enum.Enum):
    """
    Enum for litellm + litellm-adjacent services (redis/postgres/etc.)
    """

    REDIS = "redis"
    DB = "postgres"
    LITELLM = "self"


class ServiceLoggerPayload(BaseModel):
    """
    The payload logged during service success/failure
    """

    is_error: bool = Field(description="did an error occur")
    error: Optional[str] = Field(None, description="what was the error")
    service: ServiceTypes = Field(description="who is this for? - postgres/redis")
    duration: float = Field(description="How long did the request take?")
    call_type: str = Field(description="The call of the service, being made")

    def to_json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception as e:
            # if using pydantic v1
            return self.dict(**kwargs)
