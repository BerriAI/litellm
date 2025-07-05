from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class GuardrailConfigModel(BaseModel, Generic[T]):
    """Base model for guardrail configuration"""

    optional_params: T = Field(
        description="Optional parameters for the guardrail",
    )
