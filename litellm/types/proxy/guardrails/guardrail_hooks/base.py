from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class GuardrailConfigModel(BaseModel, Generic[T], ABC):
    """Base model for guardrail configuration"""

    optional_params: T = Field(
        description="Optional parameters for the guardrail",
    )

    @staticmethod
    @abstractmethod
    def ui_friendly_name() -> str:
        """UI-friendly name for the guardrail"""
        pass
