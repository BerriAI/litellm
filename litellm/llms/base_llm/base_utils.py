from abc import ABC, abstractmethod
from typing import List, Optional

from litellm.types.utils import ModelInfoBase


class BaseLLMModelInfo(ABC):
    @abstractmethod
    def get_model_info(
        self,
        model: str,
        existing_model_info: Optional[ModelInfoBase] = None,
    ) -> Optional[ModelInfoBase]:
        pass

    @abstractmethod
    def get_models(self) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        pass
