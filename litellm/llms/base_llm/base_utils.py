from abc import ABC, abstractmethod

from litellm.types.utils import ModelInfoBase


class BaseLLMModelInfo(ABC):
    @abstractmethod
    def get_model_info(self, model: str) -> ModelInfoBase:
        pass
