from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import OpenAISpeechOptionalParams
from litellm.types.utils import FileTypes, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseSpeechConfig(BaseConfig, ABC):
    @abstractmethod
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAISpeechOptionalParams]:
        pass

    @abstractmethod
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map the OpenAI params to the Whisper params
        """
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params
