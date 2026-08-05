"""
Translate from OpenAI's `/v1/chat/completions` to Neuralwatt's `/v1/chat/completions`
"""

from typing import TYPE_CHECKING, Any, Optional

import httpx
from pydantic import BaseModel, ValidationError

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai_like.chat.transformation import OpenAILikeChatConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class NeuralwattEnergyUsage(BaseModel):
    energy_joules: Optional[float] = None
    energy_kwh: Optional[float] = None
    duration_seconds: Optional[float] = None
    measurement_available: Optional[bool] = None
    attribution_method: Optional[str] = None


class _NeuralwattChatResponseEnvelope(BaseModel):
    energy: Optional[NeuralwattEnergyUsage] = None


class NeuralwattChatConfig(OpenAILikeChatConfig):
    """
    Neuralwatt is OpenAI-compatible with standard endpoints
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "neuralwatt"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        passed_api_base = api_base
        api_base = api_base or get_secret_str("NEURALWATT_API_BASE") or "https://api.neuralwatt.com/v1"  # type: ignore
        dynamic_api_key = api_key
        if passed_api_base is None or api_key:
            dynamic_api_key = api_key or get_secret_str("NEURALWATT_API_KEY")
        return api_base, dynamic_api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        model_response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        energy = _parse_energy(raw_response)
        if energy is not None:
            model_response._hidden_params["energy"] = energy.model_dump(exclude_none=True)
        return model_response


def _parse_energy(raw_response: httpx.Response) -> Optional[NeuralwattEnergyUsage]:
    try:
        return _NeuralwattChatResponseEnvelope.model_validate(raw_response.json()).energy
    except (ValueError, ValidationError):
        return None
