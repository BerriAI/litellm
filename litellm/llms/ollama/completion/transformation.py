import types
from typing import TYPE_CHECKING, Any, List, Optional, Union

from httpx._models import Headers, Response

import litellm
from litellm.llms.base_llm.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ModelInfo,
    ModelResponse,
    ProviderField,
    StreamingChoices,
)

from ..common_utils import OllamaError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OllamaConfig(BaseConfig):
    """
    Reference: https://github.com/ollama/ollama/blob/main/docs/api.md#parameters

    The class `OllamaConfig` provides the configuration for the Ollama's API interface. Below are the parameters:

    - `mirostat` (int): Enable Mirostat sampling for controlling perplexity. Default is 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0. Example usage: mirostat 0

    - `mirostat_eta` (float): Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. Default: 0.1. Example usage: mirostat_eta 0.1

    - `mirostat_tau` (float): Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. Default: 5.0. Example usage: mirostat_tau 5.0

    - `num_ctx` (int): Sets the size of the context window used to generate the next token. Default: 2048. Example usage: num_ctx 4096

    - `num_gqa` (int): The number of GQA groups in the transformer layer. Required for some models, for example it is 8 for llama2:70b. Example usage: num_gqa 1

    - `num_gpu` (int): The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable. Example usage: num_gpu 0

    - `num_thread` (int): Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance. It is recommended to set this value to the number of physical CPU cores your system has (as opposed to the logical number of cores). Example usage: num_thread 8

    - `repeat_last_n` (int): Sets how far back for the model to look back to prevent repetition. Default: 64, 0 = disabled, -1 = num_ctx. Example usage: repeat_last_n 64

    - `repeat_penalty` (float): Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. Default: 1.1. Example usage: repeat_penalty 1.1

    - `temperature` (float): The temperature of the model. Increasing the temperature will make the model answer more creatively. Default: 0.8. Example usage: temperature 0.7

    - `seed` (int): Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. Example usage: seed 42

    - `stop` (string[]): Sets the stop sequences to use. Example usage: stop "AI assistant:"

    - `tfs_z` (float): Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. Default: 1. Example usage: tfs_z 1

    - `num_predict` (int): Maximum number of tokens to predict when generating text. Default: 128, -1 = infinite generation, -2 = fill context. Example usage: num_predict 42

    - `top_k` (int): Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. Default: 40. Example usage: top_k 40

    - `top_p` (float): Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. Default: 0.9. Example usage: top_p 0.9

    - `system` (string): system prompt for model (overrides what is defined in the Modelfile)

    - `template` (string): the full prompt or prompt template (overrides what is defined in the Modelfile)
    """

    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    num_gqa: Optional[int] = None
    num_gpu: Optional[int] = None
    num_thread: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[list] = (
        None  # stop is a list based on this - https://github.com/ollama/ollama/pull/442
    )
    tfs_z: Optional[float] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    system: Optional[str] = None
    template: Optional[str] = None

    def __init__(
        self,
        mirostat: Optional[int] = None,
        mirostat_eta: Optional[float] = None,
        mirostat_tau: Optional[float] = None,
        num_ctx: Optional[int] = None,
        num_gqa: Optional[int] = None,
        num_gpu: Optional[int] = None,
        num_thread: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[list] = None,
        tfs_z: Optional[float] = None,
        num_predict: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="base_url",
                field_type="string",
                field_description="Your Ollama API Base",
                field_value="http://10.10.11.249:11434",
            )
        ]

    def get_supported_openai_params(self, model: str):
        return [
            "max_tokens",
            "stream",
            "top_p",
            "temperature",
            "seed",
            "frequency_penalty",
            "stop",
            "response_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["num_predict"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "seed":
                optional_params["seed"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "frequency_penalty":
                optional_params["repeat_penalty"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "response_format" and isinstance(value, dict):
                if value["type"] == "json_object":
                    optional_params["format"] = "json"

        return optional_params

    def _supports_function_calling(self, ollama_model_info: dict) -> bool:
        """
        Check if the 'template' field in the ollama_model_info contains a 'tools' or 'function' key.
        """
        _template: str = str(ollama_model_info.get("template", "") or "")
        return "tools" in _template.lower()

    def _get_max_tokens(self, ollama_model_info: dict) -> Optional[int]:
        _model_info: dict = ollama_model_info.get("model_info", {})

        for k, v in _model_info.items():
            if "context_length" in k:
                return v
        return None

    def get_model_info(self, model: str) -> ModelInfo:
        """
        curl http://localhost:11434/api/show -d '{
          "name": "mistral"
        }'
        """
        if model.startswith("ollama/") or model.startswith("ollama_chat/"):
            model = model.split("/", 1)[1]
        api_base = get_secret_str("OLLAMA_API_BASE") or "http://localhost:11434"

        try:
            response = litellm.module_level_client.post(
                url=f"{api_base}/api/show",
                json={"name": model},
            )
        except Exception as e:
            raise Exception(
                f"OllamaError: Error getting model info for {model}. Set Ollama API Base via `OLLAMA_API_BASE` environment variable. Error: {e}"
            )

        model_info = response.json()

        _max_tokens: Optional[int] = self._get_max_tokens(model_info)

        return ModelInfo(
            key=model,
            litellm_provider="ollama",
            mode="chat",
            supported_openai_params=self.get_supported_openai_params(model=model),
            supports_function_calling=self._supports_function_calling(model_info),
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            max_tokens=_max_tokens,
            max_input_tokens=_max_tokens,
            max_output_tokens=_max_tokens,
        )

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        return messages

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return OllamaError(
            status_code=status_code, message=error_message, headers=headers
        )

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: str,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        raise NotImplementedError("transformation currently done in handler.py")

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError("transformation currently done in handler.py")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        raise NotImplementedError("validation currently done in handler.py")
