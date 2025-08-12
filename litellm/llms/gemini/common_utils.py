import base64
import datetime
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class GeminiError(BaseLLMException):
    pass


class GeminiModelInfo(BaseLLMModelInfo):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Google AI Studio sends api key in query params"""
        return headers

    @property
    def api_version(self) -> str:
        return "v1beta"

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or get_secret_str("GEMINI_API_BASE")
            or "https://generativelanguage.googleapis.com"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or (get_secret_str("GOOGLE_API_KEY")) or (get_secret_str("GEMINI_API_KEY"))

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model.replace("gemini/", "")

    def process_model_name(self, models: List[Dict[str, str]]) -> List[str]:
        litellm_model_names = []
        for model in models:
            stripped_model_name = model["name"].replace("models/", "")
            litellm_model_name = "gemini/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(api_key)
        endpoint = f"/{self.api_version}/models"
        if api_base is None or api_key is None:
            raise ValueError(
                "GEMINI_API_BASE or GEMINI_API_KEY/GOOGLE_API_KEY is not set. Please set the environment variable, to query Gemini's `/models` endpoint."
            )

        response = litellm.module_level_client.get(
            url=f"{api_base}{endpoint}?key={api_key}",
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Gemini. Status code: {response.status_code}, Response: {response.json()}"
            )

        models = response.json()["models"]

        litellm_model_names = self.process_model_name(models)
        return litellm_model_names

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return GeminiError(
            status_code=status_code, message=error_message, headers=headers
        )
    
    def get_token_counter(self) -> Optional[BaseTokenCounter]:
        """
        Factory method to create a token counter for this provider.
        
        Returns:
            Optional TokenCounterInterface implementation for this provider,
            or None if token counting is not supported.
        """
        return GoogleAIStudioTokenCounter()


def encode_unserializable_types(
    data: Dict[str, object], depth: int = 0
) -> Dict[str, object]:
    """Converts unserializable types in dict to json.dumps() compatible types.

    This function is called in models.py after calling convert_to_dict(). The
    convert_to_dict() can convert pydantic object to dict. However, the input to
    convert_to_dict() is dict mixed of pydantic object and nested dict(the output
    of converters). So they may be bytes in the dict and they are out of
    `ser_json_bytes` control in model_dump(mode='json') called in
    `convert_to_dict`, as well as datetime deserialization in Pydantic json mode.

    Returns:
      A dictionary with json.dumps() incompatible type (e.g. bytes datetime)
      to compatible type (e.g. base64 encoded string, isoformat date string).
    """
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return data
    processed_data: dict[str, object] = {}
    if not isinstance(data, dict):
        return data
    for key, value in data.items():
        if isinstance(value, bytes):
            processed_data[key] = base64.urlsafe_b64encode(value).decode("ascii")
        elif isinstance(value, datetime.datetime):
            processed_data[key] = value.isoformat()
        elif isinstance(value, dict):
            processed_data[key] = encode_unserializable_types(value, depth + 1)
        elif isinstance(value, list):
            if all(isinstance(v, bytes) for v in value):
                processed_data[key] = [
                    base64.urlsafe_b64encode(v).decode("ascii") for v in value
                ]
            if all(isinstance(v, datetime.datetime) for v in value):
                processed_data[key] = [v.isoformat() for v in value]
            else:
                processed_data[key] = [
                    encode_unserializable_types(v, depth + 1) for v in value
                ]
        else:
            processed_data[key] = value
    return processed_data


def get_api_key_from_env() -> Optional[str]:
    return get_secret_str("GOOGLE_API_KEY") or get_secret_str("GEMINI_API_KEY")


class GoogleAIStudioTokenCounter(BaseTokenCounter):
    """Token counter implementation for Anthropic provider."""
    
    def supports_provider(
        self, 
        deployment: Optional[Dict[str, Any]] = None,
        from_endpoint: bool = False
    ) -> bool:
        if not from_endpoint:
            return False
            
        if deployment is None:
            return False
            
        full_model = deployment.get("litellm_params", {}).get("model", "")
        is_gemini_provider = full_model.startswith("gemini/") or "gemini" in full_model.lower()
        
        return is_gemini_provider
    
    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
    ) -> Optional[Dict[str, Any]]:
        from litellm.llms.gemini.count_tokens.handler import GoogleAIStudioTokenCounter
        deployment = deployment or {}
        deployment_litellm_params = deployment.get("litellm_params", {})
        count_tokens_params = {
            "model": model_to_use,
            "contents": contents,
        }
        count_tokens_params.update(deployment_litellm_params)
        result = await GoogleAIStudioTokenCounter().acount_tokens(
            **count_tokens_params,
        )
        
        if result is not None:
            return {
                "total_tokens": result["total_tokens"],
                "request_model": request_model,
                "model_used": model_to_use,
                "tokenizer_type": result["tokenizer_used"],
            }
        
        return None