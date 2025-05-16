import base64
import datetime
from typing import Dict, List, Optional, Union

import httpx

import litellm
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
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
        return api_key or (get_secret_str("GEMINI_API_KEY"))

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model.replace("gemini/", "")

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(api_key)
        endpoint = f"/{self.api_version}/models"
        if api_base is None or api_key is None:
            raise ValueError(
                "GEMINI_API_BASE or GEMINI_API_KEY is not set. Please set the environment variable, to query Gemini's `/models` endpoint."
            )

        response = litellm.module_level_client.get(
            url=f"{api_base}{endpoint}?key={api_key}",
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Gemini. Status code: {response.status_code}, Response: {response.json()}"
            )

        models = response.json()["models"]

        litellm_model_names = []
        for model in models:
            stripped_model_name = model["name"].strip("models/")
            litellm_model_name = "gemini/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return GeminiError(
            status_code=status_code, message=error_message, headers=headers
        )


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
