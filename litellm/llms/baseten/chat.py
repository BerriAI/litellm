from typing import Final

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str

BASETEN_MODEL_API_BASE: Final = "https://inference.baseten.co/v1"


class BasetenConfig(OpenAIGPTConfig):
    """
    Reference: https://inference.baseten.co/v1

    Below are the parameters:
    """

    max_tokens: int | None = None
    response_format: dict | None = None
    seed: int | None = None
    stream: bool | None = None
    top_p: int | None = None
    tool_choice: str | None = None
    tools: list | None = None
    user: str | None = None
    presence_penalty: int | None = None
    frequency_penalty: int | None = None
    stream_options: dict | None = None

    def __init__(
        self,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        seed: int | None = None,
        stop: list | None = None,
        stream: bool | None = None,
        temperature: float | None = None,
        top_p: int | None = None,
        tool_choice: str | None = None,
        tools: list | None = None,
        user: str | None = None,
        presence_penalty: int | None = None,
        frequency_penalty: int | None = None,
        stream_options: dict | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model
        """
        return [
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "tool_choice",
            "tools",
            "user",
            "presence_penalty",
            "frequency_penalty",
            "stream_options",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(self, api_base: str, api_key: str) -> tuple:
        """
        Get the OpenAI compatible provider info for Baseten
        """
        # Default to Model API
        default_api_base: Final = BASETEN_MODEL_API_BASE
        default_api_key: Final = api_key or "BASETEN_API_KEY"

        return default_api_base, default_api_key

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("BASETEN_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("BASETEN_API_BASE") or BASETEN_MODEL_API_BASE

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        """
        Calls Baseten's Model API `/v1/models` endpoint and returns the live catalog.
        """
        resolved_api_key: Final = self.get_api_key(api_key)
        if resolved_api_key is None:
            raise ValueError(
                "BASETEN_API_KEY is not set. Please set the environment variable, to query Baseten's `/models` endpoint."
            )
        resolved_api_base: Final = (self.get_api_base(api_base) or BASETEN_MODEL_API_BASE).rstrip("/")
        response: Final = litellm.module_level_client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped params
            url=f"{resolved_api_base.removesuffix('/v1')}/v1/models",
            headers={"Authorization": f"Bearer {resolved_api_key}"},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Baseten. Status code: {response.status_code}, Response: {response.text}"
            )
        return ["baseten/" + model["id"] for model in response.json()["data"]]

    @staticmethod
    def is_dedicated_deployment(model: str) -> bool:
        """
        Check if the model is a dedicated deployment (8-digit alphanumeric code)
        """
        # Remove 'baseten/' prefix if present
        model_id: Final = model.replace("baseten/", "")

        # Check if it's an 8-digit alphanumeric code
        import re

        return bool(re.match(r"^[a-zA-Z0-9]{8}$", model_id))

    @staticmethod
    def get_api_base_for_model(model: str) -> str:
        """
        Get the appropriate API base URL for the given model
        """
        if BasetenConfig.is_dedicated_deployment(model):
            # Extract the model ID (remove 'baseten/' prefix if present)
            model_id: Final = model.replace("baseten/", "")
            return f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
        else:
            # Use Model API
            return BASETEN_MODEL_API_BASE
