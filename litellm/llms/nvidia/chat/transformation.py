"""
Nvidia NIM endpoint: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer 

This is OpenAI compatible 

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""

from typing import Optional, Union
import litellm
from litellm.secret_managers.main import get_secret, get_secret_str
import json
import requests
import os

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.nvidia.statics import determine_model, MODEL_TABLE

class NvidiaConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer

    The class `NvidiaNimConfig` provides configuration for the Nvidia NIM's Chat Completions API interface. Below are the parameters:
    """

    temperature: Optional[int] = None
    top_p: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, list]] = None

    def __init__(
        self,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        
        dynamic_api_key = get_secret_str("NVIDIA_API_KEY") or get_secret_str("NVIDIA_NIM_API_KEY")
        self.dynamic_api_key = dynamic_api_key

        litellm.nvidia_models = self.available_models()
        litellm.model_list += litellm.nvidia_models

    @classmethod
    def get_config(cls):
        return super().get_config()
    
    def available_models(self) -> list:
        '''Get Available NVIDIA models.'''
        api_base = (
                    get_secret("NVIDIA_API_BASE")  
                    or get_secret("NVIDIA_BASE_URL") 
                    or get_secret("NVIDIA_NIM_API_BASE") 
                    or "https://integrate.api.nvidia.com/v1" # type: ignore
                )

        headers = {
        'Content-Type': 'application/json',
        'Authorization': self.dynamic_api_key
        }
        try:
            response = requests.request("GET", os.path.join(api_base, "models"), headers=headers)
            response.raise_for_status()

            return [item["id"] for item in json.loads(response.text)["data"]]
        except Exception as e:
            raise e
        



    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model


        Updated on July 5th, 2024 - based on https://docs.api.nvidia.com/nim/reference
        ToDo: Update this to use the new Nvidia NIM API
        """
        parms = []
        if model_cls := determine_model(model):
            if model_cls.supports_tools:
                parms += ["tools", "tool_choice"]
            if model_cls.supports_structured_output:
                parms += ["response_format"]
        if model in [
            "google/recurrentgemma-2b",
            "google/gemma-2-27b-it",
            "google/gemma-2-9b-it",
            "gemma-2-9b-it",
        ]:
            parms += ["stream", "temperature", "top_p", "max_tokens", "stop", "seed"]
        elif model == "nvidia/nemotron-4-340b-instruct":
            parms += [
                "stream",
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
            ]
        elif model == "nvidia/nemotron-4-340b-reward":
            parms += [
                "stream",
            ]
        elif model in ["google/codegemma-1.1-7b"]:
            # most params - but no 'seed' :(
            parms += [
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
            ]
        else:
            parms += [
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
                "seed",
            ]

        return parms

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params
