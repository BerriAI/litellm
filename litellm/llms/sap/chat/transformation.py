"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration Service`v2/completion`
"""
from typing import List, Optional, Union, Dict, Tuple, Any, TYPE_CHECKING, Iterator, AsyncIterator
from functools import cached_property
import litellm
import httpx


from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

from ..credentials import get_token_creator
from .models import (
    SAPMessage,
    SAPAssistantMessage,
    SAPToolChatMessage,
    ChatCompletionTool,
    ResponseFormatJSONSchema,
    ResponseFormat,
    SAPUserMessage,
    GroundingModuleConfig,
    OrchestrationRequest
)
from .handler import GenAIHubOrchestrationError, AsyncSAPStreamIterator, SAPStreamIterator

def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump(by_alias=True)


class GenAIHubOrchestrationConfig(OpenAIGPTConfig):
    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None  #
    model_version: str = "latest"

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        self.token_creator = None
        self._base_url = None
        self._resource_group = None

    def run_env_setup(self, service_key: Optional[str] = None) -> None:
        try:
            self.token_creator, self._base_url, self._resource_group = get_token_creator(service_key) # type: ignore
        except ValueError as err:
            raise GenAIHubOrchestrationError(status_code=400, message=err.args[0])


    @property
    def headers(self) -> Dict[str, str]:
        if self.token_creator is None:
            self.run_env_setup()
        access_token = self.token_creator() # type: ignore
        return {
            "Authorization": access_token,
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
            "AI-Client-Type": "LiteLLM",
        }

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            self.run_env_setup()
        return self._base_url # type: ignore


    @property
    def resource_group(self) -> str:
        if self._resource_group is None:
            self.run_env_setup()
        return self._resource_group # type: ignore

    @cached_property
    def deployment_url(self) -> str:
        # Keep a short, tight client lifecycle here to avoid fd leaks
        client = litellm.module_level_client
        # with httpx.Client(timeout=30) as client:
        deployments = client.get(
            f"{self.base_url}/lm/deployments", headers=self.headers
        ).json()
        valid: List[Tuple[str, str]] = []
        for dep in deployments.get("resources", []):
            if dep.get("scenarioId") == "orchestration":
                cfg = client.get(
                    f'{self.base_url}/lm/configurations/{dep["configurationId"]}',
                    headers=self.headers,
                ).json()
                if cfg.get("executableId") == "orchestration":
                    valid.append((dep["deploymentUrl"], dep["createdAt"]))
            # newest first
        return sorted(valid, key=lambda x: x[1], reverse=True)[0][0]

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model):
        params = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "prediction",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "extra_headers",
            "parallel_tool_calls",
            "response_format",
            "timeout",
        ]
        if (
            model.startswith('anthropic')
            or model.startswith("amazon")
            or model.startswith("cohere")
            or model.startswith("alephalpha")
            or model == "gpt-4"
        ):
            params.remove("response_format")
        if model.startswith("gemini") or model.startswith("amazon"):
            params.remove("tool_choice")
        return params

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
        if api_key:
            self.run_env_setup(api_key)
        return self.headers

    def get_complete_url(
            self,
            api_base: Optional[str],
            api_key: Optional[str],
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: Optional[bool] = None,
    ):
        api_base_ = f"{self.deployment_url}/v2/completion"
        return api_base_

    def transform_request(
        self,
        model: str,
        messages: List[Dict[str, str]], # type: ignore
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        optional_params.pop("deployment_url", None)
        model_version = optional_params.pop("model_version", "latest")
        template = messages

        tools_ = optional_params.pop("tools", [])
        if tools_ != []:
            tools = {"tools": tools_}
        else:
            tools = {}

        response_format = optional_params.pop("response_format", {})
        resp_type = response_format.get("type", None)
        if resp_type:
            if resp_type== "json_schema":
                response_format = validate_dict(response_format, ResponseFormatJSONSchema)
            else:
                response_format = validate_dict(response_format, ResponseFormat)
            response_format = {"response_format": response_format}
        optional_params.pop("stream", False)
        stream_config = {}
        if "stream_options" in optional_params:
            stream_options = optional_params.pop("stream_options", {})
            stream_config["chunk_size"] = stream_options.get("chunk_size", 100)
            if "delimiters" in stream_options:
                stream_config["delimiters"] = stream_options.get("delimiters")

        placeholder_defaults = optional_params.pop("placeholder_defaults", {})
        if placeholder_defaults:
            placeholder_defaults = {"defaults": placeholder_defaults}

        placeholder_values = optional_params.pop("placeholder_values", {})
        if placeholder_values:
            placeholder_values = {"placeholder_values": placeholder_values}

        optional_modules = {}
        optional_modules_lst = ["grounding", "masking", "filtering", "translation"]
        for module in optional_modules_lst:
            if optional_params.get(module, None):
                optional_modules[module] = optional_params.pop(module)

        fallback_modules = optional_params.pop("fallback_modules", [])
        modules = [
                    {
                    "prompt_templating": {
                        "prompt": {
                            "template": template,
                            **placeholder_defaults,
                            **tools,
                            **response_format
                        },
                        "model": {
                            "name": model,
                            "params": optional_params,
                            "version": model_version,
                        },
                    },
                    **optional_modules
                }
                ]
        for modules_dict in fallback_modules:
            fallback_model = modules_dict.pop("model")
            fallback_model_version = modules_dict.pop("model_version", "latest")
            fallback_template = modules_dict.pop("messages", [])
            fallback_tools_ = modules_dict.pop("tools", [])
            if fallback_tools_ != []:
                fallback_tools = {"tools": fallback_tools_}
            else:
                fallback_tools = {}

            fallback_response_format = modules_dict.pop("response_format", {})
            fallback_resp_type = fallback_response_format.get("type", None)
            if fallback_resp_type:
                if fallback_resp_type == "json_schema":
                    fallback_response_format = validate_dict(response_format, ResponseFormatJSONSchema)
                else:
                    fallback_response_format = validate_dict(response_format, ResponseFormat)
                fallback_response_format = {"response_format": fallback_response_format}

            fallback_placeholder_defaults = modules_dict.pop("placeholder_defaults", {})
            if fallback_placeholder_defaults:
                fallback_placeholder_defaults = {"placeholder_defaults": fallback_placeholder_defaults}

            fallback_optional_modules = {}
            for module in optional_modules_lst:
                if modules_dict.get(module, None):
                    fallback_optional_modules[module] = modules_dict.pop(module)

            modules.append(
                {
                    "prompt_templating": {
                        "prompt": {
                            "template": fallback_template,
                            **fallback_placeholder_defaults,
                            **fallback_tools,
                            **fallback_response_format
                        },
                        "model": {
                            "name": fallback_model,
                            "params": modules_dict,
                            "version": fallback_model_version,
                        },
                    },
                    **fallback_optional_modules
                }
            )



        request_body = {
            "config": {
                "modules": modules,
                "stream": stream_config,
            },
            **placeholder_values,
        }

        validate_dict(request_body, OrchestrationRequest)
        print(request_body)

        return request_body

    def transform_response(
            self,
            model: str,
            raw_response: httpx.Response,
            model_response: ModelResponse,
            logging_obj: LiteLLMLoggingObj,
            request_data: dict,
            messages: List[AllMessageValues],
            optional_params: dict,
            litellm_params: dict,
            encoding: Any,
            api_key: Optional[str] = None,
            json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        return ModelResponse.model_validate(raw_response.json()["final_result"])

    def get_model_response_iterator(
            self,
            streaming_response: Union[Iterator[str], AsyncIterator[str], "ModelResponse"],
            sync_stream: bool,
            json_mode: Optional[bool] = False,
    ):
        if sync_stream:
            return SAPStreamIterator(response=streaming_response) # type: ignore
        else:
            return AsyncSAPStreamIterator(response=streaming_response) # type: ignore
