"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration Service`v2/completion`
"""
from typing import (
    List,
    Optional,
    Union,
    Dict,
    Tuple,
    Any,
    TYPE_CHECKING,
    Iterator,
    AsyncIterator,
)
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
    ResponseFormatJSONSchema,
    ResponseFormat,
    OrchestrationRequest,
    ChatCompletionTool,
)
from .handler import (
    GenAIHubOrchestrationError,
    AsyncSAPStreamIterator,
    SAPStreamIterator,
)


def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump(by_alias=True, exclude_unset=True)


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
            self.token_creator, self._base_url, self._resource_group = get_token_creator(service_key)  # type: ignore
        except ValueError as err:
            raise GenAIHubOrchestrationError(status_code=400, message=err.args[0])

    @property
    def headers(self) -> Dict[str, str]:
        if self.token_creator is None:
            self.run_env_setup()
        access_token = self.token_creator()  # type: ignore
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
        return self._base_url  # type: ignore

    @property
    def resource_group(self) -> str:
        if self._resource_group is None:
            self.run_env_setup()
        return self._resource_group  # type: ignore

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
        # Remove response_format for providers that don't support it on SAP GenAI Hub
        if (
            model.startswith("amazon")
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

    def _build_prompt_module(
        self,
        model_name: str,
        template_messages: List[Dict[str, str]],
        params: dict,
    ) -> dict:
        # Filter strict for GPT models only - SAP AI Core doesn't accept it as a model param
        # LangChain agents pass strict=true at top level, which fails for GPT models
        # Anthropic models accept strict, so preserve it for them
        if model_name.startswith("gpt") and "strict" in params:
            params.pop("strict")

        model_version = params.pop("model_version", "latest")

        tools_ = params.pop("tools", [])
        tools_ = [validate_dict(tool, ChatCompletionTool) for tool in tools_]
        tools = {"tools": tools_} if tools_ else {}

        response_format = params.pop("response_format", {})
        resp_type = response_format.get("type", None)
        if resp_type:
            if resp_type == "json_schema":
                response_format = validate_dict(
                    response_format, ResponseFormatJSONSchema
                )
            else:
                response_format = validate_dict(response_format, ResponseFormat)
            response_format = {"response_format": response_format}
        else:
            response_format = {}

        placeholder_defaults = params.pop("placeholder_defaults", {})
        placeholder_defaults = (
            {"defaults": placeholder_defaults} if placeholder_defaults else {}
        )

        optional_modules = {}
        optional_modules_lst = ["grounding", "masking", "filtering", "translation"]
        for module in optional_modules_lst:
            if params.get(module, None) is not None:
                optional_modules[module] = params.pop(module)

        return {
            "prompt_templating": {
                "prompt": {
                    "template": template_messages,
                    **placeholder_defaults,
                    **tools,
                    **response_format,
                },
                "model": {
                    "name": model_name,
                    "params": params,
                    "version": model_version,
                },
            },
            **optional_modules,
        }

    def transform_request(
        self,
        model: str,
        messages: List[Dict[str, str]],  # type: ignore
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        optional_params = dict(optional_params)
        optional_params.pop("deployment_url", None)

        template = messages

        optional_params.pop("stream", False)
        stream_config = {}
        if "stream_options" in optional_params:
            stream_options = optional_params.pop("stream_options", {})
            if "chunk_size" in stream_options:
                stream_config["chunk_size"] = stream_options.get("chunk_size")
            if "delimiters" in stream_options:
                stream_config["delimiters"] = stream_options.get("delimiters")

        placeholder_values = optional_params.pop("placeholder_values", None)
        placeholder_values = (
            {"placeholder_values": placeholder_values}
            if placeholder_values is not None
            else {}
        )

        fallback_modules = optional_params.pop("fallback_sap_modules", [])

        modules = [
            self._build_prompt_module(
                model_name=model,
                template_messages=template,
                params=dict(optional_params),
            )
        ]

        for modules_dict in fallback_modules:
            modules_dict = dict(modules_dict)
            fallback_model = modules_dict.pop("model", None)
            if fallback_model is None:
                raise ValueError(
                    "Each entry in `fallback_sap_modules` must include a 'model' key."
                )
            if fallback_model.startswith("sap/"):
                fallback_model = fallback_model[4:]
            fallback_template = modules_dict.pop("messages", [])

            modules.append(
                self._build_prompt_module(
                    model_name=fallback_model,
                    template_messages=fallback_template,
                    params=modules_dict,
                )
            )

        if fallback_modules:
            modules_payload = modules
        else:
            modules_payload = modules[0]  # type: ignore

        request_body = {
            "config": {
                "modules": modules_payload,
                **({"stream": stream_config} if stream_config else {}),
            },
            **placeholder_values,
        }

        body = validate_dict(request_body, OrchestrationRequest)

        return body

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
        response = ModelResponse.model_validate(raw_response.json()["final_result"])

        # Strip markdown code blocks if JSON response_format was used with Anthropic models
        # SAP GenAI Hub with Anthropic models sometimes wraps JSON in ```json ... ```
        # based on prompt phrasing. GPT/Gemini models don't exhibit this behavior,
        # so we gate the stripping to avoid accidentally modifying valid responses.
        response_format = optional_params.get("response_format", {})
        if response_format.get("type") in ("json_object", "json_schema"):
            if model.startswith("anthropic"):
                response = self._strip_markdown_json(response)

        return response

    def _strip_markdown_json(self, response: ModelResponse) -> ModelResponse:
        """Strip markdown code block wrapper from JSON content if present.

        SAP GenAI Hub with Anthropic models sometimes returns JSON wrapped in
        markdown code blocks (```json ... ```) depending on prompt phrasing.
        This method strips that wrapper to ensure consistent JSON output.
        """
        import re

        for choice in response.choices or []:
            if choice.message and choice.message.content:
                content = choice.message.content.strip()
                # Match ```json ... ``` or ``` ... ```
                match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", content, re.DOTALL)
                if match:
                    choice.message.content = match.group(1).strip()

        return response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], "ModelResponse"],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        if sync_stream:
            return SAPStreamIterator(response=streaming_response)  # type: ignore
        else:
            return AsyncSAPStreamIterator(response=streaming_response)  # type: ignore
