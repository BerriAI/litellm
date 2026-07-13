"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration Service`v2/completion`
"""

from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import httpx
import litellm

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

from ..credentials import get_token_creator
from .handler import (
    AsyncSAPStreamIterator,
    GenAIHubOrchestrationError,
    SAPStreamIterator,
)
from .models import (
    ChatCompletionTool,
    OrchestrationRequest,
    ResponseFormat,
    ResponseFormatJSONSchema,
    SAPAssistantMessage,
    SAPMessage,
    SAPToolChatMessage,
    SAPUserMessage,
)

_SAP_VENDOR_PREFIXES = (
    "anthropic--",
    "amazon--",
    "cohere--",
    "mistralai--",
    "nvidia--",
    "alephalpha--",
)


def _vendor(model: str) -> str:
    for prefix in _SAP_VENDOR_PREFIXES:
        if model.startswith(prefix):
            return prefix[:-2]  # strip trailing "--"
    return ""


def _canonical_model_name(model: str) -> str:
    prefix = _vendor(model)
    return model[len(prefix) + 2 :] if prefix else model


def _sap_supports_reasoning_effort(model: str) -> bool:
    vendor = _vendor(model)
    if vendor == "anthropic":
        canonical = _canonical_model_name(model)
        return canonical.startswith("claude-4") or canonical.startswith("claude-3-7")
    if vendor:
        return False
    # no vendor prefix: bare model names from Azure OpenAI / OpenAI on SAP
    # o-series (o1, o3, o4-mini, …) and gpt-5* family both support reasoning_effort
    canonical = model
    if len(canonical) > 1 and canonical[0] == "o" and canonical[1].isdigit():
        return True
    return canonical == "gpt-5" or canonical.startswith("gpt-5-") or canonical.startswith("gpt-5.")


def _sap_supports_thinking(model: str) -> bool:
    vendor = _vendor(model)
    if vendor == "cohere":
        return "reasoning" in _canonical_model_name(model)
    return _sap_supports_reasoning_effort(model)


# Keys routed outside SAP orchestration `model.params` (prompt, stream, fallbacks, etc.)
_SAP_MODEL_PARAMS_EXCLUDED_KEYS: FrozenSet[str] = frozenset(
    {
        "tools",
        "tool_choice",
        "stream_options",
        "fallback_sap_modules",
        "placeholder_values",
        "model_version",
    }
)


def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump(by_alias=True, exclude_unset=True)


def _messages_to_sap_template(messages: List[Dict[str, str]]) -> list:  # type: ignore[type-arg]
    template = []
    for message in messages:
        if message["role"] == "user":
            template.append(validate_dict(message, SAPUserMessage))
        elif message["role"] == "assistant":
            template.append(validate_dict(message, SAPAssistantMessage))
        elif message["role"] == "tool":
            template.append(validate_dict(message, SAPToolChatMessage))
        else:
            template.append(validate_dict(message, SAPMessage))
    return template


def _tools_response_format_and_stream(optional_params: dict, model_params: dict) -> Tuple[dict, dict, dict]:
    tools_ = optional_params.pop("tools", [])
    tools_ = [validate_dict(tool, ChatCompletionTool) for tool in tools_]
    tools: dict = {"tools": tools_} if tools_ else {}

    response_format = model_params.pop("response_format", {})
    resp_type = response_format.get("type", None)
    if resp_type:
        if resp_type == "json_schema":
            response_format = validate_dict(response_format, ResponseFormatJSONSchema)
        else:
            response_format = validate_dict(response_format, ResponseFormat)
        response_format = {"response_format": response_format}

    model_params.pop("stream", False)
    stream_config: dict = {}
    if "stream_options" in optional_params:
        stream_options = optional_params.pop("stream_options", {})
        if "chunk_size" in stream_options:
            stream_config["chunk_size"] = stream_options.get("chunk_size")
        if "delimiters" in stream_options:
            stream_config["delimiters"] = stream_options.get("delimiters")

    return tools, response_format, stream_config


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
    reasoning_effort: str | dict | None = (
        None  # pep604 style intentional: ruff-strict-budget forbids new Optional[...] violations
    )
    thinking: dict | None = None  # same

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
        access_token = self.token_creator()  # pyright: ignore[reportOptionalCall]  # run_env_setup set it or raised
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
        deployments = client.get(f"{self.base_url}/lm/deployments", headers=self.headers).json()
        valid: List[Tuple[str, str]] = []
        for dep in deployments.get("resources", []):
            if dep.get("scenarioId") == "orchestration":
                cfg = client.get(
                    f"{self.base_url}/lm/configurations/{dep['configurationId']}",
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
        if _sap_supports_reasoning_effort(model):
            params.append("reasoning_effort")
        if _sap_supports_thinking(model):
            params.append("thinking")
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
                response_format = validate_dict(response_format, ResponseFormatJSONSchema)
            else:
                response_format = validate_dict(response_format, ResponseFormat)
            response_format = {"response_format": response_format}
        else:
            response_format = {}

        placeholder_defaults = params.pop("placeholder_defaults", {})
        placeholder_defaults = {"defaults": placeholder_defaults} if placeholder_defaults else {}

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

        template = _messages_to_sap_template(messages)

        placeholder_values = optional_params.pop("placeholder_values", None)
        fallback_modules = optional_params.pop("fallback_sap_modules", [])

        optional_params.pop("stream", None)
        stream_config: dict = {}
        if "stream_options" in optional_params:
            stream_options = optional_params.pop("stream_options", {})
            if "chunk_size" in stream_options:
                stream_config["chunk_size"] = stream_options["chunk_size"]
            if "delimiters" in stream_options:
                stream_config["delimiters"] = stream_options["delimiters"]

        optional_params.pop("tool_choice", None)

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
                raise ValueError("Each entry in `fallback_sap_modules` must include a 'model' key.")
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

        config_payload: Dict[str, Any] = {
            "modules": modules if len(modules) > 1 else modules[0],
        }
        if stream_config:
            config_payload["stream"] = stream_config

        request_body: Dict[str, Any] = {"config": config_payload}
        if placeholder_values is not None:
            request_body["placeholder_values"] = placeholder_values

        body = validate_dict(request_body, OrchestrationRequest)

        return body

    @staticmethod
    def _normalize_gemini_thinking(raw: dict) -> dict:
        """Convert list-shaped reasoning_content from Gemini to thinking_blocks.

        AI Core forwards Gemini's thinking tokens as:
          message.reasoning_content = [{"thought": "...", "signature": "..."}]

        ModelResponse.reasoning_content is typed Optional[str], so model_validate
        hard-fails on a list. Map to thinking_blocks (already typed for this shape)
        and set reasoning_content to the concatenated thought text so callers that
        only read the string field still get something useful.
        """
        for choice in raw.get("choices", []):
            msg = choice.get("message", {})
            rc = msg.get("reasoning_content")
            if not isinstance(rc, list):
                continue
            thinking_blocks = [
                {
                    "type": "thinking",
                    "thinking": item.get("thought", ""),
                    "signature": item.get("signature"),
                }
                for item in rc
                if isinstance(item, dict)
            ]
            msg["thinking_blocks"] = thinking_blocks
            msg["reasoning_content"] = "\n".join(b["thinking"] for b in thinking_blocks if b["thinking"]) or None
        return raw

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
        final_result = self._normalize_gemini_thinking(raw_response.json()["final_result"])
        response = ModelResponse.model_validate(final_result)

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
