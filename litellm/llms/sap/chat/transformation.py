"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration Service`v2/completion`
"""

import re
from collections.abc import AsyncIterator, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any, Final, Union

import httpx

import litellm
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    import tiktoken

    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
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

# Keys routed outside SAP orchestration `model.params` (prompt, stream, fallbacks, etc.)
_SAP_MODEL_PARAMS_EXCLUDED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "tools",
        "stream_options",
        "fallback_sap_modules",
        "placeholder_values",
        "model_version",
        "timeout",
        "max_retries",
    }
)

# ---------------------------------------------------------------------------
# SAP capability registry
# ---------------------------------------------------------------------------
# Models that accept reasoning_effort / thinking parameters on SAP GenAI Hub.
_REASONING_MODELS: re.Pattern[str] = re.compile(  # rebind-ok: module-level pattern constant; assignment is a one-time initialisation
    r"^(?:anthropic--claude-(?:4(?:\.[5-9])?|3-7)|o\d|gpt-5(?:[.\-]|$)|cohere--\S*reasoning\S*)"
)


def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump(by_alias=True, exclude_unset=True)


def _build_model_details(
    model_name: str,
    model_version: str,
    params: dict,  # mutable-ok: model params dict forwarded directly to wire payload
    timeout: int | None,
    max_retries: int | None,
) -> dict:  # mutable-ok: wire serialization helper; dict is the required output shape for JSON encoding
    """Build the model dict for the orchestration request, adding optional fields only when set."""
    model_details: dict = {  # mutable-ok: ephemeral wire dict built in one shot before JSON encoding
        "name": model_name,
        "params": params,
        "version": model_version,
    }
    if timeout is not None:
        model_details["timeout"] = timeout
    if max_retries is not None:
        model_details["max_retries"] = max_retries
    return model_details


def _messages_to_sap_template(messages: list[dict[str, str]]) -> list:
    template: Final = []
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


def _tools_response_format_and_stream(optional_params: dict, model_params: dict) -> tuple[dict, dict, dict]:
    tools_ = optional_params.pop("tools", [])
    tools_ = [validate_dict(tool, ChatCompletionTool) for tool in tools_]
    tools: Final[dict] = {"tools": tools_} if tools_ else {}

    response_format = model_params.pop("response_format", {})
    resp_type: Final = response_format.get("type", None)
    if resp_type:
        if resp_type == "json_schema":
            response_format = validate_dict(response_format, ResponseFormatJSONSchema)
        else:
            response_format = validate_dict(response_format, ResponseFormat)
        response_format = {"response_format": response_format}

    model_params.pop("stream", False)
    stream_config: Final[dict] = {}
    if "stream_options" in optional_params:
        stream_options: Final = optional_params.pop("stream_options", {})
        if "chunk_size" in stream_options:
            stream_config["chunk_size"] = stream_options.get("chunk_size")
        if "delimiters" in stream_options:
            stream_config["delimiters"] = stream_options.get("delimiters")

    return tools, response_format, stream_config


class GenAIHubOrchestrationConfig(OpenAIGPTConfig):
    frequency_penalty: int | None = None
    function_call: str | dict | None = None
    functions: list | None = None
    logit_bias: dict | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    stop: str | list | None = None
    temperature: int | None = None
    top_p: int | None = None
    response_format: dict | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None
    model_version: str = "latest"

    def __init__(
        self,
        frequency_penalty: int | None = None,
        function_call: str | dict | None = None,
        functions: list | None = None,
        logit_bias: dict | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: int | None = None,
        stop: str | list | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        response_format: dict | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        self.token_creator = None
        self._base_url = None
        self._resource_group = None

    def run_env_setup(self, service_key: str | None = None) -> None:
        try:
            self.token_creator, self._base_url, self._resource_group = get_token_creator(service_key)
        except ValueError as err:
            raise GenAIHubOrchestrationError(status_code=400, message=err.args[0])

    @property
    def headers(self) -> dict[str, str]:
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
        return self._base_url

    @property
    def resource_group(self) -> str:
        if self._resource_group is None:
            self.run_env_setup()
        return self._resource_group

    @cached_property
    def deployment_url(self) -> str:
        # Keep a short, tight client lifecycle here to avoid fd leaks
        client: Final = litellm.module_level_client
        # with httpx.Client(timeout=30) as client:
        deployments: Final = client.get(f"{self.base_url}/lm/deployments", headers=self.headers).json()
        valid: Final[list[tuple[str, str]]] = []
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
        params: Final = [
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
            "function_call",
            "functions",
            "extra_headers",
            "parallel_tool_calls",
            "response_format",
            "timeout",
            "max_retries",
            "model_version",
            "user",
        ]
        # Remove response_format for providers that don't support it on SAP GenAI Hub
        if (
            model.startswith("amazon")
            or model.startswith("cohere")
            or model.startswith("alephalpha")
            or model == "gpt-4"
        ):
            params.remove("response_format")
        if self._sap_supports_reasoning(model):
            params.extend(["reasoning_effort", "thinking"])  # mutable-ok: params list is local to this call; extend is the standard list append
        return params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        if api_key:
            self.run_env_setup(api_key)
        return self.headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ):
        api_base_: Final = f"{self.deployment_url}/v2/completion"
        return api_base_

    def _build_prompt_module(
        self,
        model_name: str,
        template_messages: list[dict[str, str]],
        params: dict,
    ) -> dict:
        # Filter strict for GPT models only - SAP AI Core doesn't accept it as a model param
        # LangChain agents pass strict=true at top level, which fails for GPT models
        # Anthropic models accept strict, so preserve it for them
        if model_name.startswith("gpt") and "strict" in params:
            params.pop("strict")

        model_version: Final = params.pop("model_version", "latest")
        timeout = params.pop("timeout", None)  # rebind-ok: popped from params for conditional inclusion below
        max_retries = params.pop("max_retries", None)  # rebind-ok: popped from params for conditional inclusion below

        tools_ = params.pop("tools", [])  # mutable-ok: intermediate list; immediately validated on the next line
        tools_ = [validate_dict(tool, ChatCompletionTool) for tool in tools_]  # mutable-ok: validated tool dicts; folded into wire payload below  # rebind-ok: rebound to validated list
        tools: Final = {"tools": tools_} if tools_ else {}

        response_format = params.pop("response_format", {})  # mutable-ok: ephemeral wire dict consumed immediately  # rebind-ok: popped for type inspection below
        resp_type: Final = response_format.get("type", None)
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

        optional_modules: Final = {}
        optional_modules_lst: Final = ["grounding", "masking", "filtering", "translation"]
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
                "model": _build_model_details(model_name, model_version, params, timeout, max_retries),
            },
            **optional_modules,
        }

    def transform_request(
        self,
        model: str,
        messages: list[dict[str, str]],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        optional_params = dict(optional_params)
        optional_params.pop("deployment_url", None)

        template: Final = _messages_to_sap_template(messages)

        placeholder_values: Final = optional_params.pop("placeholder_values", None)
        fallback_modules: Final = optional_params.pop("fallback_sap_modules", [])

        optional_params.pop("stream", None)
        stream_config: Final[dict] = {}
        if "stream_options" in optional_params:
            stream_options: Final = optional_params.pop("stream_options", {})
            if "chunk_size" in stream_options:
                stream_config["chunk_size"] = stream_options["chunk_size"]
            if "delimiters" in stream_options:
                stream_config["delimiters"] = stream_options["delimiters"]

        modules: Final = [
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
            fallback_model = fallback_model.removeprefix("sap/")
            fallback_template = modules_dict.pop("messages", [])

            modules.append(
                self._build_prompt_module(
                    model_name=fallback_model,
                    template_messages=fallback_template,
                    params=modules_dict,
                )
            )

        config_payload: Final[dict[str, Any]] = {
            "modules": modules if len(modules) > 1 else modules[0],
        }
        if stream_config:
            config_payload["stream"] = stream_config

        request_body: Final[dict[str, Any]] = {"config": config_payload}
        if placeholder_values is not None:
            request_body["placeholder_values"] = placeholder_values

        body: Final = validate_dict(request_body, OrchestrationRequest)

        return body

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
        encoding: "tiktoken.Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        final_result = raw_response.json()["final_result"]  # rebind-ok: extracted from raw response for normalisation then model_validate

        # SAP's orchestration service returns Gemini thinking as a list of
        # {"content": str, "signature": str} objects under reasoning_content.
        # Normalize to litellm's standard fields before model_validate, which
        # crashes on a list-shaped reasoning_content.
        if model.startswith("gemini"):
            self._normalize_gemini_reasoning(final_result)

        response = ModelResponse.model_validate(final_result)  # rebind-ok: constructed from final_result; referenced below for hidden params

        # Strip markdown code blocks if JSON response_format was used with Anthropic models
        # SAP GenAI Hub with Anthropic models sometimes wraps JSON in ```json ... ```
        # based on prompt phrasing. GPT/Gemini models don't exhibit this behavior,
        # so we gate the stripping to avoid accidentally modifying valid responses.
        response_format: Final = optional_params.get("response_format", {})
        if response_format.get("type") in ("json_object", "json_schema"):
            if model.startswith("anthropic"):
                response = self._strip_markdown_json(response)

        return response

    @staticmethod
    def _sap_supports_reasoning(model: str) -> bool:
        """Return True if *model* accepts reasoning_effort / thinking on SAP GenAI Hub."""
        return bool(_REASONING_MODELS.match(model))

    @staticmethod
    def _normalize_gemini_reasoning(final_result: dict) -> None:  # mutable-ok: final_result mutated in place to prepare for model_validate
        """Normalize SAP Gemini reasoning_content from list to litellm's standard fields.

        SAP's orchestration service returns Gemini thinking as a list of
        {"content": str, "signature": str} objects under reasoning_content.
        litellm uses two parallel fields for this:
        - thinking_blocks: list of ChatCompletionThinkingBlock dicts, preserving
          signature tokens that Gemini requires to replay thinking in multi-turn
          requests and that the Responses API serializes for round-trip preservation.
        - reasoning_content: plain str derived from the same blocks, exposed to
          callers and used by streaming accumulators and the Responses API.
        Both must be populated; populating only the string loses the signatures.
        """
        for choice in final_result.get("choices") or []:
            msg = (choice.get("message") or {}) if isinstance(choice, dict) else {}
            rc = msg.get("reasoning_content")
            if not isinstance(rc, list):
                continue
            blocks: list[dict] = []  # mutable-ok: local accumulator; assigned once to msg field below
            texts: list[str] = []  # mutable-ok: local accumulator; joined to string below
            for item in rc:
                if not isinstance(item, dict):
                    continue
                thought = item.get("content", "")
                block: dict = {"type": "thinking", "thinking": thought}  # mutable-ok: local block dict; signature optionally appended then collected
                if item.get("signature") is not None:
                    block["signature"] = item["signature"]  # mutable-ok: block is a local dict
                blocks.append(block)
                texts.append(thought)
            msg["thinking_blocks"] = (
                blocks if blocks else None
            )  # rebind-ok: normalizing SAP Gemini list to ChatCompletionThinkingBlock list
            msg["reasoning_content"] = "\n\n".join(texts) or None  # rebind-ok: normalizing SAP Gemini list to plain str

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
        json_mode: bool | None = False,
    ):
        if sync_stream:
            return SAPStreamIterator(response=streaming_response)
        else:
            return AsyncSAPStreamIterator(response=streaming_response)
