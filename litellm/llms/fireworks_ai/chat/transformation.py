import json
from typing import Any, List, Literal, Optional, Tuple, Union, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.common_utils import unpack_legacy_defs
from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionImageObject,
    ChatCompletionToolParam,
    OpenAIChatCompletionToolParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    ProviderSpecificModelInfo,
)
from litellm.utils import (
    get_model_cost_mutation_generation,
    supports_function_calling,
    supports_reasoning,
    supports_tool_choice,
)

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import FireworksAIException


class FireworksAIConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.fireworks.ai/api-reference/post-chatcompletions

    The class `FireworksAIConfig` provides configuration for the Fireworks's Chat Completions API interface. Below are the parameters:
    """

    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    response_format: Optional[dict] = None
    user: Optional[str] = None
    logprobs: Optional[int] = None
    reasoning_effort: Optional[str] = None

    # Non OpenAI parameters - Fireworks AI only params
    prompt_truncate_length: Optional[int] = None
    context_length_exceeded_behavior: Optional[Literal["error", "truncate"]] = None

    def __init__(
        self,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        response_format: Optional[dict] = None,
        user: Optional[str] = None,
        logprobs: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        prompt_truncate_length: Optional[int] = None,
        context_length_exceeded_behavior: Optional[Literal["error", "truncate"]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        # Base parameters supported by all models
        supported_params = [
            "stream",
            "max_completion_tokens",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "stop",
            "response_format",
            "user",
            "logprobs",
            "prompt_truncate_length",
            "context_length_exceeded_behavior",
        ]

        # Only add tools for models that support function calling
        if supports_function_calling(model=model, custom_llm_provider="fireworks_ai"):
            supported_params.append("tools")
            supported_params.append("parallel_tool_calls")
        else:
            # Historically every Fireworks model advertised tool support, so a
            # JSON entry that flips `supports_function_calling` to false will
            # silently drop `tools` from requests. Surface this so users can
            # tell why their tool calls suddenly stop working.
            verbose_logger.debug(
                "fireworks_ai model %r is marked as not supporting "
                "function calling in model_prices_and_context_window.json; "
                "`tools` and `parallel_tool_calls` will be dropped from the "
                "request.",
                model,
            )

        # Only add tool_choice for models that explicitly support it
        if supports_tool_choice(model=model, custom_llm_provider="fireworks_ai"):
            supported_params.append("tool_choice")

        # Only add reasoning_effort for models that support it
        if supports_reasoning(model=model, custom_llm_provider="fireworks_ai"):
            supported_params.append("reasoning_effort")

        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        is_tools_set = any(
            param == "tools" and value is not None
            for param, value in non_default_params.items()
        )

        for param, value in non_default_params.items():
            if param == "tool_choice":
                if value == "required":
                    # relevant issue: https://github.com/BerriAI/litellm/issues/4416
                    optional_params["tool_choice"] = "any"
                else:
                    # pass through the value of tool choice
                    optional_params["tool_choice"] = value
            elif param == "response_format":
                if (
                    is_tools_set
                ):  # fireworks ai doesn't support tools and response_format together
                    optional_params = self._add_response_format_to_tools(
                        optional_params=optional_params,
                        value=value,
                        is_response_format_supported=False,
                        enforce_tool_choice=False,  # tools and response_format are both set, don't enforce tool_choice
                    )
                else:
                    optional_params["response_format"] = value
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value

        return optional_params

    def _add_transform_inline_image_block(
        self,
        content: ChatCompletionImageObject,
        model: str,
        disable_add_transform_inline_image_block: Optional[bool],
    ) -> ChatCompletionImageObject:
        """
        Add transform_inline to the image_url (allows non-vision models to parse documents/images/etc.)
        - ignore if model is a vision model
        - ignore if user has disabled this feature
        """
        if (
            "vision" in model or disable_add_transform_inline_image_block
        ):  # allow user to toggle this feature.
            return content
        if isinstance(content["image_url"], str):
            # Skip base64 data URLs — appending #transform=inline corrupts the
            # base64 payload and causes an "Incorrect padding" decode error on
            # the Fireworks side.  Data URLs are already inlined by definition.
            # Lower-case before checking: URI schemes are case-insensitive (RFC 3986).
            if not content["image_url"].lower().startswith("data:"):
                content["image_url"] = f"{content['image_url']}#transform=inline"
        elif isinstance(content["image_url"], dict):
            url = content["image_url"]["url"]
            if not url.lower().startswith("data:"):
                content["image_url"]["url"] = f"{url}#transform=inline"
        return content

    def _transform_tools(
        self, tools: List[OpenAIChatCompletionToolParam]
    ) -> List[OpenAIChatCompletionToolParam]:
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool["function"]
            function.pop("strict", None)
            params = function.get("parameters")
            if isinstance(params, dict):
                unpack_legacy_defs(params)
        return tools

    def _transform_messages_helper(
        self, messages: List[AllMessageValues], model: str, litellm_params: dict
    ) -> List[AllMessageValues]:
        """
        Add 'transform=inline' to the url of the image_url
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            filter_value_from_dict,
            migrate_file_to_image_url,
        )

        disable_add_transform_inline_image_block = cast(
            Optional[bool],
            litellm_params.get("disable_add_transform_inline_image_block")
            or litellm.disable_add_transform_inline_image_block,
        )
        ## For any 'file' message type with pdf content, move to 'image_url' message type
        for message in messages:
            if message["role"] == "user":
                _message_content = message.get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    for idx, content in enumerate(_message_content):
                        if content["type"] == "file":
                            _message_content[idx] = migrate_file_to_image_url(content)
        for message in messages:
            if message["role"] == "user":
                _message_content = message.get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    for content in _message_content:
                        if content["type"] == "image_url":
                            content = self._add_transform_inline_image_block(
                                content=content,
                                model=model,
                                disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
                            )
            filter_value_from_dict(cast(dict, message), "cache_control")
            # Remove fields not permitted by FireworksAI (additionalProperties: false
            # on their ChatMessage schema) that may cause:
            # "Extra inputs are not permitted, field: 'messages[n].<field>'"
            if isinstance(message, dict):
                m = cast(dict, message)
                m.pop("provider_specific_fields", None)
                m.pop("thinking_blocks", None)

        return messages

    # Cached index of fireworks_ai/* entries from litellm.model_cost. Building
    # this index requires a full scan of model_cost (tens of thousands of
    # entries), so we memoize it. The cache key is (id(model_cost),
    # mutation_generation): the generation counter is bumped on every
    # register_model / reload path, so add+remove or in-place value
    # replacement (which can leave id and len unchanged) still invalidates.
    _fireworks_index_cache: Optional[Tuple[int, int, List[Tuple[str, dict]]]] = None

    @classmethod
    def _get_fireworks_index(cls) -> List[Tuple[str, dict]]:
        model_cost = litellm.model_cost
        signature = (id(model_cost), get_model_cost_mutation_generation())
        cached = cls._fireworks_index_cache
        if (
            cached is not None
            and cached[0] == signature[0]
            and cached[1] == signature[1]
        ):
            return cached[2]

        index: List[Tuple[str, dict]] = []
        for key, model_info in model_cost.items():
            if not key.startswith("fireworks_ai/"):
                continue
            if not isinstance(model_info, dict):
                continue
            key_short = key[len("fireworks_ai/") :]
            if key_short.startswith("accounts/fireworks/models/"):
                key_short = key_short[len("accounts/fireworks/models/") :]
            if not key_short:
                continue
            index.append((key_short, model_info))

        cls._fireworks_index_cache = (signature[0], signature[1], index)
        return index

    @staticmethod
    def _matches_on_hyphen_boundary(short_name: str, key_short: str) -> bool:
        """Return True if `key_short` appears in `short_name` aligned to
        hyphen-separated word boundaries (or end-of-string). This avoids
        spurious substring matches like `"some-model"` matching
        `"awesome-model"`."""
        if short_name == key_short:
            return True
        if short_name.startswith(key_short + "-"):
            return True
        if short_name.endswith("-" + key_short):
            return True
        return ("-" + key_short + "-") in short_name

    def _get_model_cost_capability(self, model: str, capability: str) -> Optional[bool]:
        short_name = model
        if short_name.startswith("fireworks_ai/"):
            short_name = short_name[len("fireworks_ai/") :]
        if short_name.startswith("accounts/fireworks/models/"):
            short_name = short_name[len("accounts/fireworks/models/") :]

        candidate_keys = [
            model,
            f"fireworks_ai/{short_name}",
            f"fireworks_ai/accounts/fireworks/models/{short_name}",
        ]

        for candidate_key in candidate_keys:
            model_info = litellm.model_cost.get(candidate_key)
            if model_info is not None and model_info.get(capability) is not None:
                return cast(Optional[bool], model_info.get(capability))

        # Fallback: preserve historical substring matching for model name
        # variants (e.g. fine-tuned or regionally-suffixed versions of a
        # known model). Pick the *longest* matching entry so a more specific
        # known model (e.g. "qwen3-8b-instruct") wins over a less specific
        # one (e.g. "qwen3-8b") when the query model is more specific still.
        # Use hyphen-aligned matching to avoid false positives where a short
        # known model name is an unrelated substring of a longer one.
        best_match_short: Optional[str] = None
        best_match_value: Optional[bool] = None
        for key_short, model_info in self._get_fireworks_index():
            if model_info.get(capability) is None:
                continue
            if not self._matches_on_hyphen_boundary(short_name, key_short):
                continue
            if best_match_short is None or len(key_short) > len(best_match_short):
                best_match_short = key_short
                best_match_value = cast(Optional[bool], model_info.get(capability))

        return best_match_value

    def get_provider_info(self, model: str) -> ProviderSpecificModelInfo:
        supports_function_calling_value = self._get_model_cost_capability(
            model=model, capability="supports_function_calling"
        )
        supports_reasoning_value = self._get_model_cost_capability(
            model=model, capability="supports_reasoning"
        )

        provider_specific_model_info: ProviderSpecificModelInfo = {
            "supports_function_calling": True,
            "supports_prompt_caching": True,  # https://docs.fireworks.ai/guides/prompt-caching
            "supports_pdf_input": True,  # via document inlining
            "supports_vision": True,  # via document inlining
        }

        if supports_function_calling_value is not None:
            provider_specific_model_info["supports_function_calling"] = (
                supports_function_calling_value
            )

        # Only include supports_reasoning if True
        if supports_reasoning_value:
            provider_specific_model_info["supports_reasoning"] = (
                supports_reasoning_value
            )

        return provider_specific_model_info

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if not model.startswith("accounts/") and "#" not in model:
            if model.endswith("-fast"):
                model = f"accounts/fireworks/routers/{model}"
            else:
                model = f"accounts/fireworks/models/{model}"
        messages = self._transform_messages_helper(
            messages=messages, model=model, litellm_params=litellm_params
        )
        if "tools" in optional_params and optional_params["tools"] is not None:
            tools = self._transform_tools(tools=optional_params["tools"])
            optional_params["tools"] = tools
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _handle_message_content_with_tool_calls(
        self,
        message: Message,
        tool_calls: Optional[List[ChatCompletionToolParam]],
    ) -> Message:
        """
        Fireworks AI sends tool calls in the content field instead of tool_calls

        Relevant Issue: https://github.com/BerriAI/litellm/issues/7209#issuecomment-2813208780
        """
        if (
            tool_calls is not None
            and message.content is not None
            and message.tool_calls is None
        ):
            try:
                function = Function(**json.loads(message.content))
                if function.name != RESPONSE_FORMAT_TOOL_NAME and function.name in [
                    tool["function"]["name"] for tool in tool_calls
                ]:
                    tool_call = ChatCompletionMessageToolCall(
                        function=function, id=str(uuid.uuid4()), type="function"
                    )
                    message.tool_calls = [tool_call]

                    message.content = None
            except Exception:
                pass

        return message

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
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise FireworksAIException(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        raw_response_headers = dict(raw_response.headers)

        additional_headers = get_response_headers(raw_response_headers)

        response = ModelResponse(**completion_response)

        if response.model is not None:
            response.model = "fireworks_ai/" + response.model

        ## FIREWORKS AI sends tool calls in the content field instead of tool_calls
        for choice in response.choices:
            cast(Choices, choice).message = (
                self._handle_message_content_with_tool_calls(
                    message=cast(Choices, choice).message,
                    tool_calls=optional_params.get("tools", None),
                )
            )

        response._hidden_params = {"additional_headers": additional_headers}

        return response

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("FIREWORKS_API_BASE")
            or "https://api.fireworks.ai/inference/v1"
        )  # type: ignore
        dynamic_api_key = api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
        return api_base, dynamic_api_key

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        api_base, api_key = self._get_openai_compatible_provider_info(
            api_base=api_base, api_key=api_key
        )
        if api_base is None or api_key is None:
            raise ValueError(
                "FIREWORKS_API_BASE or FIREWORKS_API_KEY is not set. Please set the environment variable, to query Fireworks AI's `/models` endpoint."
            )

        account_id = get_secret_str("FIREWORKS_ACCOUNT_ID")
        if account_id is None:
            raise ValueError(
                "FIREWORKS_ACCOUNT_ID is not set. Please set the environment variable, to query Fireworks AI's `/models` endpoint."
            )

        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        response = litellm.module_level_client.get(
            url=f"{base}/v1/accounts/{account_id}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Fireworks AI. Status code: {response.status_code}, Response: {response.json()}"
            )

        models = response.json()["models"]

        return ["fireworks_ai/" + model["name"] for model in models]

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
