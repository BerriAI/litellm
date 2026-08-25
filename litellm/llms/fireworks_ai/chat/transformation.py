import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Final, Literal, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import unpack_legacy_defs
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolParam,
    OpenAIChatCompletionToolParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    ProviderSpecificModelInfo,
)
from litellm.utils import (
    get_model_cost_mutation_generation,
    supports_function_calling,
    supports_reasoning,
    supports_tool_choice,
)

from ...openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
    OpenAIGPTConfig,
)
from ..common_utils import (
    FireworksAIException,
    FireworksAIMixin,
    resolve_fireworks_resource_name,
)


def _extract_fireworks_hidden_params(payload: dict) -> dict:
    """
    Collect Fireworks-specific response fields (perf_metrics, prompt_token_ids,
    per-choice raw_output and token_ids) from a non-streaming completion payload
    or a single streaming chunk, so the same data lands in ``_hidden_params`` on
    both response paths.
    """
    choices: Final = [c for c in (payload.get("choices") or []) if isinstance(c, dict)]
    top_level: Final = {
        f"fireworks_{field}": payload[field] for field in ("perf_metrics", "prompt_token_ids") if field in payload
    }
    per_choice: Final = {
        f"fireworks_{dest}": [c[field] for c in choices if field in c]
        for field, dest in (("raw_output", "raw_outputs"), ("token_ids", "token_ids"))
        if any(field in c for c in choices)
    }
    return {**top_level, **per_choice}


def _json_schema_response_format(schema: object, name: str) -> Mapping[str, object]:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}  # mutable-ok: JSON request body


EFFORT_KWARG_KEYS: Final = frozenset({"enable_thinking", "thinking", "reasoning_budget", "low_effort"})


def _bool_from_kwargs(kwargs: Mapping[str, object], keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        value = kwargs.get(key)
        if isinstance(value, bool):
            return value
    return None


def effort_from_chat_template_kwargs(kwargs: Mapping[str, object]) -> object:
    enable_thinking: Final = _bool_from_kwargs(kwargs, ("enable_thinking", "thinking"))
    if enable_thinking is False:
        return "none"
    budget: Final = kwargs.get("reasoning_budget")
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
        return int(budget)
    low_effort: Final = _bool_from_kwargs(kwargs, ("low_effort",))
    if low_effort is True:
        return "low"
    return None


NIM_VLLM_STRIP_PARAMS: Final = frozenset(
    {
        "stop_token_ids",
        "include_stop_str_in_output",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "best_of",
        "use_beam_search",
        "guided_decoding_backend",
        "guided_regex",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
        "detokenize",
        "allowed_token_ids",
        "bad_words",
        "include_reasoning",
        "nvext",
    }
)

_EXTRA_BODY_CONSUMED_PARAMS: Final = (
    frozenset({"truncate_prompt_tokens", "chat_template_kwargs", "guided_json", "guided_grammar", "guided_choice"})
    | NIM_VLLM_STRIP_PARAMS
)


class FireworksAIConfig(FireworksAIMixin, OpenAIGPTConfig):
    """
    Reference: https://docs.fireworks.ai/api-reference/post-chatcompletions

    The class `FireworksAIConfig` provides configuration for the Fireworks's Chat Completions API interface. Below are the parameters:
    """

    tools: list | None = None
    tool_choice: str | dict | None = None
    max_tokens: int | None = None
    temperature: int | None = None
    top_p: int | None = None
    top_k: int | None = None
    frequency_penalty: int | None = None
    presence_penalty: int | None = None
    n: int | None = None
    stop: str | list | None = None
    response_format: dict | None = None
    user: str | None = None
    logprobs: int | None = None
    reasoning_effort: str | None = None

    prompt_truncate_len: int | None = None
    context_length_exceeded_behavior: Literal["error", "truncate"] | None = None

    def __init__(
        self,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        top_k: int | None = None,
        frequency_penalty: int | None = None,
        presence_penalty: int | None = None,
        n: int | None = None,
        stop: str | list | None = None,
        response_format: dict | None = None,
        user: str | None = None,
        logprobs: int | None = None,
        reasoning_effort: str | None = None,
        prompt_truncate_len: int | None = None,
        context_length_exceeded_behavior: Literal["error", "truncate"] | None = None,
    ) -> None:
        OpenAIGPTConfig.__init__(
            self,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
        )
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

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
        api_key = self._get_api_key(api_key)
        if api_key is None:
            raise ValueError("FIREWORKS_API_KEY is not set")

        validated_headers: Final = OpenAIGPTConfig.validate_environment(
            self,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        return self._add_session_affinity_header(validated_headers, litellm_params)

    def get_supported_openai_params(self, model: str):
        # Base parameters supported by all models
        supported_params: Final = [
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
            "prompt_truncate_len",
            "context_length_exceeded_behavior",
            "seed",
            "top_logprobs",
            "min_p",
            "typical_p",
            "repetition_penalty",
            "mirostat_target",
            "mirostat_lr",
            "logit_bias",
            "echo",
            "echo_last",
            "ignore_eos",
            "prompt_cache_key",
            "prompt_cache_isolation_key",
            "raw_output",
            "perf_metrics_in_response",
            "return_token_ids",
            "safe_tokenization",
            "service_tier",
            "speculation",
            "prediction",
            "stream_options",
            "sampling_mask",
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

        # Only add reasoning params for models that support it
        if supports_reasoning(model=model, custom_llm_provider="fireworks_ai"):
            supported_params.append("reasoning_effort")
            supported_params.append("reasoning_history")
            supported_params.append("thinking")

        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        is_tools_set: Final = any(param == "tools" and value is not None for param, value in non_default_params.items())
        if non_default_params.get("thinking") is not None and non_default_params.get("reasoning_effort") is not None:
            raise litellm.BadRequestError(
                message=(
                    "Fireworks AI chat completions does not support specifying both "
                    "`thinking` and `reasoning_effort` in the same request."
                ),
                model=model,
                llm_provider="fireworks_ai",
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
                if is_tools_set:  # fireworks ai doesn't support tools and response_format together
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
            elif param == "reasoning_effort":
                if value is True:
                    optional_params["reasoning_effort"] = "medium"
                elif value is False:
                    optional_params["reasoning_effort"] = "none"
                elif value != "auto":
                    optional_params["reasoning_effort"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value

        return optional_params

    def map_extra_body_params(
        self, optional_params: Mapping[str, object], model: str
    ) -> dict:  # mutable-ok: http handler pops extra_body off the returned dict
        extra_body: Final = optional_params.get("extra_body")
        if not isinstance(extra_body, dict):
            return dict(optional_params)  # mutable-ok: JSON request body

        stripped: Final = tuple(sorted(k for k in extra_body if k in NIM_VLLM_STRIP_PARAMS))
        if stripped:
            verbose_logger.debug(
                "fireworks_ai does not support NIM/vLLM params %s for model=%s; dropping them from the request.",
                stripped,
                model,
            )
        promoted: Final = (
            *self._translate_truncate_prompt_tokens(extra_body, optional_params),
            *self._translate_chat_template_kwargs(extra_body, optional_params, model),
            *self.translate_guided_params(extra_body, optional_params),
        )
        if "response_format" in extra_body and "response_format" in optional_params:
            verbose_logger.debug(
                "fireworks_ai dropping extra_body.response_format; the top-level response_format takes precedence."
            )
        remaining: Final = tuple(
            (k, v)
            for k, v in extra_body.items()
            if k not in _EXTRA_BODY_CONSUMED_PARAMS
            and (k != "response_format" or "response_format" not in optional_params)
        )
        base: Final = {k: v for k, v in optional_params.items() if k != "extra_body"}  # mutable-ok: JSON request body
        return {  # mutable-ok: JSON request body
            **base,
            **dict(promoted),  # mutable-ok: JSON request body
            **({"extra_body": dict(remaining)} if remaining else {}),  # mutable-ok: JSON request body
        }

    @staticmethod
    def _translate_truncate_prompt_tokens(
        extra_body: Mapping[str, object], optional_params: Mapping[str, object]
    ) -> tuple[tuple[str, object], ...]:
        if extra_body.get("truncate_prompt_tokens") is None:
            return ()
        if "prompt_truncate_len" in extra_body or "prompt_truncate_len" in optional_params:
            verbose_logger.debug(
                "fireworks_ai ignoring truncate_prompt_tokens; explicit prompt_truncate_len takes precedence."
            )
            return ()
        return (("prompt_truncate_len", extra_body["truncate_prompt_tokens"]),)

    def _translate_chat_template_kwargs(
        self, extra_body: Mapping[str, object], optional_params: Mapping[str, object], model: str
    ) -> tuple[tuple[str, object], ...]:
        chat_template_kwargs: Final = extra_body.get("chat_template_kwargs")
        if chat_template_kwargs is None:
            return ()
        if not isinstance(chat_template_kwargs, dict):
            verbose_logger.debug(
                "fireworks_ai dropping chat_template_kwargs for model=%s; expected an object, got %s.",
                model,
                type(chat_template_kwargs).__name__,
            )
            return ()
        other_keys: Final = tuple(sorted(k for k in chat_template_kwargs if k not in EFFORT_KWARG_KEYS))
        if other_keys:
            verbose_logger.debug(
                "fireworks_ai does not support chat_template_kwargs keys %s for model=%s; dropping them.",
                other_keys,
                model,
            )
        if any(key in optional_params or key in extra_body for key in ("reasoning_effort", "thinking")):
            verbose_logger.debug(
                "fireworks_ai ignoring chat_template_kwargs; explicit reasoning_effort/thinking takes precedence."
            )
            return ()
        effort: Final = effort_from_chat_template_kwargs(chat_template_kwargs)
        if effort is None:
            return ()
        if not supports_reasoning(model=model, custom_llm_provider="fireworks_ai"):
            verbose_logger.debug(
                "fireworks_ai model %r does not support reasoning; dropping chat_template_kwargs effort keys.",
                model,
            )
            return ()
        return (("reasoning_effort", effort),)

    @staticmethod
    def translate_guided_params(
        extra_body: Mapping[str, object], optional_params: Mapping[str, object]
    ) -> tuple[tuple[str, object], ...]:
        has_guided: Final = any(
            extra_body.get(key) is not None for key in ("guided_json", "guided_grammar", "guided_choice")
        )
        if not has_guided:
            return ()
        if "response_format" in optional_params or "response_format" in extra_body:
            verbose_logger.debug(
                "fireworks_ai ignoring guided decoding params; explicit response_format takes precedence."
            )
            return ()
        if extra_body.get("guided_json") is not None:
            return (("response_format", _json_schema_response_format(extra_body["guided_json"], "response")),)
        if extra_body.get("guided_grammar") is not None:
            grammar_response_format: Final = {  # mutable-ok: JSON request body
                "type": "grammar",
                "grammar": extra_body["guided_grammar"],
            }
            return (("response_format", grammar_response_format),)
        choice_schema: Final = {  # mutable-ok: JSON request body
            "type": "string",
            "enum": extra_body["guided_choice"],
        }
        return (("response_format", _json_schema_response_format(choice_schema, "choice")),)

    def _transform_tools(self, tools: list[OpenAIChatCompletionToolParam]) -> list[OpenAIChatCompletionToolParam]:
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
        self, messages: list[AllMessageValues], model: str, litellm_params: dict
    ) -> list[AllMessageValues]:
        """
        Strip fields not permitted by FireworksAI from messages.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            filter_value_from_dict,
        )

        supports_vision_value: Final = self._get_model_cost_capability_exact(model=model, capability="supports_vision")
        for message in messages:
            if message["role"] == "user":
                _message_content = message.get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    for content in _message_content:
                        if not isinstance(content, dict):
                            continue
                        if content.get("type") == "file":
                            raise litellm.BadRequestError(
                                message=(
                                    "Fireworks AI chat completions does not support "
                                    "file content blocks. For PDFs, convert pages to "
                                    "images and send image_url blocks to a Fireworks "
                                    "vision model, or extract text before calling a "
                                    "text-only model."
                                ),
                                model=model,
                                llm_provider="fireworks_ai",
                            )
                        if content.get("type") == "image_url" and supports_vision_value is False:
                            raise litellm.BadRequestError(
                                message=(
                                    f"Fireworks AI model {model} does not support "
                                    "image inputs. Use a Fireworks vision model or "
                                    "remove image_url content blocks."
                                ),
                                model=model,
                                llm_provider="fireworks_ai",
                            )
            filter_value_from_dict(cast(dict, message), "cache_control")
            # Remove fields not permitted by FireworksAI (additionalProperties: false
            # on their ChatMessage schema) that may cause:
            # "Extra inputs are not permitted, field: 'messages[n].<field>'"
            if isinstance(message, dict):
                m = cast(dict, message)
                m.pop("provider_specific_fields", None)
                m.pop("thinking_blocks", None)
                m.pop("reasoning_content", None)

        return messages

    # Cached index of fireworks_ai/* entries from litellm.model_cost. Building
    # this index requires a full scan of model_cost (tens of thousands of
    # entries), so we memoize it. The cache key is (id(model_cost),
    # mutation_generation): the generation counter is bumped on every
    # register_model / reload path, so add+remove or in-place value
    # replacement (which can leave id and len unchanged) still invalidates.
    _fireworks_index_cache: tuple[int, int, list[tuple[str, dict]]] | None = None

    @classmethod
    def _get_fireworks_index(cls) -> list[tuple[str, dict]]:
        model_cost: Final = litellm.model_cost
        signature: Final = (id(model_cost), get_model_cost_mutation_generation())
        cached: Final = cls._fireworks_index_cache
        if cached is not None and cached[0] == signature[0] and cached[1] == signature[1]:
            return cached[2]

        index: Final[list[tuple[str, dict]]] = []
        for key, model_info in model_cost.items():
            if not key.startswith("fireworks_ai/"):
                continue
            if not isinstance(model_info, dict):
                continue
            key_short = key[len("fireworks_ai/") :]
            key_short = key_short.removeprefix("accounts/fireworks/models/")
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

    @staticmethod
    def _short_model_name(model: str) -> str:
        short_name = model
        short_name = short_name.removeprefix("fireworks_ai/")
        short_name = short_name.removeprefix("accounts/fireworks/models/")
        return short_name

    def _get_model_cost_capability_exact(self, model: str, capability: str) -> bool | None:
        short_name: Final = self._short_model_name(model)
        candidate_keys: Final = (
            model,
            f"fireworks_ai/{short_name}",
            f"fireworks_ai/accounts/fireworks/models/{short_name}",
        )
        for candidate_key in candidate_keys:
            model_info = litellm.model_cost.get(candidate_key)
            if model_info is not None and model_info.get(capability) is not None:
                return cast(bool | None, model_info.get(capability))
        return None

    def _get_model_cost_capability(self, model: str, capability: str) -> bool | None:
        exact: Final = self._get_model_cost_capability_exact(model=model, capability=capability)
        if exact is not None:
            return exact

        # Fallback: substring matching for model name variants (e.g. fine-tuned
        # or regionally-suffixed versions of a known model). Pick the *longest*
        # matching entry so a more specific known model (e.g. "qwen3-8b-instruct")
        # wins over a less specific one (e.g. "qwen3-8b"). Hyphen-aligned matching
        # avoids false positives where a short known name is an unrelated
        # substring of a longer one. This stays a soft signal: capability-gated
        # hard rejections use the exact lookup so a fuzzy match never blocks a
        # custom deployment.
        short_name: Final = self._short_model_name(model)
        matches: Final = [
            (key_short, cast(bool | None, model_info.get(capability)))
            for key_short, model_info in self._get_fireworks_index()
            if model_info.get(capability) is not None and self._matches_on_hyphen_boundary(short_name, key_short)
        ]
        if not matches:
            return None
        return max(matches, key=lambda match: len(match[0]))[1]

    def get_provider_info(self, model: str) -> ProviderSpecificModelInfo:
        supports_function_calling_value: Final = self._get_model_cost_capability(
            model=model, capability="supports_function_calling"
        )
        supports_reasoning_value: Final = self._get_model_cost_capability(model=model, capability="supports_reasoning")
        supports_vision_value: Final = self._get_model_cost_capability(model=model, capability="supports_vision")
        supports_pdf_input_value: Final = self._get_model_cost_capability(model=model, capability="supports_pdf_input")

        provider_specific_model_info: Final[ProviderSpecificModelInfo] = {
            "supports_function_calling": True,
            "supports_prompt_caching": True,  # https://docs.fireworks.ai/guides/prompt-caching
        }

        if supports_function_calling_value is not None:
            provider_specific_model_info["supports_function_calling"] = supports_function_calling_value

        # Only include supports_reasoning if True
        if supports_reasoning_value:
            provider_specific_model_info["supports_reasoning"] = supports_reasoning_value

        if supports_vision_value is not None:
            provider_specific_model_info["supports_vision"] = supports_vision_value

        if supports_pdf_input_value is not None:
            provider_specific_model_info["supports_pdf_input"] = supports_pdf_input_value

        return provider_specific_model_info

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        resolved_model: Final = resolve_fireworks_resource_name(model)
        messages = self._transform_messages_helper(
            messages=messages, model=resolved_model, litellm_params=litellm_params
        )
        if "tools" in optional_params and optional_params["tools"] is not None:
            tools: Final = self._transform_tools(tools=optional_params["tools"])
            optional_params["tools"] = tools
        if optional_params.get("stream"):
            stream_options: Final = optional_params.get("stream_options")
            if stream_options is None:
                optional_params["stream_options"] = {"include_usage": True}
            elif stream_options.get("include_usage") is not False:
                optional_params["stream_options"] = {
                    **stream_options,
                    "include_usage": True,
                }
        return super().transform_request(
            model=resolved_model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _handle_message_content_with_tool_calls(
        self,
        message: Message,
        tool_calls: list[ChatCompletionToolParam] | None,
    ) -> Message:
        """
        Fireworks AI sends tool calls in the content field instead of tool_calls

        Relevant Issue: https://github.com/BerriAI/litellm/issues/7209#issuecomment-2813208780
        """
        if tool_calls is not None and message.content is not None and message.tool_calls is None:
            try:
                function: Final = Function(**json.loads(message.content))
                if function.name != RESPONSE_FORMAT_TOOL_NAME and function.name in [
                    tool["function"]["name"] for tool in tool_calls
                ]:
                    tool_call = ChatCompletionMessageToolCall(function=function, id=str(uuid.uuid4()), type="function")
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
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
            completion_response: Final = raw_response.json()
        except Exception as e:
            response_headers: Final = getattr(raw_response, "headers", None)
            raise FireworksAIException(
                message=f"Unable to get json response - {e}, Original Response: {raw_response.text}",
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        raw_response_headers: Final = dict(raw_response.headers)

        additional_headers: Final = get_response_headers(raw_response_headers)

        response: Final = ModelResponse(**completion_response)

        if response.model is not None:
            response.model = "fireworks_ai/" + response.model

        ## FIREWORKS AI sends tool calls in the content field instead of tool_calls
        for choice in response.choices:
            cast(Choices, choice).message = self._handle_message_content_with_tool_calls(
                message=cast(Choices, choice).message,
                tool_calls=optional_params.get("tools", None),
            )

        response._hidden_params = {
            "additional_headers": additional_headers,
            **_extract_fireworks_hidden_params(completion_response),
        }

        return response

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> Any:
        return FireworksAIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("FIREWORKS_API_BASE") or "https://api.fireworks.ai/inference/v1"
        dynamic_api_key: Final = api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
        return api_base, dynamic_api_key

    def get_models(self, api_key: str | None = None, api_base: str | None = None):
        api_base, api_key = self._get_openai_compatible_provider_info(api_base=api_base, api_key=api_key)
        if api_base is None or api_key is None:
            raise ValueError(
                "FIREWORKS_API_BASE or FIREWORKS_API_KEY is not set. Please set the environment variable, to query Fireworks AI's `/models` endpoint."
            )

        account_id: Final = get_secret_str("FIREWORKS_ACCOUNT_ID")
        if account_id is None:
            raise ValueError(
                "FIREWORKS_ACCOUNT_ID is not set. Please set the environment variable, to query Fireworks AI's `/models` endpoint."
            )

        base = api_base.rstrip("/")
        base = base.removesuffix("/v1")
        response: Final = litellm.module_level_client.get(
            url=f"{base}/v1/accounts/{account_id}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Fireworks AI. Status code: {response.status_code}, Response: {response.json()}"
            )

        models: Final = response.json()["models"]

        return ["fireworks_ai/" + model["name"] for model in models]

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )


class FireworksAIChatCompletionStreamingHandler(OpenAIChatCompletionStreamingHandler):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        parsed: Final = super().chunk_parser(chunk)
        fireworks_fields: Final = _extract_fireworks_hidden_params(chunk)
        if fireworks_fields:
            parsed.provider_specific_fields = {
                **(getattr(parsed, "provider_specific_fields", None) or {}),
                **fireworks_fields,
            }
        return parsed
