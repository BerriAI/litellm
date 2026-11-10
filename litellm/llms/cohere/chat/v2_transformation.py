import itertools
import reprlib
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import httpx

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import (
    as_object_list,
    as_str_mapping,
    drops_unsupported_params,
    map_finish_reason,
)
from litellm.litellm_core_utils.reasoning_effort_utils import warn_dropped_reasoning
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.rust_bridge.host.model_cost import VersionCached
from litellm.types.llms.cohere import CohereV2ChatResponse
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
    ChatCompletionToolCallChunk,
)
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import CohereError, CohereV2ModelResponseIterator
from ..common_utils import validate_environment as cohere_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_COHERE_TOOL_CHOICE: Final = MappingProxyType({"none": "NONE", "required": "REQUIRED"})
_COHERE_TOOL_CHOICE_VALUES: Final = frozenset(_COHERE_TOOL_CHOICE.values())
_COHERE_THINKING_BUDGET: Final = MappingProxyType(
    {
        "minimal": DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
        "low": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
        "medium": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
        "high": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
        "xhigh": DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
        "max": DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    }
)
_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})
_RESPONSE_FORMAT_EXCLUSIVE_PARAMS: Final = ("tools", "documents")
_MODELS_REJECTING_RESPONSE_FORMAT_BESIDE_TOOLS: Final = frozenset({"command-r-08-2024", "command-r-plus-08-2024"})
_SCHEMA_KEYWORDS_COHERE_REJECTS: Final = frozenset(
    {
        "allOf",
        "oneOf",
        "not",
        "minimum",
        "maximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    }
)
_PATTERN_SYNTAX_COHERE_REJECTS: Final = ("(?=", "(?!", "(?<=", "(?<!")
_COHERE_PROVIDERS: Final = ("cohere_chat", "cohere")
_SCHEMA_DEPTH_CHECKED: Final = 64


def _cohere_model_id(model: str) -> str:
    return model.removeprefix("v2/")


def _cohere_model_ids(model: str, base_model: object) -> tuple[str, ...]:
    names: Final = (model, _cohere_model_id(model), f"v2/{_cohere_model_id(model)}", base_model)
    return tuple(dict.fromkeys(name for name in names if isinstance(name, str)))


@VersionCached
def _any_model_supports(capability: Callable[..., bool], model_ids: tuple[str, ...]) -> bool:
    return any(
        capability(model=model_id, custom_llm_provider=provider)
        for model_id, provider in itertools.product(model_ids, _COHERE_PROVIDERS)
    )


def _function_name(tool: object) -> object:
    function: Final = as_str_mapping((as_str_mapping(tool) or _NOTHING).get("function"))
    return None if function is None else function.get("name")


def _sole_tool_name(tools: object) -> object:
    match as_object_list(tools):
        case [tool]:
            return _function_name(tool)
        case _:
            return None


def _tool_choice_params(tool_choice: object, tools: object) -> Mapping[str, object]:
    if tool_choice == "auto":
        return _NOTHING
    if isinstance(tool_choice, str):
        return MappingProxyType({"tool_choice": _COHERE_TOOL_CHOICE.get(tool_choice, tool_choice)})
    named: Final = _function_name(tool_choice)
    forces_sole_tool: Final = named is not None and named == _sole_tool_name(tools)
    return MappingProxyType({"tool_choice": "REQUIRED" if forces_sole_tool else tool_choice})


def _effort_level(reasoning_effort: object) -> object:
    fields: Final = as_str_mapping(reasoning_effort)
    return reasoning_effort if fields is None else fields.get("effort")


def _token_budget(effort: str, max_tokens: object) -> int:
    budget: Final = _COHERE_THINKING_BUDGET[effort]
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
        return min(budget, max_tokens)
    return budget


def _thinking_params(reasoning_effort: object, max_tokens: object) -> Mapping[str, object]:
    effort: Final = _effort_level(reasoning_effort)
    match effort:
        case "default":
            return _NOTHING
        case "none":
            return MappingProxyType({"thinking": {"type": "disabled"}})
        case str() if effort in _COHERE_THINKING_BUDGET:
            return MappingProxyType(
                {"thinking": {"type": "enabled", "token_budget": _token_budget(effort, max_tokens)}}
            )
        case _:
            return MappingProxyType({"reasoning_effort": reasoning_effort})


def _cohere_json_object(schema: object) -> Mapping[str, object]:
    schema_field: Final = {} if schema is None else {"json_schema": schema}
    return MappingProxyType({"response_format": {"type": "json_object", **schema_field}})


def _response_format_params(response_format: object) -> Mapping[str, object]:
    fields: Final = as_str_mapping(response_format) or _NOTHING
    match fields.get("type"):
        case "text":
            return _NOTHING
        case "json_object":
            return _cohere_json_object(fields.get("json_schema"))
        case "json_schema":
            json_schema: Final = as_str_mapping(fields.get("json_schema")) or _NOTHING
            return _cohere_json_object(json_schema.get("schema"))
        case _:
            return MappingProxyType({"response_format": response_format})


def _cohere_v2_native_params(non_default_params: Mapping[str, object]) -> Mapping[str, object]:
    tool_choice: Final = (
        _tool_choice_params(non_default_params["tool_choice"], non_default_params.get("tools"))
        if "tool_choice" in non_default_params
        else _NOTHING
    )
    max_tokens: Final = non_default_params.get("max_completion_tokens", non_default_params.get("max_tokens"))
    thinking: Final = (
        _thinking_params(non_default_params["reasoning_effort"], max_tokens)
        if "reasoning_effort" in non_default_params
        else _NOTHING
    )
    response_format: Final = (
        _response_format_params(non_default_params["response_format"])
        if "response_format" in non_default_params
        else _NOTHING
    )
    return {**tool_choice, **thinking, **response_format}


def _beside_tools_on_rejecting_model(optional_params: Mapping[str, object], model_ids: tuple[str, ...]) -> bool:
    beside_tools: Final = any(optional_params.get(param) for param in _RESPONSE_FORMAT_EXCLUSIVE_PARAMS)
    return beside_tools and any(
        model_id.rsplit("/", 1)[-1] in _MODELS_REJECTING_RESPONSE_FORMAT_BESIDE_TOOLS for model_id in model_ids
    )


def _subschemas(schema: Mapping[str, object]) -> Iterator[object]:
    yield from (as_str_mapping(schema.get("properties")) or _NOTHING).values()
    yield from as_object_list(schema.get("anyOf")) or ()
    yield from as_object_list(schema.get("prefixItems")) or ()
    yield schema.get("items")
    yield schema.get("additionalProperties")


def _pattern_cohere_rejects(pattern: object) -> bool:
    if not isinstance(pattern, str):
        return False
    anchored: Final = pattern.startswith("^") or (pattern.endswith("$") and not pattern.endswith("\\$"))
    return anchored or any(syntax in pattern for syntax in _PATTERN_SYNTAX_COHERE_REJECTS)


def _schema_feature_cohere_rejects(schema: object, depth: int = 0) -> str | None:
    fields: Final = as_str_mapping(schema)
    if fields is None or depth > _SCHEMA_DEPTH_CHECKED:
        return None
    keyword: Final = next((key for key in fields if key in _SCHEMA_KEYWORDS_COHERE_REJECTS), None)
    if keyword is not None:
        return f"'{keyword}'"
    if _pattern_cohere_rejects(fields.get("pattern")):
        return "a pattern with ^, $, or a lookaround"
    nested: Final = (_schema_feature_cohere_rejects(subschema, depth + 1) for subschema in _subschemas(fields))
    return next((feature for feature in nested if feature is not None), None)


def _rejection(
    param: str, value: object, optional_params: Mapping[str, object], model_ids: tuple[str, ...]
) -> str | None:
    model: Final = model_ids[0]
    match param:
        case "tool_choice" if not (optional_params.get("tools") or optional_params.get("functions")):
            return None
        case "tool_choice" if not (isinstance(value, str) and value in _COHERE_TOOL_CHOICE_VALUES):
            return (
                "Cohere's tool_choice takes only 'auto', 'none', 'required', "
                f"or a function when it is the only tool, got {reprlib.repr(value)}."
            )
        case "tool_choice" if not _any_model_supports(litellm.utils.supports_tool_choice, model_ids):
            return f"Cohere model {model} does not support tool_choice."
        case "reasoning_effort":
            return f"Cohere does not support reasoning_effort={reprlib.repr(value)}."
        case "response_format" if (as_str_mapping(value) or _NOTHING).get("type") != "json_object":
            return "Cohere's response_format takes only 'text', 'json_object', or 'json_schema'."
        case "response_format" if _beside_tools_on_rejecting_model(optional_params, model_ids):
            return f"Cohere model {model} does not take response_format beside tools or documents."
        case "response_format" if (
            feature := _schema_feature_cohere_rejects((as_str_mapping(value) or _NOTHING).get("json_schema"))
        ) is not None:
            return f"Cohere's response_format schema does not support {feature}."
        case _:
            return None


def _logged_value(param: str, value: object) -> object:
    fields: Final = as_str_mapping(value)
    return fields.get("type") if param == "response_format" and fields is not None else value


def _translated_to_thinking(param: str, value: object) -> bool:
    return param == "reasoning_effort" and "reasoning_effort" not in _thinking_params(value, None)


def _allowed_openai_params(litellm_params: Mapping[str, object]) -> frozenset[str]:
    allowed: Final = as_object_list(litellm_params.get("allowed_openai_params")) or ()
    return frozenset(param for param in allowed if isinstance(param, str))


def _params_cohere_takes(
    model: str, optional_params: Mapping[str, object], litellm_params: Mapping[str, object]
) -> dict[str, object]:
    model_ids: Final = _cohere_model_ids(model, litellm_params.get("base_model"))
    allowed: Final = _allowed_openai_params(litellm_params)
    reasons: Final = {
        param: _rejection(param, value, optional_params, model_ids)
        for param, value in optional_params.items()
        if param not in allowed
    }
    rejected: Final = {param: reason for param, reason in reasons.items() if reason is not None}
    if rejected and not drops_unsupported_params(litellm_params.get("drop_params")):
        raise UnsupportedParamsError(
            message=(
                f"{' '.join(rejected.values())} "
                "To drop unsupported openai params from the call, set `litellm.drop_params = True`"
            ),
            status_code=400,
        )
    for param, reason in rejected.items():
        warn_dropped_reasoning(
            param, _logged_value(param, optional_params[param]), model_ids[0], reason, drop_params_gated=True
        )
    return {
        param: value
        for param, value in optional_params.items()
        if param not in rejected and not _translated_to_thinking(param, value)
    }


class CohereV2ChatConfig(OpenAIGPTConfig):
    """
    Configuration class for Cohere's API interface.

    Args:
        preamble (str, optional): When specified, the default Cohere preamble will be replaced with the provided one.
        chat_history (List[Dict[str, str]], optional): A list of previous messages between the user and the model.
        generation_id (str, optional): Unique identifier for the generated reply.
        response_id (str, optional): Unique identifier for the response.
        conversation_id (str, optional): An alternative to chat_history, creates or resumes a persisted conversation.
        prompt_truncation (str, optional): Dictates how the prompt will be constructed. Options: 'AUTO', 'AUTO_PRESERVE_ORDER', 'OFF'.
        connectors (List[Dict[str, str]], optional): List of connectors (e.g., web-search) to enrich the model's reply.
        search_queries_only (bool, optional): When true, the response will only contain a list of generated search queries.
        documents (List[Dict[str, str]], optional): A list of relevant documents that the model can cite.
        temperature (float, optional): A non-negative float that tunes the degree of randomness in generation.
        max_tokens (int, optional): The maximum number of tokens the model will generate as part of the response.
        k (int, optional): Ensures only the top k most likely tokens are considered for generation at each step.
        p (float, optional): Ensures that only the most likely tokens, with total probability mass of p, are considered for generation.
        frequency_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        presence_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        tools (List[Dict[str, str]], optional): A list of available tools (functions) that the model may suggest invoking.
        tool_results (List[Dict[str, Any]], optional): A list of results from invoking tools.
        seed (int, optional): A seed to assist reproducibility of the model's response.
    """

    preamble: str | None = None
    chat_history: list | None = None
    generation_id: str | None = None
    response_id: str | None = None
    conversation_id: str | None = None
    prompt_truncation: str | None = None
    connectors: list | None = None
    search_queries_only: bool | None = None
    documents: list | None = None
    temperature: int | None = None
    max_tokens: int | None = None
    k: int | None = None
    p: int | None = None
    frequency_penalty: int | None = None
    presence_penalty: int | None = None
    tools: list | None = None
    tool_results: list | None = None
    seed: int | None = None

    def __init__(
        self,
        preamble: str | None = None,
        chat_history: list | None = None,
        generation_id: str | None = None,
        response_id: str | None = None,
        conversation_id: str | None = None,
        prompt_truncation: str | None = None,
        connectors: list | None = None,
        search_queries_only: bool | None = None,
        documents: list | None = None,
        temperature: int | None = None,
        max_tokens: int | None = None,
        k: int | None = None,
        p: int | None = None,
        frequency_penalty: int | None = None,
        presence_penalty: int | None = None,
        tools: list | None = None,
        tool_results: list | None = None,
        seed: int | None = None,
    ) -> None:
        locals_: Final[Mapping[str, object]] = MappingProxyType(locals())
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

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
        return cohere_validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
        )

    def get_supported_openai_params(self, model: str) -> list[str]:
        model_ids: Final = _cohere_model_ids(model, None)
        json_mode: Final = (
            ["response_format"] if _any_model_supports(litellm.supports_response_schema, model_ids) else []
        )
        reasoning: Final = ["reasoning_effort"] if _any_model_supports(litellm.supports_reasoning, model_ids) else []
        return [
            "stream",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "tools",
            "tool_choice",
            "seed",
            "extra_headers",
            *json_mode,
            *reasoning,
        ]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "max_tokens" and "max_completion_tokens" not in non_default_params:
                optional_params["max_tokens"] = value
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "n":
                optional_params["num_generations"] = value
            if param == "top_p":
                optional_params["p"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "stop":
                optional_params["stop_sequences"] = [value] if isinstance(value, str) else value
            if param == "tools":
                optional_params["tools"] = value
            if param == "seed":
                optional_params["seed"] = value
        return {**optional_params, **_cohere_v2_native_params(non_default_params)}

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict,
    ) -> dict:
        """
        Cohere v2 chat api is in openai format, so we can use the openai transform request function to transform the request.
        """
        data: Final = super().transform_request(
            model,
            messages,
            _params_cohere_takes(model, optional_params, litellm_params),
            litellm_params,
            headers,
        )

        return data

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
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        try:
            raw_response_json: Final = raw_response.json()
        except Exception:
            raise CohereError(message=raw_response.text, status_code=raw_response.status_code)

        try:
            cohere_v2_chat_response: Final = CohereV2ChatResponse(**raw_response_json)
        except Exception:
            raise CohereError(message=raw_response.text, status_code=422)

        cohere_content: Final = cohere_v2_chat_response["message"].get("content", None)
        if cohere_content is not None:
            model_response.choices[0].message.content = "".join(
                [content.get("text", "") for content in cohere_content if content is not None]
            )

        ## ADD CITATIONS AS ANNOTATIONS
        annotations: list[ChatCompletionAnnotation] | None = None
        citations = None

        if "message" in cohere_v2_chat_response and "citations" in cohere_v2_chat_response["message"]:
            citations = cohere_v2_chat_response["message"]["citations"]

        if citations:
            annotations = self._translate_citations_to_openai_annotations(citations)

        ## Tool calling response
        cohere_tools_response: Final = cohere_v2_chat_response["message"].get("tool_calls", [])
        if cohere_tools_response is not None and cohere_tools_response != []:
            # convert cohere_tools_response to OpenAI response format
            tool_calls: Final[list[ChatCompletionToolCallChunk]] = []
            for index, tool in enumerate(cohere_tools_response):
                tool_call: ChatCompletionToolCallChunk = {
                    **tool,
                    "index": index,
                }
                tool_calls.append(tool_call)
            _message: Final = litellm.Message(
                tool_calls=tool_calls,
                content=None,
                annotations=annotations,
            )
            model_response.choices[0].message = _message
        else:
            if annotations:
                current_message: Final = model_response.choices[0].message
                current_message.annotations = annotations

        ## CALCULATING USAGE - use cohere `billed_units` for returning usage
        token_usage: Final = cohere_v2_chat_response["usage"].get("tokens", {})
        prompt_tokens: Final = token_usage.get("input_tokens", 0)
        completion_tokens: Final = token_usage.get("output_tokens", 0)

        model_response.choices[0].finish_reason = map_finish_reason(  # rebind-ok: ModelResponse is filled in place
            cohere_v2_chat_response.get("finish_reason", "COMPLETE")
        )
        model_response.created = int(time.time())
        model_response.model = model
        usage: Final = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ):
        return CohereV2ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete URL for Cohere v2 chat completion.
        The api_base should already include the full path.
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return CohereError(status_code=status_code, message=error_message)

    def _translate_citations_to_openai_annotations(self, citations: list[dict]) -> list[ChatCompletionAnnotation]:
        """
        Transform Cohere citations to OpenAI annotations format.

        Creates separate annotations for each source in a citation, allowing multiple
        annotations with the same start/end index if they reference different sources.

        Args:
            citations: List of Cohere citation objects with format:
                {
                    "start": int,
                    "end": int,
                    "text": str,
                    "sources": [
                        {
                            "type": "document",
                            "document": {
                                "title": str,
                                "snippet": str,
                                ...
                            },
                            "id": str
                        }
                    ]
                }

        Returns:
            List of OpenAI ChatCompletionAnnotation objects (one per source)
        """
        annotations: Final[list[ChatCompletionAnnotation]] = []

        for citation in citations:
            start_index = citation.get("start", 0)
            end_index = citation.get("end", 0)

            # Extract source information - loop through all sources
            sources = citation.get("sources", [])
            if not sources:
                continue

            # Create an annotation for each source
            for source in sources:
                if source.get("type") == "document" and "document" in source:
                    document = source["document"]
                    title = document.get("title", "")
                    url = source.get("url") or f"source:{source.get('id', 'unknown')}"

                    url_citation: ChatCompletionAnnotationURLCitation = {
                        "start_index": start_index,
                        "end_index": end_index,
                        "title": title,
                        "url": url,
                    }

                    annotation: ChatCompletionAnnotation = {
                        "type": "url_citation",
                        "url_citation": url_citation,
                    }

                    annotations.append(annotation)

        return annotations
