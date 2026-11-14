import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.cohere import CohereV2ChatResponse
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
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


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    model_dump: Final = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped: Final = model_dump(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _sanitize_cohere_v2_tool_call(tool_call: object) -> Mapping[str, object] | None:
    raw: Final = _as_mapping(tool_call)
    if raw is None:
        return None
    fn_raw: Final = _as_mapping(raw.get("function"))
    sanitized_fn: Final = (
        {k: fn_raw[k] for k in ("name", "arguments") if k in fn_raw and fn_raw[k] is not None}
        if fn_raw is not None
        else raw.get("function")
    )
    return {
        **({"id": raw["id"]} if raw.get("id") is not None else {}),
        **({"type": raw["type"]} if raw.get("type") is not None else {}),
        **({"function": sanitized_fn} if sanitized_fn is not None else {}),
    }


def _is_thinking_content_item(item: object) -> bool:
    mapped: Final = _as_mapping(item)
    return mapped is not None and mapped.get("type") in ("thinking", "redacted_thinking")


def _text_from_content_item(item: object) -> str:
    if isinstance(item, str):
        return item
    mapped: Final = _as_mapping(item)
    if mapped is not None and mapped.get("type") == "text":
        text: Final = mapped.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_assistant_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return "".join(_text_from_content_item(item) for item in content)
    return ""


def _strip_thinking_from_content(content: object) -> object:
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return [item for item in content if not _is_thinking_content_item(item)]
    return content


def _sanitize_cohere_v2_message(message: object) -> object:
    raw: Final = _as_mapping(message)
    if raw is None:
        return message
    role: Final = raw.get("role")
    if role == "assistant":
        raw_tool_calls: Final = raw.get("tool_calls")
        tool_calls: Final = (
            [
                sanitized
                for tc in raw_tool_calls
                if (sanitized := _sanitize_cohere_v2_tool_call(tc)) is not None
            ]
            if isinstance(raw_tool_calls, Sequence) and not isinstance(raw_tool_calls, (str, bytes))
            else []
        )
        raw_content: Final = raw.get("content")
        explicit_plan: Final = raw.get("tool_plan")
        citations: Final = raw.get("citations")
        if tool_calls:
            tool_plan: Final = (
                explicit_plan
                if isinstance(explicit_plan, str) and explicit_plan
                else _extract_assistant_text(raw_content)
            )
            return {
                "role": "assistant",
                "tool_calls": tool_calls,
                **({"tool_plan": tool_plan} if tool_plan else {}),
                **({"citations": citations} if citations is not None else {}),
            }
        cleaned_content: Final = _strip_thinking_from_content(raw_content)
        return {
            "role": "assistant",
            **({"content": cleaned_content} if cleaned_content is not None else {}),
            **({"tool_plan": explicit_plan} if explicit_plan is not None else {}),
            **({"citations": citations} if citations is not None else {}),
        }
    if role == "tool":
        return {
            "role": "tool",
            **({"tool_call_id": raw["tool_call_id"]} if raw.get("tool_call_id") is not None else {}),
            **({"content": raw["content"]} if raw.get("content") is not None else {}),
        }
    return {
        **({"role": role} if role is not None else {}),
        **({"content": raw["content"]} if raw.get("content") is not None else {}),
    }


def _sanitize_cohere_v2_request_data(data: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: BaseConfig transform_request returns a mutable dict
    raw_messages: Final = data.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        return dict(data)
    return {
        **data,
        "messages": [_sanitize_cohere_v2_message(msg) for msg in raw_messages],
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
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
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
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Cohere v2 chat api is in openai format, so we can use the openai transform request function to transform the request.
        """
        data: Final = super().transform_request(model, messages, optional_params, litellm_params, headers)

        return _sanitize_cohere_v2_request_data(data)

    async def async_transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: matches OpenAIGPTConfig.async_transform_request signature
        optional_params: dict,  # mutable-ok: matches OpenAIGPTConfig.async_transform_request signature
        litellm_params: dict,  # mutable-ok: matches OpenAIGPTConfig.async_transform_request signature
        headers: dict,  # mutable-ok: matches OpenAIGPTConfig.async_transform_request signature
    ) -> dict:  # mutable-ok: matches OpenAIGPTConfig.async_transform_request signature
        data: Final = await super().async_transform_request(
            model, messages, optional_params, litellm_params, headers
        )

        return _sanitize_cohere_v2_request_data(data)

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
            tool_calls: Final = [{**tool} for tool in cohere_tools_response]
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
