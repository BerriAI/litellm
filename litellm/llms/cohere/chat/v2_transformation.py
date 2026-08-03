import time
from collections.abc import AsyncIterator, Iterator
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import httpx

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
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

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

#: Params Cohere v2 accepts under their OpenAI names.
PASSTHROUGH_PARAMS = frozenset(
    {
        "stream",
        "temperature",
        "frequency_penalty",
        "presence_penalty",
        "tools",
        "seed",
        "logprobs",
        "response_format",
        "thinking",
    }
)

#: Cohere only accepts these two; `auto` is what it does when the field is absent.
TOOL_CHOICE_BY_OPENAI_NAME = MappingProxyType({"required": "REQUIRED", "none": "NONE"})

#: OpenAI `reasoning_effort` -> Cohere `thinking`. Unlisted efforts stay unset.
THINKING_BY_REASONING_EFFORT = MappingProxyType(
    {
        "none": MappingProxyType({"type": "disabled"}),
        "minimal": MappingProxyType(
            {"type": "enabled", "token_budget": DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET}
        ),
        "low": MappingProxyType({"type": "enabled", "token_budget": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET}),
        "medium": MappingProxyType(
            {"type": "enabled", "token_budget": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET}
        ),
        "high": MappingProxyType({"type": "enabled", "token_budget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET}),
    }
)


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
        locals_ = locals()
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
        supported_params = [
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
            "logprobs",
            "response_format",
            "extra_headers",
        ]

        if self._is_reasoning_model(model):
            supported_params.extend(("thinking", "reasoning_effort"))

        return supported_params

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Cohere exposes ``thinking`` on the command-a-reasoning family.

        Read from the id: no Cohere entry in the model map carries
        ``supports_reasoning``, so there is nothing to consult.
        """
        return "command-a-reasoning" in model.split("/")[-1].lower()

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param in PASSTHROUGH_PARAMS:
                optional_params[param] = value
            if param == "max_tokens" and "max_completion_tokens" not in non_default_params:
                optional_params["max_tokens"] = value
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "n":
                optional_params["num_generations"] = value
            if param == "top_p":
                optional_params["p"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "tool_choice" and isinstance(value, str):
                mapped_tool_choice = TOOL_CHOICE_BY_OPENAI_NAME.get(value)
                if mapped_tool_choice is not None:
                    optional_params["tool_choice"] = mapped_tool_choice
            if param == "reasoning_effort" and isinstance(value, str):
                thinking = THINKING_BY_REASONING_EFFORT.get(value)
                if thinking is not None:
                    optional_params["thinking"] = thinking
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
        # `top_k` is a provider-specific kwarg, so it bypasses `map_openai_params`
        # and arrives under its OpenAI name. Cohere calls the field `k`.
        if "top_k" in optional_params:
            optional_params["k"] = optional_params.pop("top_k")

        data = super().transform_request(model, messages, optional_params, litellm_params, headers)

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
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise CohereError(message=raw_response.text, status_code=raw_response.status_code)

        try:
            cohere_v2_chat_response = CohereV2ChatResponse(**raw_response_json)  # type: ignore
        except Exception:
            raise CohereError(message=raw_response.text, status_code=422)

        cohere_content = cohere_v2_chat_response["message"].get("content", None)
        if cohere_content is not None:
            model_response.choices[0].message.content = "".join(  # type: ignore
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
        cohere_tools_response = cohere_v2_chat_response["message"].get("tool_calls", [])
        if cohere_tools_response is not None and cohere_tools_response != []:
            # convert cohere_tools_response to OpenAI response format
            tool_calls: list[ChatCompletionToolCallChunk] = []
            for index, tool in enumerate(cohere_tools_response):
                tool_call: ChatCompletionToolCallChunk = {
                    **tool,  # type: ignore
                    "index": index,
                }
                tool_calls.append(tool_call)
            _message = litellm.Message(
                tool_calls=tool_calls,
                content=None,
                annotations=annotations,
            )
            model_response.choices[0].message = _message  # type: ignore
        else:
            if annotations:
                current_message = model_response.choices[0].message  # type: ignore
                current_message.annotations = annotations

        ## CALCULATING USAGE - use cohere `billed_units` for returning usage
        token_usage = cohere_v2_chat_response["usage"].get("tokens", {})
        prompt_tokens = token_usage.get("input_tokens", 0)
        completion_tokens = token_usage.get("output_tokens", 0)

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
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
        annotations: list[ChatCompletionAnnotation] = []

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
