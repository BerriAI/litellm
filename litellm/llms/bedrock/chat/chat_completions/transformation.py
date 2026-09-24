"""
Native OpenAI Chat Completions on Amazon Bedrock Runtime.

AWS serves this surface at
``https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions``
for the models whose price-map ``supported_endpoints`` lists ``/v1/chat/completions``
(Grok 4.6, gpt-oss, the GPT-5.6 family): chat completions stay chat completions
instead of being rewritten to Converse.

Usage: model="us.xai.grok-4.6", model="bedrock/openai.gpt-oss-20b-1:0" or
model="bedrock/global.openai.gpt-5.6-sol". Explicit ``bedrock/converse/...``
still uses Converse, and so does a request that needs a Converse-only feature
(``bedrock_request_needs_converse`` in ``common_utils``).
"""

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from typing_extensions import assert_never

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError, split_bedrock_region_path
from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream

if TYPE_CHECKING:
    import tiktoken

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

REASONING_OPEN_TAG: Final = "<reasoning>"
REASONING_CLOSE_TAG: Final = "</reasoning>"

CHAT_COMPLETIONS_REFUSED_PARAMS_BY_FAMILY: Final = MappingProxyType(
    {
        "openai.gpt-5": frozenset(("frequency_penalty", "presence_penalty", "stop", "logprobs", "top_logprobs")),
        "openai.gpt-oss": frozenset(("logit_bias",)),
        "xai.": frozenset(("frequency_penalty", "presence_penalty")),
    }
)


def chat_completions_params_refused_for(model: str) -> frozenset[str]:
    """The OpenAI params AWS's Chat Completions endpoint rejects for this model whatever else the request says.

    Each family answers them with a 400 (GPT-5.6, gpt-oss) or a 503 (Grok), where Converse dropped the same
    params under ``drop_params``, so the native config leaves them out of its supported list and the usual
    drop-or-raise handling applies before the request reaches AWS.
    """
    model_id: Final = split_bedrock_region_path(model)[1]
    return frozenset().union(
        *(refused for family, refused in CHAT_COMPLETIONS_REFUSED_PARAMS_BY_FAMILY.items() if family in model_id)
    )


def _held_close_tag_prefix(text: str) -> int:
    return next(
        (
            size
            for size in range(min(len(text), len(REASONING_CLOSE_TAG) - 1), 0, -1)
            if REASONING_CLOSE_TAG.startswith(text[-size:])
        ),
        0,
    )


@dataclass(frozen=True, slots=True)
class ReasoningTagSplitter:
    """
    The same split for a stream of content deltas, where a tag can arrive across chunks.

    ``feed`` returns the next state plus the reasoning and content text the delta contributes;
    ``flush`` releases what the stream ended on before a tag resolved.
    """

    phase: Literal["start", "reasoning", "after_close", "content"] = "start"
    pending: str = ""

    def feed(self, text: str) -> tuple["ReasoningTagSplitter", str, str]:
        match self.phase:
            case "content":
                return self, "", text
            case "after_close":
                content: Final = text.lstrip()
                return (replace(self, phase="content") if content else self), "", content
            case "start":
                return self._feed_start(self.pending + text)
            case "reasoning":
                return self._feed_reasoning(self.pending + text)
            case _:
                assert_never(self.phase)

    def _feed_start(self, buffered: str) -> tuple["ReasoningTagSplitter", str, str]:
        if buffered.startswith(REASONING_OPEN_TAG):
            return replace(self, phase="reasoning", pending="")._feed_reasoning(buffered[len(REASONING_OPEN_TAG) :])
        if REASONING_OPEN_TAG.startswith(buffered):
            return replace(self, pending=buffered), "", ""
        return replace(self, phase="content", pending=""), "", buffered

    def _feed_reasoning(self, buffered: str) -> tuple["ReasoningTagSplitter", str, str]:
        close_at: Final = buffered.find(REASONING_CLOSE_TAG)
        if close_at >= 0:
            after_close: Final = replace(self, phase="after_close", pending="")
            next_state, _, content = after_close.feed(buffered[close_at + len(REASONING_CLOSE_TAG) :])
            return next_state, buffered[:close_at], content
        held: Final = _held_close_tag_prefix(buffered)
        return replace(self, pending=buffered[len(buffered) - held :]), buffered[: len(buffered) - held], ""

    def flush(self) -> tuple["ReasoningTagSplitter", str, str]:
        drained: Final = replace(self, phase="content", pending="")
        if self.phase == "reasoning":
            return drained, self.pending, ""
        return drained, "", self.pending


def _split_streamed_content(
    splitter: ReasoningTagSplitter, content: str | None, finished: bool
) -> tuple[ReasoningTagSplitter, str, str]:
    fed_state, fed_reasoning, fed_content = splitter.feed(content or "")
    if not finished:
        return fed_state, fed_reasoning, fed_content
    drained, flushed_reasoning, flushed_content = fed_state.flush()
    return drained, fed_reasoning + flushed_reasoning, fed_content + flushed_content


def split_reasoning_tag(content: str) -> tuple[str | None, str]:
    """
    Split gpt-oss's inline ``<reasoning>...</reasoning>`` prefix out of a complete message.

    Runs the streaming splitter over the whole message, so a streamed and a non-streamed
    response to the same completion split identically. Returns ``(None, content)`` when the
    message does not start with the tag.
    """
    _, reasoning, body = _split_streamed_content(ReasoningTagSplitter(), content, finished=True)
    return reasoning or None, body


class BedrockRuntimeChatCompletionsStreamingHandler(OpenAIChatCompletionStreamingHandler):
    """OpenAI chunk parsing plus the ``<reasoning>`` split, tracked per choice index."""

    def __init__(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> None:
        super().__init__(streaming_response=streaming_response, sync_stream=sync_stream, json_mode=json_mode)
        self._splitters: Mapping[int, ReasoningTagSplitter] = MappingProxyType({})

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:  # mutable-ok: BaseModelResponseIterator signature
        parsed: Final = super().chunk_parser(chunk)
        for choice in parsed.choices:
            next_state, reasoning, content = _split_streamed_content(
                self._splitters.get(choice.index, ReasoningTagSplitter()),
                choice.delta.content,
                choice.finish_reason is not None,
            )
            self._splitters = MappingProxyType({**self._splitters, choice.index: next_state})
            if reasoning:
                choice.delta.reasoning_content = f"{getattr(choice.delta, 'reasoning_content', None) or ''}{reasoning}"
            if content or choice.delta.content is not None:
                choice.delta.content = content
        return parsed


def with_max_completion_tokens(params: Mapping[str, object]) -> Mapping[str, object]:
    """
    Send the caller's ``max_tokens`` as ``max_completion_tokens``.

    Every model on this surface accepts ``max_completion_tokens`` and the GPT-5.6 family
    rejects ``max_tokens``; an explicit ``max_completion_tokens`` wins when both are set.
    """
    if "max_tokens" not in params:
        return params
    return MappingProxyType(
        {
            key: value
            for key, value in (("max_completion_tokens", params["max_tokens"]), *params.items())
            if key != "max_tokens"
        }
    )


class AmazonBedrockRuntimeChatCompletionsConfig(OpenAILikeChatConfig):
    def __init__(self, aws_signer: BaseAWSLLM | None = None) -> None:
        super().__init__()
        self._aws_signer: Final = aws_signer or BaseAWSLLM()

    @property
    def custom_llm_provider(self) -> str | None:
        return "bedrock"

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: BaseConfig signature
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        stream: bool | None = None,
    ) -> str:
        if api_base is not None and "chat/completions" in api_base:
            return api_base.rstrip("/")
        aws_region_name: Final = self._aws_signer._get_aws_region_name(  # pyright: ignore[reportPrivateUsage]  # BaseAWSLLM has no public region resolver
            optional_params=self._params_with_region_from_path(optional_params, model), model=model
        )
        endpoint_url, _ = self._aws_signer.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.get("aws_bedrock_runtime_endpoint"),
            aws_region_name=aws_region_name,
        )
        base: Final = endpoint_url.rstrip("/")
        if base.endswith("/openai/v1/chat/completions"):
            return base
        if base.endswith("/openai/v1"):
            return f"{base}/chat/completions"
        return f"{base}/openai/v1/chat/completions"

    def _params_with_region_from_path(
        self, optional_params: dict, model: str | None
    ) -> dict:  # mutable-ok: BaseAWSLLM's region resolver and signer take a plain dict
        region_from_path, _ = split_bedrock_region_path(model or "")
        if region_from_path is None or optional_params.get("aws_region_name") is not None:
            return optional_params
        return {**optional_params, "aws_region_name": region_from_path}  # mutable-ok: BaseAWSLLM takes a plain dict

    def sign_request(
        self,
        headers: dict,  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        request_data: dict,  # mutable-ok: BaseConfig signature
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:  # mutable-ok: BaseConfig signature
        return self._aws_signer._sign_request(  # pyright: ignore[reportPrivateUsage]  # BaseAWSLLM has no public signer
            service_name="bedrock",
            headers=headers,
            optional_params=self._params_with_region_from_path(optional_params, model),
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = False,
    ) -> dict:  # mutable-ok: BaseConfig signature
        mapped: Final = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
            replace_max_completion_tokens_with_max_tokens=replace_max_completion_tokens_with_max_tokens,
        )
        return dict(with_max_completion_tokens(mapped))  # mutable-ok: get_optional_params keeps filling this dict

    def _inference_params(
        self, optional_params: Mapping[str, object]
    ) -> dict[str, object]:  # mutable-ok: BaseConfig signature of transform_request
        return {  # mutable-ok: OpenAILikeChatConfig.transform_request takes a plain dict
            key: value
            for key, value in optional_params.items()
            if key not in self._aws_signer.aws_authentication_params
        }

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        headers: dict,  # mutable-ok: BaseConfig signature
    ) -> dict:  # mutable-ok: BaseConfig signature
        return super().transform_request(
            model=split_bedrock_region_path(model)[1],
            messages=messages,
            optional_params=self._inference_params(optional_params),
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        headers: dict,  # mutable-ok: BaseConfig signature
    ) -> dict:  # mutable-ok: BaseConfig signature
        return await super().async_transform_request(
            model=split_bedrock_region_path(model)[1],
            messages=messages,
            optional_params=self._inference_params(optional_params),
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,  # mutable-ok: BaseConfig signature
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        encoding: "tiktoken.Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        response: Final = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        for choice in response.choices:
            if not isinstance(choice, Choices) or not isinstance(choice.message.content, str):
                continue
            reasoning, content = split_reasoning_tag(choice.message.content)
            if reasoning is not None:
                choice.message.reasoning_content = (
                    f"{getattr(choice.message, 'reasoning_content', None) or ''}{reasoning}"
                )
            choice.message.content = content
        return response

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseConfig signature
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseConfig signature
        validated: Final = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        project_id: Final = litellm_params.get("aws_bedrock_project_id")
        if not project_id:
            return validated
        return {**validated, "OpenAI-Project": project_id}  # mutable-ok: BaseConfig signature returns a dict

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseConfig signature
        refused: Final = frozenset(("n", *chat_completions_params_refused_for(model)))
        base_params: Final = tuple(
            param for param in super().get_supported_openai_params(model) if param not in refused
        )
        reasoning_param: Final = (
            ("reasoning_effort",)
            if "reasoning_effort" not in base_params
            and litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider)
            else ()
        )
        return [*base_params, *reasoning_param]  # mutable-ok: BaseConfig signature returns a list

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> BedrockRuntimeChatCompletionsStreamingHandler:
        return BedrockRuntimeChatCompletionsStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
