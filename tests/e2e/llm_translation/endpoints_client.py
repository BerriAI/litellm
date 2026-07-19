"""Client for the non-chat inference endpoints (responses, messages, rerank,
embeddings, audio speech, image generation).

Each test registers the deployment it needs through /model/new (deleted on
teardown), so nothing is hardcoded into the gateway config, then drives the
endpoint with `send` and parses the provider-native body with a suite-local model
so the assertion is on real content, not just a 200.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from proxy_client import ProxyClient
from e2e_http import StreamingResponse
from models import ChatMessage, LiteLLMParamsBody


class FunctionParameterProperty(BaseModel):
    type: str
    description: str | None = None


class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, FunctionParameterProperty]
    required: list[str] = []


class ResponsesFunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: FunctionParameters


class ResponsesInputTextPart(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class ResponsesInputImagePart(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str


ResponsesInputContentPart = ResponsesInputTextPart | ResponsesInputImagePart


class ResponsesInputMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: list[ResponsesInputContentPart]


ResponsesInput = str | list[ResponsesInputMessage]


class ResponsesRequest(BaseModel):
    model: str
    input: ResponsesInput
    instructions: str | None = None
    stream: bool = False
    tools: list[ResponsesFunctionTool] | None = None


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[ChatMessage]


class CacheControl(BaseModel):
    type: str = "ephemeral"


class TextBlock(BaseModel):
    type: str = "text"
    text: str
    cache_control: CacheControl | None = None


class RichMessage(BaseModel):
    role: str
    content: list[TextBlock]


class RichMessagesRequest(BaseModel):
    model: str
    max_tokens: int = 64
    system: list[TextBlock]
    messages: list[RichMessage]


class EmbeddingsRequest(BaseModel):
    model: str
    input: str


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str


class ImageRequest(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"


class ResponsesOutputContent(BaseModel):
    type: str | None = None
    text: str | None = None


class ResponsesOutputItem(BaseModel):
    type: str | None = None
    content: list[ResponsesOutputContent] = []
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None


class ResponsesResult(BaseModel):
    id: str | None = None
    status: str | None = None
    model: str | None = None
    output: list[ResponsesOutputItem] = []

    @property
    def text(self) -> str:
        return "".join(
            content.text or "" for item in self.output for content in item.content
        )

    @property
    def function_calls(self) -> tuple[ResponsesOutputItem, ...]:
        return tuple(
            item
            for item in self.output
            if item.type == "function_call"
            and item.name is not None
            and item.arguments is not None
        )


class ResponsesStreamEvent(BaseModel):
    event_id: str | None = None


class ResponsesStreamEventType(BaseModel):
    type: str


class ResponsesOutputTextDeltaEvent(ResponsesStreamEvent):
    type: Literal["response.output_text.delta"]
    delta: str


class AnthropicContentBlock(BaseModel):
    type: str | None = None
    text: str | None = None


class MessagesUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResult(BaseModel):
    id: str | None = None
    role: str | None = None
    model: str | None = None
    content: list[AnthropicContentBlock] = []
    usage: MessagesUsage = MessagesUsage()

    @property
    def text(self) -> str:
        return "".join(block.text or "" for block in self.content)


class EmbeddingItem(BaseModel):
    embedding: list[float] = []


class EmbeddingsResult(BaseModel):
    data: list[EmbeddingItem] = []

    @property
    def first_vector(self) -> tuple[float, ...]:
        return tuple(self.data[0].embedding) if self.data else ()


class RerankItem(BaseModel):
    index: int | None = None
    relevance_score: float | None = None


class RerankResult(BaseModel):
    results: list[RerankItem] = []


class ImageItem(BaseModel):
    url: str | None = None
    b64_json: str | None = None


class ImagesResult(BaseModel):
    data: list[ImageItem] = []


@dataclass(frozen=True, slots=True)
class EndpointsClient:
    proxy: ProxyClient

    def create_model(self, model_name: str, litellm_params: LiteLLMParamsBody) -> str:
        return self.proxy.create_model(model_name, litellm_params)

    def delete_model(self, model_id: str) -> None:
        self.proxy.delete_model(model_id)

    def _send(
        self, path: str, key: str, body: BaseModel, *, stream: bool = False
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            path,
            headers=self.proxy.transport.bearer(key),
            json=body,
            stream=stream,
        )

    def responses(
        self, key: str, model: str, text: str, *, stream: bool = False
    ) -> StreamingResponse:
        return self._send(
            "/v1/responses",
            key,
            ResponsesRequest(
                model=model,
                input=text,
                instructions="You are a helpful assistant",
                stream=stream,
            ),
            stream=stream,
        )

    def responses_vision(
        self, key: str, model: str, text: str, image_url: str
    ) -> StreamingResponse:
        return self._send(
            "/v1/responses",
            key,
            ResponsesRequest(
                model=model,
                input=[
                    ResponsesInputMessage(
                        content=[
                            ResponsesInputTextPart(text=text),
                            ResponsesInputImagePart(image_url=image_url),
                        ]
                    )
                ],
                instructions="You are a helpful assistant",
            ),
        )

    def responses_with_tools(
        self, key: str, model: str, text: str, tools: list[ResponsesFunctionTool]
    ) -> StreamingResponse:
        return self._send(
            "/v1/responses",
            key,
            ResponsesRequest(
                model=model,
                input=text,
                instructions="You are a helpful assistant",
                tools=tools,
            ),
        )

    def messages(
        self, key: str, model: str, text: str, *, max_tokens: int = 64
    ) -> StreamingResponse:
        return self._send(
            "/v1/messages",
            key,
            MessagesRequest(
                model=model,
                max_tokens=max_tokens,
                messages=[ChatMessage(role="user", content=text)],
            ),
        )

    def embeddings(self, key: str, model: str, text: str) -> StreamingResponse:
        return self._send("/embeddings", key, EmbeddingsRequest(model=model, input=text))

    def rerank(
        self, key: str, model: str, query: str, documents: list[str], top_n: int
    ) -> StreamingResponse:
        return self._send(
            "/v1/rerank",
            key,
            RerankRequest(model=model, query=query, documents=documents, top_n=top_n),
        )

    def audio_speech(
        self, key: str, model: str, text: str, *, voice: str = "alloy"
    ) -> StreamingResponse:
        return self._send(
            "/v1/audio/speech", key, SpeechRequest(model=model, input=text, voice=voice)
        )

    def images(self, key: str, model: str, prompt: str) -> StreamingResponse:
        return self._send(
            "/v1/images/generations", key, ImageRequest(model=model, prompt=prompt)
        )


def build_endpoints_client(proxy: ProxyClient) -> EndpointsClient:
    return EndpointsClient(proxy=proxy)
