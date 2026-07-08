"""Client for the non-chat inference endpoints (responses, messages, rerank,
embeddings, audio speech, image generation).

Each test registers the deployment it needs through /model/new (deleted on
teardown), so nothing is hardcoded into the gateway config, then drives the
endpoint with `send` and parses the provider-native body with a suite-local model
so the assertion is on real content, not just a 200.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from e2e_gateway import Gateway, build_gateway
from e2e_http import StreamingResponse
from models import ChatMessage, LiteLLMParamsBody


class ResponsesRequest(BaseModel):
    model: str
    input: str
    instructions: str | None = None


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[ChatMessage]


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


class AnthropicContentBlock(BaseModel):
    type: str | None = None
    text: str | None = None


class MessagesResult(BaseModel):
    id: str | None = None
    role: str | None = None
    model: str | None = None
    content: list[AnthropicContentBlock] = []

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
    gateway: Gateway

    def create_model(self, model_name: str, litellm_params: LiteLLMParamsBody) -> str:
        return self.gateway.create_model(model_name, litellm_params)

    def delete_model(self, model_id: str) -> None:
        self.gateway.delete_model(model_id)

    def _send(self, path: str, key: str, body: BaseModel) -> StreamingResponse:
        return self.gateway.transport.send(
            path, headers=self.gateway.transport.bearer(key), json=body
        )

    def responses(self, key: str, model: str, text: str) -> StreamingResponse:
        return self._send(
            "/v1/responses",
            key,
            ResponsesRequest(
                model=model, input=text, instructions="You are a helpful assistant"
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


def build_endpoints_client() -> EndpointsClient:
    return EndpointsClient(gateway=build_gateway())
