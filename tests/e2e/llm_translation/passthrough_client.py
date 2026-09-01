"""Client for LLM-translation e2e tests over the proxy's passthrough endpoints.

A passthrough request is sent in the PROVIDER's native format (Gemini
generateContent, Anthropic /v1/messages) to the proxy, which forwards it to the
provider and still logs a SpendLogs row (call_type="pass_through_endpoint"). The
litellm virtual key is passed as the provider key; the proxy swaps in the real env
credential. SpendLogs.request_id == the x-litellm-call-id response header. The
native request models are co-located here because only this suite uses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

from pydantic import BaseModel, Field
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from e2e_config import ws_base_url
from proxy_client import ProxyClient
from e2e_http import FileUploadForm, Headers, NoBody, Result, StreamingResponse
from models import ChatMessage


class JsonSchemaProperty(BaseModel):
    type: str


class JsonSchema(BaseModel):
    type: str
    properties: dict[str, JsonSchemaProperty]
    required: list[str]


class GeminiHeaders(Headers):
    x_goog_api_key: str = Field(serialization_alias="x-goog-api-key")
    content_type: str = Field(
        default="application/json", serialization_alias="Content-Type"
    )
    tags: str | None = None


class AnthropicHeaders(Headers):
    x_api_key: str = Field(serialization_alias="x-api-key")
    anthropic_version: str = Field(
        default="2023-06-01", serialization_alias="anthropic-version"
    )
    content_type: str = Field(
        default="application/json", serialization_alias="Content-Type"
    )
    tags: str | None = None


class VertexHeaders(Headers):
    # Only the litellm virtual key; the /vertex_ai passthrough mints the Vertex token
    # from the proxy's own service account (the deployment marked use_in_pass_through),
    # so no upstream Authorization bearer is sent from the client.
    x_litellm_api_key: str = Field(serialization_alias="x-litellm-api-key")
    content_type: str = Field(
        default="application/json", serialization_alias="Content-Type"
    )


class AltSseParams(BaseModel):
    alt: str = "sse"


class GeminiPart(BaseModel):
    text: str


class GeminiContent(BaseModel):
    role: str = "user"
    parts: list[GeminiPart]


class GeminiFunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: JsonSchema


class GeminiTool(BaseModel):
    function_declarations: list[GeminiFunctionDeclaration] = Field(
        serialization_alias="functionDeclarations"
    )


class GeminiGenerateBody(BaseModel):
    contents: list[GeminiContent]
    tools: list[GeminiTool] | None = None


class AnthropicTool(BaseModel):
    name: str
    description: str
    input_schema: JsonSchema


class AnthropicMessageBody(BaseModel):
    model: str
    max_tokens: int
    messages: list[ChatMessage]
    tools: list[AnthropicTool] | None = None
    stream: bool = False


class OpenAIChatBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    # Passthrough sends this body to OpenAI untranslated, so it has to satisfy
    # OpenAI's current contract directly: newer models reject `max_tokens` with
    # "Unsupported parameter: 'max_tokens' is not supported with this model. Use
    # 'max_completion_tokens' instead." litellm's drop_params/translation does not
    # apply on this route.
    max_completion_tokens: int = 64


class PassthroughFileObject(BaseModel):
    id: str
    object: str | None = None
    purpose: str | None = None
    filename: str | None = None
    bytes: int | None = None


class PassthroughFileDeleted(BaseModel):
    id: str
    deleted: bool


class PassthroughListEntry(BaseModel):
    id: str


class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ResponsesObject(BaseModel):
    id: str
    usage: ResponsesUsage | None = None


class ResponsesStreamEvent(BaseModel):
    """One SSE frame of a native Responses stream. Only the terminal frames carry a
    `response`, so it stays optional and the deltas validate as themselves."""

    type: str
    response: ResponsesObject | None = None


def completed_responses_object(result: StreamingResponse) -> ResponsesObject | None:
    """The `response.completed` frame's response object, or None if the stream never
    completed. Its `id` is what the spend row is keyed by on this route, and its
    usage is what the row is priced from."""
    events = (
        ResponsesStreamEvent.model_validate_json(payload)
        for payload in result.stream_events
    )
    completed = tuple(
        event.response
        for event in events
        if event.type == "response.completed" and event.response is not None
    )
    return completed[-1] if completed else None


class OpenAIResponsesBody(BaseModel):
    model: str
    input: str
    stream: bool = False


class OpenAIEmbeddingBody(BaseModel):
    model: str
    input: str


class WebsocketEnvelope(BaseModel):
    """The one field every provider event carries, so the first frame off a
    passthrough socket identifies itself without the suite parsing raw dicts."""

    type: str


class WebsocketHandshake(BaseModel):
    """What the proxy did with a websocket upgrade on a passthrough prefix.

    `rejected_status` is the HTTP status of a refused upgrade: a prefix carrying no
    websocket route answers 403, before any socket exists. `first_event_type` is the
    type of the first frame an accepted socket delivered, which is None when the
    provider waits for the client to speak first.
    """

    rejected_status: int | None = None
    first_event_type: str | None = None


class PassthroughBatchList(BaseModel):
    """OpenAI's own batch page, relayed verbatim. `object` is required so a body
    that is not an OpenAI list fails validation instead of passing vacuously."""

    object: str
    data: list[PassthroughListEntry]


def _tags_header(tags: list[str] | None) -> str | None:
    return ",".join(tags) if tags else None


@dataclass(frozen=True, slots=True)
class PassthroughClient:
    proxy: ProxyClient

    # ---- Gemini native passthrough (/gemini/v1beta/...) -----------------

    def gemini_generate(
        self,
        key: str,
        model: str,
        text: str,
        *,
        tools: list[GeminiTool] | None = None,
        tags: list[str] | None = None,
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            f"/gemini/v1beta/models/{model}:generateContent",
            headers=GeminiHeaders(x_goog_api_key=key, tags=_tags_header(tags)),
            json=GeminiGenerateBody(
                contents=[GeminiContent(parts=[GeminiPart(text=text)])], tools=tools
            ),
        )

    def gemini_stream(
        self, key: str, model: str, text: str, *, tags: list[str] | None = None
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            f"/gemini/v1beta/models/{model}:streamGenerateContent",
            headers=GeminiHeaders(x_goog_api_key=key, tags=_tags_header(tags)),
            json=GeminiGenerateBody(
                contents=[GeminiContent(parts=[GeminiPart(text=text)])]
            ),
            params=AltSseParams(),
            stream=True,
        )

    # ---- Vertex AI native passthrough (/vertex_ai/v1/projects/...) -------

    def vertex_generate(
        self, key: str, project: str, location: str, model: str, text: str
    ) -> StreamingResponse:
        path = (
            f"/vertex_ai/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent"
        )
        return self.proxy.transport.send(
            path,
            headers=VertexHeaders(x_litellm_api_key=key),
            json=GeminiGenerateBody(
                contents=[GeminiContent(parts=[GeminiPart(text=text)])]
            ),
        )

    # ---- Anthropic native passthrough (/anthropic/v1/messages) ----------

    def anthropic_message(
        self,
        key: str,
        model: str,
        text: str,
        *,
        max_tokens: int = 64,
        tools: list[AnthropicTool] | None = None,
        stream: bool = False,
        tags: list[str] | None = None,
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            "/anthropic/v1/messages",
            headers=AnthropicHeaders(x_api_key=key, tags=_tags_header(tags)),
            json=AnthropicMessageBody(
                model=model,
                max_tokens=max_tokens,
                messages=[ChatMessage(role="user", content=text)],
                tools=tools,
                stream=stream,
            ),
            stream=stream,
        )

    # ---- OpenAI file/batch routes under /openai_passthrough -------------
    #
    # Relayed to OpenAI untouched, which is the whole point of the prefix: the
    # customer opts out of the gateway's managed-file handling here.

    def openai_passthrough_upload_file(
        self, key: str, *, content: bytes, filename: str
    ) -> Result[PassthroughFileObject]:
        return self.proxy.transport.upload(
            "/openai_passthrough/v1/files",
            headers=self.proxy.transport.bearer(key),
            form=FileUploadForm(purpose="batch"),
            filename=filename,
            content=content,
            response_type=PassthroughFileObject,
        )

    def openai_passthrough_delete_file(
        self, key: str, file_id: str
    ) -> Result[PassthroughFileDeleted]:
        return self.proxy.transport.delete(
            f"/openai_passthrough/v1/files/{file_id}",
            headers=self.proxy.transport.bearer(key),
            json=NoBody(),
            response_type=PassthroughFileDeleted,
        )

    def openai_passthrough_list_batches(self, key: str) -> Result[PassthroughBatchList]:
        return self.proxy.transport.get(
            "/openai_passthrough/v1/batches",
            headers=self.proxy.transport.bearer(key),
            params=NoBody(),
            response_type=PassthroughBatchList,
        )

    # ---- OpenAI inference routes under /openai_passthrough -------------
    #
    # Relayed to OpenAI verbatim, but still costed by the gateway: the customer
    # budgets against this traffic, so a 200 that logs no spend is money the
    # gateway never sees.

    def openai_passthrough_responses(
        self, key: str, model: str, text: str, *, stream: bool = False
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            "/openai_passthrough/v1/responses",
            headers=self.proxy.transport.bearer(key),
            json=OpenAIResponsesBody(model=model, input=text, stream=stream),
            stream=stream,
        )

    def openai_passthrough_embed(
        self, key: str, model: str, text: str
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            "/openai_passthrough/v1/embeddings",
            headers=self.proxy.transport.bearer(key),
            json=OpenAIEmbeddingBody(model=model, input=text),
        )

    def openai_chat(
        self, key: str, model: str, text: str, *, max_completion_tokens: int = 64
    ) -> StreamingResponse:
        return self.proxy.transport.send(
            "/openai/v1/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=OpenAIChatBody(
                model=model,
                max_completion_tokens=max_completion_tokens,
                messages=[ChatMessage(role="user", content=text)],
            ),
        )

    # ---- OpenAI websocket passthrough ----------------------------------
    #
    # The same prefixes over an upgrade instead of a POST, for the provider APIs
    # that only speak websocket (realtime, responses.connect).

    def openai_passthrough_websocket(
        self,
        key: str,
        path: str,
        *,
        model: str | None = None,
        open_timeout: float = 30.0,
        first_event_timeout: float = 30.0,
    ) -> WebsocketHandshake:
        query = f"?{urlencode({'model': model})}" if model is not None else ""
        try:
            connection = connect(
                f"{ws_base_url()}{path}{query}",
                additional_headers={"Authorization": f"Bearer {key}"},
                open_timeout=open_timeout,
            )
        except InvalidStatus as rejected:
            return WebsocketHandshake(rejected_status=rejected.response.status_code)
        with connection:
            try:
                frame = connection.recv(timeout=first_event_timeout)
            except TimeoutError:
                return WebsocketHandshake()
            text = frame.decode("utf-8") if isinstance(frame, bytes) else frame
            return WebsocketHandshake(
                first_event_type=WebsocketEnvelope.model_validate_json(text).type
            )


def build_client(proxy: ProxyClient) -> PassthroughClient:
    return PassthroughClient(proxy=proxy)
