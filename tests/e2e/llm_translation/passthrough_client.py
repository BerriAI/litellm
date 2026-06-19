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

from pydantic import BaseModel, Field

from e2e_gateway import Gateway, build_gateway
from e2e_http import Headers, StreamingResponse
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


def _tags_header(tags: list[str] | None) -> str | None:
    return ",".join(tags) if tags else None


@dataclass(frozen=True, slots=True)
class PassthroughClient:
    gateway: Gateway

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
        return self.gateway.transport.send(
            f"/gemini/v1beta/models/{model}:generateContent",
            headers=GeminiHeaders(x_goog_api_key=key, tags=_tags_header(tags)),
            json=GeminiGenerateBody(
                contents=[GeminiContent(parts=[GeminiPart(text=text)])], tools=tools
            ),
        )

    def gemini_stream(
        self, key: str, model: str, text: str, *, tags: list[str] | None = None
    ) -> StreamingResponse:
        return self.gateway.transport.send(
            f"/gemini/v1beta/models/{model}:streamGenerateContent",
            headers=GeminiHeaders(x_goog_api_key=key, tags=_tags_header(tags)),
            json=GeminiGenerateBody(
                contents=[GeminiContent(parts=[GeminiPart(text=text)])]
            ),
            params=AltSseParams(),
            stream=True,
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
        return self.gateway.transport.send(
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


def build_client() -> PassthroughClient:
    return PassthroughClient(gateway=build_gateway())
