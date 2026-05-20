"""
LiteLLM MCP Server implementation.

Creates an MCP server that exposes LiteLLM's core functionality as tools:
- chat_completion: Chat with any LLM via LiteLLM
- embedding: Generate embeddings
- image_generation: Generate images
- text_completion: Text completions
- transcription: Audio transcription
- rerank: Rerank documents
- list_models: List available models
"""

import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from litellm.litellm_mcp_server.tool_schemas import (
    CHAT_COMPLETION_SCHEMA,
    EMBEDDINGS_SCHEMA,
    IMAGE_GENERATION_SCHEMA,
    LIST_MODELS_SCHEMA,
    RERANK_SCHEMA,
    TEXT_COMPLETION_SCHEMA,
    TRANSCRIPTION_SCHEMA,
)

logger = logging.getLogger(__name__)

TOOL_NAME_CHAT_COMPLETION = "litellm_chat_completion"
TOOL_NAME_EMBEDDING = "litellm_embedding"
TOOL_NAME_IMAGE_GENERATION = "litellm_image_generation"
TOOL_NAME_TEXT_COMPLETION = "litellm_text_completion"
TOOL_NAME_TRANSCRIPTION = "litellm_transcription"
TOOL_NAME_RERANK = "litellm_rerank"
TOOL_NAME_LIST_MODELS = "litellm_list_models"


def create_mcp_server() -> Server:
    """Create and configure a LiteLLM MCP server with all available tools.

    Returns:
        A configured ``mcp.server.Server`` instance with LiteLLM tools
        registered.
    """
    server = Server(
        name="litellm-mcp-server",
        version="1.0.0",
    )

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name=TOOL_NAME_CHAT_COMPLETION,
                description=(
                    "Send a chat completion request to any LLM via LiteLLM's "
                    "unified API. Supports 100+ providers including OpenAI, "
                    "Anthropic, Azure, Google, AWS Bedrock, and more."
                ),
                inputSchema=CHAT_COMPLETION_SCHEMA,
            ),
            Tool(
                name=TOOL_NAME_EMBEDDING,
                description=(
                    "Generate embeddings for text using any embedding model "
                    "supported by LiteLLM. Useful for semantic search, "
                    "clustering, and similarity comparisons."
                ),
                inputSchema=EMBEDDINGS_SCHEMA,
            ),
            Tool(
                name=TOOL_NAME_IMAGE_GENERATION,
                description=(
                    "Generate images from text descriptions using models like "
                    "DALL-E. Returns image URLs or base64-encoded images."
                ),
                inputSchema=IMAGE_GENERATION_SCHEMA,
            ),
            Tool(
                name=TOOL_NAME_TEXT_COMPLETION,
                description=(
                    "Generate text completions using non-chat models. "
                    "Suitable for code completion, text generation, and other "
                    "non-conversational tasks."
                ),
                inputSchema=TEXT_COMPLETION_SCHEMA,
            ),
            Tool(
                name=TOOL_NAME_TRANSCRIPTION,
                description=(
                    "Transcribe audio files to text using models like "
                    "Whisper. Supports multiple languages and audio formats."
                ),
                inputSchema=TRANSCRIPTION_SCHEMA,
            ),
            Tool(
                name=TOOL_NAME_RERANK,
                description=(
                    "Rerank a list of documents based on their relevance to "
                    "a query. Useful for improving search result quality."
                ),
                inputSchema=RERANK_SCHEMA,
            ),
            Tool(
                name=TOOL_NAME_LIST_MODELS,
                description=(
                    "List available LLM models supported by LiteLLM, "
                    "optionally filtered by provider."
                ),
                inputSchema=LIST_MODELS_SCHEMA,
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> CallToolResult:
        arguments = arguments or {}
        try:
            if name == TOOL_NAME_CHAT_COMPLETION:
                return await _handle_chat_completion(arguments)
            elif name == TOOL_NAME_EMBEDDING:
                return await _handle_embedding(arguments)
            elif name == TOOL_NAME_IMAGE_GENERATION:
                return await _handle_image_generation(arguments)
            elif name == TOOL_NAME_TEXT_COMPLETION:
                return await _handle_text_completion(arguments)
            elif name == TOOL_NAME_TRANSCRIPTION:
                return await _handle_transcription(arguments)
            elif name == TOOL_NAME_RERANK:
                return await _handle_rerank(arguments)
            elif name == TOOL_NAME_LIST_MODELS:
                return await _handle_list_models(arguments)
            else:
                return _error_result(f"Unknown tool: {name}")
        except Exception as e:
            logger.exception("Error calling tool %s", name)
            return _error_result(str(e))

    return server


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=f"Error: {message}")],
        isError=True,
    )


def _text_result(data: Any) -> CallToolResult:
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, indent=2, default=str)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
    )


async def _handle_chat_completion(arguments: Dict[str, Any]) -> CallToolResult:
    import litellm

    model = arguments.get("model")
    messages = arguments.get("messages")

    if not model:
        return _error_result("'model' is required")
    if not messages:
        return _error_result("'messages' is required")

    kwargs: Dict[str, Any] = {"model": model, "messages": messages}

    optional_params = [
        "temperature",
        "max_tokens",
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "api_base",
        "api_key",
        "api_version",
    ]
    for param in optional_params:
        if param in arguments:
            kwargs[param] = arguments[param]

    response = await litellm.acompletion(**kwargs)
    return _text_result(response.model_dump())


async def _handle_embedding(arguments: Dict[str, Any]) -> CallToolResult:
    import litellm

    model = arguments.get("model")
    input_text = arguments.get("input")

    if not model:
        return _error_result("'model' is required")
    if input_text is None:
        return _error_result("'input' is required")

    kwargs: Dict[str, Any] = {"model": model, "input": input_text}

    for param in ["api_base", "api_key"]:
        if param in arguments:
            kwargs[param] = arguments[param]

    response = await litellm.aembedding(**kwargs)
    return _text_result(response.model_dump())


async def _handle_image_generation(arguments: Dict[str, Any]) -> CallToolResult:
    import litellm

    prompt = arguments.get("prompt")
    if not prompt:
        return _error_result("'prompt' is required")

    kwargs: Dict[str, Any] = {"prompt": prompt}

    optional_params = ["model", "n", "size", "quality", "api_base", "api_key"]
    for param in optional_params:
        if param in arguments:
            kwargs[param] = arguments[param]

    response = await litellm.aimage_generation(**kwargs)
    return _text_result(response.model_dump())


async def _handle_text_completion(arguments: Dict[str, Any]) -> CallToolResult:
    import litellm

    model = arguments.get("model")
    prompt = arguments.get("prompt")

    if not model:
        return _error_result("'model' is required")
    if not prompt:
        return _error_result("'prompt' is required")

    kwargs: Dict[str, Any] = {"model": model, "prompt": prompt}

    optional_params = [
        "temperature",
        "max_tokens",
        "top_p",
        "stop",
        "api_base",
        "api_key",
    ]
    for param in optional_params:
        if param in arguments:
            kwargs[param] = arguments[param]

    response = await litellm.atext_completion(**kwargs)
    return _text_result(response.model_dump())


async def _handle_transcription(arguments: Dict[str, Any]) -> CallToolResult:
    import litellm

    model = arguments.get("model")
    file_path = arguments.get("file")

    if not model:
        return _error_result("'model' is required")
    if not file_path:
        return _error_result("'file' is required")

    kwargs: Dict[str, Any] = {"model": model, "file": open(file_path, "rb")}

    optional_params = [
        "language",
        "prompt",
        "response_format",
        "temperature",
        "api_base",
        "api_key",
    ]
    for param in optional_params:
        if param in arguments:
            kwargs[param] = arguments[param]

    response = await litellm.atranscription(**kwargs)
    return _text_result(response.model_dump())


async def _handle_rerank(arguments: Dict[str, Any]) -> CallToolResult:
    import litellm

    model = arguments.get("model")
    query = arguments.get("query")
    documents = arguments.get("documents")

    if not model:
        return _error_result("'model' is required")
    if not query:
        return _error_result("'query' is required")
    if not documents:
        return _error_result("'documents' is required")

    kwargs: Dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
    }

    optional_params = ["top_n", "api_base", "api_key"]
    for param in optional_params:
        if param in arguments:
            kwargs[param] = arguments[param]

    response = await litellm.arerank(**kwargs)
    return _text_result(response.model_dump())


async def _handle_list_models(arguments: Dict[str, Any]) -> CallToolResult:
    import litellm

    provider_filter = arguments.get("provider")

    model_names = list(litellm.model_cost.keys())

    if provider_filter:
        provider_lower = provider_filter.lower()
        model_names = [
            m
            for m in model_names
            if m.startswith(f"{provider_lower}/") or provider_lower in m.split("/")[0]
        ]

    model_names.sort()

    result = {
        "total_models": len(model_names),
        "models": model_names[:100],
    }
    if len(model_names) > 100:
        result["note"] = (
            f"Showing first 100 of {len(model_names)} models. "
            f"Use the 'provider' filter to narrow results."
        )

    return _text_result(result)
