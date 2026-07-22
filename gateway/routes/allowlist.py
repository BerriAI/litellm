"""Path allowlist for the gateway component.

The gateway exposes the LLM data-plane surface: chat/completions, embeddings,
audio, batches, files, fine-tuning, rerank, ocr, rag, video, search, image,
responses, vector stores, passthrough providers, realtime websockets, MCP
tool-call endpoints, and operational endpoints (/health, /metrics).

Any path not listed here is dropped from the gateway process so management/UI
endpoints don't ride on the same pods.

Versioned data-plane paths are enumerated explicitly rather than allowing a
blanket `/v1/` or `/v2/` prefix — those broad prefixes would otherwise also
match management routes like `/v1/access_group`, `/v1/tool/{tool_name}/logs`,
`/v2/key/info`, etc.
"""

GATEWAY_PATH_PREFIXES: tuple[str, ...] = (
    # OpenAI-compatible data-plane surface (versioned + unversioned)
    "/v1/chat/",
    "/chat/",
    "/v1/completions",
    "/completions",
    "/v1/embeddings",
    "/embeddings",
    "/v1/moderations",
    "/moderations",
    "/v1/audio/",
    "/audio/",
    "/v1/images/",
    "/images/",
    "/v1/files",
    "/files",
    "/v1/batches",
    "/batches",
    "/v1/fine_tuning/",
    "/fine_tuning/",
    "/v1/fine-tuning/",
    "/fine-tuning/",
    "/v1/responses",
    "/responses",
    "/v1/threads",
    "/threads",
    "/v1/assistants",
    "/assistants",
    "/v1/vector_stores",
    "/vector_stores",
    "/v1/indexes",
    "/v1/models",
    "/models",
    "/openai/",
    "/engines/",
    # Anthropic / agentic data-plane surface
    "/v1/messages",
    "/messages",
    "/v1/skills",
    "/v1/a2a/",
    # LiteLLM-native LLM surface
    "/v1/rerank",
    "/v2/rerank",
    "/rerank",
    "/v1/ocr",
    "/ocr",
    "/v1/rag/",
    "/rag/",
    "/v1/video",
    "/v1/videos",
    "/video/",
    "/videos",
    "/v1/search",
    "/search",
    "/v1/containers",
    "/containers",
    "/v1/evals",
    "/v1/memory",
    "/queue/chat/",
    # Google data plane (v1beta is the Google AI Studio version)
    "/v1beta/",
    "/interactions",
    # Provider passthrough
    "/anthropic/",
    "/azure/",
    "/azure_ai/",
    "/aws/",
    "/bedrock/",
    "/cohere/",
    "/gemini/",
    "/google/",
    "/vertex_ai/",
    "/vertex-ai/",
    "/assemblyai/",
    "/eu.assemblyai/",
    "/langfuse/",
    "/vllm/",
    "/mistral/",
    "/groq/",
    "/voyage/",
    "/cursor/",
    "/milvus/",
    "/openai_passthrough/",
    # Dynamic provider / toolset passthrough (path templates)
    "/{provider}/",
    "/toolset/",
    # Realtime / streaming
    "/v1/realtime",
    "/realtime",
    # Health & ops
    "/health",
    "/metrics",
    "/watsonx",
)

GATEWAY_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/routes",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/test",
        "/collector/spend-logs",
    }
)

GATEWAY_MOUNT_PATHS: frozenset[str] = frozenset(
    {
        "/metrics",
    }
)
