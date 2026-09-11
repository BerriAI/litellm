"""Path allowlist for the gateway component.

The gateway exposes the LLM data-plane surface: chat/completions, embeddings,
audio, batches, files, fine-tuning, rerank, ocr, rag, video, search, image,
responses, vector stores, passthrough providers, realtime websockets, MCP
tool-call endpoints, agent (A2A) endpoints, and operational endpoints
(/health, /metrics).

Any path not listed here is dropped from the gateway process so management/UI
endpoints don't ride on the same pods.

The surface is partitioned into workloads (``llm``, ``mcp``, ``agent``) so a
gateway pod can be dedicated to one of them via ``LITELLM_GATEWAY_WORKLOAD``.
The default workload ``all`` serves the whole surface. Ops paths are served by
every workload.

Versioned data-plane paths are enumerated explicitly rather than allowing a
blanket `/v1/` or `/v2/` prefix — those broad prefixes would otherwise also
match management routes like `/v1/access_group`, `/v1/tool/{tool_name}/logs`,
`/v2/key/info`, etc.
"""

from typing import Final, Literal, get_args

GatewayWorkload = Literal["all", "llm", "mcp", "agent"]
GATEWAY_WORKLOADS: Final[frozenset[str]] = frozenset(get_args(GatewayWorkload))
GATEWAY_WORKLOAD_ENV_VAR: Final = "LITELLM_GATEWAY_WORKLOAD"

GATEWAY_LLM_PATH_PREFIXES: Final[tuple[str, ...]] = (
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
    "/v1/messages",
    "/messages",
    "/v1/skills",
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
    "/comprehendmedical",
    "/cohere/",
    "/gemini/",
    "/gigachat/",
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
    "/{provider}/",
    "/v1/realtime",
    "/realtime",
    "/watsonx",
)

GATEWAY_MCP_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "/mcp",
    "/toolset/",
    "/{mcp_server_name}/mcp",
)

GATEWAY_AGENT_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "/v1/a2a",
    "/a2a",
)

GATEWAY_OPS_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "/health",
    "/metrics",
)

GATEWAY_PATH_PREFIXES: tuple[str, ...] = (
    *GATEWAY_LLM_PATH_PREFIXES,
    *GATEWAY_MCP_PATH_PREFIXES,
    *GATEWAY_AGENT_PATH_PREFIXES,
    *GATEWAY_OPS_PATH_PREFIXES,
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
    }
)

GATEWAY_OPS_MOUNT_PATHS: Final[frozenset[str]] = frozenset({"/metrics"})
GATEWAY_MCP_MOUNT_PATHS: Final[frozenset[str]] = frozenset({"/mcp"})
GATEWAY_MOUNT_PATHS: frozenset[str] = GATEWAY_OPS_MOUNT_PATHS | GATEWAY_MCP_MOUNT_PATHS


def _under(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(p) for p in prefixes)


def _path_workload(path: str) -> GatewayWorkload | Literal["ops"] | None:
    if path in GATEWAY_EXACT_PATHS or _under(path, GATEWAY_OPS_PATH_PREFIXES):
        return "ops"
    if _under(path, GATEWAY_MCP_PATH_PREFIXES):
        return "mcp"
    if _under(path, GATEWAY_AGENT_PATH_PREFIXES):
        return "agent"
    if _under(path, GATEWAY_LLM_PATH_PREFIXES):
        return "llm"
    return None


def gateway_serves_path(workload: GatewayWorkload, path: str) -> bool:
    owner: Final = _path_workload(path)
    if owner is None:
        return False
    return owner == "ops" or workload == "all" or owner == workload


def gateway_serves_mount(workload: GatewayWorkload, path: str) -> bool:
    if path in GATEWAY_OPS_MOUNT_PATHS:
        return True
    return path in GATEWAY_MCP_MOUNT_PATHS and workload in ("all", "mcp")


def parse_gateway_workload(raw: str | None) -> GatewayWorkload:
    value: Final = (raw or "all").strip().lower()
    match value:
        case "all" | "llm" | "mcp" | "agent":
            return value
        case _:
            raise ValueError(f"{GATEWAY_WORKLOAD_ENV_VAR}={raw!r} is not one of {', '.join(sorted(GATEWAY_WORKLOADS))}")
