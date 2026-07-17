"""Generate the LLM denominator from the product surface instead of hand-listing it.

The set of LLM cells we want covered is derived, not curated: the endpoint, route,
capability and streaming vocabularies live in `schema.py`, and which capabilities are
real for a given route is read from `model_prices_and_context_window.json` (the same
metadata the proxy ships). Adding a provider capability there, or a value to the schema
vocabulary, grows the denominator on its own; a test PR never edits it.

Scope: this generates the conversational core (chat_completions, messages, responses),
whose ids follow the `llm.<endpoint>.<route>.<capability>.<streaming>.works` grammar.
The Anthropic-format `messages` surface is the Claude Code compatibility matrix, so its
capabilities are the CLI feature set rather than model flags and are not gated by the
json. Non-core LLM endpoints (batches, files, rerank, embeddings, audio, images) use an
operation grammar the vocabulary does not enumerate, and the behavior modules (mgmt, mcp,
reliability, quota, logging, guardrail, other) have no clean cartesian in the schema; both
are carried by the curated overlay for now, and wiring their own product-surface sources
(route table, guardrail_hooks/, integrations/, router_strategy/) into generation is the
documented follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from .schema import LlmCapability, LlmEndpoint, LlmRoute, LlmStreaming, format_llm_id

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PRICES_PATH = REPO_ROOT / "model_prices_and_context_window.json"

CLAUDE_CODE_ROUTES: tuple[LlmRoute, ...] = (
    "anthropic",
    "azure_foundry",
    "bedrock_converse",
    "bedrock_invoke",
    "vertex",
)

ROUTE_PROVIDERS: dict[LlmRoute, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "azure_foundry": ("azure_ai",),
    "azure_openai": ("azure",),
    "bedrock_converse": ("bedrock_converse", "bedrock"),
    "cohere": ("cohere", "cohere_chat"),
    "openai": ("openai",),
    "together_ai": ("together_ai",),
}
VERTEX_PROVIDER_PREFIXES: tuple[str, ...] = ("vertex_ai", "gemini")


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """One capability's generation rule: the model-metadata flag that proves a route
    supports it (None when it is always emitted), which endpoints it applies to, and
    whether it has a streaming variant."""

    capability: LlmCapability
    flag: str | None
    endpoints: frozenset[LlmEndpoint]
    streamable: bool


_CONVERSATIONAL: frozenset[LlmEndpoint] = frozenset({"chat_completions", "messages", "responses"})
_MESSAGES_ONLY: frozenset[LlmEndpoint] = frozenset({"messages"})
_CHAT_ONLY: frozenset[LlmEndpoint] = frozenset({"chat_completions"})

CAPABILITY_SPECS: tuple[CapabilitySpec, ...] = (
    CapabilitySpec("basic", None, _CONVERSATIONAL, streamable=True),
    CapabilitySpec("tool_use", "supports_function_calling", _CONVERSATIONAL, streamable=True),
    CapabilitySpec("vision", "supports_vision", _CONVERSATIONAL, streamable=False),
    CapabilitySpec("structured_output", "supports_response_schema", _CONVERSATIONAL, streamable=False),
    CapabilitySpec("thinking", "supports_reasoning", _CONVERSATIONAL, streamable=False),
    CapabilitySpec("prompt_cache_5m", "supports_prompt_caching", _CONVERSATIONAL, streamable=False),
    CapabilitySpec("service_tier", None, _CHAT_ONLY, streamable=False),
    CapabilitySpec("prompt_cache_1h", "supports_prompt_caching", _MESSAGES_ONLY, streamable=False),
    CapabilitySpec("mid_conversation_system", "supports_mid_conversation_system", _MESSAGES_ONLY, streamable=False),
    CapabilitySpec("pdf_input", "supports_pdf_input", _MESSAGES_ONLY, streamable=False),
    CapabilitySpec("web_search", "supports_web_search", _MESSAGES_ONLY, streamable=False),
    CapabilitySpec("thinking_with_tool_use", "supports_reasoning", _MESSAGES_ONLY, streamable=False),
    CapabilitySpec("count_tokens", None, _MESSAGES_ONLY, streamable=False),
    CapabilitySpec("long_context_1m", None, _MESSAGES_ONLY, streamable=False),
    CapabilitySpec("tool_search", None, _MESSAGES_ONLY, streamable=False),
)


class ModelEntry(BaseModel):
    """Only the model_prices fields the denominator reads. Each capability flag is
    modelled explicitly so no supports_* value is threaded as an untyped dict value."""

    model_config = ConfigDict(extra="ignore")

    litellm_provider: str | None = None
    mode: str | None = None
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_response_schema: bool = False
    supports_reasoning: bool = False
    supports_prompt_caching: bool = False
    supports_pdf_input: bool = False
    supports_web_search: bool = False
    supports_mid_conversation_system: bool = False

    def enabled_flags(self) -> frozenset[str]:
        values: dict[str, bool] = {
            "supports_function_calling": self.supports_function_calling,
            "supports_vision": self.supports_vision,
            "supports_response_schema": self.supports_response_schema,
            "supports_reasoning": self.supports_reasoning,
            "supports_prompt_caching": self.supports_prompt_caching,
            "supports_pdf_input": self.supports_pdf_input,
            "supports_web_search": self.supports_web_search,
            "supports_mid_conversation_system": self.supports_mid_conversation_system,
        }
        return frozenset(flag for flag, enabled in values.items() if enabled)


_ENTRIES_ADAPTER: TypeAdapter[dict[str, ModelEntry]] = TypeAdapter(dict[str, ModelEntry])


def load_model_entries(path: Path = MODEL_PRICES_PATH) -> tuple[ModelEntry, ...]:
    entries = _ENTRIES_ADAPTER.validate_json(path.read_bytes())
    return tuple(entries.values())


def _route_of(provider: str) -> LlmRoute | None:
    """The registry route a model_prices provider maps to, or None when it is not one
    of the routes the denominator enumerates."""
    if provider.startswith(VERTEX_PROVIDER_PREFIXES):
        return "vertex"
    return next(
        (route for route, providers in ROUTE_PROVIDERS.items() if provider in providers),
        None,
    )


def _flags_by_route(entries: tuple[ModelEntry, ...]) -> dict[LlmRoute, frozenset[str]]:
    """The union of supports_* flags across every model that maps to each route: a
    route advertises a capability when any model behind it does."""
    pairs: tuple[tuple[LlmRoute, frozenset[str]], ...] = tuple(
        (route, entry.enabled_flags())
        for entry in entries
        if entry.litellm_provider is not None
        for route in (_route_of(entry.litellm_provider),)
        if route is not None
    )
    routes: frozenset[LlmRoute] = frozenset(route for route, _ in pairs)
    return {route: frozenset[str]().union(*(flags for r, flags in pairs if r == route)) for route in routes}


def _routes_with_mode(entries: tuple[ModelEntry, ...], mode: str) -> frozenset[LlmRoute]:
    return frozenset(
        route
        for entry in entries
        if entry.mode == mode and entry.litellm_provider is not None
        for route in (_route_of(entry.litellm_provider),)
        if route is not None
    )


def _routes_for_endpoint(endpoint: LlmEndpoint, entries: tuple[ModelEntry, ...]) -> tuple[LlmRoute, ...]:
    if endpoint == "messages":
        return CLAUDE_CODE_ROUTES
    if endpoint == "responses":
        return tuple(sorted(_routes_with_mode(entries, "responses")))
    return tuple(sorted(_routes_with_mode(entries, "chat")))


def _streamings(spec: CapabilitySpec) -> tuple[LlmStreaming, ...]:
    return ("nonstream", "stream") if spec.streamable else ("nonstream",)


def _available(
    endpoint: LlmEndpoint,
    spec: CapabilitySpec,
    route: LlmRoute,
    flags_by_route: dict[LlmRoute, frozenset[str]],
) -> bool:
    if endpoint == "messages":
        return True
    if spec.flag is None:
        return True
    return spec.flag in flags_by_route.get(route, frozenset())


def generate_llm_cell_ids(entries: tuple[ModelEntry, ...] | None = None) -> frozenset[str]:
    """The generated core-LLM denominator: one `...works` cell per real
    (endpoint, route, capability, streaming) combination."""
    model_entries = load_model_entries() if entries is None else entries
    flags_by_route = _flags_by_route(model_entries)
    return frozenset(
        format_llm_id(endpoint, route, spec.capability, streaming, "works")
        for spec in CAPABILITY_SPECS
        for endpoint in spec.endpoints
        for route in _routes_for_endpoint(endpoint, model_entries)
        if _available(endpoint, spec, route, flags_by_route)
        for streaming in _streamings(spec)
    )


_ENDPOINT_ROUTE_FRAGMENTS: dict[LlmEndpoint, str] = {
    "chat_completions": "/chat/completions",
    "messages": "/v1/messages",
    "responses": "/responses",
    "embeddings": "/embeddings",
    "batches": "/batches",
    "files": "/files",
    "rerank": "/rerank",
    "images_generations": "/images/generations",
    "audio_speech": "/audio/speech",
    "audio_transcriptions": "/audio/transcriptions",
    "moderations": "/moderations",
    "realtime": "/realtime",
}

ROUTE_CHECKABLE_ENDPOINTS: frozenset[LlmEndpoint] = frozenset(_ENDPOINT_ROUTE_FRAGMENTS)


def _path_serves(fragment: str, path: str) -> bool:
    """Whether a route path serves an endpoint fragment, matched on path-segment
    boundaries so `/files` does not spuriously match `/v1/filesystem`."""
    return path == fragment or path.endswith(fragment) or f"{fragment}/" in path


def route_table_endpoints() -> frozenset[LlmEndpoint] | None:
    """The LLM endpoints the proxy actually serves, read from the live route table,
    or None when litellm cannot be imported (kept off the hot path so the generator
    and its unit tests stay hermetic). Used by the collector as a drift check."""
    try:
        from litellm.proxy._types import LiteLLMRoutes
    except ImportError:
        return None
    served = tuple(str(path) for path in LiteLLMRoutes.llm_api_routes.value)
    return frozenset(
        endpoint
        for endpoint, fragment in _ENDPOINT_ROUTE_FRAGMENTS.items()
        if any(_path_serves(fragment, path) for path in served)
    )
