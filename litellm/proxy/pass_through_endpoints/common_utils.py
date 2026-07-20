from urllib.parse import urlparse

from fastapi import Request

# Hostnames that route to OpenAI-compatible APIs.
#
# `openai.com` is OpenAI proper — kept as a suffix (rather than the exact
# `api.openai.com`) so regional/alternate API subdomains stay covered. The two
# Azure domains below are *shared by
# every Azure Cognitive Service* (Speech, Vision, Language, ...), not just Azure
# OpenAI: `openai.azure.com` is the classic Azure OpenAI domain, while
# `cognitiveservices.azure.com` is used by newer "Azure AI Foundry" /
# Cognitive Services-hosted Azure OpenAI deployments. Because the hostname alone
# cannot tell Azure OpenAI apart from the other Cognitive Services on those
# domains, requests there must additionally carry an OpenAI-style path segment.
OPENAI_HOSTNAMES = ("openai.com",)
AZURE_OPENAI_HOSTNAMES = ("openai.azure.com", "cognitiveservices.azure.com")
# Path markers that identify an Azure request as Azure OpenAI rather than Speech
# / Vision / Language / ... `/openai/` is the native Azure OpenAI path prefix;
# `/v1/` is the OpenAI-v1 surface used by LiteLLM's pass-through routing. Other
# Cognitive Services use service-named prefixes and versions like `/v3.1/`,
# `/v1.0/`, so they do not collide with these markers.
AZURE_OPENAI_PATH_MARKERS = ("/openai/", "/v1/")


def hostname_matches(hostname: str, suffixes: tuple[str, ...]) -> bool:
    """True if hostname equals one of `suffixes` or is a subdomain of it.

    Uses suffix matching (not a bare substring test) so look-alikes such as
    `cognitiveservices.azure.com.attacker.example` are not accepted.
    """
    return any(hostname == suffix or hostname.endswith("." + suffix) for suffix in suffixes)


def is_openai_compatible_url(url_route: str | None) -> bool:
    """True if the URL targets an OpenAI-compatible API surface.

    For the shared Azure Cognitive Services domains we additionally require an
    OpenAI-style path segment (`/openai/` or `/v1/`) so non-OpenAI Azure services
    (Speech, Vision, Language, ...) on the same domain are not misclassified as
    OpenAI routes.
    """
    if not url_route:
        return False
    parsed_url = urlparse(url_route)
    hostname = parsed_url.hostname
    if not hostname:
        return False
    if hostname_matches(hostname, OPENAI_HOSTNAMES):
        return True
    if hostname_matches(hostname, AZURE_OPENAI_HOSTNAMES):
        return any(marker in parsed_url.path for marker in AZURE_OPENAI_PATH_MARKERS)
    return False


# Hostnames of OpenAI-*protocol* upstreams that are neither OpenAI nor Azure.
# Fireworks serves an OpenAI-compatible surface at `api.fireworks.ai/inference/v1`.
# Unlike the Azure domains there is no non-OpenAI service sharing this host, so
# no path marker is required to disambiguate it.
FIREWORKS_HOSTNAMES = ("fireworks.ai",)

# `litellm.constants.openai_compatible_providers` enumerates the third-party
# providers that speak the OpenAI wire protocol, but deliberately omits OpenAI
# itself and Azure OpenAI (they are first-class providers, not "compatible"
# ones). Pass-through cost tracking cares about the protocol, not the pedigree,
# so both sets are in scope here.
_NATIVE_OPENAI_PROVIDERS = (
    "openai",
    "text-completion-openai",
    "custom_openai",
    "openai_like",
    "azure",
    "azure_text",
)

# Fireworks serverless model ids are fully-qualified resource names —
# `accounts/{account}/models/{model}`. The price map stores them under the
# provider-prefixed key `fireworks_ai/accounts/...`, while a native pass-through
# response echoes the bare form.
FIREWORKS_PROVIDER = "fireworks_ai"
_FIREWORKS_ACCOUNTS_PREFIX = "accounts/"
_FIREWORKS_MODELS_MARKER = "/models/"


def is_fireworks_url(url_route: str | None) -> bool:
    """True if the URL targets the Fireworks AI API."""
    if not url_route:
        return False
    hostname = urlparse(url_route).hostname
    if not hostname:
        return False
    return hostname_matches(hostname, FIREWORKS_HOSTNAMES)


# Cohere's API host. Suffix-matched (like every other host predicate here) so
# regional/alternate subdomains stay covered without accepting look-alikes.
COHERE_HOSTNAMES = ("cohere.com", "cohere.ai")

# Only the chat surface streams and is reconstructable from SSE chunks by
# `CoherePassthroughLoggingHandler`. `/v1/embed` is non-streaming, and rerank /
# classify have no chunk parser — classifying those as a streaming Cohere
# endpoint would claim cost coverage that does not exist.
COHERE_STREAMING_PATHS = ("/v2/chat",)


def is_cohere_streaming_url(url_route: str | None) -> bool:
    """True if the URL is a Cohere endpoint whose stream we can cost."""
    if not url_route:
        return False
    parsed_url = urlparse(url_route)
    hostname = parsed_url.hostname
    if not hostname or not hostname_matches(hostname, COHERE_HOSTNAMES):
        return False
    return any(path in parsed_url.path for path in COHERE_STREAMING_PATHS)


def is_openai_compatible_provider(custom_llm_provider: str | None) -> bool:
    """True when `custom_llm_provider` speaks the OpenAI wire protocol.

    Pass-through cost math for OpenAI-shaped responses is already
    provider-agnostic: the response is transformed with the OpenAI config and
    the resulting usage handed to `litellm.completion_cost` along with the
    provider. Gating entry to that handler on a hardcoded hostname allow-list
    therefore excluded every other OpenAI-compatible upstream (Fireworks, Groq,
    Together, ...) for no reason, and those routes recorded $0 while still
    billing our upstream account.
    """
    if not custom_llm_provider:
        return False

    from litellm.constants import openai_compatible_providers

    provider = custom_llm_provider.strip().lower()
    return provider in _NATIVE_OPENAI_PROVIDERS or provider in openai_compatible_providers


def _is_shared_azure_cognitive_host(url_route: str | None) -> bool:
    """True for a host on the Azure domains shared with non-OpenAI services."""
    if not url_route:
        return False
    hostname = urlparse(url_route).hostname
    if not hostname:
        return False
    return hostname_matches(hostname, AZURE_OPENAI_HOSTNAMES)


def is_openai_wire_compatible_route(
    url_route: str | None,
    custom_llm_provider: str | None = None,
) -> bool:
    """True if this pass-through call should be costed by the OpenAI handler.

    Hostname classification is consulted first so OpenAI/Azure routes behave
    exactly as they did before. `custom_llm_provider` then widens the scope to
    any other OpenAI-compatible upstream — the way `is_gemini_route` /
    `is_cursor_route` already key off the provider.

    The one place the provider is NOT allowed to widen anything is the Azure
    Cognitive Services domains, which Azure OpenAI shares with Speech / Vision /
    Language. There the `/openai/` `/v1/` path marker stays authoritative: a
    `custom_llm_provider: azure` label must not turn a Speech transcription into
    a chat completion.
    """
    if is_openai_compatible_url(url_route) or is_fireworks_url(url_route):
        return True
    if _is_shared_azure_cognitive_host(url_route):
        return False
    return is_openai_compatible_provider(custom_llm_provider)


def is_fireworks_model_id(model: str | None) -> bool:
    """True for a Fireworks serverless model id, provider-prefixed or bare."""
    if not model:
        return False
    candidate = model[len(FIREWORKS_PROVIDER) + 1 :] if model.startswith(FIREWORKS_PROVIDER + "/") else model
    return candidate.startswith(_FIREWORKS_ACCOUNTS_PREFIX) and _FIREWORKS_MODELS_MARKER in candidate


def normalize_fireworks_model_id(model: str | None) -> str | None:
    """Return the bare `accounts/.../models/...` id for a Fireworks model.

    `litellm.llms.fireworks_ai.cost_calculator` is the pricing authority for
    Fireworks: it resolves the model via `get_model_info(..., "fireworks_ai")`
    (which re-adds the `fireworks_ai/` prefix the price map is keyed on) and,
    for ids absent from the map, falls back to a parameter-size tier
    (`fireworks-ai-default`, `fireworks-ai-above-16b`, ...) derived from the
    model *name*. Both steps want the bare id, so strip a redundant provider
    prefix rather than letting it double up into
    `fireworks_ai/fireworks_ai/accounts/...`.

    Non-Fireworks models are returned unchanged.
    """
    if not model or not is_fireworks_model_id(model):
        return model
    if model.startswith(FIREWORKS_PROVIDER + "/"):
        return model[len(FIREWORKS_PROVIDER) + 1 :]
    return model


def resolve_openai_passthrough_provider(
    model: str | None = None,
    custom_llm_provider: str | None = None,
    url_route: str | None = None,
) -> str:
    """Pick the provider to price an OpenAI-compatible pass-through call with.

    Generic pass-throughs (`general_settings.pass_through_endpoints`) carry no
    `custom_llm_provider` field at all, so the OpenAI handler defaulted to
    `"openai"`. For a Fireworks model that makes `completion_cost` raise "this
    model isn't mapped yet", which the handler swallows — the request is billed
    upstream and recorded at $0. Infer the provider from the unambiguous
    Fireworks model-id / hostname shapes instead of guessing "openai".
    """
    if custom_llm_provider:
        return custom_llm_provider
    if is_fireworks_model_id(model) or is_fireworks_url(url_route):
        return FIREWORKS_PROVIDER
    return "openai"


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")
