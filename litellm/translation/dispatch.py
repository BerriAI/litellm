"""The one v1/v2 fork.

A pure decision: a request stays on v1 until its provider is opted in, takes
the same-family fast path when the inbound schema already speaks the provider's
wire format (so the IR semantic re-encode is skipped), and otherwise goes
through the IR. Enabling any body-touching feature (param normalization,
guardrails, semantic cache) sets ``body_touching`` and forces the IR path even
for a same-family pair, because the fast path forwards the message payload as-is.
"""

from __future__ import annotations

from typing import Literal, get_args

from expression import case, tag, tagged_union

from .ir import UNIT, Unit

InboundSchema = Literal[
    "openai_chat", "anthropic_messages", "google_genai", "responses", "completions"
]

Provider = Literal[
    "anthropic",
    "bedrock_converse",
    "bedrock_invoke",
    "vertex_ai",
    "azure",
    "azure_ai",
    "azure_ai_anthropic",
    "openai_compat",
    "gemini",
    "vertex_anthropic",
    "xai",
    # wave-1a: the SDK-path openai-compat family (providers/compat_sdk).
    # baseten is deliberately ABSENT: its streams ride a dedicated legacy
    # CustomStreamWrapper branch (handle_baseten_chunk), not the openai
    # dialect, so it stays a typed v1 fallback (wave1a-port.md).
    "together_ai",
    "cerebras",
    "nvidia_nim",
    "lm_studio",
    "llamafile",
    "lambda_ai",
    "nebius",
    "novita",
    "wandb",
    "featherless_ai",
    "nscale",
    "hyperbolic",
    "volcengine",
    # wave-1b SDK-path shims (compat_sdk profile rows). ai21 chat traffic is
    # COERCED to "ai21_chat" upstream (get_llm_provider), so that is the
    # dispatch name; the legacy "ai21" wrapper branch never sees chat streams
    # (coercion canary in test_differential_compat_sdk_stream). aiml is
    # deliberately ABSENT: its config class is unregistered at HEAD, so v1
    # serves it through the generic openai fallback stack (drop canary).
    "ai21_chat",
    "dashscope",
    "docker_model_runner",
    "empower",
    "friendliai",
    "galadriel",
    "github",
    "inception",
    "meta_llama",
    "morph",
    "v0",
    "zai",
    "vercel_ai_gateway",
    # wave-1b JSON-registry providers (compat_sdk profile rows): only the 14
    # LlmProviders enum members ride the dynamic JSONProviderConfig; the 7
    # non-members (veniceai, abliteration, llamagate, gmi, sarvam, aihubmix,
    # crusoe) dispatch through the generic openai fallback arms in v1 and are
    # deliberately ABSENT (drop canary).
    "publicai",
    "helicone",
    "xiaomi_mimo",
    "scaleway",
    "synthetic",
    "apertis",
    "nano-gpt",
    "poe",
    "chutes",
    "assemblyai",
    "charity_engine",
    "neosantara",
    "tensormesh",
    "parasail",
    # wave-1b httpx-path shims (compat_httpx profile rows): dedicated elifs,
    # transforms LIVE, NO seam model preset (bare wire model, except the
    # compactifai/amazon-nova/lemonade request-model prefixes the family
    # response parser owns).
    "heroku",
    "bedrock_mantle",
    "minimax",
    "compactifai",
    "amazon_nova",
    "datarobot",
    "gradient_ai",
    "ovhcloud",
    "lemonade",
    # wave-2a: compat_sdk profile rows + named gates. perplexity/sambanova/
    # deepinfra/moonshot are SDK-path (seam preset + re-prefix applies);
    # cometapi is httpx-path (NO model preset, bare wire model — the xai R4
    # pin) and is a compat_httpx family row since the sibling merge, with
    # its own strict-envelope/copy-both stream policy row (LINE_PARSERS).
    "perplexity",
    "sambanova",
    "deepinfra",
    "moonshot",
    "cometapi",
    # wave-2b-beta: own-module providers (providers/<name>/). cohere and
    # cohere_chat are ONE module (main.py's elif handles both names; the v2
    # wire is the DEFAULT route at HEAD — the legacy "v1/" route predicate
    # is a typed fallback inside the cohere guard, researcher-4 §11).
    "cohere",
    "cohere_chat",
    # wave-2b-beta: mistral (httpx path, dedicated elif; bare wire model on
    # responses; the magistral reasoning-prompt injection is a typed
    # fallback inside the module — codestral reuses the v1 config but is
    # NOT a wave-2b provider and stays a v1 fallback).
    "mistral",
]

NeverPortProvider = Literal[
    "replicate",
    "predibase",
    "triton",
    "petals",
    "nlp_cloud",
    "gigachat",
    "bytez",
    "a2a",
    "langgraph",
    "langflow",
    "ragflow",
    "chatgpt",
    "vllm",
    "aiohttp_openai",
    "azure_text",
    "anthropic_text",
    "oobabooga",
    "maritalk",
    "clarifai",
    "cloudflare",
    "litellm_proxy",
    "aleph_alpha",
    "palm",
]
"""Permanent v1 fallbacks. NOT a TODO list: each name carries a reason in the
researcher-3/researcher-4 dossiers (poll-loop architectures, in-process libs,
dead vendors, agent gateways, transform-time I/O, gateway-passthrough
semantics). Promotion is a deliberate two-file edit (remove here, add to
``Provider``) that the disjointness test forces you to notice.

Deliberately OUTSIDE this Literal:

- baseten: not policy — it keeps its own re-evaluate canary
  (test_baseten_drop_canary) keyed on the legacy wrapper branch.
- cohere / cohere_chat: their DEFAULT route at HEAD is the v2 wire (wave-2
  scope); only the legacy explicit-``v1/`` route is never-port, and a provider
  Literal cannot carry a route. That predicate (``"v1/" in model`` -> typed
  fallback) belongs INSIDE the future cohere module, asserted there.
- the wave-1b drop set (aiml + the 7 non-enum JSON providers): re-evaluate
  drops with their own canaries, not permanent policy.
"""

NEVER_PORT: frozenset[str] = frozenset(get_args(NeverPortProvider))

_SAME_FAMILY: frozenset[tuple[InboundSchema, Provider]] = frozenset(
    {
        ("anthropic_messages", "anthropic"),
        ("anthropic_messages", "bedrock_invoke"),
        ("anthropic_messages", "vertex_ai"),
        ("anthropic_messages", "vertex_anthropic"),
        ("openai_chat", "openai_compat"),
        # xai is NOT same-family despite speaking openai-chat: v1's transform
        # touches the body (tools strict strip, non-user message name strip),
        # so a verbatim fast-path forward would diverge from v1.
        # The compat_sdk family is NOT same-family either: their param maps
        # touch the body (mct -> max_tokens renames, supported-list raises,
        # together's response_format pop), so a verbatim forward would
        # diverge from v1's gates.
    }
)


@tagged_union(frozen=True)
class Route:
    tag: Literal["v1", "fast_path", "v2"] = tag()

    v1: Unit = case()
    fast_path: Provider = case()
    v2: Provider = case()

    @staticmethod
    def of_v1() -> Route:
        return Route(v1=UNIT)

    @staticmethod
    def of_fast_path(provider: Provider) -> Route:
        return Route(fast_path=provider)

    @staticmethod
    def of_v2(provider: Provider) -> Route:
        return Route(v2=provider)


def route(
    *,
    schema: InboundSchema,
    provider: Provider,
    enabled_providers: frozenset[Provider],
    body_touching: bool,
) -> Route:
    if provider not in enabled_providers:
        return Route.of_v1()
    if (schema, provider) in _SAME_FAMILY and not body_touching:
        return Route.of_fast_path(provider)
    return Route.of_v2(provider)
