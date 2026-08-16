from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

TIER_NAMES: tuple[str, ...] = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
AUTOROUTER_MODEL_NAME = "autorouter"


class ConfigGenerationError(Exception):
    """Raised when an AutorouteConfig references a model the discovery step didn't find."""


class DiscoveredModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    mode: str = "chat"
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None


class _RawModelGroup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_group: str
    # Optional: some real deployments return an explicit `"mode": null` for models that
    # were registered without a mode (seen for embedding models like voyage-4-large).
    # ModelGroupInfo's own "chat" default (litellm/types/router.py) only applies when the
    # key is missing entirely, not when it's present as null, so this must tolerate None.
    mode: str | None = "chat"
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None


_RAW_MODEL_GROUPS_ADAPTER = TypeAdapter(list[_RawModelGroup])


def parse_discovered_models(raw: list[JsonValue]) -> tuple[DiscoveredModel, ...]:
    """Validate a raw `/model_group/info` response into typed models."""
    parsed = _RAW_MODEL_GROUPS_ADAPTER.validate_python(raw)
    return tuple(
        DiscoveredModel(
            name=group.model_group,
            # A null mode means the server genuinely doesn't know what this model does;
            # "unknown" (rather than guessing "chat") keeps it out of both chat_models()
            # and embedding_models() instead of risking a wrong-mode deployment.
            mode=group.mode or "unknown",
            input_cost_per_token=group.input_cost_per_token,
            output_cost_per_token=group.output_cost_per_token,
        )
        for group in parsed
    )


def chat_models(models: tuple[DiscoveredModel, ...]) -> tuple[DiscoveredModel, ...]:
    return tuple(m for m in models if m.mode == "chat")


def embedding_models(models: tuple[DiscoveredModel, ...]) -> tuple[DiscoveredModel, ...]:
    return tuple(m for m in models if m.mode == "embedding")


class HeuristicClassifier(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["heuristic"] = "heuristic"


class LLMClassifier(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["llm"] = "llm"
    model: str
    timeout_ms: int = 3000


ClassifierChoice = Union[HeuristicClassifier, LLMClassifier]


class NoSemanticMatching(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["none"] = "none"


class KeywordTierRule(BaseModel):
    model_config = ConfigDict(frozen=True)
    keywords: tuple[str, ...]
    tier: str


# Satisfies complexity_router's "semantic matching requires non-empty keyword_tier_rules"
# invariant with a sane starting point; the wizard lets the user override these per tier.
DEFAULT_KEYWORD_TIER_RULES: tuple[KeywordTierRule, ...] = (
    KeywordTierRule(keywords=("hi", "hello", "thanks"), tier="SIMPLE"),
    KeywordTierRule(keywords=("explain", "how does"), tier="MEDIUM"),
    KeywordTierRule(keywords=("refactor", "implement", "debug"), tier="COMPLEX"),
    KeywordTierRule(keywords=("step by step", "think through", "prove"), tier="REASONING"),
)


class SemanticMatching(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["semantic"] = "semantic"
    embedding_model: str
    match_threshold: float = 0.5
    keyword_tier_rules: tuple[KeywordTierRule, ...] = DEFAULT_KEYWORD_TIER_RULES


SemanticMatchingChoice = Union[NoSemanticMatching, SemanticMatching]


class AutorouteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str
    api_key: str
    # Each tier maps to a pool of one or more models; complexity_router picks randomly among
    # them per request (or, in adaptive mode, learns which to prefer within the pool).
    tiers: dict[str, tuple[str, ...]]
    default_model: str
    classifier: ClassifierChoice = Field(default_factory=HeuristicClassifier)
    semantic_matching: SemanticMatchingChoice = Field(default_factory=NoSemanticMatching)
    adaptive: bool = False


def validate_config(config: AutorouteConfig, discovered: tuple[DiscoveredModel, ...]) -> None:
    """Raise ConfigGenerationError if config references a model discovery didn't return."""
    chat_names: frozenset[str] = frozenset(m.name for m in chat_models(discovered))
    embedding_names: frozenset[str] = frozenset(m.name for m in embedding_models(discovered))

    for tier, models in config.tiers.items():
        for model in models:
            if model not in chat_names:
                raise ConfigGenerationError(f"Tier {tier} references unknown chat model '{model}'")

    if config.default_model not in chat_names:
        raise ConfigGenerationError(f"default_model '{config.default_model}' is not a known chat model")

    if isinstance(config.classifier, LLMClassifier) and config.classifier.model not in chat_names:
        raise ConfigGenerationError(f"classifier model '{config.classifier.model}' is not a known chat model")

    if (
        isinstance(config.semantic_matching, SemanticMatching)
        and config.semantic_matching.embedding_model not in embedding_names
    ):
        raise ConfigGenerationError(
            f"embedding model '{config.semantic_matching.embedding_model}' is not a known embedding model"
        )


def _litellm_proxy_deployment(name: str, base_url: str, api_key: str) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"litellm_proxy/{name}",
            "api_base": base_url,
            "api_key": api_key,
        },
    }


def build_generated_model_list(config: AutorouteConfig) -> list[JsonValue]:
    """Build the model_list for the ephemeral proxy's config.yaml.

    Every real model referenced anywhere (tier targets, classifier, embedding) is deduplicated
    to exactly one `litellm_proxy/<name>` deployment forwarding to the customer's real proxy,
    plus one `auto_router/complexity_router` deployment tying the tiers together.
    """
    referenced_names = {model for models in config.tiers.values() for model in models}
    referenced_names.add(config.default_model)
    if isinstance(config.classifier, LLMClassifier):
        referenced_names.add(config.classifier.model)
    if isinstance(config.semantic_matching, SemanticMatching):
        referenced_names.add(config.semantic_matching.embedding_model)

    proxy_deployments = [
        _litellm_proxy_deployment(name, config.base_url, config.api_key) for name in sorted(referenced_names)
    ]

    complexity_router_config: dict[str, JsonValue] = {
        "tiers": {tier: list(models) for tier, models in config.tiers.items()},
        "default_model": config.default_model,
    }
    if isinstance(config.classifier, LLMClassifier):
        complexity_router_config["classifier_type"] = "llm"
        complexity_router_config["classifier_llm_config"] = {
            "model": config.classifier.model,
            "timeout_ms": config.classifier.timeout_ms,
        }
    if isinstance(config.semantic_matching, SemanticMatching):
        complexity_router_config["semantic_keyword_matching"] = True
        complexity_router_config["embedding_model"] = config.semantic_matching.embedding_model
        complexity_router_config["match_threshold"] = config.semantic_matching.match_threshold
        complexity_router_config["keyword_tier_rules"] = [
            {"keywords": list(rule.keywords), "tier": rule.tier} for rule in config.semantic_matching.keyword_tier_rules
        ]
    if config.adaptive:
        complexity_router_config["adaptive"] = True

    auto_router_litellm_params: dict[str, JsonValue] = {
        "model": "auto_router/complexity_router",
        "complexity_router_config": complexity_router_config,
    }
    # A bare "*" model_name looks like the obvious way to catch every request Claude Code
    # might send regardless of which model it thinks it's using, but Router's auto-router
    # registry is keyed by the literal requested model string (router.py:10711-10717), not
    # resolved through pattern/wildcard matching first -- so a "*" entry here would only ever
    # match a client that literally sends model="*", never an actual wildcard catch-all. Callers
    # instead need to make Claude Code request this "autorouter" name directly (see
    # ANTHROPIC_DEFAULT_*_MODEL in settings.py's merge_claude_settings_static_token).
    return [
        *proxy_deployments,
        {"model_name": AUTOROUTER_MODEL_NAME, "litellm_params": auto_router_litellm_params},
    ]


def build_generated_proxy_config(config: AutorouteConfig, master_key: str) -> dict[str, JsonValue]:
    """Full config.yaml content for the ephemeral proxy, including its own auth key.

    master_key must live under general_settings, not litellm_settings -- the proxy server
    only ever reads general_settings.master_key (proxy_server.py:4530) to authenticate
    requests; a key placed under litellm_settings is silently ignored, leaving the proxy
    with no real auth at all.
    """
    return {
        "model_list": build_generated_model_list(config),
        "general_settings": {"master_key": master_key},
    }


__all__ = [
    "AUTOROUTER_MODEL_NAME",
    "TIER_NAMES",
    "AutorouteConfig",
    "ClassifierChoice",
    "ConfigGenerationError",
    "DEFAULT_KEYWORD_TIER_RULES",
    "DiscoveredModel",
    "HeuristicClassifier",
    "KeywordTierRule",
    "LLMClassifier",
    "NoSemanticMatching",
    "SemanticMatching",
    "SemanticMatchingChoice",
    "build_generated_model_list",
    "build_generated_proxy_config",
    "chat_models",
    "embedding_models",
    "parse_discovered_models",
    "validate_config",
]
