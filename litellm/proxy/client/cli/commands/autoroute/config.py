from typing import Dict, FrozenSet, List, Literal, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

TIER_NAMES: Tuple[str, ...] = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")


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


_RAW_MODEL_GROUPS_ADAPTER = TypeAdapter(List[_RawModelGroup])


def parse_discovered_models(raw: List[JsonValue]) -> Tuple[DiscoveredModel, ...]:
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


def chat_models(models: Tuple[DiscoveredModel, ...]) -> Tuple[DiscoveredModel, ...]:
    return tuple(m for m in models if m.mode == "chat")


def embedding_models(models: Tuple[DiscoveredModel, ...]) -> Tuple[DiscoveredModel, ...]:
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


class SemanticMatching(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["semantic"] = "semantic"
    embedding_model: str
    match_threshold: float = 0.5


SemanticMatchingChoice = Union[NoSemanticMatching, SemanticMatching]

# Satisfies complexity_router's "semantic matching requires non-empty keyword_tier_rules"
# invariant with a sane starting point; the generated config.yaml can be hand-edited afterward.
_DEFAULT_KEYWORD_TIER_RULES: Tuple[Dict[str, JsonValue], ...] = (
    {"keywords": ["hi", "hello", "thanks"], "tier": "SIMPLE"},
    {"keywords": ["explain", "how does"], "tier": "MEDIUM"},
    {"keywords": ["refactor", "implement", "debug"], "tier": "COMPLEX"},
    {"keywords": ["step by step", "think through", "prove"], "tier": "REASONING"},
)


class AutorouteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str
    api_key: str
    tiers: Dict[str, str]
    default_model: str
    classifier: ClassifierChoice = Field(default_factory=HeuristicClassifier)
    semantic_matching: SemanticMatchingChoice = Field(default_factory=NoSemanticMatching)
    adaptive: bool = False


def validate_config(config: AutorouteConfig, discovered: Tuple[DiscoveredModel, ...]) -> None:
    """Raise ConfigGenerationError if config references a model discovery didn't return."""
    chat_names: FrozenSet[str] = frozenset(m.name for m in chat_models(discovered))
    embedding_names: FrozenSet[str] = frozenset(m.name for m in embedding_models(discovered))

    for tier, model in config.tiers.items():
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


def _litellm_proxy_deployment(name: str, base_url: str, api_key: str) -> Dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"litellm_proxy/{name}",
            "api_base": base_url,
            "api_key": api_key,
        },
    }


def build_generated_model_list(config: AutorouteConfig) -> List[JsonValue]:
    """Build the model_list for the ephemeral proxy's config.yaml.

    Every real model referenced anywhere (tier targets, classifier, embedding) is deduplicated
    to exactly one `litellm_proxy/<name>` deployment forwarding to the customer's real proxy,
    plus one `auto_router/complexity_router` deployment tying the tiers together.
    """
    referenced_names = {*config.tiers.values(), config.default_model}
    if isinstance(config.classifier, LLMClassifier):
        referenced_names.add(config.classifier.model)
    if isinstance(config.semantic_matching, SemanticMatching):
        referenced_names.add(config.semantic_matching.embedding_model)

    proxy_deployments = [
        _litellm_proxy_deployment(name, config.base_url, config.api_key) for name in sorted(referenced_names)
    ]

    complexity_router_config: Dict[str, JsonValue] = {
        "tiers": dict(config.tiers),
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
        complexity_router_config["keyword_tier_rules"] = list(_DEFAULT_KEYWORD_TIER_RULES)
    if config.adaptive:
        complexity_router_config["adaptive"] = True

    auto_router_deployment: Dict[str, JsonValue] = {
        "model_name": "autorouter",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": complexity_router_config,
        },
    }
    return [*proxy_deployments, auto_router_deployment]


def build_generated_proxy_config(config: AutorouteConfig, master_key: str) -> Dict[str, JsonValue]:
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
    "TIER_NAMES",
    "ConfigGenerationError",
    "DiscoveredModel",
    "parse_discovered_models",
    "chat_models",
    "embedding_models",
    "HeuristicClassifier",
    "LLMClassifier",
    "ClassifierChoice",
    "NoSemanticMatching",
    "SemanticMatching",
    "SemanticMatchingChoice",
    "AutorouteConfig",
    "validate_config",
    "build_generated_model_list",
]
