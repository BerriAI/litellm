"""Naming contract for strategy-router (auto-router) pseudo-models.

A deployment whose ``litellm_params.model`` starts with ``auto_router/`` does not
name a provider model; the string is the discriminator that selects which
pre-routing strategy owns the deployment. This module is the single source of
truth for classifying that string (``Router._is_*_router_deployment`` delegates
here) and for checking that a client-supplied write leaves the deployment
coherent, so management endpoints can reject corruption with a 400 instead of
the router silently dropping the deployment at load time under
``ignore_invalid_deployments``.
"""

from collections.abc import Mapping
from typing import Final, Literal

AUTO_ROUTER_MODEL_PREFIX: Final = "auto_router/"

StrategyRouterKind = Literal["semantic", "complexity", "adaptive", "quality"]

STRATEGY_ROUTER_PARAM_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "auto_router_config",
        "auto_router_config_path",
        "auto_router_default_model",
        "auto_router_embedding_model",
        "auto_router_max_input_chars",
        "complexity_router_config",
        "complexity_router_default_model",
        "adaptive_router_config",
        "quality_router_config",
        "quality_router_default_model",
    }
)

_REQUIRED_FIELD_GROUPS: Final[Mapping[StrategyRouterKind, tuple[tuple[str, ...], ...]]] = {
    "semantic": (
        ("auto_router_config", "auto_router_config_path"),
        ("auto_router_default_model",),
        ("auto_router_embedding_model",),
    ),
    "complexity": (("complexity_router_config", "complexity_router_default_model"),),
    "adaptive": (("adaptive_router_config",),),
    "quality": (("quality_router_config", "quality_router_default_model"),),
}


def classify_strategy_router_model(model: str) -> StrategyRouterKind | None:
    """Classify a ``litellm_params.model`` string the way the Router does.

    Returns None for regular provider models. Mirrors Router registration
    exactly: reserved names are matched by prefix, everything else under
    ``auto_router/`` is a semantic router.
    """
    if not model.startswith(AUTO_ROUTER_MODEL_PREFIX):
        return None
    remainder: Final = model[len(AUTO_ROUTER_MODEL_PREFIX) :]
    if remainder.startswith("complexity_router"):
        return "complexity"
    if remainder.startswith("adaptive_router"):
        return "adaptive"
    if remainder.startswith("quality_router"):
        return "quality"
    return "semantic"


def validate_complexity_router_config_write(complexity_router_config: Mapping[str, object] | None) -> str | None:
    """Reject a complexity config the router would refuse to build a deployment from.

    Parsed with the router's own ``ComplexityRouterConfig`` rather than a copy of
    its rules, so the boundary rejects exactly what the load would. Plugins are
    resolved from dotted paths only on the config.yaml path, so a written config
    reaches this function in the same shape the load hands to the same model.
    Judged on the config alone: a patch may write one without naming a model, and
    the stored model is encrypted at rest, so it cannot be classified here.
    """
    from pydantic import ValidationError

    from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig

    if complexity_router_config is None:
        return None
    try:
        _ = ComplexityRouterConfig.model_validate(complexity_router_config)
    except ValidationError as exc:
        first: Final = exc.errors()[0]
        location: Final = ".".join(str(part) for part in first.get("loc", ())) or "complexity_router_config"
        return (
            f"complexity_router_config is invalid at {location}: {first.get('msg', 'invalid value')}. "
            "The router would drop this deployment at load time, so the write is rejected instead."
        )
    return None


def validate_strategy_router_model_write(model: str, present_fields: frozenset[str]) -> str | None:
    """Check that writing ``model`` leaves a deployment the router can load.

    ``present_fields`` is the set of strategy-router param fields that are
    non-None on the deployment after the write (stored fields merged with the
    incoming ones). Returns a human-readable violation, or None when coherent.
    A config's contents are ``validate_complexity_router_config_write``'s to
    judge, since a write may carry one without naming a model at all.
    """
    kind: Final = classify_strategy_router_model(model)
    if kind is None:
        offending: Final = sorted(present_fields & STRATEGY_ROUTER_PARAM_FIELDS)
        if offending:
            return (
                f"litellm_params.model='{model}' does not start with '{AUTO_ROUTER_MODEL_PREFIX}' but the "
                f"deployment carries auto-router settings ({', '.join(offending)}), so the router could not "
                f"load it. Keep the '{AUTO_ROUTER_MODEL_PREFIX}' prefix; to change the name clients call, "
                "edit the public model_name instead."
            )
        return None
    remainder: Final = model[len(AUTO_ROUTER_MODEL_PREFIX) :]
    if remainder.startswith(AUTO_ROUTER_MODEL_PREFIX):
        return (
            f"litellm_params.model='{model}' repeats the '{AUTO_ROUTER_MODEL_PREFIX}' prefix, so the router "
            f"could not load it. Use '{remainder}'; to change the name clients call, edit the public "
            "model_name instead."
        )
    if not remainder:
        return (
            f"litellm_params.model='{model}' is missing the router name after the '{AUTO_ROUTER_MODEL_PREFIX}' prefix."
        )
    missing: Final = tuple(
        " or ".join(group) for group in _REQUIRED_FIELD_GROUPS[kind] if not any(f in present_fields for f in group)
    )
    if missing:
        return (
            f"litellm_params.model='{model}' selects the {kind} router, which requires "
            f"{'; '.join(missing)} in litellm_params."
        )
    return None
