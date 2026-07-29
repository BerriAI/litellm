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

from typing import Literal, Mapping

AUTO_ROUTER_MODEL_PREFIX = "auto_router/"

StrategyRouterKind = Literal["semantic", "complexity", "adaptive", "quality"]

STRATEGY_ROUTER_PARAM_FIELDS: frozenset[str] = frozenset(
    {
        "auto_router_config",
        "auto_router_config_path",
        "auto_router_default_model",
        "auto_router_embedding_model",
        "complexity_router_config",
        "complexity_router_default_model",
        "adaptive_router_config",
        "quality_router_config",
        "quality_router_default_model",
    }
)

_REQUIRED_FIELD_GROUPS: Mapping[StrategyRouterKind, tuple[tuple[str, ...], ...]] = {
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
    remainder = model[len(AUTO_ROUTER_MODEL_PREFIX) :]
    if remainder.startswith("complexity_router"):
        return "complexity"
    if remainder.startswith("adaptive_router"):
        return "adaptive"
    if remainder.startswith("quality_router"):
        return "quality"
    return "semantic"


def validate_strategy_router_model_write(model: str, present_fields: frozenset[str]) -> str | None:
    """Check that writing ``model`` leaves a deployment the router can load.

    ``present_fields`` is the set of strategy-router param fields that are
    non-None on the deployment after the write (stored fields merged with the
    incoming ones). Returns a human-readable violation, or None when coherent.
    """
    kind = classify_strategy_router_model(model)
    if kind is None:
        offending = sorted(present_fields & STRATEGY_ROUTER_PARAM_FIELDS)
        if offending:
            return (
                f"litellm_params.model='{model}' does not start with '{AUTO_ROUTER_MODEL_PREFIX}' but the "
                f"deployment carries auto-router settings ({', '.join(offending)}), so the router could not "
                f"load it. Keep the '{AUTO_ROUTER_MODEL_PREFIX}' prefix; to change the name clients call, "
                "edit the public model_name instead."
            )
        return None
    remainder = model[len(AUTO_ROUTER_MODEL_PREFIX) :]
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
    missing = tuple(
        " or ".join(group) for group in _REQUIRED_FIELD_GROUPS[kind] if not any(f in present_fields for f in group)
    )
    if missing:
        return (
            f"litellm_params.model='{model}' selects the {kind} router, which requires "
            f"{'; '.join(missing)} in litellm_params."
        )
    return None
