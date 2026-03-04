"""
Google/Gemini model variant suffix resolution utilities.

Google exposes model variants via suffixes appended to the base model name
(e.g. ``gemini-3.1-pro-preview-customtools``).  When a request arrives for a
variant that is not explicitly configured in the router, we strip these
suffixes and fall back to the base model's deployment while forwarding the
original variant name to the upstream API.

Refs: https://github.com/BerriAI/litellm/issues/21697
"""

from typing import Callable, Dict, List, Optional, Set, Tuple

GOOGLE_MODEL_VARIANT_SUFFIXES: List[str] = ["-customtools"]

# Prefixes that identify a model as Google/Gemini/Vertex so that variant
# resolution is not accidentally applied to non-Google models.
# Only "gemini" typically matches user-facing model_name values at the
# router callsite.  The provider prefixes ("vertex_ai/", "google/", …) are
# included defensively so this function also works when called with fully-
# qualified litellm_params.model names from other code paths.
_GOOGLE_MODEL_PREFIXES: Tuple[str, ...] = (
    "gemini",
    "vertex_ai/",
    "vertex_ai_beta/",
    "google/",
)


def _is_google_model(model: str) -> bool:
    """Return True if *model* looks like a Google/Gemini/Vertex model."""
    model_lower = model.lower()
    return any(model_lower.startswith(p) for p in _GOOGLE_MODEL_PREFIXES)


def resolve_google_model_variant(
    model: str,
    model_names: Set[str],
    get_model_from_alias: Callable[[str], Optional[str]],
) -> Optional[Tuple[str, str]]:
    """
    Resolve Google/Gemini model variant suffixes.

    When a model like ``gemini-3.1-pro-preview-customtools`` is requested
    but no deployment exists, strip known variant suffixes and check
    whether the base model is registered.

    Only applies to Google/Gemini/Vertex models to avoid accidentally
    stripping suffixes from non-Google deployments.

    Parameters
    ----------
    model : str
        The requested model name (possibly with a variant suffix).
    model_names : set of str
        Set of model names currently registered in the router.
    get_model_from_alias : callable
        Function that resolves a model name via ``model_group_alias``.
        Should return ``None`` when no alias matches.

    Returns
    -------
    tuple of (str, str)
        ``(base_model_name, variant_suffix)`` when a match is found.
    None
        When no known variant suffix applies or the base model is not
        registered.
    """
    if not _is_google_model(model):
        return None

    for suffix in GOOGLE_MODEL_VARIANT_SUFFIXES:
        if model.endswith(suffix):
            base_model = model[: -len(suffix)]
            if base_model in model_names:
                return base_model, suffix
            alias_model = get_model_from_alias(base_model)
            if alias_model is not None:
                return alias_model, suffix
    return None


def build_variant_deployments(
    deployments: List[Dict],
    variant_suffix: str,
) -> List[Dict]:
    """
    Create shallow copies of *deployments* with the variant suffix appended
    to each deployment's ``litellm_params.model``.

    The original deployment dicts are never mutated.
    """
    variant_deployments: List[Dict] = []
    for dep in deployments:
        dep_copy = dep.copy()
        dep_copy["litellm_params"] = dep["litellm_params"].copy()
        dep_copy["litellm_params"]["model"] = (
            dep_copy["litellm_params"]["model"] + variant_suffix
        )
        variant_deployments.append(dep_copy)
    return variant_deployments
