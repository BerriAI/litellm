import re
from typing import Iterable, List, Optional

# Sentinels that mean "any model" in the existing key/team model lists.
_UNRESTRICTED_SENTINELS = {"*", "all-proxy-models", "all-team-models"}


def _is_unrestricted(allowed_models: List[str]) -> bool:
    # An empty list means "no restriction" in litellm, matching key/team semantics.
    if not allowed_models:
        return True
    return any(model in _UNRESTRICTED_SENTINELS for model in allowed_models)


def _matches_pattern(requested_model: str, pattern: str) -> bool:
    # Mirrors v1 is_model_allowed_by_pattern: '*' is the only wildcard.
    if "*" not in pattern:
        return False
    return bool(re.match("^" + pattern.replace("*", ".*") + "$", requested_model))


def can_call_model(
    allowed_models: Optional[List[str]],
    requested_model: str,
    model_access_groups: Optional[Iterable[str]] = None,
) -> bool:
    """Decide whether a principal with ``allowed_models`` may call ``requested_model``.

    Data-plane access is a direct membership/pattern predicate, not a policy
    engine: it runs on the inference hot path where a casbin evaluation would be
    pure overhead for what is a list check. Empty list or a sentinel means
    unrestricted; an exact name matches; a wildcard pattern (e.g. ``bedrock/*``)
    matches using v1's pattern semantics.

    ``model_access_groups`` are the access-group names ``requested_model`` belongs
    to (from the router); if the key lists any of them the call is allowed,
    mirroring v1 ``model_in_access_group``. Injected rather than resolved here so
    this stays a pure predicate.
    """
    models = list(allowed_models or [])
    if _is_unrestricted(models):
        return True
    if requested_model in models:
        return True
    if any(_matches_pattern(requested_model, model) for model in models):
        return True
    groups = model_access_groups or ()
    return any(model in groups for model in models)
