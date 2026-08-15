"""Same-provider canonical model-name resolution.

Harnesses hardcode concrete model IDs. A client that asks for
``claude-haiku-4-5-20251001`` against a gateway that serves the very same model
under the deployment name ``anthropic/claude-haiku-4-5`` gets a 403/400 today,
even though the gateway *is* serving what was asked for. Only the spelling
differs.

This module builds a ``canonical name -> model group`` index from the router's
own deployments so that spelling difference can be bridged. Two hard rules keep
the bridge from becoming a guess:

1. **Identity, not similarity.** Two names are equivalent only when the model
   cost map attests they are the same model: both present, same
   ``litellm_provider``, same ``mode``, and identical pricing. Family/version
   hops (``claude-sonnet-4-5`` -> ``claude-sonnet-5``) are never equivalences,
   because that changes which model answers.
2. **Never across providers.** The requested name's inferred provider must equal
   the target deployment's provider. The same weights on Bedrock, Vertex, and
   the first-party API differ in credentials, data residency, quota pool, and
   billing; picking between them is an operator decision, not something a
   gateway should infer. Cross-provider mapping stays available through an
   explicit ``model_group_alias``.

Resolution is a last resort: callers consult it only after every existing route
(exact name, deployment id, ``model_group_alias``, routing group, team route,
wildcard/pattern route, ``default_deployment``) has declined, so a request that
succeeds today can never be re-pointed by this module.

See also ``Router.resolve_canonical_model_name``.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import litellm
from litellm._logging import verbose_router_logger

if TYPE_CHECKING:
    from litellm.types.router import DeploymentTypedDict

# Cost-map fields that must match exactly for two names to be called the same
# model. Pricing equality is a tripwire against false identity, not the
# definition of it -- the provider/mode checks below carry that weight.
_IDENTITY_ATTESTING_FIELDS: Final[tuple[str, ...]] = (
    "litellm_provider",
    "mode",
    "input_cost_per_token",
    "output_cost_per_token",
    "max_input_tokens",
    "max_output_tokens",
)

# Sentinel stored in the index when two distinct model groups claim the same
# canonical identity. Resolution then declines rather than silently picking a
# billing path the operator never sanctioned.
_AMBIGUOUS: Final = None


def _cost_map_entry(model: str) -> Mapping[str, object] | None:
    """The cost-map entry for ``model``, or None when absent."""
    entry: Final = litellm.model_cost.get(model)
    return entry if isinstance(entry, Mapping) else None


def _same_model_per_cost_map(name_a: str, name_b: str) -> bool:
    """Whether the cost map attests ``name_a`` and ``name_b`` are one model.

    Both names must be present with identical provider, mode, and pricing. A
    customer fine-tune or an unknown vanity name is absent from the map and so
    can never be equated with anything -- which is the point.
    """
    entry_a: Final = _cost_map_entry(name_a)
    entry_b: Final = _cost_map_entry(name_b)
    if entry_a is None or entry_b is None:
        return False
    return all(entry_a.get(field) == entry_b.get(field) for field in _IDENTITY_ATTESTING_FIELDS)


def _infer_provider(model: str) -> str | None:
    """The provider LiteLLM would route ``model`` to, or None if undecidable.

    Wraps ``get_llm_provider``, which raises ``BadRequestError`` for names it
    cannot place. An undecidable name simply never participates in resolution.
    """
    try:
        _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:
        return None
    return custom_llm_provider or None


def canonicalize(model: str) -> tuple[str, str] | None:
    """Reduce ``model`` to a ``(provider, canonical_name)`` identity.

    The canonical name is the model string with any LiteLLM provider-route
    prefix removed (``anthropic/claude-opus-5`` -> ``claude-opus-5``), which is
    LiteLLM's own routing syntax rather than part of the model's identity. The
    provider is carried alongside so equality checks are always provider-scoped.

    Returns None when the provider cannot be inferred.
    """
    if not model:
        return None
    provider: Final = _infer_provider(model)
    if provider is None:
        return None
    # get_llm_provider returns the model with its routing prefix stripped, which
    # is exactly the normalization wanted here.
    try:
        stripped, _, _, _ = litellm.get_llm_provider(model=model)
    except Exception:
        return None
    return (provider, stripped or model)


def _undated_variants(model: str) -> tuple[str, ...]:
    """Plausible dated<->undated spellings of ``model``, unvalidated.

    Purely syntactic candidate generation: every candidate is still gated by
    ``_same_model_per_cost_map`` before it is treated as an equivalence, so a
    coincidental date-like suffix on an unrelated model cannot create a false
    match (it will not be in the cost map, or will not match on pricing).

    Only an 8-digit ``-YYYYMMDD`` suffix is considered. Deliberately narrower
    than the cost-lookup heuristics in ``litellm.utils`` (which strip any
    trailing ``-\\d+`` and would conflate ``gemini-1.5-pro-001`` with ``-002``):
    a wrong match there mis-prices a log line, a wrong match here serves the
    wrong model.
    """
    parts: Final = model.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return (parts[0],)
    return ()


def build_canonical_index(
    deployments: list["DeploymentTypedDict"],
) -> dict[tuple[str, str], str | None]:
    """Map ``(provider, canonical_name) -> model group`` for ``deployments``.

    A model group is indexed only when every one of its deployments agrees on
    the same canonical identity; mixed groups are skipped. When two groups claim
    one identity the entry is set to ``_AMBIGUOUS`` (None) so lookups decline.

    Never raises: a malformed deployment or cost-map entry degrades to a smaller
    index, never to a router that fails to boot.
    """
    index: dict[tuple[str, str], str | None] = {}
    group_identity: dict[str, tuple[str, str] | None] = {}

    for deployment in deployments:
        try:
            model_group = deployment.get("model_name")
            litellm_params = deployment.get("litellm_params") or {}
            underlying = litellm_params.get("model") if isinstance(litellm_params, Mapping) else None
            if not isinstance(model_group, str) or not isinstance(underlying, str):
                continue

            identity = canonicalize(underlying)
            if model_group in group_identity and group_identity[model_group] != identity:
                # Deployments in this group disagree about what they serve; the
                # group cannot stand for a single canonical identity.
                group_identity[model_group] = None
                continue
            group_identity.setdefault(model_group, identity)
        except Exception as exc:  # pragma: no cover - defensive
            verbose_router_logger.debug("canonical-resolution: skipping deployment: %s", exc)
            continue

    for model_group, identity in group_identity.items():
        if identity is None:
            continue
        provider, canonical_name = identity
        # Index the canonical spelling plus any dated<->undated sibling the cost
        # map attests is the same model.
        spellings: list[str] = [canonical_name]
        for candidate in _undated_variants(canonical_name):
            if _same_model_per_cost_map(canonical_name, candidate):
                spellings.append(candidate)
        for dated, entry in litellm.model_cost.items():
            if not isinstance(entry, Mapping) or dated in spellings:
                continue
            if _undated_variants(dated) == (canonical_name,) and _same_model_per_cost_map(canonical_name, dated):
                spellings.append(dated)

        for spelling in spellings:
            key = (provider, spelling)
            existing = index.get(key, "__absent__")
            if existing == "__absent__":
                index[key] = model_group
            elif existing != model_group:
                # Two groups, same identity: decline rather than choose.
                index[key] = _AMBIGUOUS
                verbose_router_logger.info(
                    "canonical-resolution: '%s' is served by more than one model group "
                    "(%s, %s); auto-resolution disabled for it. Add an explicit "
                    "model_group_alias to pick one.",
                    spelling,
                    existing,
                    model_group,
                )

    return index


def lookup(
    index: Mapping[tuple[str, str], str | None],
    requested_model: str,
) -> str | None:
    """The model group serving ``requested_model``, or None.

    None covers every decline: unknown provider, no identity match, or an
    ambiguous identity. Pure dict lookup after canonicalization -- no I/O.
    """
    identity: Final = canonicalize(requested_model)
    if identity is None:
        return None
    target: Final = index.get(identity)
    if target is None:
        return None
    # A request already naming its serving group is not a rewrite.
    if target == requested_model:
        return None
    return target
