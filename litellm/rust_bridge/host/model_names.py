"""The candidate names ``get_model_info`` tries for a model, for the Rust resolver"""

from typing import Final

from litellm.litellm_core_utils.get_llm_provider_logic import provider_banner_suppressed


def candidates(
    model: str, custom_llm_provider: str | None
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None, str | None]:
    """Main's ``get_potential_model_names`` with the provider banner suppressed. The names come back in the order
    ``combined_model_name``, ``region_free_combined_model_name``, ``split_model``, ``combined_stripped_model_name``,
    ``stripped_model_name``, ``provider_prefixed_model_name``, then the provider rows must match"""
    from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
    from litellm.utils import get_potential_model_names

    with provider_banner_suppressed():
        names: Final = get_potential_model_names(
            model=model, custom_llm_provider=custom_llm_provider or declared_authenticating_provider(model)
        )
    return (
        names["combined_model_name"],
        names["region_free_combined_model_name"],
        names["split_model"],
        names["combined_stripped_model_name"],
        names["stripped_model_name"],
        names["provider_prefixed_model_name"],
        names["custom_llm_provider"],
    )


def registration_candidates(
    key: str, row_provider: str | None, custom_llm_provider: str | None
) -> tuple[tuple[tuple[str, str | None, bool, bool, bool], ...], dict[str, object] | None, bool]:
    """The names a ``register_model`` row for ``key`` may inherit from, in main's order. Each is
    ``(name, provider, requires_provider, names_the_row, seeded)``. ``get_model_info``'s candidates come first, then
    ``_resolve_builtin_model_cost_entry``'s prefix, route and region stripped ones, which seed the row from the
    capability rules. Also returns that seed and whether the key's provider authenticates on provider info, which
    keeps the key's own row as the only base"""
    from litellm.constants import PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO
    from litellm.litellm_core_utils.fallback_generalizations import match_capability_generalizations
    from litellm.types.utils import LlmProvidersSet
    from litellm.utils import (
        _BEDROCK_REGION_PREFIXES,  # pyright: ignore[reportPrivateUsage]  # the host module reads main's resolver internals
        _strip_model_name,  # pyright: ignore[reportPrivateUsage]  # the host module reads main's resolver internals
    )

    provider: Final = row_provider or ""
    if provider in PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO or any(
        key.startswith(f"{skipped}/") for skipped in PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO
    ):
        return ((key, None, False, True, False),), None, True

    looked_up: Final = candidates(key, custom_llm_provider)
    lookup_provider: Final = looked_up[6]
    lookup_names: Final = (looked_up[0], key, looked_up[1], looked_up[2], looked_up[3], looked_up[4], looked_up[5])

    from litellm.llms.bedrock.common_utils import strip_bedrock_routing_prefix

    segments: Final = key.split("/")
    match_provider: Final = provider or (segments[0] if len(segments) > 1 and segments[0] in LlmProvidersSet else "")
    provider_prefix_count: Final = next(
        (idx for idx, segment in enumerate(segments[:-1]) if segment not in LlmProvidersSet), len(segments) - 1
    )
    prefix_stripped: Final = tuple("/".join(segments[idx:]) for idx in range(1, provider_prefix_count + 1))
    prefix_free: Final = prefix_stripped[-1] if prefix_stripped else key
    base: Final = (
        strip_bedrock_routing_prefix(prefix_free) if match_provider in ("bedrock", "bedrock_converse") else prefix_free
    )
    region_stripped: Final = tuple(
        base[len(region_prefix) :] for region_prefix in _BEDROCK_REGION_PREFIXES if base.startswith(region_prefix)
    )
    with provider_banner_suppressed():
        stripped: Final = _strip_model_name(model=base, custom_llm_provider=match_provider) if match_provider else base
    row_names: Final = frozenset((key, *prefix_stripped, base, stripped))
    builtin_names: Final = (key, *prefix_stripped, base, *region_stripped, stripped)
    return (
        (
            *((name, lookup_provider, False, True, False) for name in lookup_names if name is not None),
            *((name, match_provider or None, True, name in row_names, True) for name in builtin_names),
        ),
        match_capability_generalizations(key),
        False,
    )


def replay_live_deployments() -> None:
    """Re-registers the deployments of every live router after an install, main's ``_LiveDeploymentReplay``"""
    from litellm.utils import (
        _LiveDeploymentReplay,  # pyright: ignore[reportPrivateUsage]  # the host module reads main's resolver internals
    )

    if _LiveDeploymentReplay.callback is not None:
        _LiveDeploymentReplay.callback()
