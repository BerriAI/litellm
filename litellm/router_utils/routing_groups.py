"""
Validation for `router_settings.routing_groups`, shared by the Router and the
proxy's config-update endpoint so a config the UI saves cannot be one the
runtime refuses to load.
"""

from collections.abc import Sequence
from typing import Final

from litellm._logging import verbose_router_logger
from litellm.types.router import RoutingGroup, RoutingStrategy


def validate_routing_strategy(routing_strategy: RoutingStrategy | str | None) -> None:
    """
    Raises `ValueError` unless `routing_strategy` is a known strategy or None.

    See: https://github.com/BerriAI/litellm/issues/11330
    """
    if routing_strategy is None:
        return

    valid_strategy_strings: Final = ("simple-shuffle", "lar1", *(s.value for s in RoutingStrategy))
    is_valid_string: Final = isinstance(routing_strategy, str) and routing_strategy in valid_strategy_strings
    is_valid_enum: Final = isinstance(routing_strategy, RoutingStrategy)
    if not is_valid_string and not is_valid_enum:
        raise ValueError(
            f"Invalid routing_strategy: '{routing_strategy}'. "
            f"Valid options: {list(valid_strategy_strings)}. "
            f"Check 'router_settings.routing_strategy' in your config.yaml "
            f"or the 'routing_strategy' parameter if using the Router SDK directly."
        )


def parse_routing_groups(
    groups_input: Sequence[RoutingGroup | dict] | None,
    known_model_names: frozenset[str] = frozenset(),
) -> tuple[RoutingGroup, ...]:
    """
    Parses and validates `routing_groups`, raising `ValueError` on the first
    problem found. Every check runs before the caller mutates any state, so an
    invalid update can never leave a router holding a half-applied set of
    groups.
    """
    if not groups_input:
        return ()

    groups: Final = tuple(raw if isinstance(raw, RoutingGroup) else RoutingGroup(**raw) for raw in groups_input)

    if any(not group.group_name for group in groups):
        raise ValueError("routing_groups: group_name must be non-empty.")

    if any(group.group_name == "default" for group in groups):
        raise ValueError("routing_groups: 'default' is reserved for the implicit fallback group.")

    names: Final = tuple(group.group_name for group in groups)
    duplicate_names: Final = frozenset(name for name in names if names.count(name) > 1)
    if duplicate_names:
        raise ValueError(f"routing_groups: group names must be unique, duplicate group_name '{min(duplicate_names)}'.")

    for group in groups:
        validate_routing_strategy(group.routing_strategy)

    owners_by_model: Final = tuple(
        (model_name, tuple(group.group_name for group in groups if model_name in group.models))
        for model_name in dict.fromkeys(model_name for group in groups for model_name in group.models)
    )
    conflicts: Final = tuple(
        f"model_name '{model_name}' appears in {' and '.join(repr(owner) for owner in owners)}"
        for model_name, owners in owners_by_model
        if len(owners) > 1
    )
    if conflicts:
        raise ValueError(f"routing_groups: {'; '.join(conflicts)}. Each model may belong to at most one group.")

    unknown_models: Final = (
        tuple(
            (model_name, group.group_name)
            for group in groups
            for model_name in group.models
            if model_name not in known_model_names
        )
        if known_model_names
        else ()
    )
    for model_name, group_name in unknown_models:
        verbose_router_logger.warning(
            "routing_groups: model_name '%s' (group '%s') is not in model_list; "
            "the group entry will only take effect once a deployment with that "
            "model_name is added.",
            model_name,
            group_name,
        )

    return groups
