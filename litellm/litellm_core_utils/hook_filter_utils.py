import fnmatch
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.types.hook_filters import DEPLOYMENT_SCOPED_HOOK_NAMES, FILTERABLE_HOOK_NAMES, HookFilterConfig


def _matches_single(value: str | None, patterns: tuple[str, ...] | None) -> bool:
    if patterns is None:
        return True
    if value is None:
        return False
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def _matches_any_of(values: tuple[str, ...], patterns: tuple[str, ...] | None) -> bool:
    if patterns is None:
        return True
    return any(fnmatch.fnmatch(value, pattern) for value in values for pattern in patterns)


def should_run_hook_for_filters(
    hook_filter: HookFilterConfig | None,
    *,
    model: str | None,
    key_alias: str | None,
    model_tags: tuple[str, ...],
    request_tags: tuple[str, ...],
) -> bool:
    """
    True if a hook with this filter should run for this request. A filter of
    None (no hook_filters entry configured for this hook) always runs, so
    behavior is unchanged for every hook nobody has opted into filtering.
    """
    if hook_filter is None:
        return True
    return (
        _matches_single(model, hook_filter.models)
        and _matches_single(key_alias, hook_filter.key_aliases)
        and _matches_any_of(model_tags, hook_filter.model_tags)
        and _matches_any_of(request_tags, hook_filter.request_tags)
    )


def validate_hook_filters(owner_name: str, hook_filters: Mapping[str, HookFilterConfig]) -> None:
    """
    Fail fast on a misconfigured hook_filters block: an unknown hook name, or
    model_tags scoped to a hook that runs before a deployment is resolved (its
    tags don't exist yet, so the filter would either always or never match).
    """
    for hook_name, hook_filter in hook_filters.items():
        if hook_name not in FILTERABLE_HOOK_NAMES:
            raise ValueError(
                f"{owner_name}: hook_filters has unknown hook name '{hook_name}'; "
                f"valid hook names are {sorted(FILTERABLE_HOOK_NAMES)}"
            )
        if hook_filter.model_tags is not None and hook_name not in DEPLOYMENT_SCOPED_HOOK_NAMES:
            raise ValueError(
                f"{owner_name}: hook '{hook_name}' cannot filter on model_tags; deployment tags are only "
                f"resolved on {sorted(DEPLOYMENT_SCOPED_HOOK_NAMES)}"
            )


def parse_hook_filters(owner_name: str, raw_hook_filters: Mapping[str, object]) -> Mapping[str, HookFilterConfig]:
    """Builds a validated HookFilterConfig per hook name from a raw config mapping."""
    parsed: Final[Mapping[str, HookFilterConfig]] = MappingProxyType(
        {hook_name: HookFilterConfig.model_validate(raw_filter) for hook_name, raw_filter in raw_hook_filters.items()}
    )
    validate_hook_filters(owner_name, parsed)
    return parsed


def get_request_tags_for_hook_filters(data: dict) -> tuple[str, ...]:  # mutable-ok: matches external dict param
    """
    Shared by every dispatch site that builds a proxy-request-shaped ``data``
    dict (as opposed to litellm_logging.py's ``litellm_params`` shape, which
    nests the raw request under a differently-keyed ``proxy_server_request``).
    Lazily imports ``StandardLoggingPayloadSetup`` to avoid a circular import:
    ``litellm_logging.py`` imports ``call_custom_hook``, which imports this module.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    proxy_server_request: Final = data.get("metadata") or {}  # mutable-ok: read-only lookup
    tags: Final = StandardLoggingPayloadSetup._get_request_tags(  # pyright: ignore[reportPrivateUsage]  # shared
        data, proxy_server_request
    )
    return tuple(tags)
