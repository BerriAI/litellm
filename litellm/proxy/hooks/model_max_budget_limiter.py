import asyncio
import functools
import itertools
import json
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    TypeAlias,
    cast,  # noqa: TID251  # untyped callback payload requires explicit narrowing
)

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import Span
from litellm.litellm_core_utils.batch_utils import batch_charged_once_marker_key, batch_id_to_charge_once
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.llms.bedrock.common_utils import get_bedrock_base_model, get_bedrock_cross_region_inference_regions
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.hooks.model_budget_shared_windows import (
    PROXY_SHARED_WINDOWS,
    LocalTTLCache,
    SharedWindow,
    SharedWindows,
)
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting, budget_window_seconds
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import BudgetConfig, LlmProviders, StandardLoggingPayload

if TYPE_CHECKING:
    from litellm.router import Router

VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX: Final = "virtual_key_spend"
END_USER_SPEND_CACHE_KEY_PREFIX: Final = "end_user_model_spend"
USER_SPEND_CACHE_KEY_PREFIX: Final = "user_model_spend"
TEAM_SPEND_CACHE_KEY_PREFIX: Final = "team_model_spend"
TEAM_MEMBER_SPEND_CACHE_KEY_PREFIX: Final = "team_member_model_spend"

_SPEND_CACHE_KEY_PREFIXES: Final = MappingProxyType(
    {
        Litellm_EntityType.KEY: VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.USER: USER_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.END_USER: END_USER_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.TEAM: TEAM_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.TEAM_MEMBER: TEAM_MEMBER_SPEND_CACHE_KEY_PREFIX,
    }
)

_LEGACY_REQUEST_MODEL_SCOPES: Final = frozenset({Litellm_EntityType.KEY, Litellm_EntityType.END_USER})

_PER_REPLICA_SCOPES: Final = frozenset({Litellm_EntityType.END_USER})

_PROCESS_STARTED_AT: Final = time.monotonic()

_LATEST_RESET_AT: Final = datetime(9999, 12, 31, tzinfo=timezone.utc).timestamp()

_GLOB_WILDCARD: Final = "*"

ModelAccessGroups: TypeAlias = Callable[[str, str | None], Collection[str]]

_PROVIDER_PREFIXES: Final = tuple(f"{provider.value}/" for provider in LlmProviders)

_BUDGET_START_TIME_KEY_PREFIXES: Final = MappingProxyType(
    {
        Litellm_EntityType.KEY: "virtual_key_budget_start_time",
        Litellm_EntityType.USER: "user_model_budget_start_time",
        Litellm_EntityType.END_USER: "end_user_budget_start_time",
        Litellm_EntityType.TEAM: "team_model_budget_start_time",
        Litellm_EntityType.TEAM_MEMBER: "team_member_model_budget_start_time",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedModelBudget:
    """The `model_max_budget` entry a request resolved to.

    ``budget_model`` is the key as the operator configured it, not the model
    name on the request. Every counter is keyed on it so enforcement, the
    post-call increment and the `/key/info` + `/user/info` usage reads cannot
    disagree about which counter a request belongs to.
    """

    budget_model: str
    budget_config: BudgetConfig


@dataclass(frozen=True, slots=True)
class RoutedModels:
    served_names: frozenset[str]
    group_members: Mapping[str, frozenset[str]]

    def members(self, access_group: str) -> frozenset[str]:
        return self.group_members.get(access_group, frozenset())


def team_member_budget_entity_id(user_id: str, team_id: str) -> str:
    return f"{len(user_id)}:{user_id}:{team_id}"


def model_budget_spend_cache_key(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    budget_model: str,
    budget_duration: str | None,
) -> str:
    """Sole owner of the per-model spend counter key, shared by its writer and all of its readers."""
    return f"{_SPEND_CACHE_KEY_PREFIXES[entity_type]}:{entity_id}:{budget_model}:{budget_duration}"


def _legacy_request_model_spend_cache_key(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    model: str,
    resolved: ResolvedModelBudget,
) -> str | None:
    """The counter this request was billed to before the budget model owned the key, or None.

    Upgrading proxies carry live counters keyed on the model as REQUESTED
    (`openai/gpt-4`) rather than as configured (`gpt-4`), and those were the
    counters the previous version enforced on. Nothing writes that spelling once
    this version is running, so the pre-upgrade and post-upgrade counters hold
    disjoint halves of one window and adding them is the window's real spend.

    Only the key and end-user scopes ever had one. The user scope is introduced
    by this change, so it has no counter to carry.

    The carry stops one budget window after start-up, because a legacy counter
    belongs to a window that was already open when this process replaced the one
    writing it. Past that point the lookup could only ever miss.
    """
    budget_duration: Final = resolved.budget_config.budget_duration
    if entity_type not in _LEGACY_REQUEST_MODEL_SCOPES or budget_duration is None:
        return None
    if time.monotonic() - _PROCESS_STARTED_AT >= duration_in_seconds(budget_duration):
        return None
    return model_budget_spend_cache_key(
        entity_type=entity_type,
        entity_id=entity_id,
        budget_model=model,
        budget_duration=budget_duration,
    )


def model_budget_start_time_cache_key(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    budget_model: str,
    budget_duration: str | None,
) -> str:
    """Window start for one (entity, budget model) pair.

    Scoped per budget model because an entity may budget two models over
    different periods, and a shared start time lets the shorter period restart
    the longer one's window.
    """
    return f"{_BUDGET_START_TIME_KEY_PREFIXES[entity_type]}:{entity_id}:{budget_model}:{budget_duration}"


def model_access_groups_of(router: "Router", model: str, team_id: str | None = None) -> frozenset[str]:
    """Access groups serving `model`, or the team's own model of that name, leaving out any named after a model the
    router serves.

    An entry with that name already budgets that model, so reading it as the group too would quietly widen it.
    """
    return frozenset(
        group
        for group in router.get_model_access_groups(model_name=model, team_id=team_id)
        if group not in router.model_names
    )


def routed_models_of(router: "Router", team_id: str | None = None) -> RoutedModels:
    serves_every_name: Final = (
        router.default_deployment is not None or router.router_general_settings.pass_through_all_models
    )
    served_names: Final = frozenset(router.model_names) | (
        frozenset({_GLOB_WILDCARD}) if serves_every_name else frozenset()
    )
    team_members: Final = _team_public_names_by_group(router=router, team_id=team_id)
    return RoutedModels(
        served_names=served_names,
        group_members=MappingProxyType(
            {
                group: frozenset(members) | team_members.get(group, frozenset())
                for group, members in router.get_model_access_groups().items()
                if group not in served_names
            }
        ),
    )


def _team_public_names_by_group(router: "Router", team_id: str | None) -> Mapping[str, frozenset[str]]:
    if team_id is None:
        return MappingProxyType({})
    groups_by_name: Final = MappingProxyType(
        {
            name: model_access_groups_of(router=router, model=name, team_id=team_id)
            for name in router.team_public_model_names_of(team_id)
        }
    )
    return MappingProxyType(
        {
            group: frozenset(name for name, groups in groups_by_name.items() if group in groups)
            for group in frozenset[str]().union(*groups_by_name.values())
        }
    )


def proxy_model_access_groups(model: str, team_id: str | None = None) -> frozenset[str]:
    router: Final = _proxy_router()
    return frozenset() if router is None else model_access_groups_of(router=router, model=model, team_id=team_id)


def proxy_routed_models(team_id: str | None = None) -> RoutedModels:
    router: Final = _proxy_router()
    if router is None:
        return RoutedModels(served_names=frozenset(), group_members=MappingProxyType({}))
    return routed_models_of(router=router, team_id=team_id)


def _proxy_router() -> "Router | None":
    try:
        from litellm.proxy.proxy_server import llm_router
    except ImportError:
        return None
    return llm_router


def resolve_model_budget(
    model: str,
    model_max_budget: Mapping[str, object],
    model_access_groups: ModelAccessGroups = proxy_model_access_groups,
    team_id: str | None = None,
) -> ResolvedModelBudget | None:
    """Find the `model_max_budget` entry that governs `model`, or None.

    An entry named after the model wins, then one named after a model access
    group serving it, then the most specific glob (`claude-opus*`). Every model
    an access group or glob entry covers draws from that entry's one counter.
    """
    resolved: Final = (
        _usable_entry(entry=entry, raw_budget_config=model_max_budget[entry])
        for entry in _governing_entry_names(
            model=model, model_max_budget=model_max_budget, model_access_groups=model_access_groups, team_id=team_id
        )
    )
    return next((entry for entry in resolved if entry is not None), None)


def _governing_entry_names(
    model: str,
    model_max_budget: Mapping[str, object],
    model_access_groups: ModelAccessGroups,
    team_id: str | None,
) -> Iterator[str]:
    candidates: Final = _budget_model_candidates(model)
    yield from (candidate for candidate in candidates if candidate in model_max_budget)
    group_entries: Final = tuple(
        sorted(entry for entry in model_max_budget if _GLOB_WILDCARD not in entry and entry not in candidates)
    )
    if group_entries:
        groups: Final = model_access_groups(model, team_id)
        yield from (entry for entry in group_entries if entry in groups)
    yield from sorted(
        (entry for entry in model_max_budget if _GLOB_WILDCARD in entry and _glob_matches(entry, candidates)),
        key=_glob_precedence,
        reverse=True,
    )


def _glob_matches(pattern: str, names: Iterable[str]) -> bool:
    return any(_glob_matches_name(pattern=pattern, name=name) for name in names)


def _glob_matches_name(pattern: str, name: str) -> bool:
    """Whole-name match where `*` is the only wildcard, in linear time whatever the pattern holds.

    Taking each literal piece between two wildcards at its first occurrence is
    always safe, since a later occurrence only leaves less room for the rest, so
    the match never needs to backtrack the way a `.*` regex does.
    """
    if _GLOB_WILDCARD not in pattern:
        return pattern == name
    head, *middle, tail = pattern.split(_GLOB_WILDCARD)
    middle_end: Final = len(name) - len(tail)
    if middle_end < len(head) or not name.startswith(head) or not name.endswith(tail):
        return False
    return functools.reduce(_after_piece(name=name, end=middle_end), middle, len(head)) >= 0


def _after_piece(name: str, end: int) -> Callable[[int, str], int]:
    def step(position: int, piece: str) -> int:
        found: Final = name.find(piece, position, end) if position >= 0 else -1
        return found + len(piece) if found >= 0 else -1

    return step


def _glob_precedence(pattern: str) -> tuple[int, int, str]:
    """More literal characters first, then fewer wildcards, so padding a pattern with `*` never makes it win."""
    wildcards: Final = pattern.count(_GLOB_WILDCARD)
    return len(pattern) - wildcards, -wildcards, pattern


@dataclass(frozen=True, slots=True)
class _RequestForm:
    """Request names that an entry governs.

    Every such name matches all of ``name_globs`` and also resolves through
    ``candidate_globs``. Its last ``/`` segment is among them unless
    ``last_segment_open``.
    """

    name_globs: tuple[str, ...]
    candidate_globs: tuple[str, ...] = ()
    last_segment_open: bool = False


def entries_outranked_by(entry: str, model_max_budget: Collection[str], routed_models: RoutedModels) -> Iterator[str]:
    """Entries that `entry` can take precedence over for some model a request can name.

    A model or group entry is resolved before any pattern, and a pattern before
    every less specific one, so `entry` outranks a pattern that some request it
    governs also resolves to. A group entry with a wildcard route among its
    members outranks every pattern.
    """
    patterns: Final = tuple(other for other in model_max_budget if _GLOB_WILDCARD in other)
    if _GLOB_WILDCARD in entry:
        entry_precedence: Final = _glob_precedence(entry)
        pattern_forms: Final = tuple(_pattern_request_forms(pattern=entry, routed_models=routed_models))
        yield from (
            other
            for other in patterns
            if _glob_precedence(other) < entry_precedence and _resolves_any(pattern=other, forms=pattern_forms)
        )
        return
    entry_members: Final = routed_models.members(entry)
    covers_wildcard_route: Final = any(_GLOB_WILDCARD in member for member in entry_members)
    entry_forms: Final = tuple(
        _entry_request_forms(entry=entry, entry_members=entry_members, routed_models=routed_models)
    )
    yield from (other for other in patterns if covers_wildcard_route or _resolves_any(pattern=other, forms=entry_forms))
    yield from (
        other
        for other in model_max_budget
        if _GLOB_WILDCARD not in other
        and _ranks_ahead_of_group(
            entry=entry, entry_members=entry_members, group=other, group_members=routed_models.members(other)
        )
    )


def _resolves_any(pattern: str, forms: Iterable[_RequestForm]) -> bool:
    return any(_resolves(pattern=pattern, form=form) for form in forms)


def _resolves(pattern: str, form: _RequestForm) -> bool:
    return (
        _globs_share_a_name(pattern, *form.name_globs)
        or any(_globs_share_a_name(pattern, candidate) for candidate in form.candidate_globs)
        or (form.last_segment_open and "/" not in pattern and _globs_share_a_name(f"*/{pattern}", *form.name_globs))
    )


def _globs_share_a_name(*globs: str) -> bool:
    """Whether one name matches every glob.

    Globs that each hold a `*` share a name exactly when their literal heads
    are prefixes of one another and so are their literal tails: the longest
    head, every middle piece and the longest tail, in that order, matches all
    of them.
    """
    literal: Final = next((glob for glob in globs if _GLOB_WILDCARD not in glob), None)
    if literal is not None:
        return all(_glob_matches_name(pattern=glob, name=literal) for glob in globs)
    heads: Final = sorted((glob.partition(_GLOB_WILDCARD)[0] for glob in globs), key=len)
    tails: Final = sorted((glob.rpartition(_GLOB_WILDCARD)[2] for glob in globs), key=len)
    return all(longer.startswith(shorter) for shorter, longer in itertools.pairwise(heads)) and all(
        longer.endswith(shorter) for shorter, longer in itertools.pairwise(tails)
    )


def _served_form(name: str) -> _RequestForm:
    return _RequestForm(name_globs=(name,), candidate_globs=_budget_model_candidates(name))


def _wildcard_routes(routed_models: RoutedModels) -> tuple[str, ...]:
    return tuple(sorted(name for name in routed_models.served_names if _GLOB_WILDCARD in name))


def _entry_request_forms(
    entry: str, entry_members: frozenset[str], routed_models: RoutedModels
) -> Iterator[_RequestForm]:
    yield from (
        _served_form(name)
        for name in _names_governed_by(entry=entry, entry_members=entry_members, routed_models=routed_models)
    )
    routes: Final = _wildcard_routes(routed_models)
    yield from (
        _RequestForm(name_globs=(route, f"*/{entry}"), candidate_globs=_budget_model_candidates(entry))
        for route in routes
    )
    yield from _bedrock_request_forms(
        bedrock_models=tuple(model for model in _bedrock_base_models() if entry in _bedrock_names(model)),
        routes=routes,
    )


def _pattern_request_forms(pattern: str, routed_models: RoutedModels) -> Iterator[_RequestForm]:
    yield _RequestForm(name_globs=(pattern,))
    yield from (
        _served_form(name)
        for name in routed_models.served_names
        if _GLOB_WILDCARD not in name and _glob_matches(pattern, _budget_model_candidates(name))
    )
    routes: Final = _wildcard_routes(routed_models)
    yield from (_RequestForm(name_globs=(route, pattern), last_segment_open=True) for route in routes)
    if "/" not in pattern:
        yield from (_RequestForm(name_globs=(route, f"*/{pattern}"), candidate_globs=(pattern,)) for route in routes)
    yield from (
        _narrowed_to(pattern=pattern, form=form)
        for form in _bedrock_request_forms(bedrock_models=_bedrock_base_models(), routes=routes)
        if _resolves(pattern=pattern, form=form)
    )


def _narrowed_to(pattern: str, form: _RequestForm) -> _RequestForm:
    if any(_globs_share_a_name(pattern, candidate) for candidate in form.candidate_globs):
        return form
    return _RequestForm(name_globs=(*form.name_globs, pattern), candidate_globs=form.candidate_globs)


def _bedrock_request_forms(bedrock_models: Iterable[str], routes: Sequence[str]) -> Iterator[_RequestForm]:
    """Wildcard-routed request names that resolve to a Bedrock model: its id, under a cross-region prefix or with a
    context or throughput suffix, as the whole name or after any routing prefix, region path or ARN."""
    forms: Final = itertools.chain.from_iterable(
        _bedrock_model_request_forms(model=model, routes=routes) for model in bedrock_models
    )
    yield from (form for form in forms if _globs_share_a_name(*form.name_globs))


def _bedrock_model_request_forms(model: str, routes: Sequence[str]) -> Iterator[_RequestForm]:
    names: Final = _bedrock_names(model)
    return (
        _RequestForm(name_globs=(route, f"{prefix}{last_segment}"), candidate_globs=(last_segment, *names))
        for route, (last_segment, prefix) in itertools.product(
            routes, itertools.product(_bedrock_last_segments(model), ("", "*/"))
        )
    )


def _bedrock_last_segments(model: str) -> tuple[str, ...]:
    regioned: Final = (model, *(f"{region}.{model}" for region in get_bedrock_cross_region_inference_regions()))
    _, version_separator, version = model.rpartition(":")
    suffixes: Final = ("", "[*]", *((":*k",) if version_separator and version.isdigit() else ()))
    return tuple(f"{name}{suffix}" for name, suffix in itertools.product(regioned, suffixes))


def _model_cost() -> Mapping[str, object]:
    return litellm.model_cost


def _bedrock_base_models() -> tuple[str, ...]:
    return tuple(
        model
        for model, cost_entry in _model_cost().items()
        if _priced_as_bedrock(cost_entry) and get_bedrock_base_model(model) == model
    )


def _names_governed_by(entry: str, entry_members: frozenset[str], routed_models: RoutedModels) -> Iterator[str]:
    yield entry
    yield from (f"{prefix}{entry}" for prefix in _PROVIDER_PREFIXES)
    yield from (
        name
        for name in routed_models.served_names
        if _GLOB_WILDCARD not in name and entry in _budget_model_candidates(name)
    )
    yield from (member for member in entry_members if _GLOB_WILDCARD not in member)


def _ranks_ahead_of_group(entry: str, entry_members: frozenset[str], group: str, group_members: Iterable[str]) -> bool:
    return any(
        _GLOB_WILDCARD in member
        or entry in _budget_model_candidates(member)
        or (member in entry_members and entry < group)
        for member in group_members
    )


def _usable_entry(entry: str, raw_budget_config: object) -> ResolvedModelBudget | None:
    if (budget_config := _usable_budget_config(raw_budget_config)) is None:
        # An entry that will not validate cannot be keyed, so it cannot be
        # enforced or incremented. Skip to the next candidate rather than
        # raising: raising would abort every other scope's increment and turn
        # a config typo into a 500, and stopping here would let one malformed
        # specific entry disable a perfectly good bare-family budget beside
        # it. The candidate chain already falls through an ABSENT entry, and
        # an unparseable one is indistinguishable from absent to enforcement.
        # `validate_model_max_budget` rejects these on the write path, so
        # reaching here means config.yaml or a direct DB edit.
        verbose_proxy_logger.warning(
            "Ignoring unusable model_max_budget entry for %s; it cannot be enforced or tracked",
            entry,
        )
        return None
    return ResolvedModelBudget(budget_model=entry, budget_config=budget_config)


def team_model_budget_entries(
    model: str, team_model_max_budget: Mapping[str, object], key_model_max_budget: Mapping[str, object] | None
) -> Mapping[str, object]:
    """The team entries that budget `model` for this key.

    A key entry named after `model` that spend-gates it replaces the team entry named after it, so that model is
    gated on and billed to the key's entry. The team's pattern and group entries still apply beside it.
    """
    exact_candidates: Final = _budget_model_candidates(model)
    if not _key_entry_replaces_team_entry(
        model=model, exact_candidates=exact_candidates, key_model_max_budget=key_model_max_budget
    ):
        return team_model_max_budget
    return MappingProxyType(
        {entry: config for entry, config in team_model_max_budget.items() if entry not in exact_candidates}
    )


def _key_entry_replaces_team_entry(
    model: str, exact_candidates: tuple[str, ...], key_model_max_budget: Mapping[str, object] | None
) -> bool:
    if not key_model_max_budget:
        return False
    resolved: Final = resolve_model_budget(
        model=model,
        model_max_budget=MappingProxyType(
            {
                candidate: key_model_max_budget[candidate]
                for candidate in exact_candidates
                if candidate in key_model_max_budget
            }
        ),
        model_access_groups=lambda _model, _team_id: frozenset(),
    )
    return resolved is not None and _spend_gated(resolved.budget_config)


def _spend_gated(budget_config: BudgetConfig) -> bool:
    return budget_config.max_budget is not None and budget_config.max_budget >= 0


def entry_names_model(entry: str, model: str) -> bool:
    """Whether `entry` budgets `model` under the model's own name, rather than as a pattern or an access group."""
    return entry in _budget_model_candidates(model)


def _budget_model_candidates(model: str) -> tuple[str, ...]:
    """Names a budget may be configured under for a request on `model`, most specific first.

    Beyond the model as sent, a budget may be keyed on the model without its
    ``{custom_llm_provider}/`` prefix (``gpt-4o`` governs ``openai/gpt-4o``), on
    the Bedrock base model (``anthropic.claude-opus-4-8`` governs the
    cross-region ``us.anthropic.claude-opus-4-8``), or on the bare family name
    that Bedrock id shares with its direct-provider twin (``claude-opus-4-8``).
    """
    return tuple(dict.fromkeys((model, model.split("/")[-1], *_bedrock_candidates(model))))


def _bedrock_candidates(model: str) -> tuple[str, ...]:
    """Bedrock-only candidates, empty unless litellm prices `model` as a Bedrock model.

    Gating on the cost map rather than on a vendor allowlist is what makes
    splitting the leading dotted segment safe: most dotted model ids are not
    Bedrock ids at all (``azure/gpt-4.1``, ``gpt-image-1.5``), and splitting one
    of those would produce a garbage candidate.
    """
    base_model: Final = get_bedrock_base_model(model)
    return _bedrock_names(base_model) if _priced_as_bedrock(_model_cost().get(base_model)) else ()


def _bedrock_names(base_model: str) -> tuple[str, ...]:
    _, _, without_vendor = base_model.partition(".")
    return (base_model, without_vendor) if without_vendor else (base_model,)


def _priced_as_bedrock(cost_entry: object) -> bool:
    return str(_metadata_value(cost_entry, "litellm_provider") or "").startswith("bedrock")


@dataclass(frozen=True, slots=True)
class _WindowReading:
    current_spend: float
    started_at: float | None


async def build_model_max_budget_usage(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    model_max_budget: Mapping[str, object] | None,
    cache: DualCache | None,
    now: datetime | None = None,
    shared_windows: SharedWindows = PROXY_SHARED_WINDOWS,
) -> dict[str, dict[str, object]]:
    """Current-window spend per configured budget model, as `/key/info` and `/user/info` report it.

    `cache` must be the DualCache the limiter writes the counters to; callers
    read it off the limiter rather than re-deriving it, so a scope that is being
    blocked can never report zero usage.

    `budget_reset_at` is when the open window ends and its spend counter
    expires, which is the window's start plus its period. It is None when no
    window is open, because nothing has been charged since the last one ended,
    and when the window ends after year 9999, so it never resets.
    """
    if cache is None or entity_id is None or not model_max_budget:
        return {}

    budgets: Final = tuple(
        (budget_model, budget_config)
        for budget_model, raw_budget_config in model_max_budget.items()
        for budget_config in (_usable_budget_config(raw_budget_config),)
        if budget_config is not None
    )
    if not budgets:
        return {}
    spend_keys: Final = tuple(
        model_budget_spend_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            budget_model=budget_model,
            budget_duration=budget_config.budget_duration,
        )
        for budget_model, budget_config in budgets
    )
    windows: Final = await _read_windows(
        cache=cache,
        spend_keys=spend_keys,
        start_time_keys=tuple(
            model_budget_start_time_cache_key(
                entity_type=entity_type,
                entity_id=entity_id,
                budget_model=budget_model,
                budget_duration=budget_config.budget_duration,
            )
            for budget_model, budget_config in budgets
        ),
    )
    current_time: Final = (now or datetime.now(timezone.utc)).timestamp()
    shared: Final = tuple(
        [
            (
                await _open_shared_window(
                    shared_windows=shared_windows,
                    spend_key=spend_key,
                    budget_duration=budget_config.budget_duration,
                    now=current_time,
                )
                if cache.redis_cache is None
                else None
            )
            for spend_key, (_, budget_config) in zip(spend_keys, budgets, strict=True)
        ]
    )
    readings: Final = tuple(
        _fuller_reading(local=window, shared=shared_window)
        for window, shared_window in zip(windows, shared, strict=True)
    )
    return {
        budget_model: {
            "current_spend": round(reading.current_spend, 4),
            "budget_limit": budget_config.max_budget,
            "time_period": budget_config.budget_duration,
            "budget_reset_at": _window_reset_at(
                started_at=reading.started_at,
                budget_duration=budget_config.budget_duration,
                now=current_time,
            ),
        }
        for (budget_model, budget_config), reading in zip(budgets, readings, strict=True)
    }


async def _open_shared_window(
    shared_windows: SharedWindows, spend_key: str, budget_duration: str | None, now: float
) -> SharedWindow | None:
    if budget_duration is None:
        return None
    window: Final = await shared_windows.read(spend_key=spend_key, budget_duration=budget_duration)
    if window is None or not window.is_open(budget_duration=budget_duration, now=now):
        return None
    return window


def _fuller_reading(local: _WindowReading, shared: SharedWindow | None) -> _WindowReading:
    if shared is None or shared.spend <= local.current_spend:
        return local
    return _WindowReading(current_spend=shared.spend, started_at=shared.started_at)


@with_service_target("model_budgets")
async def _read_windows(
    cache: DualCache, spend_keys: Sequence[str], start_time_keys: Sequence[str]
) -> tuple[_WindowReading, ...]:
    """Every window's spend and start time in one batched read, however many models are budgeted.

    Redis holds the window total across replicas; the in-memory copy is one replica's share.
    """
    keys: Final = [*spend_keys, *start_time_keys]
    redis_cache: Final = cache.redis_cache
    if redis_cache is not None:
        shared: Final = await redis_cache.async_batch_get_cache(key_list=keys)
        return _readings(tuple(_as_float(shared.get(key)) for key in keys), window_count=len(spend_keys))
    # async_batch_get_cache returns None if it fails internally, and its result is
    # index-aligned with `keys` otherwise. An unusable result reads as a miss,
    # which is what a never-written counter already reads as.
    batched: Final = await cache.async_batch_get_cache(keys=keys)
    if not isinstance(batched, list) or len(batched) != len(keys):
        return tuple(_WindowReading(current_spend=0.0, started_at=None) for _ in spend_keys)
    return _readings(tuple(_as_float(value) for value in batched), window_count=len(spend_keys))


def _readings(values: Sequence[float | None], window_count: int) -> tuple[_WindowReading, ...]:
    """`values` holds every window's spend followed by every window's start, in the same order."""
    return tuple(
        _WindowReading(current_spend=spend or 0.0, started_at=started_at)
        for spend, started_at in zip(values[:window_count], values[window_count:], strict=True)
    )


def _window_reset_at(started_at: float | None, budget_duration: str | None, now: float) -> str | None:
    if started_at is None or budget_duration is None:
        return None
    ends_at: Final = started_at + budget_window_seconds(started_at, budget_duration)
    if ends_at <= now or ends_at > _LATEST_RESET_AT:
        return None
    return datetime.fromtimestamp(ends_at, tz=timezone.utc).isoformat()


def is_enforced_budget_entry(raw_budget_config: object) -> bool:
    return _usable_budget_config(raw_budget_config) is not None


def _usable_budget_config(raw_budget_config: object) -> BudgetConfig | None:
    try:
        budget_config: Final = BudgetConfig.model_validate(raw_budget_config)
        if budget_config.budget_duration is None:
            return None
        duration_in_seconds(budget_config.budget_duration)
    except Exception:  # noqa: BLE001  # a malformed entry must not fail the whole report
        return None
    return budget_config


def _as_spend(current_spend: object) -> float:
    try:
        return float(current_spend or 0.0)  # pyright: ignore[reportArgumentType]  # non-numeric falls to the except
    except (TypeError, ValueError):
        return 0.0


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # pyright: ignore[reportArgumentType]  # non-numeric falls to the except
    except (TypeError, ValueError):
        return None


def _resolve_entity_model_budgets(
    model: str,
    entity_budgets: Iterable[tuple[Litellm_EntityType, str | None, object]],
    model_access_groups: ModelAccessGroups,
    team_id: str | None,
) -> tuple[tuple[Litellm_EntityType, str, ResolvedModelBudget], ...]:
    """Drop the scopes that do not budget `model`, keeping only what can be incremented."""
    return tuple(
        (entity_type, entity_id, resolved)
        for entity_type, entity_id, model_max_budget in entity_budgets
        if entity_id is not None and isinstance(model_max_budget, Mapping) and model_max_budget
        for resolved in (
            resolve_model_budget(
                model=model,
                model_max_budget=model_max_budget,
                model_access_groups=model_access_groups,
                team_id=team_id,
            ),
        )
        if resolved is not None and resolved.budget_config.budget_duration is not None
    )


def _requested_model_name(
    metadata: Mapping[str, object], standard_logging_payload: StandardLoggingPayload
) -> str | None:
    """The model name the request was gated on: a team-scoped deployment's public name, since the router renames
    its model group to a unique internal one, else the model group, else the deployment's model."""
    team_public_model_name: Final = _metadata_value(metadata.get("model_info"), "team_public_model_name")
    if (
        isinstance(team_public_model_name, str)
        and team_public_model_name
        and _GLOB_WILDCARD not in team_public_model_name
    ):
        return team_public_model_name
    return standard_logging_payload.get("model_group") or standard_logging_payload.get("model")


def _metadata_value(metadata: object, key: str) -> object | None:
    if not isinstance(metadata, Mapping):
        return None
    metadata_mapping: Final[Mapping[str, object]] = cast(  # cast-ok: callback metadata has string keys
        Mapping[str, object], metadata
    )
    return metadata_mapping.get(key)


def _metadata_string_value(metadata: object, key: str) -> str | None:
    value: Final = _metadata_value(metadata, key)
    return value if isinstance(value, str) else None


class _PROXY_VirtualKeyModelMaxBudgetLimiter(RouterBudgetLimiting):
    """
    Handles budgets for model + virtual key

    Example: key=$LITELLM_API_KEY, model=gpt-4o, max_budget=100, time_period=1d
    """

    def __init__(
        self,
        dual_cache: DualCache,
        clock: Callable[[], float] = time.time,
        model_access_groups: ModelAccessGroups = proxy_model_access_groups,
        shared_windows: SharedWindows = PROXY_SHARED_WINDOWS,
    ):
        self.dual_cache = dual_cache
        self._clock = clock
        self._model_access_groups = model_access_groups
        self._shared_windows = shared_windows
        self.redis_increment_operation_queue = []
        self._redis_increment_queue_lock = asyncio.Lock()
        self._redis_increment_flush_lock = asyncio.Lock()
        self._detached_increment_operations = None
        self._retired = False
        self._flush_after_retire = None
        self.deployment_budget_config = None

    @with_service_target("model_budgets")
    async def is_key_within_model_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> bool:
        """
        Check if the user_api_key_dict is within the model budget

        Raises:
            BudgetExceededError: If the user_api_key_dict has exceeded the model budget
        """
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.KEY,
            entity_id=user_api_key_dict.token,
            team_id=user_api_key_dict.team_id,
            model_max_budget=user_api_key_dict.model_max_budget,
            model=model,
            exceeded_message=(
                f"LiteLLM Virtual Key: {user_api_key_dict.token}, key_alias: {user_api_key_dict.key_alias}, "
                f"exceeded budget for model={model}"
            ),
        )

    @with_service_target("model_budgets")
    async def get_fallback_model_within_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> str | None:
        budget_fallbacks: Final[dict[str, list[str]]] = user_api_key_dict.budget_fallbacks or {}
        for fallback_model in budget_fallbacks.get(model, []):
            try:
                await self.is_key_within_model_budget(user_api_key_dict=user_api_key_dict, model=fallback_model)
                return fallback_model
            except litellm.BudgetExceededError:
                continue
        return None

    @with_service_target("model_budgets")
    async def is_user_within_model_budget(
        self,
        user_id: str,
        user_model_max_budget: Mapping[str, object],
        model: str,
        team_id: str | None = None,
    ) -> bool:
        """
        Check if the internal user is within the model budget

        Raises:
            BudgetExceededError: If the user has exceeded the model budget
        """
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.USER,
            entity_id=user_id,
            team_id=team_id,
            model_max_budget=user_model_max_budget,
            model=model,
            exceeded_message=f"LiteLLM User: {user_id}, exceeded budget for model={model}",
        )

    @with_service_target("model_budgets")
    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: Mapping[str, object],
        model: str,
        team_id: str | None = None,
    ) -> bool:
        """
        Check if the end_user is within the model budget

        Raises:
            BudgetExceededError: If the end_user has exceeded the model budget
        """
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.END_USER,
            entity_id=end_user_id,
            team_id=team_id,
            model_max_budget=end_user_model_max_budget,
            model=model,
            exceeded_message=f"LiteLLM End User: {end_user_id}, exceeded budget for model={model}",
        )

    @with_service_target("model_budgets")
    async def is_team_within_model_budget(
        self,
        team_id: str,
        team_model_max_budget: Mapping[str, object],
        key_model_max_budget: Mapping[str, object] | None,
        model: str,
    ) -> bool:
        """
        Check if the team is within the model budget, leaving out a team entry named after `model` that the key's
        own `model_max_budget` replaces

        Raises:
            BudgetExceededError: If the team has exceeded the model budget
        """
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.TEAM,
            entity_id=team_id,
            team_id=team_id,
            model_max_budget=team_model_budget_entries(
                model=model, team_model_max_budget=team_model_max_budget, key_model_max_budget=key_model_max_budget
            ),
            model=model,
            exceeded_message=f"LiteLLM Team: {team_id}, exceeded budget for model={model}",
        )

    @with_service_target("model_budgets")
    async def is_team_member_within_model_budget(
        self,
        user_id: str,
        team_id: str,
        team_member_model_max_budget: Mapping[str, object],
        model: str,
    ) -> bool:
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.TEAM_MEMBER,
            entity_id=team_member_budget_entity_id(user_id=user_id, team_id=team_id),
            team_id=team_id,
            model_max_budget=team_member_model_max_budget,
            model=model,
            exceeded_message=(
                f"LiteLLM Team Member: user={user_id}, team={team_id}, exceeded budget for model={model}"
            ),
        )

    async def _is_entity_within_model_budget(
        self,
        entity_type: Litellm_EntityType,
        entity_id: str | None,
        team_id: str | None,
        model_max_budget: Mapping[str, object] | None,
        model: str,
        exceeded_message: str,
    ) -> bool:
        if not model_max_budget:
            return True
        resolved: Final = resolve_model_budget(
            model=model,
            model_max_budget=model_max_budget,
            model_access_groups=self._model_access_groups,
            team_id=team_id,
        )
        if resolved is None:
            verbose_proxy_logger.debug("Model %s not found in %s model_max_budget", model, entity_type.value)
            return True

        max_budget: Final = resolved.budget_config.max_budget
        if max_budget is None or max_budget < 0:
            return True

        local_spend: Final = await self._get_spend_for_model_budget(
            entity_type=entity_type,
            entity_id=entity_id,
            model=model,
            resolved=resolved,
        )
        current_spend: Final = (
            local_spend
            if local_spend >= max_budget
            else await self._with_shared_spend(
                entity_type=entity_type, entity_id=entity_id, resolved=resolved, local_spend=local_spend
            )
        )
        if current_spend >= max_budget:
            raise litellm.BudgetExceededError(
                message=(
                    exceeded_message
                    if resolved.budget_model in _budget_model_candidates(model)
                    else f"{exceeded_message}, budget_model={resolved.budget_model}"
                ),
                current_cost=current_spend,
                max_budget=max_budget,
                entity_type=entity_type.value,
                entity_id=entity_id,
            )
        return True

    async def _get_spend_for_model_budget(
        self,
        entity_type: Litellm_EntityType,
        entity_id: str | None,
        model: str,
        resolved: ResolvedModelBudget,
    ) -> float:
        """Spend charged to this budget in the current window, legacy counter included.

        A counter that was never written is zero spend, not unknown spend. The
        distinction only shows up at a zero-dollar cap, where skipping the
        comparison would let the strictest possible limit admit every request.
        """
        spend_key: Final = model_budget_spend_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            budget_model=resolved.budget_model,
            budget_duration=resolved.budget_config.budget_duration,
        )
        legacy_spend_key: Final = _legacy_request_model_spend_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            model=model,
            resolved=resolved,
        )
        current_spend: Final = _as_spend(await self._cached_spend(spend_key))
        if legacy_spend_key is None or legacy_spend_key == spend_key:
            return current_spend
        return current_spend + _as_spend(await self._cached_spend(legacy_spend_key))

    async def _with_shared_spend(
        self,
        entity_type: Litellm_EntityType,
        entity_id: str | None,
        resolved: ResolvedModelBudget,
        local_spend: float,
    ) -> float:
        """Without Redis each replica's counter holds only its own share, so the window every replica adds to
        in the database is the floor."""
        if not self._shares_windows_in_database(entity_type):
            return local_spend
        shared: Final = await _open_shared_window(
            shared_windows=self._shared_windows,
            spend_key=model_budget_spend_cache_key(
                entity_type=entity_type,
                entity_id=entity_id,
                budget_model=resolved.budget_model,
                budget_duration=resolved.budget_config.budget_duration,
            ),
            budget_duration=resolved.budget_config.budget_duration,
            now=self._clock(),
        )
        return local_spend if shared is None else max(local_spend, shared.spend)

    async def _cached_spend(self, spend_key: str) -> float | None:
        redis_cache: Final = self.dual_cache.redis_cache
        if redis_cache is None:
            return await self.dual_cache.async_get_cache(key=spend_key)
        return await redis_cache.async_get_cache(key=spend_key)

    @with_service_target("model_budgets")
    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        return healthy_deployments

    @with_service_target("model_budgets")
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        Track spend for virtual key + model in DualCache

        Example: key=$LITELLM_API_KEY, model=gpt-4o, max_budget=100, time_period=1d
        """
        verbose_proxy_logger.debug("in RouterBudgetLimiting.async_log_success_event")
        standard_logging_payload: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)
        if standard_logging_payload is None:
            verbose_proxy_logger.debug(
                "Skipping _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event: standard_logging_payload is None"
            )
            return

        _litellm_params: Final[dict] = kwargs.get("litellm_params", {}) or {}
        _metadata: Final[dict] = StandardLoggingPayloadSetup.merge_litellm_metadata(
            _litellm_params  # pyright: ignore[reportUnknownArgumentType]  # callback metadata is partially untyped
        )
        payload_metadata: Final = standard_logging_payload.get("metadata") or {}

        model: Final = _requested_model_name(
            metadata=_metadata,  # pyright: ignore[reportUnknownArgumentType]  # callback metadata is partially untyped
            standard_logging_payload=standard_logging_payload,
        )
        if model is None:
            return

        response_cost: Final[float] = standard_logging_payload.get("response_cost", 0)
        key_model_max_budget: Final = _metadata.get("user_api_key_model_max_budget")
        team_member_user_id: Final = _metadata_string_value(
            payload_metadata,  # pyright: ignore[reportUnknownArgumentType]  # callback metadata is partially untyped
            "user_api_key_user_id",
        )
        team_member_team_id: Final = _metadata_string_value(
            payload_metadata,  # pyright: ignore[reportUnknownArgumentType]  # callback metadata is partially untyped
            "user_api_key_team_id",
        )
        team_model_max_budget: Final = _metadata_value(
            _metadata,  # pyright: ignore[reportUnknownArgumentType]  # callback metadata is partially untyped
            "user_api_key_team_model_max_budget",
        )
        team_member_entity_id: Final = (
            team_member_budget_entity_id(user_id=team_member_user_id, team_id=team_member_team_id)
            if team_member_user_id is not None and team_member_team_id is not None
            else None
        )
        entity_budgets: Final = (
            (
                Litellm_EntityType.KEY,
                payload_metadata.get("user_api_key_hash"),
                key_model_max_budget,
            ),
            (
                Litellm_EntityType.TEAM,
                team_member_team_id,
                (
                    team_model_budget_entries(
                        model=model,
                        team_model_max_budget=team_model_max_budget,
                        key_model_max_budget=(
                            key_model_max_budget if isinstance(key_model_max_budget, Mapping) else None
                        ),
                    )
                    if isinstance(team_model_max_budget, Mapping)
                    else team_model_max_budget
                ),
            ),
            (
                Litellm_EntityType.USER,
                payload_metadata.get("user_api_key_user_id"),
                _metadata.get("user_api_key_user_model_max_budget"),
            ),
            (
                Litellm_EntityType.END_USER,
                standard_logging_payload.get("end_user") or payload_metadata.get("user_api_key_end_user_id"),
                _metadata.get("user_api_key_end_user_model_max_budget"),
            ),
            (
                Litellm_EntityType.TEAM_MEMBER,
                team_member_entity_id,
                _metadata_value(
                    _metadata,  # pyright: ignore[reportUnknownArgumentType]  # callback metadata is partially untyped
                    "user_api_key_team_member_model_max_budget",
                ),
            ),
        )

        resolved_budgets: Final = _resolve_entity_model_budgets(
            model=model,
            entity_budgets=entity_budgets,
            model_access_groups=self._model_access_groups,
            team_id=team_member_team_id,
        )
        if not resolved_budgets:
            verbose_proxy_logger.debug(
                "Not running _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event: "
                "no key, team, user or end-user model_max_budget covers model=%s",
                model,
            )
            return

        batch_id: Final = batch_id_to_charge_once(
            call_type=kwargs.get("call_type"),
            response_obj=response_obj,
            response_cost=response_cost,
        )
        for entity_type, entity_id, resolved in resolved_budgets:
            await self._charge_entity(
                entity_type=entity_type,
                entity_id=entity_id,
                resolved=resolved,
                response_cost=response_cost,
                batch_id=batch_id,
            )

        if self.dual_cache.redis_cache is not None:
            await self._push_in_memory_increments_to_redis()

        verbose_proxy_logger.debug(
            "current state of in memory cache %s",
            json.dumps(self.dual_cache.in_memory_cache.cache_dict, indent=4, default=str),
        )

    async def _charge_entity(
        self,
        entity_type: Litellm_EntityType,
        entity_id: str | None,
        resolved: ResolvedModelBudget,
        response_cost: float,
        batch_id: str | None,
    ) -> None:
        budget_duration: Final = resolved.budget_config.budget_duration
        if budget_duration is None:
            return
        spend_key: Final = model_budget_spend_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            budget_model=resolved.budget_model,
            budget_duration=budget_duration,
        )
        if batch_id is not None and not await self._claim_batch_charge(
            spend_key=spend_key,
            batch_id=batch_id,
            ttl_seconds=duration_in_seconds(budget_duration),
        ):
            return
        start_time_key: Final = model_budget_start_time_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            budget_model=resolved.budget_model,
            budget_duration=budget_duration,
        )
        shares_window: Final = self._shares_windows_in_database(entity_type)
        if shares_window:
            await self._join_shared_window(
                spend_key=spend_key, start_time_key=start_time_key, budget_duration=budget_duration
            )
        await self._increment_spend_for_key(
            budget_config=resolved.budget_config,
            spend_key=spend_key,
            start_time_key=start_time_key,
            response_cost=response_cost,
        )
        if shares_window:
            await self._share_spend(
                spend_key=spend_key,
                start_time_key=start_time_key,
                budget_duration=budget_duration,
                response_cost=response_cost,
            )

    def _shares_windows_in_database(self, entity_type: Litellm_EntityType) -> bool:
        """End-user ids are whatever the client sends, so their windows stay on each replica rather than adding a
        database row per id."""
        return self.dual_cache.redis_cache is None and entity_type not in _PER_REPLICA_SCOPES

    async def _join_shared_window(self, spend_key: str, start_time_key: str, budget_duration: str) -> None:
        """Without Redis, start this replica's window where the database says every replica's window starts.

        Every replica then sends the same window start with its spend, so no flush reads as a newer window that
        replaces the spend the others already added. While the database cannot say, this replica keeps its spend
        unshared and sends it once a later charge joins the shared window.
        """
        now: Final = self._clock()
        local: Final = _local_windows(self.dual_cache)
        unshared_key: Final = _unshared_spend_key(start_time_key)
        if local.get_cache(key=unshared_key) is None and _open_local_window_start(
            local=local, start_time_key=start_time_key, budget_duration=budget_duration, now=now
        ):
            return
        started_at: Final = await self._shared_windows.open(
            spend_key=spend_key, budget_duration=budget_duration, now=now
        )
        unshared_spend: Final = _as_float(local.get_cache(key=unshared_key))
        local_window_open: Final = _open_local_window_start(
            local=local, start_time_key=start_time_key, budget_duration=budget_duration, now=now
        )
        if local_window_open and unshared_spend is None:
            return
        if not local_window_open:
            _start_local_window(
                local=local,
                spend_key=spend_key,
                start_time_key=start_time_key,
                started_at=now if started_at is None else started_at,
                budget_duration=budget_duration,
                now=now,
                spend=0.0,
            )
            if started_at is None:
                local.set_cache(key=unshared_key, value=0.0, ttl=budget_window_seconds(now, budget_duration))
            else:
                local.delete_cache(key=unshared_key)
            return
        if started_at is None or unshared_spend is None:
            return
        _start_local_window(
            local=local,
            spend_key=spend_key,
            start_time_key=start_time_key,
            started_at=started_at,
            budget_duration=budget_duration,
            now=now,
            spend=_as_float(local.get_cache(key=spend_key)) or 0.0,
        )
        local.delete_cache(key=unshared_key)
        await self._shared_windows.add(
            spend_key=spend_key, budget_duration=budget_duration, started_at=started_at, spend=unshared_spend
        )

    async def _share_spend(
        self, spend_key: str, start_time_key: str, budget_duration: str, response_cost: float
    ) -> None:
        now: Final = self._clock()
        local: Final = _local_windows(self.dual_cache)
        unshared_key: Final = _unshared_spend_key(start_time_key)
        unshared_spend: Final = _as_float(local.get_cache(key=unshared_key))
        if unshared_spend is not None:
            local.set_cache(
                key=unshared_key,
                value=unshared_spend + response_cost,
                ttl=budget_window_seconds(now, budget_duration),
            )
            return
        await self._shared_windows.add(
            spend_key=spend_key,
            budget_duration=budget_duration,
            started_at=await self._get_or_set_budget_start_time(
                start_time_key=start_time_key,
                current_time=now,
                ttl_seconds=budget_window_seconds(now, budget_duration),
            ),
            spend=response_cost,
        )

    async def _claim_batch_charge(self, spend_key: str, batch_id: str, ttl_seconds: int) -> bool:
        marker_key: Final = batch_charged_once_marker_key(spend_key=spend_key, batch_id=batch_id)
        polls: Final = await self.dual_cache.async_increment_cache(
            key=marker_key, value=1, ttl=ttl_seconds, refresh_ttl=True
        )
        return polls == 1


def _local_windows(dual_cache: DualCache) -> LocalTTLCache:
    return dual_cache.in_memory_cache


def _open_local_window_start(local: LocalTTLCache, start_time_key: str, budget_duration: str, now: float) -> bool:
    started_at: Final = _as_float(local.get_cache(key=start_time_key))
    return started_at is not None and now - started_at <= budget_window_seconds(started_at, budget_duration)


def _start_local_window(
    local: LocalTTLCache,
    spend_key: str,
    start_time_key: str,
    started_at: float,
    budget_duration: str,
    now: float,
    spend: float,
) -> None:
    ttl: Final = started_at + budget_window_seconds(started_at, budget_duration) - now
    local.delete_cache(key=spend_key)
    local.delete_cache(key=start_time_key)
    local.set_cache(key=spend_key, value=spend, ttl=ttl)
    local.set_cache(key=start_time_key, value=started_at, ttl=ttl)


def _unshared_spend_key(start_time_key: str) -> str:
    return f"{start_time_key}:unshared"


PROXY_VirtualKeyModelMaxBudgetLimiter = _PROXY_VirtualKeyModelMaxBudgetLimiter
