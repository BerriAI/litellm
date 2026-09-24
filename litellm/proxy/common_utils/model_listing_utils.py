"""Team-scoped (BYOK) model-name translation for the model listing endpoints.

`/v1/models`, `/models`, and `GET /v1/models/{id}` should surface the public
`team_public_model_name` rather than the internal routing key
`model_name_{team_id}_{uuid}`, consistent with `/v1/model/info`. The internal
key still routes regardless; this is a presentation-layer swap only and does not
touch access-group or auth semantics (see issue #28382). Operators can pin the
legacy internal names with `general_settings.use_team_public_model_name: false`.
"""

from __future__ import annotations

import re
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from pydantic import TypeAdapter, ValidationError

import litellm

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.proxy.model_listing import ModelInfoResponse

CLAUDE_CODE_PICKER_PATTERN: Final = re.compile(r"claude|anthropic", re.IGNORECASE)
GATEWAY_CLIENT_HEADER: Final = "x-gateway-client"
CLAUDE_CODE_CLIENT: Final = "claude-code"
_CLAUDE_CODE_ALIAS_PREFIX: Final = "claude-router-"
_ONE_MILLION_SUFFIX: Final = "[1m]"
_ONE_MILLION_TOKENS: Final = 1_000_000
_ALIAS_ENTRIES: Final = TypeAdapter(Mapping[object, object])
_NO_ALIASES: Final[Mapping[str, str]] = MappingProxyType({})


def configured_display_names(
    entries: Sequence[tuple[str, str]],
    llm_router: Router | None,
) -> Mapping[str, str]:
    """response_id -> configured `model_info.display_name` for the listing entries
    that have one.

    Metadata is looked up by each entry's internal lookup id (so team-scoped rows
    resolve), while the returned map is keyed by the public response id the
    Anthropic-shaped listing is built from. Entries without a configured name are
    omitted so the listing falls back to the id itself.
    """
    if llm_router is None:
        return MappingProxyType({})
    resolved: Final = (
        (response_id, llm_router.get_configured_display_name(lookup_id)) for response_id, lookup_id in entries
    )
    return MappingProxyType(
        {response_id: display_name for response_id, display_name in resolved if display_name is not None}
    )


def _unmarked(name: str) -> str:
    return name[: -len(_ONE_MILLION_SUFFIX)] if name.lower().endswith(_ONE_MILLION_SUFFIX) else name


def _compatibility_id(model_id: str) -> str:
    return f"{_CLAUDE_CODE_ALIAS_PREFIX}{model_id.encode().hex()}"


def _decoded_compatibility_id(view_id: str) -> str | None:
    encoded: Final = _unmarked(view_id).removeprefix(_CLAUDE_CODE_ALIAS_PREFIX)
    if encoded == _unmarked(view_id):
        return None
    try:
        model_id: Final = bytes.fromhex(encoded).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    return model_id if _compatibility_id(model_id) == _unmarked(view_id) else None


def claude_code_model_id(
    model_id: str,
    max_input_tokens: float | None,
    routing_names: Container[str],
) -> str:
    """The collision-free id Claude Code's picker lists a model under."""
    if "*" in model_id:
        return model_id
    shaped: Final = model_id if CLAUDE_CODE_PICKER_PATTERN.search(model_id) else _compatibility_id(model_id)
    one_million: Final = max_input_tokens is not None and max_input_tokens >= _ONE_MILLION_TOKENS
    marked: Final = (
        f"{shaped}{_ONE_MILLION_SUFFIX}" if one_million and not shaped.lower().endswith(_ONE_MILLION_SUFFIX) else shaped
    )
    return next(
        (
            name
            for name in (marked, shaped)
            if name == model_id or claude_code_group_name(name, routing_names) == model_id
        ),
        model_id,
    )


def claude_code_group_name(view_id: str, routing_names: Container[str]) -> str | None:
    """Decode a canonical compatibility id only when no configured route claims it."""
    if view_id in routing_names:
        return None
    unmarked: Final = _unmarked(view_id)
    if unmarked != view_id and unmarked in routing_names:
        return unmarked
    model_id: Final = _decoded_compatibility_id(view_id)
    return model_id if model_id and model_id in routing_names else None


def is_claude_code_client(headers: Mapping[str, str]) -> bool:
    """Claude Code itself, or a client asking for its view of the listing the way Ramp Router's does"""
    from litellm.llms.anthropic.common_utils import is_claude_code_user_agent

    return (
        is_claude_code_user_agent(headers.get("user-agent", ""))
        or headers.get(GATEWAY_CLIENT_HEADER, "").lower() == CLAUDE_CODE_CLIENT
    )


def claude_code_view_ids(
    rows: Sequence[ModelInfoResponse],
    headers: Mapping[str, str],
    routing_names: Container[str],
) -> Mapping[str, str]:
    """served id -> Claude Code id for the requested listing view"""
    if not is_claude_code_client(headers):
        return MappingProxyType({})
    return MappingProxyType(
        {row["id"]: claude_code_model_id(row["id"], row.get("max_input_tokens"), routing_names) for row in rows}
    )


@dataclass(frozen=True, slots=True)
class ClaudeCodeRoutingNames:
    """Existing routes always own their names, including aliases and wildcard routes."""

    llm_router: Router | None
    team_id: str | None = None
    alias_maps: tuple[object, ...] = ()

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        if name in litellm.model_alias_map or any(
            isinstance(aliases, Mapping) and name in aliases for aliases in self.alias_maps
        ):
            return True
        if self.llm_router is None:
            return False
        return (
            name in self.llm_router.model_group_alias
            or self.llm_router.has_model_id(name)
            or bool(self.llm_router.get_candidate_model_ids_for_route(name, self.team_id))
        )


def caller_alias_maps(
    key_aliases: object,
    team_aliases: object,
    key_team_id: str | None,
    listed_team_id: str | None,
) -> tuple[object, ...]:
    """The alias maps `/chat/completions` rewrites this caller's model through, in the order
    it applies them: the team's first, only when listing the team the key authenticated as,
    then the key's own twice, once in `add_litellm_data_to_request` and once more in
    `common_processing_pre_call_logic`."""
    if listed_team_id is not None and listed_team_id != key_team_id:
        return (key_aliases, key_aliases)
    return (team_aliases, key_aliases, key_aliases)


def _alias_map(aliases: object) -> Mapping[str, str]:
    try:
        entries: Final = _ALIAS_ENTRIES.validate_python(aliases, strict=True)
    except ValidationError:
        return _NO_ALIASES
    return MappingProxyType(
        {alias: target for alias, target in entries.items() if isinstance(alias, str) and isinstance(target, str)}
    )


def _alias_names(alias_maps: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(alias for aliases in alias_maps for alias in aliases))


def _rewrite(model_id: str, alias_maps: Sequence[Mapping[str, str]]) -> str | None:
    target: Final = reduce(lambda name, aliases: aliases.get(name, name), alias_maps, model_id)
    return None if target == model_id else target


def alias_target(model_id: str, alias_maps: Sequence[object]) -> str | None:
    """The model group `/chat/completions` rewrites `model_id` to through the caller's key and
    team aliases, applied in `alias_maps` order, else None."""
    return _rewrite(model_id, tuple(_alias_map(aliases) for aliases in alias_maps))


def alias_listing_entries(
    entries: Sequence[tuple[str, str]],
    alias_maps: Sequence[object],
) -> tuple[tuple[str, str], ...]:
    """`entries` plus one `(alias, lookup_id)` row per key or team alias whose target is
    listed. An alias colliding with a listed id keeps the listed entry."""
    maps: Final = tuple(_alias_map(aliases) for aliases in alias_maps)
    lookup_by_response: Final = MappingProxyType(dict(entries))
    lookup_ids: Final = frozenset(lookup_by_response.values())
    targets: Final = MappingProxyType(
        {alias: _rewrite(alias, maps) for alias in _alias_names(maps) if alias not in lookup_by_response}
    )
    added: Final = tuple(
        (alias, lookup_by_response.get(target, target))
        for alias, target in targets.items()
        if target is not None and (target in lookup_by_response or target in lookup_ids)
    )
    return (*entries, *added)


def claude_code_requested_group(
    requested: str,
    llm_router: Router,
    team_id: str | None,
    alias_maps: tuple[object, ...] = (),
) -> str | None:
    return claude_code_group_name(requested, ClaudeCodeRoutingNames(llm_router, team_id, alias_maps))


class TeamModelNameTranslator:
    """Translates internal team routing keys to their public names for the model
    listing/retrieve responses. Stateless; the live router and general_settings
    are injected per call so the unit tests can drive it without globals.
    """

    @staticmethod
    def _internal_public_pair(model: object) -> tuple[str, str] | None:
        """`(internal_routing_key, public_name)` for a team-scoped row, else None."""
        if not isinstance(model, dict):
            return None
        model_dict: Final = cast(dict[str, object], model)  # any-ok: checked
        model_info_raw: Final[object] = model_dict.get("model_info")
        if not isinstance(model_info_raw, Mapping):
            return None
        model_info: Final = cast(Mapping[str, object], model_info_raw)  # any-ok: checked
        team_id: Final = model_info.get("team_id")
        team_public: Final = model_info.get("team_public_model_name")
        name: Final = model_dict.get("model_name")
        if (
            isinstance(team_id, str)
            and isinstance(team_public, str)
            and isinstance(name, str)
            and team_id
            and team_public
            and name.startswith(f"model_name_{team_id}_")
        ):
            return name, team_public
        return None

    @staticmethod
    def _is_enabled(general_settings: Mapping[str, object]) -> bool:
        return general_settings.get("use_team_public_model_name", True) is not False

    @staticmethod
    def build_internal_to_public_map(
        llm_router: Router | None,
        general_settings: Mapping[str, object],
    ) -> dict[str, str]:
        """Internal team routing key -> public `team_public_model_name`.

        Empty when disabled via the legacy flag, the router is absent, or the
        router model list is malformed.
        """
        if llm_router is None or not TeamModelNameTranslator._is_enabled(general_settings):
            return {}
        router_model_list: Final = llm_router.get_model_list()
        if not isinstance(router_model_list, list):
            return {}
        return dict(
            pair
            for pair in (TeamModelNameTranslator._internal_public_pair(model) for model in router_model_list)
            if pair is not None
        )

    @staticmethod
    def _response_to_lookup_map(
        model_names: list[str],
        internal_to_public: dict[str, str],
    ) -> dict[str, str]:
        """Map each public response id to the first internal lookup id seen in
        `model_names`, preserving first-occurrence order. First-wins keeps list
        and retrieve in agreement on which accessible deployment a shared public
        id resolves to: a global iterated before a colliding team alias stays
        the listed entry, and sibling team rows collapse to their first
        occurrence.
        """
        result: Final[dict[str, str]] = {}
        for name in model_names:
            result.setdefault(internal_to_public.get(name, name), name)
        return result

    @staticmethod
    def listing_entries(
        model_names: list[str],
        llm_router: Router | None,
        general_settings: Mapping[str, object],
    ) -> list[tuple[str, str]]:
        """`(response_id, metadata_lookup_id)` for each listed model, de-duplicated
        by response_id while preserving order.

        For team-scoped rows `response_id` is the public name shown to the client,
        while `metadata_lookup_id` stays the internal routing key so downstream
        metadata/fallback lookups (keyed by the routing name) still resolve. The
        lookup id is always one of `model_names` (the caller's accessible set), so
        a public name shared across teams never resolves to another team's
        internal key. Both ids are identical for unmapped names (globals,
        access-group keys).
        """
        internal_to_public: Final = TeamModelNameTranslator.build_internal_to_public_map(llm_router, general_settings)
        if not internal_to_public:
            return [(name, name) for name in model_names]
        return list(TeamModelNameTranslator._response_to_lookup_map(model_names, internal_to_public).items())

    @staticmethod
    def translate_listing(
        model_names: list[str],
        llm_router: Router | None,
        general_settings: Mapping[str, object],
    ) -> list[str]:
        """Public-name view of `model_names` (the `response_id` of each listing
        entry). Sibling deployments sharing a public name collapse to one entry
        while preserving order; unmapped names pass through.
        """
        return [
            entry[0] for entry in TeamModelNameTranslator.listing_entries(model_names, llm_router, general_settings)
        ]

    @staticmethod
    def resolve_public_name(
        model_id: str,
        available_models: list[str],
        llm_router: Router | None,
        general_settings: Mapping[str, object],
    ) -> str:
        """Resolve a public team name back to the internal routing key the router
        indexes by, so `GET /v1/models/{id}` accepts the name the listing returns.

        Resolution is restricted to `available_models` (the caller's accessible
        set) so colliding public names across teams never resolve across an access
        boundary. Uses the same first-occurrence dedup as `listing_entries` so a
        public id advertised by `/v1/models` resolves to the same internal
        deployment that the listing's metadata was built from. Returns `model_id`
        unchanged when it is not an accessible public team name (already-internal
        names and globals pass through).
        """
        internal_to_public: Final = TeamModelNameTranslator.build_internal_to_public_map(llm_router, general_settings)
        if not internal_to_public:
            return model_id
        return TeamModelNameTranslator._response_to_lookup_map(available_models, internal_to_public).get(
            model_id, model_id
        )
