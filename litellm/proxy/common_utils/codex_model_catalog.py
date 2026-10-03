"""Codex-native model catalog for the model listing endpoints.

Codex CLI discovers a provider's models with ``GET <model_catalog_url>?client_version=<its version>``
(the provider's ``model_catalog_url`` pointed at ``/v1/models``, with its ``api_key_model_discovery``
feature on) and decodes the body as its own ``ModelsResponse``, ``{"models": [ModelInfo, ...]}``,
never the OpenAI list shape. This module builds that body from the rows ``/v1/models`` already
lists: a model Codex knows keeps Codex's own entry for it, any other model gets the fallback entry
Codex itself uses for an unknown slug, and ``model_info.service_tiers`` becomes the entry's
``service_tiers``, which Codex turns into slash commands (``/ultrafast``) that send ``service_tier``
upstream.

The stock entries are Codex 0.159.3's bundled catalog, vendored unchanged from
https://github.com/openai/codex/blob/rust-v0.159.3/codex-rs/models-manager/models.json (Apache-2.0)
as ``codex_bundled_models_0.159.3.json``, and ``codex_base_instructions.md`` is
``codex-rs/models-manager/prompt.md`` at the same tag. Refresh both by downloading those two paths at
a newer tag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import accumulate, pairwise
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.proxy.model_listing import ModelInfoResponse

CODEX_BASE_INSTRUCTIONS_PATH: Final = Path(__file__).with_name("codex_base_instructions.md")
CODEX_BUNDLED_MODELS_PATH: Final = Path(__file__).with_name("codex_bundled_models_0.159.3.json")
CODEX_CATALOG_BYTE_LIMIT: Final = 1024 * 1024
CODEX_CHAT_MODES: Final = frozenset({"chat", "responses"})

_TierId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_BODY_PREFIX: Final = '{"models":['
_BODY_SUFFIX: Final = "]}"


class CodexServiceTier(BaseModel):
    """A `ModelServiceTier` as Codex reads it: the slash command is `name` lowercased, and toggling
    it sends `id` as the request's `service_tier`."""

    model_config = ConfigDict(frozen=True)

    id: _TierId
    name: str
    description: str


class _ConfiguredTier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: _TierId
    name: str | None = None
    description: str | None = None


_CONFIGURED_TIERS: Final = TypeAdapter(tuple[_TierId | _ConfiguredTier, ...])


class CodexTruncationPolicy(BaseModel):
    mode: Literal["bytes"] = "bytes"
    limit: int = 10_000


class CodexFallbackModel(BaseModel):
    """One `ModelInfo` entry of a Codex model catalog for a model Codex does not know.

    Every field that some Codex release since `model_catalog_json` appeared
    (0.105.0) deserializes without a default is spelled out here, so one catalog
    parses on all of them; the values match the fallback metadata Codex uses for
    a model slug it does not know, so picking such a proxy model behaves the
    same as `codex -m` did.
    """

    slug: str
    display_name: str
    description: NoneType = None
    supported_reasoning_levels: tuple[()] = ()
    shell_type: Literal["unified_exec"] = "unified_exec"
    visibility: Literal["list"] = "list"
    supported_in_api: Literal[True] = True
    priority: int
    service_tiers: tuple[CodexServiceTier, ...] = ()
    availability_nux: NoneType = None
    upgrade: NoneType = None
    support_verbosity: Literal[False] = False
    supports_reasoning_summaries: Literal[False] = False
    supports_parallel_tool_calls: Literal[False] = False
    default_verbosity: NoneType = None
    apply_patch_tool_type: NoneType = None
    truncation_policy: CodexTruncationPolicy = CodexTruncationPolicy()
    experimental_supported_tools: tuple[()] = ()
    context_window: int | None
    base_instructions: str


class CodexStockUpgrade(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str


class CodexStockModel(BaseModel):
    """One `ModelInfo` entry as Codex ships it or prints it from `codex debug models`.

    Only the fields the listing rewrites are named; everything else that release
    knows about the model (its reasoning levels, prompt, tool support) rides
    along untouched, whatever the release's schema.
    """

    model_config = ConfigDict(extra="allow")

    slug: str
    display_name: str
    priority: int
    visibility: str
    supported_in_api: bool = True
    upgrade: CodexStockUpgrade | None = None
    service_tiers: tuple[CodexServiceTier, ...] = ()
    default_service_tier: str | None = None


class CodexStockCatalog(BaseModel):
    models: tuple[CodexStockModel, ...]


CodexCatalogEntry = CodexFallbackModel | CodexStockModel


@dataclass(frozen=True, slots=True)
class CodexCatalogRow:
    """What the catalog needs to know about one listed model: its public id, the listing's mode and
    input limit, the upstream model behind it, and the `model_info` values the operator set;
    `service_tiers` holds one raw `model_info.service_tiers` value per deployment behind the id, None
    where a deployment sets none."""

    id: str
    mode: str | None = None
    max_input_tokens: int | None = None
    upstream_model: str | None = None
    display_name: str | None = None
    service_tiers: tuple[object, ...] = ()


@cache
def bundled_codex_models() -> Mapping[str, CodexStockModel]:
    """Codex's bundled catalog by slug, read once from the vendored file."""
    catalog: Final = CodexStockCatalog.model_validate_json(CODEX_BUNDLED_MODELS_PATH.read_text(encoding="utf-8"))
    return MappingProxyType({model.slug: model for model in catalog.models})


@cache
def codex_base_instructions() -> str:
    return CODEX_BASE_INSTRUCTIONS_PATH.read_text(encoding="utf-8")


def _tier(item: str | _ConfiguredTier, known: Mapping[str, CodexServiceTier]) -> CodexServiceTier:
    if isinstance(item, str) and item in known:
        return known[item]
    configured: Final = _ConfiguredTier(id=item) if isinstance(item, str) else item
    return CodexServiceTier(
        id=configured.id,
        name=configured.name if configured.name is not None else configured.id.capitalize(),
        description=(
            configured.description
            if configured.description is not None
            else f"Sends service_tier={configured.id} upstream"
        ),
    )


def _first_by_id(tiers: Sequence[CodexServiceTier]) -> tuple[CodexServiceTier, ...]:
    return tuple(tier for index, tier in enumerate(tiers) if tier.id not in {seen.id for seen in tiers[:index]})


_NO_KNOWN_TIERS: Final[Mapping[str, CodexServiceTier]] = MappingProxyType({})


def _deployment_tiers(
    raw: object, model_id: str, known: Mapping[str, CodexServiceTier]
) -> tuple[CodexServiceTier, ...]:
    try:
        items: Final = _CONFIGURED_TIERS.validate_python(raw)
    except ValidationError as e:
        verbose_proxy_logger.warning(
            "model_info.service_tiers for %s offers no tier, expected a list of tier ids or {id, name, description} objects: %s",
            model_id,
            e.errors()[0]["msg"],
        )
        return ()
    return _first_by_id(tuple(_tier(item, known) for item in items))


def configured_service_tiers(
    per_deployment: Sequence[object], model_id: str, known: Mapping[str, CodexServiceTier] = _NO_KNOWN_TIERS
) -> tuple[CodexServiceTier, ...] | None:
    """The tiers `model_info.service_tiers` configures for `model_id`, given each of its deployments'
    raw value (None where unset), or None when no deployment sets one.

    Accepts tier id strings and `{id, name, description}` objects. A string naming one of the
    model's `known` tiers (the ones Codex ships for it) keeps that tier's name and description, so
    `"priority"` stays Codex's `/fast`; any other string `x` is the command `/x` described as
    "Sends service_tier=x upstream". A duplicate id keeps its first entry. A request to `model_id`
    can route to any of its deployments, so a tier is offered only when every deployment lists it,
    in the first deployment's order; a deployment that sets none leaves nothing to offer. Each of
    those cases is logged, and an invalid value on any deployment is logged and offers no tier for
    the whole model, stock tiers included, so one typo never fails the listing and never offers a
    tier nobody declared.
    """
    declared: Final = tuple(raw for raw in per_deployment if raw is not None)
    if not declared:
        return None
    parsed: Final = tuple(_deployment_tiers(raw, model_id, known) for raw in declared)
    if len(parsed) < len(per_deployment):
        verbose_proxy_logger.warning(
            "model_info.service_tiers for %s is set on %d of its %d deployments, so no tier is offered: "
            "a tier is offered only when every deployment lists it",
            model_id,
            len(parsed),
            len(per_deployment),
        )
        return ()
    tier_ids: Final = tuple(frozenset(tier.id for tier in tiers) for tiers in parsed)
    shared: Final = tier_ids[0].intersection(*tier_ids[1:])
    offered: Final = tuple(tier for tier in parsed[0] if tier.id in shared)
    if len(offered) < len(parsed[0]):
        verbose_proxy_logger.warning(
            "model_info.service_tiers for %s differs between its deployments, offering only the tiers every one lists: %s",
            model_id,
            ", ".join(tier.id for tier in offered) or "none",
        )
    return offered


def _codex_can_drive(row: CodexCatalogRow) -> bool:
    return "*" not in row.id and (row.mode is None or row.mode in CODEX_CHAT_MODES)


def _stock_for(row: CodexCatalogRow, stock: Mapping[str, CodexStockModel]) -> CodexStockModel | None:
    listed: Final = stock.get(row.id)
    if listed is not None or row.upstream_model is None:
        return listed
    upstream: Final = stock.get(row.upstream_model)
    return upstream if upstream is not None else stock.get(row.upstream_model.partition("/")[2])


def _entry(
    index: int,
    row: CodexCatalogRow,
    stock: Mapping[str, CodexStockModel],
    served: frozenset[str],
    instructions: str,
) -> CodexCatalogEntry:
    stock_model: Final = _stock_for(row, stock)
    stock_tiers: Final = MappingProxyType(
        {tier.id: tier for tier in (stock_model.service_tiers if stock_model is not None else ())}
    )
    configured: Final = configured_service_tiers(row.service_tiers, row.id, stock_tiers)
    if stock_model is None:
        return CodexFallbackModel(
            slug=row.id,
            display_name=row.display_name if row.display_name is not None else row.id,
            priority=index,
            service_tiers=configured if configured is not None else (),
            context_window=row.max_input_tokens,
            base_instructions=instructions,
        )
    upgrade: Final = (
        stock_model.upgrade if stock_model.upgrade is not None and stock_model.upgrade.model in served else None
    )
    stock_name: Final = stock_model.display_name if row.id == stock_model.slug else row.id
    listing: Final = {
        "slug": row.id,
        "display_name": row.display_name if row.display_name is not None else stock_name,
        "priority": index,
        "visibility": "list",
        "supported_in_api": True,
        "upgrade": upgrade,
    }
    if configured is None:
        return stock_model.model_copy(update=listing)
    configured_ids: Final = frozenset(tier.id for tier in configured)
    return stock_model.model_copy(
        update={
            **listing,
            "service_tiers": configured,
            "default_service_tier": (
                stock_model.default_service_tier if stock_model.default_service_tier in configured_ids else None
            ),
        }
    )


def _serialized(entry: CodexCatalogEntry) -> str:
    """A stock entry keeps exactly the fields its source printed (an older Codex may know no
    `service_tiers`), a fallback entry spells out every field Codex requires."""
    return entry.model_dump_json(exclude_unset=isinstance(entry, CodexStockModel))


def _bytes_left_after(left: int, size: int) -> int:
    return left - size if size <= left else left


def _fits(serialized: Sequence[str], byte_limit: int | None) -> tuple[bool, ...]:
    """Whether each entry, taken in order, fits in what is left of a body of at most `byte_limit`
    bytes: an entry larger than the bytes left is passed over and a smaller one after it is still
    taken, each entry counted with its separating comma and the last one's comma given back."""
    if byte_limit is None:
        return (True,) * len(serialized)
    budget: Final = byte_limit - len(_BODY_PREFIX) - len(_BODY_SUFFIX) + 1
    left: Final = tuple(
        accumulate((len(entry.encode()) + 1 for entry in serialized), _bytes_left_after, initial=budget)
    )
    return tuple(after < before for before, after in pairwise(left))


def _kept_positions(
    entries: Sequence[CodexCatalogEntry], serialized: Sequence[str], byte_limit: int | None
) -> frozenset[int]:
    """The listing positions that survive the byte limit: an entry offering a service tier is taken
    ahead of one offering none, each group in listing order, since a model without a tier is driven
    by name just as well off Codex's bundled catalog; an entry too large for the bytes left never
    costs the smaller ones after it their place."""
    survival: Final = sorted(range(len(entries)), key=lambda position: (not entries[position].service_tiers, position))
    fits: Final = _fits(tuple(serialized[position] for position in survival), byte_limit)
    return frozenset(position for position, fit in zip(survival, fits, strict=True) if fit)


@dataclass(frozen=True, slots=True)
class CodexCatalogBody:
    """A `ModelsResponse` body, the ids it lists, and the chat-capable ids the byte limit left out."""

    json: str
    listed: tuple[str, ...]
    left_out: tuple[str, ...]


def codex_models_response_json(
    rows: Sequence[CodexCatalogRow],
    *,
    stock: Mapping[str, CodexStockModel] | None = None,
    instructions: str | None = None,
    byte_limit: int | None = CODEX_CATALOG_BYTE_LIMIT,
) -> CodexCatalogBody:
    """Codex's `ModelsResponse` body for `rows`, in listing order, cut to Codex's byte limit.

    Only chat-capable rows are listed (Codex cannot drive an embedding or image model, and a
    wildcard id is no model), their order is their `priority`. Codex drops a catalog over its
    byte limit silently and keeps its bundled one, so the body holds only the entries that fit,
    the ones offering a service tier taken first and the rest in listing order, an entry too large
    for the bytes left passed over, every kept entry at its listing position; `byte_limit=None`
    keeps every entry.
    """
    known: Final = bundled_codex_models() if stock is None else stock
    prompt: Final = codex_base_instructions() if instructions is None else instructions
    drivable: Final = tuple(row for row in rows if _codex_can_drive(row))
    served: Final = frozenset(row.id for row in drivable)
    entries: Final = tuple(_entry(index, row, known, served, prompt) for index, row in enumerate(drivable))
    serialized: Final = tuple(_serialized(entry) for entry in entries)
    kept: Final = _kept_positions(entries, serialized, byte_limit)
    return CodexCatalogBody(
        json=f"{_BODY_PREFIX}{','.join(serialized[position] for position in sorted(kept))}{_BODY_SUFFIX}",
        listed=tuple(row.id for position, row in enumerate(drivable) if position in kept),
        left_out=tuple(row.id for position, row in enumerate(drivable) if position not in kept),
    )


def _catalog_row(
    row: ModelInfoResponse, lookup_id: str, llm_router: Router | None, team_id: str | None
) -> CodexCatalogRow:
    return CodexCatalogRow(
        id=row["id"],
        mode=row.get("mode"),
        max_input_tokens=row.get("max_input_tokens"),
        upstream_model=llm_router.get_routable_upstream_model(lookup_id, team_id) if llm_router is not None else None,
        display_name=llm_router.get_configured_display_name(lookup_id) if llm_router is not None else None,
        service_tiers=llm_router.get_configured_service_tiers(lookup_id, team_id) if llm_router is not None else (),
    )


def codex_catalog_rows(
    rows: Sequence[ModelInfoResponse],
    entries: Sequence[tuple[str, str]],
    llm_router: Router | None,
    team_id: str | None = None,
) -> tuple[CodexCatalogRow, ...]:
    """`rows` joined with the router's configured metadata, looked up by each entry's internal id so
    team-scoped rows resolve the way the Anthropic listing's display names do, a `model_group_alias`
    reading its target's deployments under its own name; `team_id` is the requesting key's team, so the
    upstream model that picks Codex's stock entry and the tiers are read only off the deployments its
    requests can route to."""
    lookup_ids: Final = MappingProxyType(dict(entries))
    return tuple(_catalog_row(row, lookup_ids.get(row["id"], row["id"]), llm_router, team_id) for row in rows)


def codex_model_list_body(
    rows: Sequence[ModelInfoResponse],
    entries: Sequence[tuple[str, str]],
    llm_router: Router | None,
    team_id: str | None = None,
) -> str:
    """The `/v1/models?client_version=...` body, logging the models Codex's byte limit left out."""
    body: Final = codex_models_response_json(codex_catalog_rows(rows, entries, llm_router, team_id))
    if body.left_out:
        verbose_proxy_logger.warning(
            "Codex model catalog cut at %d bytes, left out: %s. Models offering a service tier are kept first, "
            "then model_list order; list the models Codex users need first in model_list",
            CODEX_CATALOG_BYTE_LIMIT,
            ", ".join(body.left_out),
        )
    return body.json
