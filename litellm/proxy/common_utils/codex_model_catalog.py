"""Codex-native model catalog for the model listing endpoints.

Codex CLI discovers a provider's models with ``GET <model_catalog_url>?client_version=<its version>``
(the provider's ``model_catalog_url`` pointed at ``/v1/models``, with its ``api_key_model_discovery``
feature on) and decodes the body as its own ``ModelsResponse``, ``{"models": [ModelInfo, ...]}``,
never the OpenAI list shape. Codex then ignores its own bundled catalog, so this body is all it knows
about the provider's models. It is built from the rows ``/v1/models`` already lists: a model Codex
knows, by its id or by the upstream model its deployments call once the provider's own prefixes are
stripped, keeps Codex's own entry for it, out of the catalog Codex itself uses for the provider those
deployments call (its Bedrock catalogs for ``bedrock`` and ``bedrock_mantle``, its OpenAI one otherwise), minus any service tier some deployment behind it prices tokens
but not that tier for; any other model gets an entry translated from the capabilities its deployments' model info
declares (see ``codex_model_translation``). ``model_info.service_tiers`` becomes the entry's
``service_tiers``, which Codex turns into slash commands (``/ultrafast``) that send ``service_tier``
upstream. A ``model_info.max_input_tokens`` the operator sets replaces a known model's
``context_window``, up to the most Codex lets its own ``model_context_window`` raise that model to.
A translated entry offers the model group's ``supported_reasoning_efforts`` as its reasoning levels,
trimmed to the ones the requesting Codex release can parse.

The stock entries are Codex 0.166.0's bundled catalog, vendored unchanged from
https://github.com/openai/codex/blob/rust-v0.166.0/codex-rs/models-manager/models.json (Apache-2.0)
as ``codex_bundled_models_0.166.0.json``, and ``codex_base_instructions.md`` is
``codex-rs/models-manager/prompt.md`` at the same tag. Refresh both by downloading those two paths at
a newer tag.

Codex builds its Bedrock catalogs in code rather than shipping them as files, so the ones for Bedrock's
runtime and Mantle endpoints are vendored from Codex 0.166.0's own output, with the ``base_instructions``
copy of each entry's ``model_messages.instructions_template`` that it adds for older clients dropped:
``codex -c 'model_provider="amazon-bedrock-runtime"' debug models | jq 'del(.models[].base_instructions)'``
as ``codex_bedrock_runtime_models_0.166.0.json``, and ``model_provider="amazon-bedrock"`` the same way as
``codex_bedrock_mantle_models_0.166.0.json``. Refresh them at the same release as the bundled catalog,
which they take each entry's ``minimal_client_version`` from, and recheck
``CODEX_BEDROCK_MODELS_MINIMAL_CLIENT_VERSION`` against the fields the new output leaves out.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from itertools import accumulate, pairwise
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.codex_model_translation import (
    TOKEN_PRICE_KEY,
    CodexDeploymentCapabilities,
    CodexFallbackModel,
    CodexServiceTier,
    TierId,
    codex_fallback_entry,
    combined_capabilities,
    no_priced_deployment_lacks_tier,
    tier_price_keys,
)
from litellm.proxy.common_utils.codex_version import codex_version_at_least
from litellm.types.llms.base_model import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.router_api import RouterRegistry
    from litellm.types.proxy.model_listing import ModelInfoResponse

CODEX_BASE_INSTRUCTIONS_PATH: Final = Path(__file__).with_name("codex_base_instructions.md")
CODEX_BUNDLED_MODELS_PATH: Final = Path(__file__).with_name("codex_bundled_models_0.166.0.json")
CodexBedrockEndpoint: TypeAlias = Literal["bedrock", "bedrock_mantle"]
CODEX_BEDROCK_MODELS_MINIMAL_CLIENT_VERSION: Final = "0.148.0"
CODEX_BEDROCK_MODELS_PATHS: Final[Mapping[CodexBedrockEndpoint, Path]] = MappingProxyType(
    {
        "bedrock": Path(__file__).with_name("codex_bedrock_runtime_models_0.166.0.json"),
        "bedrock_mantle": Path(__file__).with_name("codex_bedrock_mantle_models_0.166.0.json"),
    }
)
CODEX_CATALOG_BYTE_LIMIT: Final = 1024 * 1024
_FIRST_CODEX_WITHOUT_CATALOG_BYTE_LIMIT: Final = "0.166.0-alpha.1"
CODEX_CHAT_MODES: Final = frozenset({"chat", "responses"})

_BODY_PREFIX: Final = '{"models":['
_BODY_SUFFIX: Final = "]}"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_OBJECTS: Final = TypeAdapter(tuple[dict[str, object], ...])
_NO_UPDATE: Final[Mapping[str, object]] = MappingProxyType({})


class _ConfiguredTier(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: TierId
    name: str | None = None
    description: str | None = None


_CONFIGURED_TIERS: Final = TypeAdapter(tuple[TierId | _ConfiguredTier, ...])


class CodexStockUpgrade(LiteLLMBaseModel):
    model_config = ConfigDict(extra="allow")

    model: str


class CodexReasoningLevel(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    effort: str


class CodexStockModel(LiteLLMBaseModel):
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
    additional_speed_tiers: tuple[str, ...] = ()
    default_service_tier: str | None = None
    supports_search_tool: bool = False
    supported_reasoning_levels: tuple[CodexReasoningLevel, ...] = ()
    context_window: int | None = None
    max_context_window: int | None = None
    minimal_client_version: str | tuple[int, ...] | None = None


class CodexStockCatalog(LiteLLMBaseModel):
    models: tuple[CodexStockModel, ...]


CodexCatalogEntry = CodexFallbackModel | CodexStockModel
CodexAliasDefaults: TypeAlias = Literal["codex", "deployment"]
_NATIVE_ALIAS_DEFAULTS: Final[CodexAliasDefaults] = "deployment"


@dataclass(frozen=True, slots=True)
class CodexCatalogRow:
    """What the catalog needs to know about one listed model: its public id, the listing's mode and
    input limit, the upstream model behind it and the provider its deployment declares, every routable
    deployment's `upstreams` (model and declared provider), the reasoning effort its deployments pin, and the
    `model_info` values the operator set; `service_tiers` holds one raw `model_info.service_tiers`
    value per deployment behind the id, None where a deployment sets none, and
    `configured_max_input_tokens` one `model_info.max_input_tokens` per deployment the same way.
    `deployment_capabilities` holds what each deployment's model info, merged with the cost map,
    declares. `supported_reasoning_efforts` is the model group's, None when unknown."""

    id: str
    mode: str | None = None
    max_input_tokens: int | None = None
    upstream_model: str | None = None
    upstream_provider: str | None = None
    upstreams: tuple[tuple[str, str | None], ...] = ()
    reasoning_effort: str | None = None
    display_name: str | None = None
    service_tiers: tuple[object, ...] = ()
    supported_reasoning_efforts: tuple[str, ...] | None = None
    search_tool_support: tuple[bool | None, ...] = ()
    configured_max_input_tokens: tuple[int | None, ...] = ()
    deployment_capabilities: tuple[CodexDeploymentCapabilities, ...] = ()


@cache
def bundled_codex_models() -> Mapping[str, CodexStockModel]:
    """Codex's bundled catalog by slug, read once from the vendored file."""
    catalog: Final = CodexStockCatalog.model_validate_json(CODEX_BUNDLED_MODELS_PATH.read_text(encoding="utf-8"))
    return MappingProxyType({model.slug: model for model in catalog.models})


def _with_minimal_client_version(model: CodexStockModel, built_from: str | tuple[int, ...] | None) -> CodexStockModel:
    source_minimum: Final = built_from if not isinstance(built_from, tuple) else ".".join(map(str, built_from))
    minimum: Final = (
        source_minimum
        if codex_version_at_least(source_minimum, CODEX_BEDROCK_MODELS_MINIMAL_CLIENT_VERSION)
        else CODEX_BEDROCK_MODELS_MINIMAL_CLIENT_VERSION
    )
    return model.model_copy(update={"minimal_client_version": minimum})


@cache
def bundled_bedrock_codex_models(endpoint: CodexBedrockEndpoint) -> Mapping[str, CodexStockModel]:
    """Codex's catalog for a Bedrock endpoint by slug, read once from the vendored file. `codex debug models`
    prints no `minimal_client_version`, so each entry takes the later of the bundled entry's Codex built it from
    and `CODEX_BEDROCK_MODELS_MINIMAL_CLIENT_VERSION`, the first release that decodes an entry without the fields
    `debug models` leaves out, such as `supports_parallel_tool_calls`."""
    from litellm.llms.bedrock.common_utils import bedrock_model_name_without_provider

    catalog: Final = CodexStockCatalog.model_validate_json(
        CODEX_BEDROCK_MODELS_PATHS[endpoint].read_text(encoding="utf-8")
    )
    built_from: Final = MappingProxyType(
        {slug: model.minimal_client_version for slug, model in bundled_codex_models().items()}
    )
    return MappingProxyType(
        {
            model.slug: _with_minimal_client_version(
                model, built_from.get(bedrock_model_name_without_provider(model.slug))
            )
            for model in catalog.models
        }
    )


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


def offered_search_tool(per_deployment: Sequence[bool | None]) -> bool | None:
    if False in per_deployment:
        return False
    return True if per_deployment and None not in per_deployment else None


def _codex_can_drive(model_id: str, mode: str | None) -> bool:
    return "*" not in model_id and (mode is None or mode in CODEX_CHAT_MODES)


@cache
def _provider_model_names() -> Mapping[str, Callable[[str], str]]:
    """Imported on first use for the same reason as `tier_price_keys`."""
    from litellm.llms.bedrock.common_utils import bedrock_model_name_without_provider

    return MappingProxyType(
        {"bedrock": bedrock_model_name_without_provider, "bedrock_mantle": bedrock_model_name_without_provider}
    )


def _provider_model_name(upstream_model: str, upstream_provider: str | None) -> str:
    """The upstream model without the prefixes its provider adds, the provider being the one the deployment
    declares in `custom_llm_provider`, else the model's `provider/` prefix."""
    from litellm.litellm_core_utils.get_llm_provider_logic import declared_provider

    provider: Final = declared_provider(upstream_model, upstream_provider)
    if provider is None:
        return upstream_model
    model: Final = upstream_model.removeprefix(f"{provider}/")
    name_for: Final = _provider_model_names().get(provider)
    return model if name_for is None else name_for(model)


@dataclass(frozen=True, slots=True)
class _StockMatch:
    """Codex's entry for a row, and whether the row lists it under another name than Codex's (an alias)."""

    model: CodexStockModel
    alias: bool


def _on_endpoint(
    declared: Sequence[tuple[str | None, str]], endpoint: CodexBedrockEndpoint
) -> tuple[CodexBedrockEndpoint, str] | None:
    return next(((endpoint, model) for provider, model in declared if provider == endpoint), None)


def _bedrock_upstream(row: CodexCatalogRow) -> tuple[CodexBedrockEndpoint, str] | None:
    """The Bedrock endpoint a request to the row can reach whose Codex catalog promises the least, with the model
    a deployment on it calls. A request can route to any deployment, and Codex's Bedrock entries leave out what
    Bedrock does not support yet, so one Bedrock deployment puts the whole row on its catalog, the runtime one
    (no tool search) ahead of Mantle's."""
    from litellm.litellm_core_utils.get_llm_provider_logic import declared_provider

    first: Final = () if row.upstream_model is None else ((row.upstream_model, row.upstream_provider),)
    declared: Final = tuple((declared_provider(model, provider), model) for model, provider in (*first, *row.upstreams))
    on_bedrock: Final = (_on_endpoint(declared, endpoint) for endpoint in CODEX_BEDROCK_MODELS_PATHS)
    return next((found for found in on_bedrock if found is not None), None)


def _bedrock_stock_for(row_id: str, endpoint: CodexBedrockEndpoint, upstream_model: str) -> _StockMatch | None:
    """Codex's entry for the row's id or the Bedrock model as written, else for the same model under any of
    Bedrock's routing and region prefixes (`eu.openai.gpt-6.1-sol` takes `global.openai.gpt-6.1-sol`'s)."""
    from litellm.llms.bedrock.common_utils import bedrock_model_name_without_provider

    stock: Final = bundled_bedrock_codex_models(endpoint)
    names: Final = (row_id, upstream_model.removeprefix(f"{endpoint}/"))
    exact: Final = next((stock[name] for name in names if name in stock), None)
    model_names: Final = tuple(bedrock_model_name_without_provider(name) for name in names)
    same_model: Final = (
        next((model for slug, model in stock.items() if bedrock_model_name_without_provider(slug) == name), None)
        for name in model_names
    )
    found: Final = exact if exact is not None else next((model for model in same_model if model is not None), None)
    if found is None:
        return None
    return _StockMatch(found, model_names[0] != bedrock_model_name_without_provider(found.slug))


def _stock_for(row: CodexCatalogRow, stock: Mapping[str, CodexStockModel]) -> _StockMatch | None:
    """Codex's entry for the row's id, else for its upstream model as written, else for that model's name
    once its provider's own prefixes are stripped (`openai/gpt-6.1-sol` is `gpt-6.1-sol`), out of Codex's
    Bedrock catalog when a deployment behind the row is on Bedrock and `stock` otherwise. A Bedrock model that
    catalog does not know has no entry, since `stock`'s would promise what Bedrock does not support."""
    bedrock: Final = _bedrock_upstream(row)
    if bedrock is not None:
        return _bedrock_stock_for(row.id, *bedrock)
    names: Final = (
        (row.id,)
        if row.upstream_model is None
        else (row.id, row.upstream_model, _provider_model_name(row.upstream_model, row.upstream_provider))
    )
    found: Final = next((stock[name] for name in names if name in stock), None)
    return None if found is None else _StockMatch(found, row.id != found.slug)


def _readable_by(stock_model: CodexStockModel, client_version: str | None) -> bool:
    """Whether Codex's own server would hand this entry to the requesting release: one at or past the entry's
    `minimal_client_version`. An older release can fail to decode it, and one entry it cannot decode (a reasoning
    effort it has no name for) fails the whole catalog, so it gets a translated entry. No version keeps the entry."""
    minimum: Final = stock_model.minimal_client_version
    if minimum is None or client_version is None:
        return True
    return codex_version_at_least(client_version, minimum if isinstance(minimum, str) else ".".join(map(str, minimum)))


def _tier_fields(stock_model: CodexStockModel, tiers: tuple[CodexServiceTier, ...]) -> Mapping[str, object]:
    """The tier fields of an entry offering `tiers`. Codex turns `/fast` on off a speed tier as much as off a
    service tier, so a speed tier stays only while a service tier billed the same way is still offered, and the
    default tier only while it is still offered."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_service_tier_cost_key

    offered: Final = frozenset(tier.id for tier in tiers)
    offered_prices: Final = frozenset(get_service_tier_cost_key(TOKEN_PRICE_KEY, tier) for tier in offered)
    speed_tiers: Final = tuple(
        tier
        for tier in stock_model.additional_speed_tiers
        if get_service_tier_cost_key(TOKEN_PRICE_KEY, tier) in offered_prices - {TOKEN_PRICE_KEY}
    )
    return MappingProxyType(
        {
            "service_tiers": tiers,
            **({"additional_speed_tiers": speed_tiers} if stock_model.additional_speed_tiers else {}),
            "default_service_tier": (
                stock_model.default_service_tier if stock_model.default_service_tier in offered else None
            ),
        }
    )


def _priced_tiers(row: CodexCatalogRow, stock_model: CodexStockModel) -> Mapping[str, object]:
    """Codex's own tiers for the model without the ones some deployment behind the row prices tokens but not
    that tier for, or nothing when none is dropped. Sending a tier the upstream does not sell fails the
    request, and the cost map prices a tier for exactly the models that sell it, while a deployment it has no
    token price for at all says nothing either way."""
    deployments: Final = row.deployment_capabilities
    tiers: Final = tuple(
        tier for tier in stock_model.service_tiers if no_priced_deployment_lacks_tier(tier.id, deployments)
    )
    return _NO_UPDATE if tiers == stock_model.service_tiers else _tier_fields(stock_model, tiers)


def _alias_context_window(row: CodexCatalogRow, stock_model: CodexStockModel) -> Mapping[str, int]:
    if row.max_input_tokens is None or stock_model.context_window is None:
        return MappingProxyType({})
    return MappingProxyType({"context_window": min(stock_model.context_window, row.max_input_tokens)})


def _configured_window(row: CodexCatalogRow, stock_model: CodexStockModel) -> Mapping[str, int]:
    """The `context_window` the operator's `model_info.max_input_tokens` gives a known model's entry,
    or nothing, which keeps `stock_model`'s own.

    A request to the row can route to any of its deployments, so the window is the smallest one of them
    takes: its `max_input_tokens` where the operator set a positive one, Codex's own window where not. A
    limit set on only some deployments can therefore lower the window but never raise it. The window never
    goes past `stock_model`'s `max_context_window`, the most Codex lets its own `model_context_window`
    setting raise that model to."""
    per_deployment: Final = row.configured_max_input_tokens
    declared: Final = tuple(limit for limit in per_deployment if limit is not None and limit > 0)
    unset_window: Final = () if len(declared) == len(per_deployment) else (stock_model.context_window,)
    if not declared or None in unset_window:
        return MappingProxyType({})
    smallest: Final = min(limit for limit in (*declared, *unset_window) if limit is not None)
    ceiling: Final = stock_model.max_context_window
    return MappingProxyType({"context_window": smallest if ceiling is None else min(smallest, ceiling)})


def _alias_reasoning_level(row: CodexCatalogRow, stock_model: CodexStockModel) -> str | None:
    efforts: Final = frozenset(level.effort for level in stock_model.supported_reasoning_levels)
    return row.reasoning_effort if row.reasoning_effort in efforts else None


def _deployment_defaults(row: CodexCatalogRow, stock_model: CodexStockModel) -> Mapping[str, object]:
    """What an alias's deployments decide in place of the defaults Codex picks for the model they route to:
    no service tier, verbosity, upgrade nudge or startup promo of Codex's own, so what the deployment
    pins (or OpenAI's API default) stays in force, the pinned reasoning effort when Codex offers it and
    none otherwise, and a window no larger than the alias's own limit."""
    return MappingProxyType(
        {
            **_tier_fields(stock_model, ()),
            "default_verbosity": None,
            "upgrade": None,
            "availability_nux": None,
            "default_reasoning_level": _alias_reasoning_level(row, stock_model),
            **_alias_context_window(row, stock_model),
        }
    )


def _entry(
    index: int,
    row: CodexCatalogRow,
    stock: Mapping[str, CodexStockModel],
    served: frozenset[str],
    instructions: str,
    alias_defaults: CodexAliasDefaults,
    client_version: str | None,
) -> CodexCatalogEntry:
    matched: Final = _stock_for(row, stock)
    stock_model: Final = matched.model if matched is not None and _readable_by(matched.model, client_version) else None
    stock_tiers: Final = MappingProxyType(
        {tier.id: tier for tier in (stock_model.service_tiers if stock_model is not None else ())}
    )
    configured: Final = configured_service_tiers(row.service_tiers, row.id, stock_tiers)
    search_tool: Final = offered_search_tool(row.search_tool_support)
    if stock_model is None:
        return codex_fallback_entry(
            slug=row.id,
            display_name=row.display_name if row.display_name is not None else row.id,
            priority=index,
            capabilities=combined_capabilities(row.deployment_capabilities, row.max_input_tokens),
            instructions=instructions,
            service_tiers=configured if configured is not None else (),
            supports_search_tool=search_tool is True,
            reasoning_efforts=row.supported_reasoning_efforts or (),
            client_version=client_version,
        )
    upgrade: Final = (
        stock_model.upgrade if stock_model.upgrade is not None and stock_model.upgrade.model in served else None
    )
    is_alias: Final = matched is not None and matched.alias
    deployment_decides: Final = is_alias and alias_defaults == "deployment"
    stock_name: Final = row.id if is_alias else stock_model.display_name
    listing: Final = {
        "slug": row.id,
        "display_name": row.display_name if row.display_name is not None else stock_name,
        "priority": index,
        "visibility": "list",
        "supported_in_api": True,
        "upgrade": upgrade,
        **({} if search_tool is None else {"supports_search_tool": search_tool}),
        **(_deployment_defaults(row, stock_model) if deployment_decides else {}),
        **_configured_window(row, stock_model),
    }
    if configured is None:
        return stock_model.model_copy(update={**_priced_tiers(row, stock_model), **listing})
    return stock_model.model_copy(
        update={
            **listing,
            **_tier_fields(stock_model, configured),
            **({"default_service_tier": None} if deployment_decides else {}),
        }
    )


def _serialized(entry: CodexCatalogEntry) -> str:
    """A stock entry keeps exactly the fields its source printed (an older Codex may know no
    `service_tiers`), a fallback entry spells out every field Codex requires."""
    return entry.model_dump_json(exclude_unset=isinstance(entry, CodexStockModel))


def _bytes_left_after(left: int, size: int) -> int:
    return left - size if size <= left else left


def _fits(serialized: Sequence[str], budget: int | None) -> tuple[bool, ...]:
    """Whether each entry, taken in order, fits in what is left of `budget` bytes: an entry larger
    than the bytes left is passed over and a smaller one after it is still taken, each entry
    counted with its separating comma."""
    if budget is None:
        return (True,) * len(serialized)
    left: Final = tuple(
        accumulate((len(entry.encode()) + 1 for entry in serialized), _bytes_left_after, initial=budget)
    )
    return tuple(after < before for before, after in pairwise(left))


def _kept_positions(
    entries: Sequence[CodexCatalogEntry], serialized: Sequence[str], budget: int | None
) -> frozenset[int]:
    """The listing positions that survive the byte limit: an entry offering a service tier is taken
    ahead of one offering none, each group in listing order, since a model without a tier is driven
    by name just as well off Codex's bundled catalog; an entry too large for the bytes left never
    costs the smaller ones after it their place."""
    survival: Final = sorted(range(len(entries)), key=lambda position: (not entries[position].service_tiers, position))
    fits: Final = _fits(tuple(serialized[position] for position in survival), budget)
    return frozenset(position for position, fit in zip(survival, fits, strict=True) if fit)


def _entries(
    drivable: Sequence[CodexCatalogRow],
    stock: Mapping[str, CodexStockModel] | None,
    instructions: str | None,
    first_priority: int,
    alias_defaults: CodexAliasDefaults,
    client_version: str | None,
) -> tuple[CodexCatalogEntry, ...]:
    known: Final = bundled_codex_models() if stock is None else stock
    prompt: Final = codex_base_instructions() if instructions is None else instructions
    served: Final = frozenset(row.id for row in drivable)
    return tuple(
        _entry(first_priority + index, row, known, served, prompt, alias_defaults, client_version)
        for index, row in enumerate(drivable)
    )


@dataclass(frozen=True, slots=True)
class CodexCatalogBody:
    """A `ModelsResponse` body, the ids it lists, and the chat-capable ids the byte limit left out."""

    json: str
    listed: tuple[str, ...]
    left_out: tuple[str, ...]


def codex_catalog_byte_limit(client_version: str | None) -> int | None:
    """None, no limit, for a Codex that reads a catalog of any size since https://github.com/openai/codex/pull/56525.
    An older Codex, or one sending no version, drops a catalog over 1 MiB. 0.165.0 still has the limit."""
    return (
        None
        if codex_version_at_least(client_version, _FIRST_CODEX_WITHOUT_CATALOG_BYTE_LIMIT)
        else CODEX_CATALOG_BYTE_LIMIT
    )


def codex_models_response_json(
    rows: Sequence[CodexCatalogRow],
    *,
    stock: Mapping[str, CodexStockModel] | None = None,
    instructions: str | None = None,
    byte_limit: int | None = CODEX_CATALOG_BYTE_LIMIT,
    alias_defaults: CodexAliasDefaults = "codex",
    client_version: str | None = None,
) -> CodexCatalogBody:
    """Codex's `ModelsResponse` body for `rows`, in listing order, cut to Codex's byte limit.

    Only chat-capable rows are listed (Codex cannot drive an embedding or image model, and a
    wildcard id is no model), their order is their `priority`. Codex drops a catalog over its
    byte limit silently and keeps its bundled one, so the body holds only the entries that fit,
    the ones offering a service tier taken first and the rest in listing order, an entry too large
    for the bytes left passed over, every kept entry at its listing position; `byte_limit=None`
    keeps every entry. A row whose deployments set `model_info.max_input_tokens` gets the smallest
    window any of them takes as its known model's `context_window`, capped at that model's
    `max_context_window`. A row listed under another name than the Codex model its `upstream_model` names (an alias) takes that model's
    entry; `alias_defaults="deployment"` drops the defaults Codex picks for that model in favor of
    what the alias's deployments pin, while `"codex"` keeps them, as Codex would use them for the
    model's own name. Codex's own service tiers stay on a known model's entry only when no deployment
    behind the row prices tokens without pricing them, and only while the operator configures none. `client_version` is the
    requesting Codex's version, which decides the reasoning levels a fallback entry may offer; None counts
    as a version too old for custom levels.
    """
    drivable: Final = tuple(row for row in rows if _codex_can_drive(row.id, row.mode))
    entries: Final = _entries(drivable, stock, instructions, 0, alias_defaults, client_version)
    serialized: Final = tuple(_serialized(entry) for entry in entries)
    kept: Final = _kept_positions(
        entries, serialized, None if byte_limit is None else byte_limit - len(_BODY_PREFIX) - len(_BODY_SUFFIX) + 1
    )
    return CodexCatalogBody(
        json=f"{_BODY_PREFIX}{','.join(serialized[position] for position in sorted(kept))}{_BODY_SUFFIX}",
        listed=tuple(row.id for position, row in enumerate(drivable) if position in kept),
        left_out=tuple(row.id for position, row in enumerate(drivable) if position not in kept),
    )


def _priority(model: Mapping[str, object]) -> int:
    priority: Final = model.get("priority")
    return priority if isinstance(priority, int) else -1


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _entry_json(entry: CodexCatalogEntry) -> Mapping[str, object]:
    return _JSON_OBJECT.validate_python(entry.model_dump(mode="json", exclude_unset=isinstance(entry, CodexStockModel)))


def merged_codex_catalog(
    upstream_body: bytes,
    rows: Sequence[CodexCatalogRow],
    *,
    stock: Mapping[str, CodexStockModel] | None = None,
    instructions: str | None = None,
    byte_limit: int | None = CODEX_CATALOG_BYTE_LIMIT,
    client_version: str | None = None,
) -> str | None:
    try:
        top: Final = _JSON_OBJECT.validate_json(upstream_body)
        upstream: Final = _JSON_OBJECTS.validate_python(top.get("models"))
    except ValidationError:
        return None
    slugs: Final = frozenset(model.get("slug") for model in upstream)
    added: Final = tuple(row for row in rows if _codex_can_drive(row.id, row.mode) and row.id not in slugs)
    first_priority: Final = max((_priority(model) for model in upstream), default=-1) + 1
    entries: Final = _entries(added, stock, instructions, first_priority, _NATIVE_ALIAS_DEFAULTS, client_version)
    added_models: Final = tuple(_entry_json(entry) for entry in entries)
    serialized: Final = tuple(_compact_json(model) for model in added_models)
    budget: Final = None if byte_limit is None else byte_limit - len(_compact_json({**top, "models": upstream}))
    kept: Final = tuple(sorted(_kept_positions(entries, serialized, budget)))
    if len(kept) < len(entries):
        verbose_proxy_logger.warning(
            "Codex native model catalog left out %d of %d LiteLLM models to stay within Codex's %d byte limit",
            len(entries) - len(kept),
            len(entries),
            byte_limit,
        )
    if not kept:
        return None
    return _compact_json({**top, "models": [*upstream, *(added_models[position] for position in kept)]})


def _canonical_value(value: object) -> object:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else repr(value)


def codex_catalog_etag(upstream_etag: str, rows: Sequence[CodexCatalogRow]) -> str:
    canonical: Final = json.dumps(
        [asdict(row) for row in rows], sort_keys=True, separators=(",", ":"), default=_canonical_value
    )
    digest: Final = hashlib.sha256(json.dumps([upstream_etag, canonical]).encode()).hexdigest()
    return f'"{digest}"'


def upstream_codex_slugs(upstream_body: bytes) -> frozenset[str] | None:
    try:
        upstream: Final = _JSON_OBJECTS.validate_python(_JSON_OBJECT.validate_json(upstream_body).get("models"))
    except ValidationError:
        return None
    slugs: Final = (model.get("slug") for model in upstream)
    return frozenset(slug for slug in slugs if isinstance(slug, str))


def _group_reasoning_efforts(
    row: ModelInfoResponse, lookup_id: str, registry: RouterRegistry | None, team_id: str | None
) -> tuple[str, ...] | None:
    if registry is None or not _codex_can_drive(row["id"], row.get("mode")):
        return None
    group: Final = registry.group(lookup_id, include_hidden=True, team_id=team_id)
    return group.supported_reasoning_efforts if group is not None else None


def _shared_pinned_effort(pinned: tuple[object, ...]) -> str | None:
    first: Final = next(iter(pinned), None)
    return first if isinstance(first, str) and all(effort == first for effort in pinned) else None


def deployment_capabilities(model_info: Mapping[str, object] | None, model_id: str) -> CodexDeploymentCapabilities:
    """One deployment's model info, merged with the cost map, as the capabilities a Codex entry describes.
    Model info that does not validate declares nothing, logged, so one bad value never fails the listing."""
    if model_info is None:
        return CodexDeploymentCapabilities()
    prices_tokens: Final = model_info.get(TOKEN_PRICE_KEY) is not None
    priced: Final = frozenset(key for key in tier_price_keys() if model_info.get(key) is not None)
    try:
        return CodexDeploymentCapabilities.model_validate(
            {**model_info, "prices_tokens": prices_tokens, "priced_service_tier_keys": priced}
        )
    except ValidationError as e:
        verbose_proxy_logger.warning(
            "model info of a deployment of %s declares no capability to Codex: %s", model_id, e.errors()[0]["msg"]
        )
        return CodexDeploymentCapabilities(prices_tokens=prices_tokens, priced_service_tier_keys=priced)


def _deployment_capabilities(
    lookup_id: str, registry: RouterRegistry | None, team_id: str | None
) -> tuple[CodexDeploymentCapabilities, ...]:
    if registry is None:
        return ()
    return tuple(
        deployment_capabilities(model_info, lookup_id)
        for model_info in registry.group(lookup_id, field="deployment_model_info", team_id=team_id)
    )


def _catalog_row(
    row: ModelInfoResponse, lookup_id: str, label_id: str, llm_router: Router | None, team_id: str | None
) -> CodexCatalogRow:
    registry: Final = llm_router.registry() if llm_router is not None else None
    return CodexCatalogRow(
        id=row["id"],
        mode=row.get("mode"),
        max_input_tokens=row.get("max_input_tokens"),
        upstream_model=registry.resolve(lookup_id, kind="upstream", team_id=team_id) if registry is not None else None,
        upstream_provider=(
            registry.resolve(lookup_id, kind="upstream_provider", team_id=team_id) if registry is not None else None
        ),
        upstreams=registry.resolve(lookup_id, kind="upstreams", team_id=team_id) if registry is not None else (),
        display_name=registry.group(label_id, field="display_name") if registry is not None else None,
        reasoning_effort=(
            _shared_pinned_effort(registry.group(lookup_id, field="pinned_reasoning_efforts", team_id=team_id))
            if registry is not None
            else None
        ),
        service_tiers=registry.group(lookup_id, field="service_tiers", team_id=team_id) if registry is not None else (),
        supported_reasoning_efforts=_group_reasoning_efforts(row, lookup_id, registry, team_id),
        search_tool_support=(
            registry.resolve(lookup_id, kind="search_tool_support", team_id=team_id) if registry is not None else ()
        ),
        configured_max_input_tokens=(
            registry.group(lookup_id, field="max_input_tokens", team_id=team_id) if registry is not None else ()
        ),
        deployment_capabilities=_deployment_capabilities(lookup_id, registry, team_id),
    )


def codex_catalog_rows(
    rows: Sequence[ModelInfoResponse],
    entries: Sequence[tuple[str, str]],
    llm_router: Router | None,
    team_id: str | None = None,
    label_entries: Sequence[tuple[str, str]] | None = None,
) -> tuple[CodexCatalogRow, ...]:
    """`rows` joined with the router's configured metadata, looked up by each entry's internal id so
    team-scoped rows resolve the way the Anthropic listing's display names do, a `model_group_alias`
    reading its target's deployments under its own name; `team_id` is the requesting key's team, so the
    upstream model that picks Codex's stock entry, the tiers, the configured input limits and each
    deployment's capabilities are read only off the deployments its requests can route to, and the
    reasoning efforts are the ones
    `/model_group/info` reports for the entry. Display names are read under `label_entries`' ids when given."""
    lookup_ids: Final = MappingProxyType(dict(entries))
    label_ids: Final = lookup_ids if label_entries is None else MappingProxyType(dict(label_entries))
    return tuple(
        _catalog_row(
            row, lookup_ids.get(row["id"], row["id"]), label_ids.get(row["id"], row["id"]), llm_router, team_id
        )
        for row in rows
    )


def codex_model_list_body(
    rows: Sequence[ModelInfoResponse],
    entries: Sequence[tuple[str, str]],
    llm_router: Router | None,
    team_id: str | None = None,
    label_entries: Sequence[tuple[str, str]] | None = None,
    *,
    client_version: str | None = None,
    stock: Mapping[str, CodexStockModel] | None = None,
    instructions: str | None = None,
) -> str:
    """The `/v1/models?client_version=...` body, logging the models Codex's byte limit left out. `stock` and
    `instructions` are the requesting Codex release's catalog and prompt, the bundled ones when None."""
    byte_limit: Final = codex_catalog_byte_limit(client_version)
    body: Final = codex_models_response_json(
        codex_catalog_rows(rows, entries, llm_router, team_id, label_entries),
        stock=stock,
        instructions=instructions,
        byte_limit=byte_limit,
        client_version=client_version,
    )
    if body.left_out:
        verbose_proxy_logger.warning(
            "Codex model catalog cut at %d bytes, left out: %s. Models offering a service tier are kept first, "
            "then model_list order; list the models Codex users need first in model_list",
            byte_limit,
            ", ".join(body.left_out),
        )
    return body.json


def codex_entry_names_default_reasoning_level(
    model_id: str, llm_router: Router, team_id: str | None, client_version: str | None
) -> bool:
    """Whether the native catalog entry for `model_id` names a `default_reasoning_level`. Codex's model picker
    sends reasoning effort `none` for a model whose entry names none, while showing that choice as
    "default"."""
    listed: Final[ModelInfoResponse] = {"id": model_id, "object": "model", "created": 0, "owned_by": ""}
    (entry,) = _entries(
        (_catalog_row(listed, model_id, model_id, llm_router, team_id),),
        None,
        "",
        0,
        _NATIVE_ALIAS_DEFAULTS,
        client_version,
    )
    return _entry_json(entry).get("default_reasoning_level") is not None
