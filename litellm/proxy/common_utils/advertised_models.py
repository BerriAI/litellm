"""Catalog-only entries appended to the model listing.

`general_settings.advertised_models` lets an operator advertise a model id on
`GET /v1/models` without registering a routable deployment for it. The use case
is a model clients reach on their own, e.g. a realtime endpoint over a
websocket: it belongs in the client's model picker, but the proxy never serves
it, and inventing a placeholder deployment just to make it appear would leave a
live route that fails confusingly when someone calls it.

Discovery only, in both directions. Nothing here registers a route, so a request
naming a catalog id still fails as an unknown model, and `GET /v1/models/{id}`
still answers 404: the catalog says what exists, not what this proxy serves.

A catalog entry never displaces a routed one, and never resurrects one. The
reserved set is every name the router knows plus every id already listed, not
just what this caller can see, so an entry cannot re-expose a model that team
scoping, a pause or a health filter had hidden from them, nor impersonate it
under a different owner. A repeated id is listed once.

Every row is marked `catalog_only`, so a client picking models out of the
listing can tell an advertised id from one the proxy will actually route.

A row carries exactly what the operator declared. Nothing is inferred from the
cost map, so an entry that happens to share a name with a known model does not
silently pick up that model's mode or token limits, and an entry the cost map
cannot resolve does not drive `get_llm_provider` into printing its provider list
to stdout on every listing request.

Misconfiguration fails open: an `advertised_models` value that does not validate
is logged and skipped, leaving the listing exactly as it would have been.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MODEL_CREATED_AT_TIME
from litellm.proxy._types import AdvertisedModel
from litellm.types.proxy.model_listing import ModelInfoResponse

if TYPE_CHECKING:
    from litellm.router import Router

ADVERTISED_MODELS_SETTING: Final = "advertised_models"

_ADVERTISED_MODELS_ADAPTER: Final = TypeAdapter(tuple[AdvertisedModel, ...])


def configured_advertised_models(general_settings: Mapping[str, object]) -> tuple[AdvertisedModel, ...]:
    """Typed `advertised_models` entries, or none when unset or malformed."""
    configured: Final = general_settings.get(ADVERTISED_MODELS_SETTING)
    if configured is None:
        return ()
    try:
        return _ADVERTISED_MODELS_ADAPTER.validate_python(configured)
    except ValidationError as validation_error:
        verbose_proxy_logger.warning(
            "general_settings.%s is not a list of {id, owned_by} entries, skipping it: %s",
            ADVERTISED_MODELS_SETTING,
            validation_error,
        )
        return ()


def _listing_row(entry: AdvertisedModel) -> ModelInfoResponse:
    row: Final[ModelInfoResponse] = {
        "id": entry.id,
        "object": "model",
        "created": DEFAULT_MODEL_CREATED_AT_TIME,
        "owned_by": entry.owned_by,
        "catalog_only": True,
    }
    return row


def _reserved_ids(listed_ids: Collection[str], llm_router: Router | None) -> frozenset[str]:
    """Ids a catalog entry may not claim.

    Every name the router knows counts, not just the ids this caller can see, so
    an entry cannot re-expose a deployment that team scoping, a pause or a health
    filter had already removed from their listing.
    """
    if llm_router is None:
        return frozenset(listed_ids)
    return frozenset(listed_ids).union(llm_router.get_model_names(), llm_router.get_model_access_groups())


def _first_per_id(entries: tuple[AdvertisedModel, ...]) -> tuple[AdvertisedModel, ...]:
    """The first entry declared for each id, in configured order."""
    latest_wins: Final = MappingProxyType({entry.id: entry for entry in reversed(entries)})
    ordered_ids: Final = tuple(dict.fromkeys(entry.id for entry in entries))
    return tuple(latest_wins[entry_id] for entry_id in ordered_ids)


def advertised_model_rows(
    general_settings: Mapping[str, object],
    listed_ids: Collection[str],
    llm_router: Router | None = None,
) -> tuple[ModelInfoResponse, ...]:
    """Listing rows for the configured catalog entries, in configured order."""
    configured: Final = configured_advertised_models(general_settings)
    if not configured:
        return ()
    reserved: Final = _reserved_ids(listed_ids, llm_router)
    candidates: Final = tuple(entry for entry in configured if entry.id not in reserved)
    return tuple(_listing_row(entry) for entry in _first_per_id(candidates))
