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

A catalog entry never displaces a routed one. An id already in the listing wins,
so a typo here cannot mask a working model or impersonate it, and a repeated id
is listed once.

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
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MODEL_CREATED_AT_TIME
from litellm.proxy._types import AdvertisedModel
from litellm.types.proxy.model_listing import ModelInfoResponse

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
    }
    return row


def advertised_model_rows(
    general_settings: Mapping[str, object],
    listed_ids: Collection[str],
) -> tuple[ModelInfoResponse, ...]:
    """Listing rows for the configured catalog entries, in configured order.

    `listed_ids` are the ids the listing already carries; entries naming one of
    them are dropped so a routed model is never displaced.
    """
    already_listed: Final = frozenset(listed_ids)
    candidates: Final = tuple(
        entry for entry in configured_advertised_models(general_settings) if entry.id not in already_listed
    )
    return tuple(
        _listing_row(entry)
        for index, entry in enumerate(candidates)
        if all(earlier.id != entry.id for earlier in candidates[:index])
    )
