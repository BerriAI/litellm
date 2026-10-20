"""Serve test-chosen entries as ``litellm.model_cost`` without writing into the read-only view."""

from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.rust_bridge._native import ModelCatalog
from litellm.rust_bridge.model_catalog import ModelCatalogSlot
from litellm.utils import invalidate_model_cost_caches

TEST_OWNER: Final = "test"


def serve_model_cost(monkeypatch: pytest.MonkeyPatch, entries: Mapping[str, Mapping[str, object]]) -> None:
    """Serve exactly ``entries`` as ``litellm.model_cost`` for the rest of the test."""
    slot: Final = ModelCatalogSlot(ModelCatalog.empty())
    slot.register(TEST_OWNER, entries)
    monkeypatch.setattr(litellm, "model_catalog", slot)
    invalidate_model_cost_caches()


def add_model_cost(entries: Mapping[str, Mapping[str, object]]) -> None:
    """Merge ``entries`` into the served catalog as written, bypassing ``register_model``'s resolution.

    The unit conftest gives every test its own catalog, so the entries are gone after the test.
    """
    litellm.model_catalog.register(TEST_OWNER, entries)
    invalidate_model_cost_caches()


def serve_model_cost_without(monkeypatch: pytest.MonkeyPatch, *keys: str) -> None:
    """Serve the current catalog minus ``keys`` for the rest of the test."""
    removed: Final = frozenset(keys)
    serve_model_cost(
        monkeypatch, {key: dict(entry) for key, entry in litellm.model_cost.items() if key not in removed}
    )
