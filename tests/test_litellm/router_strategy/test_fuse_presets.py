import json
from importlib.resources import files
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.router_strategy.complexity_router.fuse_presets import get_fuse_presets, resolve_fuse_profile


def test_catalog_is_loaded_once_and_preserves_bundled_content() -> None:
    get_fuse_presets.cache_clear()
    first: Final = get_fuse_presets()
    second: Final = get_fuse_presets()
    assert first is second
    bundled: Final = json.loads(
        files("litellm.router_strategy.complexity_router").joinpath("fuse_presets.json").read_text(encoding="utf-8")
    )
    assert first.model_dump(mode="json") == bundled
    entries: Final = (*first.models, *first.harnesses)
    assert len({entry.id for entry in entries}) == len(entries)
    assert len(first.models) == 9
    assert len(first.harnesses) == 5
    assert all(entry.sources and all(source.startswith("https://") for source in entry.sources) for entry in entries)


def test_every_catalog_entry_resolves_without_changing_custom_ownership() -> None:
    catalog: Final = get_fuse_presets()
    for entry in catalog.models:
        assert resolve_fuse_profile(None, entry.id, "model") == entry.text
        assert resolve_fuse_profile("Custom text", entry.id, "model") == "Custom text"
    for entry in catalog.harnesses:
        assert resolve_fuse_profile(None, entry.id, "harness") == entry.text
        assert resolve_fuse_profile("Custom text", entry.id, "harness") == "Custom text"
    assert resolve_fuse_profile("Custom text", None, "model") == "Custom text"
    assert resolve_fuse_profile("Custom text", None, "harness") == "Custom text"


def test_cached_catalog_and_records_cannot_be_modified() -> None:
    catalog: Final = get_fuse_presets()
    for record, field in ((catalog, "version"), (catalog.models[0], "text"), (catalog.harnesses[0], "text")):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(record, field, "Changed")
