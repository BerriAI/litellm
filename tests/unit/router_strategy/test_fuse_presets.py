import json
from hashlib import sha256
from importlib.resources import files
from typing import Final, Literal

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
    assert all(entry.sources and all(source.startswith("https://") for source in entry.sources) for entry in entries)


@pytest.mark.parametrize(
    ("kind", "preset_id", "expected_digest"),
    (
        ("model", "gpt-6-astra-v1", "a9403b0c00ea64081b7b08b5b968850670f3a047d219a7e0668f2169146ae96e"),
        ("model", "gpt-5.6-sol-v1", "2b91a6c43e0e93183aaaf9c355e1bbb8ed2e9817aab6b0c2f50148f53a23247b"),
        ("model", "gpt-5.6-luna-v1", "fff94a9e01bf4519798d5be4e76a3f9d57b75a2d9966a59dc92cbfeb5cd08d07"),
        ("model", "gpt-5.6-terra-v1", "75de040f3bea841fa4764885738303893ee7ac0804aed1e932cd3959185ff893"),
        ("model", "claude-haiku-4-5-v1", "91c1920953073462b6b70ef810596a5325f08286b5e62630cff47938fc4157db"),
        ("model", "claude-sonnet-5-v1", "133f4414c644a707cd8cf565a486153856f4836ca4e4f75ee0553f2b7a1e3663"),
        ("model", "claude-opus-5-v1", "9cbfcae45d2e3a2575e44ce5adf618f56614abff4b3221d35900c647200b99ef"),
        ("model", "claude-fable-5-v1", "25c275d7403f1572ffb4fe899d5feecd9a434ebdc37b4dd9ef601a8ecf4850fc"),
        ("model", "claude-fable-5-1-v1", "37693107c878ab6266530395bbdc2d2813d676d179bf281d05e5a5ec1b9d4c60"),
        ("harness", "unspecified-v1", "d9eb30b61509456f0c71ca805b33d821cab6605578d567a29ab421d8f602ce7b"),
        ("harness", "claude-code-v1", "7ee8e9d50f1cf44a8a58461efff66d6182f245d25499702c144d1c642c101ed9"),
        ("harness", "codex-cli-v1", "0678047e34562ef05b5e2fba099c1f9e5876304f7eaf3b0d8c3809e707eb3311"),
        ("harness", "opencode-v1", "8b6cc240d90091ac2ef9b374b535f981a55abb91e25d4c04fdb9fc206eeb907e"),
        ("harness", "mini-swe-agent-v1", "21e2dc4a8a2320a5a554a498b30516326dc3592ebf20b0f4db20ddb33e879a39"),
    ),
)
def test_existing_preset_text_is_unchanged(
    kind: Literal["model", "harness"], preset_id: str, expected_digest: str
) -> None:
    text: Final = resolve_fuse_profile(None, preset_id, kind)
    assert text is not None
    assert sha256(text.encode("utf-8")).hexdigest() == expected_digest


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
