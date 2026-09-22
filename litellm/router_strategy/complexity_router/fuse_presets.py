from functools import lru_cache
from importlib.resources import files
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ProfileText: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class FuseModelPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    text: ProfileText
    sources: tuple[str, ...] = Field(min_length=1)
    model: str


class FuseHarnessPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    text: ProfileText
    sources: tuple[str, ...] = Field(min_length=1)


class FusePresetCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    models: tuple[FuseModelPreset, ...]
    harnesses: tuple[FuseHarnessPreset, ...]


@lru_cache(maxsize=1)
def get_fuse_presets() -> FusePresetCatalog:
    return FusePresetCatalog.model_validate_json(
        files(__package__).joinpath("fuse_presets.json").read_text(encoding="utf-8")
    )


def resolve_fuse_profile(text: str | None, preset_id: str | None, kind: Literal["model", "harness"]) -> str | None:
    if preset_id is None:
        return text
    catalog: Final = get_fuse_presets()
    presets: Final = catalog.models if kind == "model" else catalog.harnesses
    preset: Final = next((entry for entry in presets if entry.id == preset_id), None)
    if preset is None:
        return None
    return text if text is not None else preset.text
