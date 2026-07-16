"""Load Claude Code matrix deployments from test_config.yaml for /model/new."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from models import LiteLLMParamsBody

CONFIG_PATH = Path(__file__).resolve().parent / "test_config.yaml"

_YAML_TO_PYDANTIC_ALIASES = {
    "vertex_ai_project": "vertex_project",
    "vertex_ai_location": "vertex_location",
    "vertex_ai_credentials": "vertex_credentials",
}


@dataclass(frozen=True, slots=True)
class CompatDeployment:
    model_name: str
    litellm_params: LiteLLMParamsBody


def _normalize_params(raw: Mapping[str, object]) -> dict[str, object]:
    return {_YAML_TO_PYDANTIC_ALIASES.get(key, key): value for key, value in raw.items()}


ConfigReader = Callable[[Path], str]


def _default_reader(path: Path) -> str:
    return path.read_text()


def load_all_deployments(
    config_path: Path = CONFIG_PATH,
    reader: ConfigReader = _default_reader,
) -> tuple[CompatDeployment, ...]:
    doc = yaml.safe_load(reader(config_path))
    if not isinstance(doc, dict):
        return ()
    model_list = doc.get("model_list") or []
    if not isinstance(model_list, list):
        return ()
    return tuple(
        CompatDeployment(
            model_name=str(entry["model_name"]),
            litellm_params=LiteLLMParamsBody(
                **_normalize_params(cast(Mapping[str, object], entry["litellm_params"]))
            ),
        )
        for entry in model_list
        if isinstance(entry, dict) and "model_name" in entry and "litellm_params" in entry
    )


def all_expected_model_names(
    *,
    config_path: Path = CONFIG_PATH,
    reader: ConfigReader = _default_reader,
) -> frozenset[str]:
    return frozenset(d.model_name for d in load_all_deployments(config_path, reader))
