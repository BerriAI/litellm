"""Load the claude_code compat matrix's deployment list from
``test_config.yaml``.

``test_config.yaml`` is the ground-truth config the stage deployment
uses; parsing it at fixture time means a change there (new tier, tier
retirement, provider swap, endpoint rename) reaches the fixture with
no extra edit. A drift-check test asserts every ``*_MODELS`` list
referenced by the compat cells is covered by the yaml, so a cell that
adds a probe for a name the yaml doesn't know about fails loudly at
collection instead of at 400-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import yaml

from models import LiteLLMParamsBody

CONFIG_PATH = Path(__file__).resolve().parent / "test_config.yaml"


@dataclass(frozen=True, slots=True)
class CompatDeployment:
    model_name: str
    litellm_params: LiteLLMParamsBody


# The yaml uses ``vertex_ai_*`` for the vertex project/location fields
# (that is the spelling the proxy config file historically standardized
# on), while ``LiteLLMParamsBody`` names them without the ``_ai`` infix
# (matching the proxy's DB column). Both spellings resolve at call time
# on the proxy side, but pydantic silently drops unknown fields, so a
# raw ``LiteLLMParamsBody(**entry)`` would produce a body with the
# vertex project stripped - the resulting deployment 400s at
# ``/v1/messages`` with "Invalid model name". Normalize the yaml keys
# to the pydantic names in one place.
_YAML_TO_PYDANTIC_ALIASES = {
    "vertex_ai_project": "vertex_project",
    "vertex_ai_location": "vertex_location",
    "vertex_ai_credentials": "vertex_credentials",
}


def _normalize_params(raw: Mapping[str, object]) -> dict[str, object]:
    return {_YAML_TO_PYDANTIC_ALIASES.get(k, k): v for k, v in raw.items()}


ConfigReader = Callable[[Path], str]


def _default_reader(path: Path) -> str:
    return path.read_text()


def load_all_deployments(
    config_path: Path = CONFIG_PATH,
    reader: ConfigReader = _default_reader,
) -> tuple[CompatDeployment, ...]:
    """Every deployment declared in the yaml, in file order."""
    doc = yaml.safe_load(reader(config_path))
    model_list = doc.get("model_list") or []
    return tuple(
        CompatDeployment(
            model_name=entry["model_name"],
            litellm_params=LiteLLMParamsBody(
                **_normalize_params(entry["litellm_params"])
            ),
        )
        for entry in model_list
    )


def all_expected_model_names(
    *,
    config_path: Path = CONFIG_PATH,
    reader: ConfigReader = _default_reader,
) -> frozenset[str]:
    """Every virtual name the compat matrix declares - the ground truth
    the cells are supposed to probe. Used by the drift-check test."""
    return frozenset(
        d.model_name for d in load_all_deployments(config_path, reader)
    )
