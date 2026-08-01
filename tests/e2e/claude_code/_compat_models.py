"""Load and register the claude_code compat matrix's deployment list
from ``test_config.yaml``.

``test_config.yaml`` is the ground-truth config the stage deployment
uses; parsing it at fixture time means a change there (new tier, tier
retirement, provider swap, endpoint rename) reaches the fixture with
no extra edit. A drift-check test asserts every ``*_MODELS`` list
referenced by the compat cells is covered by the yaml, so a cell that
adds a probe for a name the yaml doesn't know about fails loudly at
collection instead of at 400-time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

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


@dataclass(frozen=True, slots=True)
class DeploymentFailure:
    model_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class RegistrationOutcome:
    model_ids: tuple[str, ...]
    failures: tuple[DeploymentFailure, ...]


RegisterDeployment = Callable[[CompatDeployment], str]
DeleteModel = Callable[[str], None]


def register_deployments(
    deployments: Sequence[CompatDeployment],
    register: RegisterDeployment,
) -> RegistrationOutcome:
    """Register every deployment concurrently and collect the outcome.

    Registration is one ``/model/new`` POST plus a poll until the data
    plane lists the model, so fifteen of them in sequence cost fifteen
    round trips before the first cell can run. They are independent, so
    they go out at once and the run pays roughly one.

    Errors stay values: a deployment the proxy can't serve (missing
    provider credential, bad params) lands in ``failures`` instead of
    aborting the batch, because the cells that target it should fail
    loudly on their own rather than taking the whole matrix down.
    """
    if not deployments:
        return RegistrationOutcome(model_ids=(), failures=())

    def _one(deployment: CompatDeployment) -> str | DeploymentFailure:
        try:
            return register(deployment)
        except Exception as exc:
            return DeploymentFailure(
                model_name=deployment.model_name,
                reason=f"{type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=len(deployments)) as pool:
        outcomes = tuple(pool.map(_one, deployments))

    return RegistrationOutcome(
        model_ids=tuple(o for o in outcomes if isinstance(o, str)),
        failures=tuple(o for o in outcomes if isinstance(o, DeploymentFailure)),
    )


def unregister_deployments(
    model_ids: Sequence[str],
    delete: DeleteModel,
) -> tuple[DeploymentFailure, ...]:
    """Delete every registered deployment concurrently; return what failed.

    Teardown runs after the last cell, so its cost is pure wall time at
    the end of a run. Failures come back as values because one flaky
    delete must not mask the test results the run just produced.
    """
    if not model_ids:
        return ()

    def _one(model_id: str) -> DeploymentFailure | None:
        try:
            delete(model_id)
        except Exception as exc:
            return DeploymentFailure(
                model_name=model_id,
                reason=f"{type(exc).__name__}: {exc}",
            )
        return None

    with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:
        return tuple(o for o in pool.map(_one, model_ids) if o is not None)


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
