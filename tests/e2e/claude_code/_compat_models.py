"""Load the claude_code compat matrix's deployment list from
``test_config.yaml`` and select the subset whose provider credentials
are actually set in the current environment.

Design goal: keep exactly one source of truth for the compat matrix's
model_list. ``test_config.yaml`` is the config the stage deployment
uses; parsing it at fixture time means a change there (new tier, tier
retirement, provider swap, endpoint rename) reaches this fixture with
no extra edit. A drift-check test asserts every ``*_MODELS`` list
referenced by the compat cells is covered by the yaml, so a cell that
adds a probe for a name the yaml doesn't know about fails loudly at
collection instead of at 400-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from models import LiteLLMParamsBody

CONFIG_PATH = Path(__file__).resolve().parent / "test_config.yaml"


@dataclass(frozen=True, slots=True)
class CompatDeployment:
    model_name: str
    litellm_params: LiteLLMParamsBody


_OS_ENV_PREFIX = "os.environ/"


def _referenced_env_vars(params: Mapping[str, object]) -> tuple[str, ...]:
    """Collect the environment variables an ``os.environ/FOO`` param
    reference points at. If a provider needs credentials in the ambient
    process environment (bedrock via IMDS, vertex via ADC) the yaml
    simply omits those fields, so this returns an empty tuple and the
    caller has to decide separately whether the ambient path is set up.
    """
    return tuple(
        value[len(_OS_ENV_PREFIX):]
        for value in params.values()
        if isinstance(value, str) and value.startswith(_OS_ENV_PREFIX)
    )


# Providers that authenticate through the *ambient* process environment
# (no ``os.environ/FOO`` reference in the yaml, credentials picked up by
# the AWS/GCP SDKs directly). If any of these variables is set, we count
# the ambient path as configured and register those deployments. Empty
# means: skip those cells locally, they will 400 with "Invalid model
# name" the same way any other unregistered model does.
_AMBIENT_AWS_ENV = ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_ROLE_ARN")
_AMBIENT_GCP_ENV = ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT")


def _is_bedrock(params: Mapping[str, object]) -> bool:
    model = params.get("model")
    return isinstance(model, str) and model.startswith("bedrock/")


def _is_vertex(params: Mapping[str, object]) -> bool:
    model = params.get("model")
    return isinstance(model, str) and model.startswith("vertex_ai/")


def _has_all(env: Mapping[str, str], names: Iterable[str]) -> bool:
    return all(env.get(name) for name in names)


def _has_any(env: Mapping[str, str], names: Iterable[str]) -> bool:
    return any(env.get(name) for name in names)


def _is_available(
    params: Mapping[str, object], env: Mapping[str, str]
) -> bool:
    """True if every credential the deployment needs is present in ``env``.

    Rules, in this order:
    - Any ``os.environ/FOO`` references in the yaml must be set. Missing
      one is a hard skip for that deployment.
    - Bedrock deployments additionally need one of the ambient AWS creds.
    - Vertex deployments additionally need one of the ambient GCP creds.
    """
    if not _has_all(env, _referenced_env_vars(params)):
        return False
    if _is_bedrock(params) and not _has_any(env, _AMBIENT_AWS_ENV):
        return False
    if _is_vertex(params) and not _has_any(env, _AMBIENT_GCP_ENV):
        return False
    return True


# The yaml uses ``vertex_ai_*`` for the vertex project/location fields
# (that is the spelling the proxy config file historically standardized
# on), while ``LiteLLMParamsBody`` names them without the ``_ai`` infix
# (matching the proxy's DB column). Both spellings resolve at call time
# on the proxy side, but pydantic silently drops unknown fields, so a
# raw ``LiteLLMParamsBody(**entry)`` would produce a body with the
# vertex project stripped — the resulting deployment 400s at
# ``/v1/messages`` with "Invalid model name". Normalize the yaml keys
# to the pydantic names in one place.
_YAML_TO_PYDANTIC_ALIASES = {
    "vertex_ai_project": "vertex_project",
    "vertex_ai_location": "vertex_location",
    "vertex_ai_credentials": "vertex_credentials",
}


def _normalize_params(raw: Mapping[str, object]) -> dict[str, object]:
    return {_YAML_TO_PYDANTIC_ALIASES.get(k, k): v for k, v in raw.items()}


def load_all_deployments(config_path: Path = CONFIG_PATH) -> tuple[CompatDeployment, ...]:
    """Parse every ``model_list`` entry from the yaml, in file order."""
    doc = yaml.safe_load(config_path.read_text())
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


def _raw_params_by_name(
    config_path: Path,
) -> dict[str, Mapping[str, object]]:
    """Return the *raw* (yaml-shape) params for each deployment so the
    availability check sees the exact ``os.environ/FOO`` references the
    yaml declared (pydantic strips unknown fields, which would hide
    references the ``_is_available`` check needs to see)."""
    doc = yaml.safe_load(config_path.read_text())
    return {entry["model_name"]: entry["litellm_params"] for entry in doc.get("model_list") or []}


def deployments_for_available_providers(
    env: Mapping[str, str],
    *,
    config_path: Path = CONFIG_PATH,
) -> tuple[CompatDeployment, ...]:
    """Subset of ``load_all_deployments`` the current env can serve."""
    raw = _raw_params_by_name(config_path)
    return tuple(
        d
        for d in load_all_deployments(config_path)
        if _is_available(raw[d.model_name], env)
    )


def all_expected_model_names(
    *, config_path: Path = CONFIG_PATH
) -> frozenset[str]:
    """Every virtual name the compat matrix declares — the ground truth
    the cells are supposed to probe. Used by the drift-check test."""
    return frozenset(d.model_name for d in load_all_deployments(config_path))
