"""Shared helpers for the registry characterization tests (``test_registry_characterization*.py``)."""

import uuid
from collections.abc import Callable, Iterable, Mapping

from litellm.router_core import RouterCore
from litellm.types.router import Deployment
from tests.unit.test_router.core.scripted_host import deployment

MakeCore = Callable[..., RouterCore]

MESSAGES = [{"role": "user", "content": "hi"}]


def ids(deployments: Iterable[Mapping[str, object]] | None) -> list[str]:  # mutable-ok: a fresh list
    """The ``model_info.id`` of each deployment dict, in order."""
    return [str(entry["model_info"]["id"]) for entry in deployments or ()]  # pyright: ignore[reportIndexIssue]  # deployment dicts


def names_and_ids(
    deployments: Iterable[Mapping[str, object]] | None,
) -> list[tuple[str, str]]:  # mutable-ok: a fresh list
    """``(model_name, model_info.id)`` of each deployment dict, in order."""
    return [(str(entry["model_name"]), str(entry["model_info"]["id"])) for entry in deployments or ()]  # pyright: ignore[reportIndexIssue]  # deployment dicts


def typed(model_name: str, deployment_id: str, **params: object) -> Deployment:  # kwargs-ok: litellm_params
    """A ``Deployment`` built from the same entry ``scripted_host.deployment`` returns."""
    return Deployment(**deployment(model_name, deployment_id, **params))  # pyright: ignore[reportArgumentType]  # a model_list entry


def team_deployment(
    model_name: str, deployment_id: str, team_id: str, public_name: str, **params: object
) -> dict[str, object]:  # mutable-ok: a model_list entry  # kwargs-ok: litellm_params
    """A team-owned deployment: ``model_info.team_id`` and ``team_public_model_name`` set."""
    extra_info: dict[str, object] = dict(params.pop("model_info", None) or {})  # pyright: ignore[reportArgumentType]  # a model_info dict
    return deployment(
        model_name,
        deployment_id,
        model_info={"team_id": team_id, "team_public_model_name": public_name, **extra_info},
        **params,
    )


def unique_id(prefix: str = "reg") -> str:
    """A deployment id no other test uses, for tests that write into the process-wide ``litellm.model_cost``."""
    return f"{prefix}-{uuid.uuid4().hex}"
