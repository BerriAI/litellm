from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

EU_JEV_LATEST_MODEL: Final = "eu-jeff-latest"
_EU_JEV_CURRENT_RELEASE_NAMES: Final = frozenset((EU_JEV_LATEST_MODEL, "jeff-latest", "jeff-1.0.0"))
_EU_JEV_PREFIX: Final = "eu_jev/"


def eu_jev_registry_model(requested_model: str | None) -> str:
    release: Final = requested_model.removeprefix(_EU_JEV_PREFIX) if requested_model is not None else None
    if release is None or release in _EU_JEV_CURRENT_RELEASE_NAMES:
        return EU_JEV_LATEST_MODEL
    return release


def eu_jev_scoped_model(decision_body: Mapping[str, object]) -> str:
    requested_model: Final = decision_body.get("model")
    return f"{_EU_JEV_PREFIX}{eu_jev_registry_model(requested_model if isinstance(requested_model, str) else None)}"


def eu_jev_upstream_body(hook_body: Mapping[str, object], client_body: Mapping[str, object]) -> Mapping[str, object]:
    hook_model: Final = hook_body.get("model")
    without_model: Final = {key: value for key, value in hook_body.items() if key != "model"}
    if hook_model == eu_jev_scoped_model(client_body):
        client_model: Final = client_body.get("model")
        if client_model is None:
            return MappingProxyType(without_model)
        if isinstance(client_model, str):
            return MappingProxyType({**without_model, "model": client_model.removeprefix(_EU_JEV_PREFIX)})
        return MappingProxyType({**without_model, "model": client_model})
    if isinstance(hook_model, str):
        return MappingProxyType({**without_model, "model": hook_model.removeprefix(_EU_JEV_PREFIX)})
    return hook_body
