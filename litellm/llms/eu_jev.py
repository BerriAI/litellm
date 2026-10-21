from typing import Final

EU_JEV_LATEST_MODEL: Final = "eu-jeff-latest"
_EU_JEV_CURRENT_RELEASE_NAMES: Final = frozenset((EU_JEV_LATEST_MODEL, "jeff-latest", "jeff-1.0.0"))


def eu_jev_registry_model(requested_model: str | None) -> str:
    if requested_model is None or requested_model in _EU_JEV_CURRENT_RELEASE_NAMES:
        return EU_JEV_LATEST_MODEL
    return requested_model
