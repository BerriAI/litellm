from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from litellm.router import Router


def get_model_listing_entries(
    model_names: list[str],
    llm_router: "Router | None",
    general_settings: object,
) -> tuple[tuple[str, str], ...]:
    """Build `/v1/models` entries while keeping router lookup keys intact.

    Team-scoped BYOK deployments use internal router keys like
    `model_name_{team_id}_{uuid}`. The listing response should surface the
    public model name, but metadata lookups such as fallbacks still need the
    internal key because the router indexes those configs by routing key.
    """
    if not _should_use_team_public_model_name(general_settings) or llm_router is None:
        return _default_model_listing_entries(model_names)

    router_model_list: object = llm_router.get_model_list()
    if not isinstance(router_model_list, list):
        return _default_model_listing_entries(model_names)
    router_models = cast(list[object], router_model_list)  # any-ok: checked

    team_name_pairs = tuple(
        pair
        for model in router_models
        for pair in (_team_public_name_pair(model),)
        if pair is not None
    )
    if not team_name_pairs:
        return _default_model_listing_entries(model_names)

    return _dedupe_model_listing_entries(
        tuple(
            _model_listing_entry(model_name, team_name_pairs)
            for model_name in model_names
        )
    )


def _default_model_listing_entries(
    model_names: list[str],
) -> tuple[tuple[str, str], ...]:
    return tuple((model_name, model_name) for model_name in model_names)


def _should_use_team_public_model_name(general_settings: object) -> bool:
    if not isinstance(general_settings, Mapping):
        return True
    settings = cast(Mapping[str, object], general_settings)  # any-ok: checked
    return settings.get("use_team_public_model_name", True) is not False


def _team_public_name_pair(model: object) -> tuple[str, str] | None:
    if not isinstance(model, Mapping):
        return None
    model_dict = cast(Mapping[str, object], model)  # any-ok: checked
    model_info_raw: object = model_dict.get("model_info")
    if not isinstance(model_info_raw, Mapping):
        return None

    model_info = cast(Mapping[str, object], model_info_raw)  # any-ok: checked
    team_id = model_info.get("team_id")
    team_public = model_info.get("team_public_model_name")
    name = model_dict.get("model_name")
    if (
        isinstance(team_id, str)
        and isinstance(team_public, str)
        and isinstance(name, str)
        and name.startswith(f"model_name_{team_id}_")
    ):
        return name, team_public
    return None


def _model_listing_entry(
    model_name: str,
    team_name_pairs: tuple[tuple[str, str], ...],
) -> tuple[str, str]:
    for internal_name, public_name in team_name_pairs:
        if internal_name == model_name:
            return public_name, internal_name
    return model_name, model_name


def _dedupe_model_listing_entries(
    entries: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    deduped_entries: list[tuple[str, str]] = []
    seen_response_ids: set[str] = set()
    for entry in entries:
        response_model_id: str = entry[0]
        if response_model_id in seen_response_ids:
            continue
        seen_response_ids.add(response_model_id)
        deduped_entries.append(entry)
    return tuple(deduped_entries)
