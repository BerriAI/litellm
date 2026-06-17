"""Team-scoped (BYOK) model-name translation for the model listing endpoints.

`/v1/models`, `/models`, and `GET /v1/models/{id}` should surface the public
`team_public_model_name` rather than the internal routing key
`model_name_{team_id}_{uuid}`, consistent with `/v1/model/info`. The internal
key still routes regardless; this is a presentation-layer swap only and does not
touch access-group or auth semantics (see issue #28382). Operators can pin the
legacy internal names with `general_settings.use_team_public_model_name: false`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from litellm.router import Router


class TeamModelNameTranslator:
    """Translates internal team routing keys to their public names for the model
    listing/retrieve responses. Stateless; the live router and general_settings
    are injected per call so the unit tests can drive it without globals.
    """

    @staticmethod
    def _internal_public_pair(model: object) -> tuple[str, str] | None:
        """`(internal_routing_key, public_name)` for a team-scoped row, else None."""
        if not isinstance(model, dict):
            return None
        model_dict = cast(dict[str, object], model)  # any-ok: checked
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

    @staticmethod
    def _is_enabled(general_settings: Mapping[str, object]) -> bool:
        return general_settings.get("use_team_public_model_name", True) is not False

    @staticmethod
    def build_internal_to_public_map(
        llm_router: "Router | None",
        general_settings: Mapping[str, object],
    ) -> dict[str, str]:
        """Internal team routing key -> public `team_public_model_name`.

        Empty when disabled via the legacy flag, the router is absent, or the
        router model list is malformed.
        """
        if llm_router is None or not TeamModelNameTranslator._is_enabled(
            general_settings
        ):
            return {}
        router_model_list = llm_router.get_model_list()
        if not isinstance(router_model_list, list):
            return {}
        return dict(
            pair
            for pair in (
                TeamModelNameTranslator._internal_public_pair(model)
                for model in router_model_list
            )
            if pair is not None
        )

    @staticmethod
    def translate_listing(
        model_names: list[str],
        llm_router: "Router | None",
        general_settings: Mapping[str, object],
    ) -> list[str]:
        """Swap internal team routing keys for public names in a list response.

        Sibling deployments sharing a public name collapse to one entry while
        preserving order; unmapped names (globals, access-group keys) pass through.
        """
        internal_to_public = TeamModelNameTranslator.build_internal_to_public_map(
            llm_router, general_settings
        )
        if not internal_to_public:
            return model_names
        deduped: dict[str, None] = {
            internal_to_public.get(n, n): None for n in model_names
        }
        return list(deduped)

    @staticmethod
    def resolve_public_name(
        model_id: str,
        available_models: list[str],
        llm_router: "Router | None",
        general_settings: Mapping[str, object],
    ) -> str:
        """Resolve a public team name back to the internal routing key the router
        indexes by, so `GET /v1/models/{id}` accepts the name the listing returns.

        Resolution is restricted to `available_models` (the caller's accessible
        set) so colliding public names across teams never resolve across an access
        boundary. Returns `model_id` unchanged when it is not an accessible public
        team name (already-internal names and globals pass through).
        """
        internal_to_public = TeamModelNameTranslator.build_internal_to_public_map(
            llm_router, general_settings
        )
        if not internal_to_public:
            return model_id
        return next(
            (
                internal
                for internal in available_models
                if internal_to_public.get(internal) == model_id
            ),
            model_id,
        )
