from typing import List

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient


class AvailableTeamsDBHandler:
    @staticmethod
    async def _get_available_teams_from_db(prisma_client: PrismaClient) -> List[str]:
        litellm_settings = (
            await prisma_client.db.litellm_config.find_unique(
                where={"param_name": "litellm_settings"}
            )
            or {}
        )

        if (
            "default_internal_user_params" in litellm_settings
            and "available_teams" in litellm_settings["default_internal_user_params"]
        ):
            return litellm_settings["default_internal_user_params"]["available_teams"]
        else:
            return []

    @staticmethod
    async def _set_available_teams_in_db(
        prisma_client: PrismaClient, available_teams: List[str]
    ):
        litellm_settings = (
            await prisma_client.db.litellm_config.find_unique(
                where={"param_name": "litellm_settings"}
            )
            or {}
        )

        if "default_internal_user_params" not in litellm_settings:
            litellm_settings["default_internal_user_params"] = {}

        current_available_teams = litellm_settings["default_internal_user_params"].get(
            "available_teams", []
        )
        current_available_teams.extend(available_teams)
        litellm_settings["default_internal_user_params"][
            "available_teams"
        ] = current_available_teams

        await prisma_client.db.litellm_config.update(
            where={"param_name": "litellm_settings"},
            data={"param_value": safe_dumps(litellm_settings)},
        )
