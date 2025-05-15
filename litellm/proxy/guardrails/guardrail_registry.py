# litellm/proxy/guardrails/guardrail_registry.py

from datetime import datetime, timezone
from typing import List, Optional

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient
from litellm.types.guardrails import Guardrail, SupportedGuardrailIntegrations

from .guardrail_initializers import (
    initialize_aim,
    initialize_aporia,
    initialize_bedrock,
    initialize_guardrails_ai,
    initialize_hide_secrets,
    initialize_lakera,
    initialize_presidio,
)

guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.APORIA.value: initialize_aporia,
    SupportedGuardrailIntegrations.BEDROCK.value: initialize_bedrock,
    SupportedGuardrailIntegrations.LAKERA.value: initialize_lakera,
    SupportedGuardrailIntegrations.AIM.value: initialize_aim,
    SupportedGuardrailIntegrations.PRESIDIO.value: initialize_presidio,
    SupportedGuardrailIntegrations.HIDE_SECRETS.value: initialize_hide_secrets,
    SupportedGuardrailIntegrations.GURDRAILS_AI.value: initialize_guardrails_ai,
}


class GuardrailRegistry:
    """
    Registry for guardrails

    Handles adding, removing, and getting guardrails in DB + in memory
    """

    def __init__(self):
        pass

    ###########################################################
    ########### DB management helpers for guardrails ###########
    ############################################################
    async def add_guardrail_to_db(
        self, guardrail: Guardrail, prisma_client: PrismaClient
    ):
        """
        Add a guardrail to the database
        """
        try:
            guardrail_name = guardrail.get("guardrail_name")
            litellm_params: str = safe_dumps(guardrail.get("litellm_params", {}))
            guardrail_info: str = safe_dumps(guardrail.get("guardrail_info", {}))

            # Create guardrail in DB
            created_guardrail = await prisma_client.db.litellm_guardrailstable.create(
                data={
                    "guardrail_name": guardrail_name,
                    "litellm_params": litellm_params,
                    "guardrail_info": guardrail_info,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            # Add guardrail_id to the returned guardrail object
            guardrail_dict = dict(guardrail)
            guardrail_dict["guardrail_id"] = created_guardrail.guardrail_id

            return guardrail_dict
        except Exception as e:
            raise Exception(f"Error adding guardrail to DB: {str(e)}")

    async def delete_guardrail_from_db(
        self, guardrail_id: str, prisma_client: PrismaClient
    ):
        """
        Delete a guardrail from the database
        """
        try:
            # Delete from DB
            await prisma_client.db.litellm_guardrailstable.delete(
                where={"guardrail_id": guardrail_id}
            )

            return {"message": f"Guardrail {guardrail_id} deleted successfully"}
        except Exception as e:
            raise Exception(f"Error deleting guardrail from DB: {str(e)}")

    async def update_guardrail_in_db(
        self, guardrail_id: str, guardrail: Guardrail, prisma_client: PrismaClient
    ):
        """
        Update a guardrail in the database
        """
        try:
            guardrail_name = guardrail.get("guardrail_name")
            litellm_params = guardrail.get("litellm_params", {})
            guardrail_info = guardrail.get("guardrail_info", {})

            # Update in DB
            updated_guardrail = await prisma_client.db.litellm_guardrailstable.update(
                where={"guardrail_id": guardrail_id},
                data={
                    "guardrail_name": guardrail_name,
                    "litellm_params": litellm_params,
                    "guardrail_info": guardrail_info,
                    "updated_at": datetime.now(timezone.utc),
                },
            )

            # Convert to dict and return
            return dict(updated_guardrail)
        except Exception as e:
            raise Exception(f"Error updating guardrail in DB: {str(e)}")

    @staticmethod
    async def get_all_guardrails_from_db(
        prisma_client: PrismaClient,
    ) -> List[Guardrail]:
        """
        Get all guardrails from the database
        """
        try:
            guardrails_from_db = (
                await prisma_client.db.litellm_guardrailstable.find_many(
                    order={"created_at": "desc"},
                )
            )

            guardrails: List[Guardrail] = []
            for guardrail in guardrails_from_db:
                guardrails.append(Guardrail(**(dict(guardrail))))

            return guardrails
        except Exception as e:
            raise Exception(f"Error getting guardrails from DB: {str(e)}")

    async def get_guardrail_by_id_from_db(
        self, guardrail_id: str, prisma_client: PrismaClient
    ) -> Optional[Guardrail]:
        """
        Get a guardrail by its ID from the database
        """
        try:
            guardrail = await prisma_client.db.litellm_guardrailstable.find_unique(
                where={"guardrail_id": guardrail_id}
            )

            if not guardrail:
                return None

            return Guardrail(**(dict(guardrail)))
        except Exception as e:
            raise Exception(f"Error getting guardrail from DB: {str(e)}")

    async def get_guardrail_by_name_from_db(
        self, guardrail_name: str, prisma_client: PrismaClient
    ) -> Optional[Guardrail]:
        """
        Get a guardrail by its name from the database
        """
        try:
            guardrail = await prisma_client.db.litellm_guardrailstable.find_unique(
                where={"guardrail_name": guardrail_name}
            )

            if not guardrail:
                return None

            return Guardrail(**(dict(guardrail)))
        except Exception as e:
            raise Exception(f"Error getting guardrail from DB: {str(e)}")
