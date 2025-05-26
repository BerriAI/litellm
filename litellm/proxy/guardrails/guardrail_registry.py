# litellm/proxy/guardrails/guardrail_registry.py

import importlib
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient
from litellm.secret_managers.main import get_secret
from litellm.types.guardrails import (
    Guardrail,
    GuardrailEventHooks,
    LakeraCategoryThresholds,
    LitellmParams,
    SupportedGuardrailIntegrations,
)

from .guardrail_initializers import (
    initialize_aim,
    initialize_aporia,
    initialize_bedrock,
    initialize_guardrails_ai,
    initialize_hide_secrets,
    initialize_lakera,
    initialize_lakera_v2,
    initialize_presidio,
)

guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.APORIA.value: initialize_aporia,
    SupportedGuardrailIntegrations.BEDROCK.value: initialize_bedrock,
    SupportedGuardrailIntegrations.LAKERA.value: initialize_lakera,
    SupportedGuardrailIntegrations.LAKERA_V2.value: initialize_lakera_v2,
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
    ########### In memory management helpers for guardrails ###########
    ############################################################
    def get_initialized_guardrail_callback(
        self, guardrail_name: str
    ) -> Optional[CustomGuardrail]:
        """
        Returns the initialized guardrail callback for a given guardrail name
        """
        active_guardrails = (
            litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=CustomGuardrail
            )
        )
        for active_guardrail in active_guardrails:
            if isinstance(active_guardrail, CustomGuardrail):
                if active_guardrail.guardrail_name == guardrail_name:
                    return active_guardrail
        return None

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
            litellm_params: str = safe_dumps(dict(guardrail.get("litellm_params", {})))
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
            litellm_params: str = safe_dumps(dict(guardrail.get("litellm_params", {})))
            guardrail_info: str = safe_dumps(guardrail.get("guardrail_info", {}))

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


class InMemoryGuardrailHandler:
    """
    Class that handles initializing guardrails and adding them to the CallbackManager
    """

    def __init__(self):
        self.IN_MEMORY_GUARDRAILS: Dict[str, Guardrail] = {}
        """
        Guardrail id to Guardrail object mapping
        """

        self.guardrail_id_to_custom_guardrail: Dict[str, Optional[CustomGuardrail]] = {}
        """
        Guardrail id to CustomGuardrail object mapping
        """

    def initialize_guardrail(
        self,
        guardrail: Dict,
        config_file_path: Optional[str] = None,
    ) -> Optional[Guardrail]:
        """
        Initialize a guardrail from a dictionary and add it to the litellm callback manager

        Returns a Guardrail object if the guardrail is initialized successfully
        """
        guardrail_id = guardrail.get("guardrail_id") or str(uuid.uuid4())
        guardrail["guardrail_id"] = guardrail_id
        if guardrail_id in self.IN_MEMORY_GUARDRAILS:
            verbose_proxy_logger.debug(
                "guardrail_id already exists in IN_MEMORY_GUARDRAILS"
            )
            return self.IN_MEMORY_GUARDRAILS[guardrail_id]

        custom_guardrail_callback: Optional[CustomGuardrail] = None
        litellm_params_data = guardrail["litellm_params"]
        verbose_proxy_logger.debug("litellm_params= %s", litellm_params_data)

        litellm_params = LitellmParams(**litellm_params_data)

        if (
            "category_thresholds" in litellm_params_data
            and litellm_params_data["category_thresholds"]
        ):
            lakera_category_thresholds = LakeraCategoryThresholds(
                **litellm_params_data["category_thresholds"]
            )
            litellm_params.category_thresholds = lakera_category_thresholds

        if litellm_params.api_key and litellm_params.api_key.startswith("os.environ/"):
            litellm_params.api_key = str(get_secret(litellm_params.api_key))

        if litellm_params.api_base and litellm_params.api_base.startswith(
            "os.environ/"
        ):
            litellm_params.api_base = str(get_secret(litellm_params.api_base))

        guardrail_type = litellm_params.guardrail
        if guardrail_type is None:
            raise ValueError("guardrail_type is required")

        initializer = guardrail_initializer_registry.get(guardrail_type)

        if initializer:
            custom_guardrail_callback = initializer(litellm_params, guardrail)
        elif isinstance(guardrail_type, str) and "." in guardrail_type:
            custom_guardrail_callback = self.initialize_custom_guardrail(
                guardrail=guardrail,
                guardrail_type=guardrail_type,
                litellm_params=litellm_params,
                config_file_path=config_file_path,
            )
        else:
            raise ValueError(f"Unsupported guardrail: {guardrail_type}")

        parsed_guardrail = Guardrail(
            guardrail_id=guardrail.get("guardrail_id"),
            guardrail_name=guardrail["guardrail_name"],
            litellm_params=litellm_params,
        )

        # store references to the guardrail in memory
        self.IN_MEMORY_GUARDRAILS[guardrail_id] = parsed_guardrail
        self.guardrail_id_to_custom_guardrail[guardrail_id] = custom_guardrail_callback

        return parsed_guardrail

    def initialize_custom_guardrail(
        self,
        guardrail: Dict,
        guardrail_type: str,
        litellm_params: LitellmParams,
        config_file_path: Optional[str] = None,
    ) -> Optional[CustomGuardrail]:
        """
        Initialize a Custom Guardrail from a python file

        This initializes it by adding it to the litellm callback manager
        """
        if not config_file_path:
            raise Exception(
                "GuardrailsAIException - Please pass the config_file_path to initialize_guardrails_v2"
            )

        _file_name, _class_name = guardrail_type.split(".")
        verbose_proxy_logger.debug(
            "Initializing custom guardrail: %s, file_name: %s, class_name: %s",
            guardrail_type,
            _file_name,
            _class_name,
        )

        directory = os.path.dirname(config_file_path)
        module_file_path = os.path.join(directory, _file_name) + ".py"

        spec = importlib.util.spec_from_file_location(_class_name, module_file_path)  # type: ignore
        if not spec:
            raise ImportError(
                f"Could not find a module specification for {module_file_path}"
            )

        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)  # type: ignore
        _guardrail_class = getattr(module, _class_name)

        mode = litellm_params.mode
        if mode is None:
            raise ValueError(
                f"mode is required for guardrail {guardrail_type} please set mode to one of the following: {', '.join(GuardrailEventHooks)}"
            )

        default_on = litellm_params.default_on
        _guardrail_callback = _guardrail_class(
            guardrail_name=guardrail["guardrail_name"],
            event_hook=mode,
            default_on=default_on,
        )
        litellm.logging_callback_manager.add_litellm_callback(_guardrail_callback)  # type: ignore

        return _guardrail_callback

    def update_in_memory_guardrail(
        self, guardrail_id: str, guardrail: Guardrail
    ) -> None:
        """
        Update a guardrail in memory

        - updates the guardrail in memory
        - updates the guardrail params in litellm.callback_manager
        """
        self.IN_MEMORY_GUARDRAILS[guardrail_id] = guardrail

        custom_guardrail_callback = self.guardrail_id_to_custom_guardrail.get(
            guardrail_id
        )
        if custom_guardrail_callback:
            updated_litellm_params = cast(
                LitellmParams, guardrail.get("litellm_params", {})
            )
            custom_guardrail_callback.update_in_memory_litellm_params(
                litellm_params=updated_litellm_params
            )

    def delete_in_memory_guardrail(self, guardrail_id: str) -> None:
        """
        Delete a guardrail in memory
        """
        self.IN_MEMORY_GUARDRAILS.pop(guardrail_id, None)

    def list_in_memory_guardrails(self) -> List[Guardrail]:
        """
        List all guardrails in memory
        """
        return list(self.IN_MEMORY_GUARDRAILS.values())

    def get_guardrail_by_id(self, guardrail_id: str) -> Optional[Guardrail]:
        """
        Get a guardrail by its ID from memory
        """
        return self.IN_MEMORY_GUARDRAILS.get(guardrail_id)


########################################################
# In Memory Guardrail Handler for LiteLLM Proxy
########################################################
IN_MEMORY_GUARDRAIL_HANDLER = InMemoryGuardrailHandler()
########################################################
