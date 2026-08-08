# litellm/proxy/guardrails/guardrail_registry.py

import importlib
import os
from datetime import datetime, timezone
from itertools import chain, count
from typing import Any, Final, Literal, Optional, cast

from pydantic import ValidationError

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.grayswan import (
    GraySwanGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.grayswan import (
    initialize_guardrail as initialize_grayswan,
)
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import (
    ToolPermissionGuardrail,
)
from litellm.proxy.types_utils.utils import get_instance_fn
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import GuardrailsRepository
from litellm.secret_managers.main import get_secret
from litellm.types.guardrails import (
    Guardrail,
    GuardrailEventHooks,
    LakeraCategoryThresholds,
    LitellmParams,
    SupportedGuardrailIntegrations,
)

from .guardrail_hooks.llm_as_a_judge import (
    initialize_guardrail as initialize_llm_as_a_judge,
)
from .guardrail_initializers import (
    initialize_bedrock,
    initialize_hide_secrets,
    initialize_lakera,
    initialize_lakera_v2,
    initialize_presidio,
    initialize_tool_permission,
)

guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.BEDROCK.value: initialize_bedrock,
    SupportedGuardrailIntegrations.LAKERA.value: initialize_lakera,
    SupportedGuardrailIntegrations.LAKERA_V2.value: initialize_lakera_v2,
    SupportedGuardrailIntegrations.PRESIDIO.value: initialize_presidio,
    SupportedGuardrailIntegrations.HIDE_SECRETS.value: initialize_hide_secrets,
    SupportedGuardrailIntegrations.TOOL_PERMISSION.value: initialize_tool_permission,
    SupportedGuardrailIntegrations.GRAYSWAN.value: initialize_grayswan,
    SupportedGuardrailIntegrations.LLM_AS_A_JUDGE.value: initialize_llm_as_a_judge,
}

CONFIG_GUARDRAIL_ID_NAMESPACE: Final = uuid.UUID("625f63f4-935a-50e5-98b5-fbe77babc74a")

guardrail_class_registry: Final[dict[str, type[CustomGuardrail]]] = {
    SupportedGuardrailIntegrations.BEDROCK.value: BedrockGuardrail,
    SupportedGuardrailIntegrations.GRAYSWAN.value: GraySwanGuardrail,
    SupportedGuardrailIntegrations.LAKERA.value: lakeraAI_Moderation,
    SupportedGuardrailIntegrations.LAKERA_V2.value: LakeraAIGuardrail,
    SupportedGuardrailIntegrations.PRESIDIO.value: _OPTIONAL_PresidioPIIMasking,
    SupportedGuardrailIntegrations.TOOL_PERMISSION.value: ToolPermissionGuardrail,
}


def get_guardrail_initializer_from_hooks():
    """
    Get guardrail initializers by discovering them from the guardrail_hooks directory structure.

    Scans the guardrail_hooks directory for subdirectories containing __init__.py files
    with either guardrail_initializer_registry or initialize_guardrail functions.

    Returns:
        Dict[str, Callable]: A dictionary mapping guardrail types to their initializer functions
    """
    discovered_initializers: Final = {}

    try:
        # Get the path to the guardrail_hooks directory
        current_dir: Final = os.path.dirname(__file__)
        hooks_dir: Final = os.path.join(current_dir, "guardrail_hooks")

        if not os.path.exists(hooks_dir):
            verbose_proxy_logger.debug("guardrail_hooks directory not found")
            return discovered_initializers

        # Scan each subdirectory in guardrail_hooks
        for item in os.listdir(hooks_dir):
            item_path = os.path.join(hooks_dir, item)

            # Skip files and __pycache__ directories
            if not os.path.isdir(item_path) or item.startswith("__"):
                continue

            # Check if the directory has an __init__.py file
            init_file = os.path.join(item_path, "__init__.py")
            if not os.path.exists(init_file):
                continue

            module_path = f"litellm.proxy.guardrails.guardrail_hooks.{item}"
            try:
                # Import the module
                verbose_proxy_logger.debug("Discovering guardrails in: %s", module_path)

                module = importlib.import_module(module_path)

                # Check for guardrail_initializer_registry dictionary
                if hasattr(module, "guardrail_initializer_registry"):
                    registry = getattr(module, "guardrail_initializer_registry")
                    if isinstance(registry, dict):
                        discovered_initializers.update(registry)
                        verbose_proxy_logger.debug(
                            "Found guardrail_initializer_registry in %s: %s", module_path, list(registry.keys())
                        )

                # Check for standalone initialize_guardrail function (fallback for directory-based guardrails)
                elif hasattr(module, "initialize_guardrail"):
                    # For directories with just initialize_guardrail, use the directory name as the key
                    initialize_fn = getattr(module, "initialize_guardrail")
                    discovered_initializers[item] = initialize_fn
                    verbose_proxy_logger.debug("Found initialize_guardrail function in %s", module_path)

            except ImportError as e:
                verbose_proxy_logger.error("Could not import %s: %s", module_path, e)
                continue
            except Exception as e:
                verbose_proxy_logger.error("Error processing %s: %s", module_path, e)
                continue

        verbose_proxy_logger.debug(
            "Discovered %s guardrail initializers: %s",
            len(discovered_initializers),
            list(discovered_initializers.keys()),
        )

    except Exception as e:
        verbose_proxy_logger.error("Error discovering guardrail initializers: %s", e)

    return discovered_initializers


def get_guardrail_class_from_hooks():
    """
    Get guardrail classes by discovering them from the guardrail_hooks directory structure.
    """
    """
    Get guardrail initializers by discovering them from the guardrail_hooks directory structure.

    Scans the guardrail_hooks directory for subdirectories containing __init__.py files
    with either guardrail_initializer_registry or initialize_guardrail functions.

    Returns:
        Dict[str, Callable]: A dictionary mapping guardrail types to their initializer functions
    """
    discovered_classes: Final = {}

    try:
        # Get the path to the guardrail_hooks directory
        current_dir: Final = os.path.dirname(__file__)
        hooks_dir: Final = os.path.join(current_dir, "guardrail_hooks")

        if not os.path.exists(hooks_dir):
            verbose_proxy_logger.debug("guardrail_hooks directory not found")
            return discovered_classes

        # Scan each subdirectory in guardrail_hooks
        for item in os.listdir(hooks_dir):
            item_path = os.path.join(hooks_dir, item)

            # Skip files and __pycache__ directories
            if not os.path.isdir(item_path) or item.startswith("__"):
                continue

            # Check if the directory has an __init__.py file
            init_file = os.path.join(item_path, "__init__.py")

            if not os.path.exists(init_file):
                continue

            module_path = f"litellm.proxy.guardrails.guardrail_hooks.{item}"

            try:
                # Import the module
                verbose_proxy_logger.debug("Discovering guardrails in: %s", module_path)

                module = importlib.import_module(module_path)

                # Check for guardrail_initializer_registry dictionary
                if hasattr(module, "guardrail_class_registry"):
                    registry = getattr(module, "guardrail_class_registry")
                    if isinstance(registry, dict):
                        discovered_classes.update(registry)

            except ImportError as e:
                verbose_proxy_logger.debug("Could not import %s: %s", module_path, e)
                continue
            except Exception as e:
                verbose_proxy_logger.exception("Error processing %s: %s", module_path, e)
                continue

    except Exception as e:
        verbose_proxy_logger.error("Error discovering guardrail initializers: %s", e)

    return discovered_classes


guardrail_class_registry.update(get_guardrail_class_from_hooks())


# Merge with dynamically discovered guardrail initializers
_discovered_initializers: Final = get_guardrail_initializer_from_hooks()

guardrail_initializer_registry.update(_discovered_initializers)


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
    def get_initialized_guardrail_callback(self, guardrail_name: str) -> CustomGuardrail | None:
        """
        Returns the initialized guardrail callback for a given guardrail name
        """
        active_guardrails = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomGuardrail)
        for active_guardrail in active_guardrails:
            if isinstance(active_guardrail, CustomGuardrail):
                if active_guardrail.guardrail_name == guardrail_name:
                    return active_guardrail
        return None

    ###########################################################
    ########### DB management helpers for guardrails ###########
    ############################################################
    async def add_guardrail_to_db(self, guardrail: Guardrail, prisma_client: PrismaClient):
        """
        Add a guardrail to the database
        """
        try:
            guardrail_name: Final = guardrail.get("guardrail_name")
            # Properly serialize LitellmParams Pydantic model to dict
            litellm_params_obj: Final[Any] = guardrail.get("litellm_params", {})
            if hasattr(litellm_params_obj, "model_dump"):
                litellm_params_dict = litellm_params_obj.model_dump()
            else:
                litellm_params_dict = dict(litellm_params_obj) if litellm_params_obj else {}
            litellm_params: Final[str] = safe_dumps(litellm_params_dict)
            guardrail_info: Final[str] = safe_dumps(guardrail.get("guardrail_info", {}))

            # Create guardrail in DB
            created_guardrail: Final = await GuardrailsRepository(prisma_client).table.create(
                data={
                    "guardrail_name": guardrail_name,
                    "litellm_params": litellm_params,
                    "guardrail_info": guardrail_info,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            # Add guardrail_id to the returned guardrail object
            guardrail_dict: Final = dict(guardrail)
            guardrail_dict["guardrail_id"] = created_guardrail.guardrail_id

            return guardrail_dict
        except Exception as e:
            raise Exception(f"Error adding guardrail to DB: {e}")

    async def delete_guardrail_from_db(self, guardrail_id: str, prisma_client: PrismaClient):
        """
        Delete a guardrail from the database
        """
        try:
            # Delete from DB
            await GuardrailsRepository(prisma_client).table.delete(where={"guardrail_id": guardrail_id})

            return {"message": f"Guardrail {guardrail_id} deleted successfully"}
        except Exception as e:
            raise Exception(f"Error deleting guardrail from DB: {e}")

    async def update_guardrail_in_db(self, guardrail_id: str, guardrail: Guardrail, prisma_client: PrismaClient):
        """
        Update a guardrail in the database
        """
        try:
            guardrail_name: Final = guardrail.get("guardrail_name")
            # Properly serialize LitellmParams Pydantic model to dict
            litellm_params_obj: Final[Any] = guardrail.get("litellm_params", {})
            if hasattr(litellm_params_obj, "model_dump"):
                litellm_params_dict = litellm_params_obj.model_dump()
            else:
                litellm_params_dict = dict(litellm_params_obj) if litellm_params_obj else {}
            litellm_params: Final[str] = safe_dumps(litellm_params_dict)
            guardrail_info: Final[str] = safe_dumps(guardrail.get("guardrail_info", {}))

            # Update in DB
            updated_guardrail: Final = await GuardrailsRepository(prisma_client).table.update(
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
            raise Exception(f"Error updating guardrail in DB: {e}")

    @staticmethod
    async def get_all_guardrails_from_db(
        prisma_client: PrismaClient,
    ) -> list[Guardrail]:
        """
        Get all active guardrails from the database.
        Only rows with status == "active" are returned (pending_review and rejected are excluded).
        """
        try:
            guardrails_from_db: Final = await GuardrailsRepository(prisma_client).table.find_many(
                where={"status": "active"},
                order={"created_at": "desc"},
            )

            guardrails: Final[list[Guardrail]] = []
            for guardrail in guardrails_from_db:
                guardrails.append(Guardrail(**(dict(guardrail))))

            return guardrails
        except Exception as e:
            raise Exception(f"Error getting guardrails from DB: {e}")

    async def get_guardrail_by_id_from_db(self, guardrail_id: str, prisma_client: PrismaClient) -> Guardrail | None:
        """
        Get a guardrail by its ID from the database
        """
        try:
            guardrail: Final = await GuardrailsRepository(prisma_client).table.find_unique(
                where={"guardrail_id": guardrail_id}
            )

            if not guardrail:
                return None

            return Guardrail(**(dict(guardrail)))
        except Exception as e:
            raise Exception(f"Error getting guardrail from DB: {e}")

    async def get_guardrail_by_name_from_db(self, guardrail_name: str, prisma_client: PrismaClient) -> Guardrail | None:
        """
        Get a guardrail by its name from the database
        """
        try:
            guardrail: Final = await GuardrailsRepository(prisma_client).table.find_unique(
                where={"guardrail_name": guardrail_name}
            )

            if not guardrail:
                return None

            return Guardrail(**(dict(guardrail)))
        except Exception as e:
            raise Exception(f"Error getting guardrail from DB: {e}")


class InMemoryGuardrailHandler:
    """
    Class that handles initializing guardrails and adding them to the CallbackManager
    """

    def __init__(self):
        self.IN_MEMORY_GUARDRAILS: dict[str, Guardrail] = {}
        """
        Guardrail id to Guardrail object mapping
        """

        self.guardrail_id_to_custom_guardrail: dict[str, CustomGuardrail | None] = {}
        """
        Guardrail id to CustomGuardrail object mapping
        """

        self._sources: dict[str, Literal["db", "config"]] = {}
        """
        Guardrail id to provenance marker. "db" entries are reconciled against
        the DB on each polling tick; "config" entries are owned by proxy_config.yaml
        and never deleted by reconciliation.
        """

    def _stable_guardrail_id(self, guardrail_name: str) -> str:
        seeds: Final = chain((guardrail_name,), (f"{guardrail_name}:{occurrence}" for occurrence in count(1)))
        candidate_ids: Final = (str(uuid.uuid5(CONFIG_GUARDRAIL_ID_NAMESPACE, seed.encode("utf-8"))) for seed in seeds)
        return next(candidate_id for candidate_id in candidate_ids if candidate_id not in self.IN_MEMORY_GUARDRAILS)

    def initialize_guardrail(
        self,
        guardrail: Guardrail,
        config_file_path: str | None = None,
        llm_router: Optional["Router"] = None,
        source: Literal["db", "config"] = "config",
    ) -> Guardrail | None:
        """
        Initialize a guardrail from a dictionary and add it to the litellm callback manager

        Returns a Guardrail object if the guardrail is initialized successfully
        """
        guardrail_id: Final = guardrail.get("guardrail_id") or self._stable_guardrail_id(guardrail["guardrail_name"])
        guardrail["guardrail_id"] = guardrail_id
        if guardrail_id in self.IN_MEMORY_GUARDRAILS:
            verbose_proxy_logger.debug("guardrail_id already exists in IN_MEMORY_GUARDRAILS")
            # Honor the caller's source even on the early-return path so a
            # racing polling tick or a hot-reload of config can correct an
            # entry's provenance.
            self._sources[guardrail_id] = source
            return self.IN_MEMORY_GUARDRAILS[guardrail_id]

        custom_guardrail_callback: CustomGuardrail | None = None
        litellm_params_data: Final = guardrail["litellm_params"]
        verbose_proxy_logger.debug("litellm_params= %s", litellm_params_data)

        if isinstance(litellm_params_data, dict):
            litellm_params = LitellmParams(**litellm_params_data)
        else:
            litellm_params = litellm_params_data

        if "category_thresholds" in litellm_params_data and litellm_params_data["category_thresholds"]:
            lakera_category_thresholds: Final = LakeraCategoryThresholds(**litellm_params_data["category_thresholds"])
            litellm_params.category_thresholds = lakera_category_thresholds

        if litellm_params.api_key and litellm_params.api_key.startswith("os.environ/"):
            litellm_params.api_key = str(get_secret(litellm_params.api_key))

        if litellm_params.api_base and litellm_params.api_base.startswith("os.environ/"):
            litellm_params.api_base = str(get_secret(litellm_params.api_base))

        guardrail_type: Final = litellm_params.guardrail

        if guardrail_type is None:
            raise ValueError("guardrail_type is required")

        initializer: Final = guardrail_initializer_registry.get(guardrail_type)

        if initializer:
            # Try to call with llm_router first, fall back to without if it fails
            import inspect

            sig: Final = inspect.signature(initializer)
            if "llm_router" in sig.parameters:
                custom_guardrail_callback = initializer(
                    litellm_params,
                    guardrail,
                    llm_router,
                )
            else:
                custom_guardrail_callback = initializer(litellm_params, guardrail)
        elif isinstance(guardrail_type, str) and "." in guardrail_type:
            custom_guardrail_callback = self.initialize_custom_guardrail(
                guardrail=cast(dict, guardrail),
                guardrail_type=guardrail_type,
                litellm_params=litellm_params,
                config_file_path=config_file_path,
            )
        else:
            raise ValueError(f"Unsupported guardrail: {guardrail_type}")

        if custom_guardrail_callback is not None:
            for scoping_param in (
                "skip_system_message_in_guardrail",
                "skip_tool_message_in_guardrail",
                "scan_only_tool_results",
            ):
                setattr(custom_guardrail_callback, scoping_param, getattr(litellm_params, scoping_param, None))
            scan_only_tool_results_enabled: Final = effective_scan_only_tool_results_for_guardrail(
                custom_guardrail_callback
            )
            if scan_only_tool_results_enabled and not custom_guardrail_callback.supports_scan_only_tool_results():
                raise ValueError(
                    f"Guardrail {guardrail['guardrail_name']}: scan_only_tool_results is enabled, but this "
                    "guardrail's role filtering never scans tool results, so no request content would ever "
                    "be scanned. Remove scan_only_tool_results or the guardrail's role-filtering option."
                )
            if scan_only_tool_results_enabled and effective_skip_tool_message_for_guardrail(custom_guardrail_callback):
                raise ValueError(
                    f"Guardrail {guardrail['guardrail_name']}: scan_only_tool_results and "
                    "skip_tool_message_in_guardrail are enabled together, which excludes every message from "
                    "scanning, so no request content would ever be scanned. Remove one of the two."
                )
            configured_run_in_parallel: Final = getattr(litellm_params, "run_in_parallel", None)
            if configured_run_in_parallel is not None:
                custom_guardrail_callback.run_in_parallel = bool(configured_run_in_parallel)

        parsed_guardrail: Final = Guardrail(
            guardrail_id=guardrail.get("guardrail_id"),
            guardrail_name=guardrail["guardrail_name"],
            litellm_params=litellm_params,
            guardrail_info=guardrail.get("guardrail_info"),
        )

        # store references to the guardrail in memory
        self.IN_MEMORY_GUARDRAILS[guardrail_id] = parsed_guardrail
        self.guardrail_id_to_custom_guardrail[guardrail_id] = custom_guardrail_callback
        self._sources[guardrail_id] = source

        return parsed_guardrail

    def initialize_custom_guardrail(
        self,
        guardrail: dict,
        guardrail_type: str,
        litellm_params: LitellmParams,
        config_file_path: str | None = None,
    ) -> CustomGuardrail | None:
        """
        Initialize a Custom Guardrail from a python file or module path

        This initializes it by adding it to the litellm callback manager
        """
        if not config_file_path:
            raise Exception("GuardrailsAIException - Please pass the config_file_path to initialize_guardrails_v2")

        verbose_proxy_logger.debug(
            "Initializing custom guardrail: %s",
            guardrail_type,
        )

        _guardrail_class: Final = get_instance_fn(guardrail_type, config_file_path=config_file_path)

        mode: Final = litellm_params.mode
        if mode is None:
            raise ValueError(
                f"mode is required for guardrail {guardrail_type} please set mode to one of the following: {', '.join(GuardrailEventHooks)}"
            )

        default_on: Final = litellm_params.default_on

        # Extract additional params from litellm_params to pass to custom guardrail
        # This matches the behavior of other guardrail initializers (e.g., initialize_lakera)
        # and aligns with the documented behavior for custom guardrails
        if hasattr(litellm_params, "model_dump"):
            extra_params = litellm_params.model_dump(exclude_none=True)
        else:
            extra_params = dict(litellm_params) if litellm_params else {}

        # Remove params that are handled explicitly or are internal
        for key in ["guardrail", "mode", "default_on"]:
            extra_params.pop(key, None)

        _guardrail_callback: Final = _guardrail_class(
            guardrail_name=guardrail["guardrail_name"],
            event_hook=mode,
            default_on=default_on,
            **extra_params,
        )
        litellm.logging_callback_manager.add_litellm_callback(_guardrail_callback)

        return _guardrail_callback

    def update_in_memory_guardrail(
        self,
        guardrail_id: str,
        guardrail: Guardrail,
        source: Literal["db", "config"] = "db",
    ) -> None:
        """
        Update a guardrail in memory

        - updates the guardrail in memory
        - updates the guardrail params in litellm.callback_manager
        """
        self.IN_MEMORY_GUARDRAILS[guardrail_id] = guardrail
        self._sources[guardrail_id] = source

        custom_guardrail_callback: Final = self.guardrail_id_to_custom_guardrail.get(guardrail_id)
        if custom_guardrail_callback:
            updated_litellm_params: Final = cast(LitellmParams, guardrail.get("litellm_params", {}))
            custom_guardrail_callback.update_in_memory_litellm_params(litellm_params=updated_litellm_params)

    def delete_in_memory_guardrail(self, guardrail_id: str) -> None:
        """
        Delete a guardrail in memory and remove from litellm callbacks.

        The callback is purged from every callback list, not just
        litellm.callbacks: request handling promotes guardrail callbacks into the
        success/failure/async lists, so removing it from only litellm.callbacks
        leaves the old instance stranded in those lists on every re-initialization.
        """
        # Remove from in-memory storage
        self.IN_MEMORY_GUARDRAILS.pop(guardrail_id, None)
        self._sources.pop(guardrail_id, None)

        custom_guardrail_callback: Final = self.guardrail_id_to_custom_guardrail.pop(guardrail_id, None)
        if custom_guardrail_callback is None:
            return

        litellm.logging_callback_manager.remove_callback_from_all_lists(custom_guardrail_callback)

    def list_in_memory_guardrails(self) -> list[Guardrail]:
        """
        List all guardrails in memory
        """
        return list(self.IN_MEMORY_GUARDRAILS.values())

    def get_guardrail_by_id(self, guardrail_id: str) -> Guardrail | None:
        """
        Get a guardrail by its ID from memory
        """
        return self.IN_MEMORY_GUARDRAILS.get(guardrail_id)

    def get_source(self, guardrail_id: str) -> Literal["db", "config"] | None:
        """
        Return the provenance of an in-memory guardrail.
        """
        return self._sources.get(guardrail_id)

    def list_config_guardrails(self) -> list[Guardrail]:
        """
        List in-memory guardrails owned by config.yaml.

        DB-sourced entries are excluded: a read surface that also queries the DB
        would double-count live ones, and a DB-sourced entry that's missing from
        the DB is stale (deleted on another pod, awaiting reconciliation here).
        """
        return [g for gid, g in self.IN_MEMORY_GUARDRAILS.items() if self._sources.get(gid) == "config"]

    def get_config_guardrail_by_id(self, guardrail_id: str) -> Guardrail | None:
        """
        Get a config-owned in-memory guardrail by its ID, or None.

        Mirrors the fallback in get_guardrail_info: a DB-sourced in-memory entry
        that missed the DB lookup is stale and must not be surfaced.
        """
        if self._sources.get(guardrail_id) != "config":
            return None
        return self.IN_MEMORY_GUARDRAILS.get(guardrail_id)

    def reconcile_db_guardrails(self, db_guardrail_ids: set[str]) -> list[str]:
        """
        Drop in-memory entries that originated from the DB but are no longer
        present in db_guardrail_ids. Config-loaded guardrails are never touched.

        Called by the periodic DB polling tick so that a guardrail deleted
        on another pod is eventually purged from this pod's memory + callbacks.
        """
        stale_ids: Final = [
            guardrail_id
            for guardrail_id, source in self._sources.items()
            if source == "db" and guardrail_id not in db_guardrail_ids
        ]
        for guardrail_id in stale_ids:
            verbose_proxy_logger.info(
                "Reconcile: removing stale DB-backed guardrail '%s' from memory (deleted in DB by another pod)",
                guardrail_id,
            )
            self.delete_in_memory_guardrail(guardrail_id)
        return stale_ids

    @staticmethod
    def _normalize_litellm_params_for_comparison(
        params: Any | None,
    ) -> dict[str, Any] | None:
        """
        Render litellm_params to a canonical dict so an in-memory LitellmParams and
        the raw dict loaded from the DB compare equal when they describe the same
        config. The in-memory side is a LitellmParams whose model_dump() carries
        every field default and coerces enums, while the DB side is the raw stored
        dict holding only the keys originally provided. Comparing those two shapes
        directly never matches, so each DB poll would re-initialize the guardrail
        forever; normalizing both through LitellmParams keeps the diff meaningful.
        """
        if params is None:
            return None
        if isinstance(params, LitellmParams):
            return params.model_dump()
        if isinstance(params, dict):
            try:
                return LitellmParams(**params).model_dump()
            except ValidationError as e:
                verbose_proxy_logger.warning(
                    "Could not normalize guardrail litellm_params for comparison; treating the guardrail as changed. Error: %s",
                    e,
                )
                return params
        return params

    def _has_guardrail_params_changed(self, guardrail_id: str, new_guardrail: Guardrail) -> bool:
        """
        Check if guardrail params or name have changed compared to in-memory version.
        Returns True if params/name changed or guardrail doesn't exist in memory.
        """
        existing: Final = self.IN_MEMORY_GUARDRAILS.get(guardrail_id)
        if existing is None:
            return True

        # Compare guardrail_name
        if existing.get("guardrail_name") != new_guardrail.get("guardrail_name"):
            return True

        # Compare litellm_params
        existing_dict: Final = self._normalize_litellm_params_for_comparison(existing.get("litellm_params"))
        new_dict: Final = self._normalize_litellm_params_for_comparison(new_guardrail.get("litellm_params"))

        # Compare and identify specific differences
        changed_fields = {}
        if existing_dict is not None and new_dict is not None:
            all_keys: Final = set(existing_dict.keys()) | set(new_dict.keys())
            for key in all_keys:
                old_val = existing_dict.get(key)
                new_val = new_dict.get(key)
                if old_val != new_val:
                    changed_fields[key] = {"old": old_val, "new": new_val}
        elif existing_dict != new_dict:
            changed_fields = {"litellm_params": {"old": existing_dict, "new": new_dict}}

        # Log differences if any found
        if changed_fields:
            verbose_proxy_logger.debug("Guardrail params changed. Differences: %s", changed_fields)

        # Return True if any fields changed
        return len(changed_fields) > 0

    def reinitialize_guardrail(
        self,
        guardrail: Guardrail,
        config_file_path: str | None = None,
        source: Literal["db", "config"] = "config",
    ) -> Guardrail | None:
        """
        Force re-initialization of a guardrail even if it exists in memory.
        Removes old callback from litellm.callbacks and creates fresh instance.
        """
        guardrail_id: Final = guardrail.get("guardrail_id")
        if not guardrail_id:
            verbose_proxy_logger.error("Cannot reinitialize guardrail without guardrail_id")
            return None

        # Remove from memory if exists (also removes from callbacks)
        if guardrail_id in self.IN_MEMORY_GUARDRAILS:
            self.delete_in_memory_guardrail(guardrail_id)

        # Initialize fresh (will add new callback to litellm.callbacks)
        return self.initialize_guardrail(guardrail=guardrail, config_file_path=config_file_path, source=source)

    def sync_guardrail_from_db(self, guardrail: Guardrail, config_file_path: str | None = None) -> Guardrail | None:
        """
        Sync a guardrail from DB - initializes if new, re-initializes if changed.
        This is the method to call during DB polling.
        """
        guardrail_id: Final = guardrail.get("guardrail_id")
        if not guardrail_id:
            verbose_proxy_logger.error("Cannot sync guardrail without guardrail_id")
            return None

        if self._has_guardrail_params_changed(guardrail_id, guardrail):
            guardrail_name: Final = guardrail.get("guardrail_name", "Unknown")
            verbose_proxy_logger.info(
                "Guardrail '%s' (ID: %s) params changed, re-initializing...", guardrail_name, guardrail_id
            )
            return self.reinitialize_guardrail(
                guardrail=guardrail,
                config_file_path=config_file_path,
                source="db",
            )

        # Params unchanged but the entry is still DB-backed; make sure the
        # source marker reflects that even if it was previously set differently
        # (e.g. a config entry whose UUID later collided with a DB row).
        self._sources[guardrail_id] = "db"
        return self.IN_MEMORY_GUARDRAILS.get(guardrail_id)


########################################################
# In Memory Guardrail Handler for LiteLLM Proxy
########################################################
IN_MEMORY_GUARDRAIL_HANDLER: Final = InMemoryGuardrailHandler()
########################################################
