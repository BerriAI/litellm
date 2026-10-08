# litellm/proxy/guardrails/guardrail_registry.py

import asyncio
import importlib
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from itertools import chain, count
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol, TypeAlias, TypeVar, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH, GUARDRAIL_ROTATION_ATTEMPTS
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.proxy.auth.master_key_boot_check import SALT_KEY_ENV_VAR
from litellm.proxy.common_utils.callback_utils import CALLBACK_VAR_ENCRYPTED_PREFIX, is_sensitive_callback_key
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
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
from litellm.proxy.guardrails.guardrail_hooks.presidio import (  # noqa: F401  # legacy module exports
    OPTIONAL_PresidioPIIMasking,
    _OPTIONAL_PresidioPIIMasking,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
)
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import (
    ToolPermissionGuardrail,
)
from litellm.proxy.types_utils.utils import get_instance_fn
from litellm.proxy.utils import PrismaClient
from litellm.repositories.prisma_protocols import TableActions
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
    configured_event_hooks,
    initialize_bedrock,
    initialize_hide_secrets,
    initialize_lakera,
    initialize_lakera_v2,
    initialize_presidio,
    initialize_tool_permission,
)

if TYPE_CHECKING:
    from prisma import models as prisma_models


class _GuardrailRowLike(Protocol):
    @property
    def guardrail_id(self) -> str: ...
    def __iter__(self) -> Iterator[tuple[str, object]]: ...


def _guardrail_table(prisma_client: PrismaClient) -> "TableActions[prisma_models.LiteLLM_GuardrailsTable]":
    """Typed view of the guardrails table actions exposed by the Prisma repository."""
    return GuardrailsRepository(prisma_client).table


_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_ARRAY: Final = TypeAdapter(list[object])


def _as_json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _as_json_array(value: object) -> list[object] | None:
    return _JSON_ARRAY.validate_python(value) if isinstance(value, list) else None


def contains_encrypted_marker(value: object, depth: int = 0) -> bool:
    """True if any string in value, at any JSON depth, starts with the encrypted-value prefix."""
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return False
    if isinstance(value, str):
        return value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX)
    json_object: Final = _as_json_object(value)
    if json_object is not None:
        return any(contains_encrypted_marker(v, depth + 1) for v in json_object.values())
    json_array: Final = _as_json_array(value)
    return json_array is not None and any(contains_encrypted_marker(item, depth + 1) for item in json_array)


def _encrypted_param(key: str, value: object, new_encryption_key: str | None, depth: int = 0) -> object:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return value
    json_object: Final = _as_json_object(value)
    if json_object is not None:
        return {k: _encrypted_param(k, v, new_encryption_key, depth + 1) for k, v in json_object.items()}
    json_array: Final = _as_json_array(value)
    if json_array is not None:
        return [_encrypted_param(key, item, new_encryption_key, depth + 1) for item in json_array]
    if not (
        isinstance(value, str)
        and value
        and is_sensitive_callback_key(key)
        and not value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX)
    ):
        return value
    try:
        return CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper(value, new_encryption_key=new_encryption_key)
    except Exception:  # noqa: BLE001  # no salt key or master key configured: store the value as written
        return value


def _decrypted_param(key: str, value: object, depth: int = 0) -> object:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return value
    json_object: Final = _as_json_object(value)
    if json_object is not None:
        return {k: _decrypted_param(k, v, depth + 1) for k, v in json_object.items()}
    json_array: Final = _as_json_array(value)
    if json_array is not None:
        return [_decrypted_param(key, item, depth + 1) for item in json_array]
    if not (isinstance(value, str) and value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX)):
        return value
    decrypted: Final = decrypt_value_helper(
        value.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX),
        key=key,
        exception_type="debug",
        return_original_value=False,
    )
    return value if decrypted is None else decrypted


def encrypt_guardrail_litellm_params(
    litellm_params: Mapping[str, object], new_encryption_key: str | None = None
) -> dict[str, object]:
    """Encrypt every string stored under a sensitive key (at any dict depth) for the guardrails table."""
    return {key: _encrypted_param(key, value, new_encryption_key) for key, value in litellm_params.items()}


def decrypt_guardrail_litellm_params(litellm_params: Mapping[str, object]) -> dict[str, object]:
    """Decrypt values written by encrypt_guardrail_litellm_params; plaintext values pass through unchanged."""
    return {key: _decrypted_param(key, value) for key, value in litellm_params.items()}


def guardrail_from_db_row(row: Iterable[tuple[str, object]]) -> Guardrail:
    """Build a Guardrail from a guardrails table row with its litellm_params decrypted."""
    fields: Final = dict(row)
    stored_params: Final = _as_json_object(fields.get("litellm_params"))
    if stored_params is None:
        return Guardrail(**fields)
    return Guardrail(**{**fields, "litellm_params": decrypt_guardrail_litellm_params(stored_params)})


async def _rotate_guardrail_row(
    prisma_client: PrismaClient,
    row: "prisma_models.LiteLLM_GuardrailsTable | None",
    encryption_key: str,
    attempts_left: int = GUARDRAIL_ROTATION_ATTEMPTS,
) -> int:
    """Re-encrypt one row's params under encryption_key with a compare-and-set on updated_at.
    A row edited since it was read is re-read and retried, up to attempts_left writes. Returns 1 when rewritten."""
    if row is None or not isinstance(row.litellm_params, Mapping):
        return 0
    rotated_params: Final = encrypt_guardrail_litellm_params(
        decrypt_guardrail_litellm_params(row.litellm_params), new_encryption_key=encryption_key
    )
    if rotated_params == row.litellm_params:
        return 0
    if await _guardrail_table(prisma_client).update_many(
        where={"guardrail_id": row.guardrail_id, "updated_at": row.updated_at},
        data={"litellm_params": safe_dumps(rotated_params)},
    ):
        return 1
    if attempts_left <= 1:
        verbose_proxy_logger.warning(
            "Guardrail %s kept changing during master key rotation; its secrets were not re-encrypted",
            row.guardrail_id,
        )
        return 0
    latest_row: Final = await _guardrail_table(prisma_client).find_unique(where={"guardrail_id": row.guardrail_id})
    return await _rotate_guardrail_row(prisma_client, latest_row, encryption_key, attempts_left - 1)


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

GuardrailCallbacks: TypeAlias = tuple[CustomGuardrail, ...]

guardrail_class_registry: Final[dict[str, type[CustomGuardrail]]] = {
    SupportedGuardrailIntegrations.BEDROCK.value: BedrockGuardrail,
    SupportedGuardrailIntegrations.GRAYSWAN.value: GraySwanGuardrail,
    SupportedGuardrailIntegrations.LAKERA.value: lakeraAI_Moderation,
    SupportedGuardrailIntegrations.LAKERA_V2.value: LakeraAIGuardrail,
    SupportedGuardrailIntegrations.PRESIDIO.value: OPTIONAL_PresidioPIIMasking,
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
                    registry: Mapping[str, Callable[..., CustomGuardrail]] | None = getattr(
                        module, "guardrail_initializer_registry", None
                    )
                    if isinstance(registry, dict):
                        discovered_initializers.update(registry)
                        verbose_proxy_logger.debug(
                            "Found guardrail_initializer_registry in %s: %s", module_path, list(registry.keys())
                        )

                # Check for standalone initialize_guardrail function (fallback for directory-based guardrails)
                elif hasattr(module, "initialize_guardrail"):
                    # For directories with just initialize_guardrail, use the directory name as the key
                    initialize_fn: Callable[..., CustomGuardrail] | None = getattr(module, "initialize_guardrail", None)
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
                    registry: Mapping[str, type[CustomGuardrail]] | None = getattr(
                        module, "guardrail_class_registry", None
                    )
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
            litellm_params_obj: Final = guardrail.get("litellm_params", {})
            if hasattr(litellm_params_obj, "model_dump"):
                litellm_params_dict = litellm_params_obj.model_dump()
            else:
                litellm_params_dict = dict(litellm_params_obj) if litellm_params_obj else {}
            litellm_params: Final[str] = safe_dumps(encrypt_guardrail_litellm_params(litellm_params_dict))
            guardrail_info: Final[str] = safe_dumps(guardrail.get("guardrail_info", {}))

            # Create guardrail in DB
            created_guardrail: Final[_GuardrailRowLike] = await _guardrail_table(prisma_client).create(
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
            await _guardrail_table(prisma_client).delete(where={"guardrail_id": guardrail_id})

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
            litellm_params_obj: Final = guardrail.get("litellm_params", {})
            if hasattr(litellm_params_obj, "model_dump"):
                litellm_params_dict = litellm_params_obj.model_dump()
            else:
                litellm_params_dict = dict(litellm_params_obj) if litellm_params_obj else {}
            litellm_params: Final[str] = safe_dumps(encrypt_guardrail_litellm_params(litellm_params_dict))
            guardrail_info: Final[str] = safe_dumps(guardrail.get("guardrail_info", {}))

            # Update in DB
            updated_guardrail: Final[_GuardrailRowLike | None] = await _guardrail_table(prisma_client).update(
                where={"guardrail_id": guardrail_id},
                data={
                    "guardrail_name": guardrail_name,
                    "litellm_params": litellm_params,
                    "guardrail_info": guardrail_info,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            if updated_guardrail is None:
                raise ValueError(f"Guardrail not found, passed guardrail_id={guardrail_id}")

            return dict(guardrail_from_db_row(updated_guardrail))
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
            guardrails_from_db: Final = await _guardrail_table(prisma_client).find_many(
                where={"status": "active"},
                order={"created_at": "desc"},
            )

            guardrails: Final[list[Guardrail]] = []
            for guardrail in guardrails_from_db:
                guardrails.append(guardrail_from_db_row(guardrail))

            return guardrails
        except Exception as e:
            raise Exception(f"Error getting guardrails from DB: {e}")

    async def get_guardrail_by_id_from_db(self, guardrail_id: str, prisma_client: PrismaClient) -> Guardrail | None:
        """
        Get a guardrail by its ID from the database
        """
        try:
            guardrail: Final = await _guardrail_table(prisma_client).find_unique(where={"guardrail_id": guardrail_id})

            if not guardrail:
                return None

            return guardrail_from_db_row(guardrail)
        except Exception as e:
            raise Exception(f"Error getting guardrail from DB: {e}")

    async def get_guardrail_by_name_from_db(self, guardrail_name: str, prisma_client: PrismaClient) -> Guardrail | None:
        """
        Get a guardrail by its name from the database
        """
        try:
            guardrail: Final = await _guardrail_table(prisma_client).find_unique(
                where={"guardrail_name": guardrail_name}
            )

            if not guardrail:
                return None

            return guardrail_from_db_row(guardrail)
        except Exception as e:
            raise Exception(f"Error getting guardrail from DB: {e}")

    @staticmethod
    async def rotate_guardrail_params_master_key(prisma_client: PrismaClient, new_master_key: str) -> int:
        """Re-encrypt every guardrail row's sensitive litellm_params under the key the proxy decrypts with after the
        rotation (LITELLM_SALT_KEY when set, otherwise new_master_key). Returns the number of rows rewritten."""
        salt_key: Final = os.environ.get(SALT_KEY_ENV_VAR)
        encryption_key: Final = new_master_key if salt_key is None else salt_key
        rows: Final = await _guardrail_table(prisma_client).find_many()
        rotated = [await _rotate_guardrail_row(prisma_client, row, encryption_key) for row in rows]
        return sum(rotated)


def _apply_configured_bool_overrides(instance: CustomGuardrail, litellm_params: LitellmParams) -> None:
    """Override the parallel/raw-scan flags only when ``litellm_params`` explicitly
    sets them, preserving whatever default the guardrail's own constructor chose
    otherwise (its constructor default may be True, so blindly copying an
    absent/None config value would silently clobber it back to False)."""
    if litellm_params.run_in_parallel is not None:
        instance.run_in_parallel = bool(litellm_params.run_in_parallel)
    if litellm_params.scan_raw_request is not None:
        instance.scan_raw_request = bool(litellm_params.scan_raw_request)


def _as_callback_tuple(
    initialized: CustomGuardrail | Sequence[CustomGuardrail] | None,
) -> GuardrailCallbacks:
    if initialized is None:
        return ()
    if isinstance(initialized, (list, tuple)):
        return tuple(initialized)
    return (initialized,)


def _logging_only_scope_error(
    custom_guardrail_callback: CustomGuardrail, guardrail_name: str, litellm_params: LitellmParams
) -> str | None:
    logging_only_scope: Final = litellm_params.logging_only_scope
    if logging_only_scope is not None and GuardrailEventHooks.logging_only.value not in configured_event_hooks(
        litellm_params.mode
    ):
        return (
            f"Guardrail {guardrail_name}: logging_only_scope is set, but mode does not include logging_only, "
            "so it would never apply. Add logging_only to mode or remove logging_only_scope."
        )
    if logging_only_scope in ("input", "output") and not custom_guardrail_callback.supports_logging_only_scope():
        return (
            f"Guardrail {guardrail_name}: logging_only_scope={logging_only_scope!r} is not supported by this "
            "guardrail, whose logging_only hook scans on its own. Remove logging_only_scope."
        )
    return None


def _configure_callback_scoping(
    custom_guardrail_callback: CustomGuardrail,
    guardrail_name: str,
    litellm_params: LitellmParams,
    *,
    reject_invalid_logging_only_scope: bool = False,
) -> None:
    logging_only_scope: Final = litellm_params.logging_only_scope
    logging_only_scope_error: Final = _logging_only_scope_error(
        custom_guardrail_callback, guardrail_name, litellm_params
    )
    if logging_only_scope_error is not None:
        if reject_invalid_logging_only_scope:
            raise ValueError(logging_only_scope_error)
        verbose_proxy_logger.error(
            "%s Ignoring logging_only_scope; the guardrail keeps its configured mode.",
            logging_only_scope_error.replace("\r", "").replace("\n", ""),
        )
        custom_guardrail_callback.logging_only_scope = None
    else:
        custom_guardrail_callback.logging_only_scope = logging_only_scope
    for scoping_param in (
        "skip_system_message_in_guardrail",
        "skip_tool_message_in_guardrail",
        "scan_only_tool_results",
    ):
        setattr(custom_guardrail_callback, scoping_param, getattr(litellm_params, scoping_param, None))
    scan_only_tool_results_enabled: Final = effective_scan_only_tool_results_for_guardrail(custom_guardrail_callback)
    if scan_only_tool_results_enabled and not custom_guardrail_callback.supports_scan_only_tool_results():
        raise ValueError(
            f"Guardrail {guardrail_name}: scan_only_tool_results is enabled, but this "
            "guardrail's role filtering never scans tool results, so no request content would ever "
            "be scanned. Remove scan_only_tool_results or the guardrail's role-filtering option."
        )
    if scan_only_tool_results_enabled and effective_skip_tool_message_for_guardrail(custom_guardrail_callback):
        raise ValueError(
            f"Guardrail {guardrail_name}: scan_only_tool_results and "
            "skip_tool_message_in_guardrail are enabled together, which excludes every message from "
            "scanning, so no request content would ever be scanned. Remove one of the two."
        )
    _apply_configured_bool_overrides(custom_guardrail_callback, litellm_params)


_ParamsT = TypeVar("_ParamsT", bound=BaseModel)


def parse_tolerant_litellm_params(
    litellm_params_data: Mapping[str, object],
    guardrail_name: str,
    params_model: type[_ParamsT] = LitellmParams,
) -> _ParamsT:
    try:
        return params_model(**litellm_params_data)
    except ValidationError as validation_error:
        if any(tuple(error["loc"]) != ("logging_only_scope",) for error in validation_error.errors()):
            raise
        verbose_proxy_logger.error(
            "Guardrail %s: logging_only_scope=%r is not one of 'input', 'output' or 'both'. "
            "Ignoring logging_only_scope; the guardrail keeps its configured mode.",
            guardrail_name.replace("\r", "").replace("\n", ""),
            str(litellm_params_data.get("logging_only_scope")).replace("\r", "").replace("\n", "")[:100],
        )
        return params_model(**MappingProxyType({**litellm_params_data, "logging_only_scope": None}))


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

        self.guardrail_id_to_sibling_callbacks: dict[str, GuardrailCallbacks] = {}  # mutable-ok: per-id registry

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
        *,
        reject_invalid_logging_only_scope: bool = False,
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

        litellm_params_data: Final = guardrail["litellm_params"]
        verbose_proxy_logger.debug("litellm_params= %s", litellm_params_data)

        if isinstance(litellm_params_data, dict):
            if reject_invalid_logging_only_scope:
                litellm_params = LitellmParams(**litellm_params_data)
            else:
                litellm_params = parse_tolerant_litellm_params(litellm_params_data, guardrail["guardrail_name"])
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

        created_callbacks: Final = self._create_callbacks(
            guardrail=guardrail,
            guardrail_type=guardrail_type,
            litellm_params=litellm_params,
            config_file_path=config_file_path,
            llm_router=llm_router,
        )
        try:
            for custom_guardrail_callback in created_callbacks:
                _configure_callback_scoping(
                    custom_guardrail_callback,
                    guardrail["guardrail_name"],
                    litellm_params,
                    reject_invalid_logging_only_scope=reject_invalid_logging_only_scope,
                )
        except Exception:
            for custom_guardrail_callback in created_callbacks:
                litellm.logging_callback_manager.remove_callback_from_all_lists(custom_guardrail_callback)
            raise

        parsed_guardrail: Final = Guardrail(
            guardrail_id=guardrail.get("guardrail_id"),
            guardrail_name=guardrail["guardrail_name"],
            litellm_params=litellm_params,
            guardrail_info=guardrail.get("guardrail_info"),
        )

        # store references to the guardrail in memory
        self.IN_MEMORY_GUARDRAILS[guardrail_id] = parsed_guardrail
        self.guardrail_id_to_custom_guardrail[guardrail_id] = created_callbacks[0] if created_callbacks else None
        self.guardrail_id_to_sibling_callbacks[guardrail_id] = created_callbacks[1:]
        self._sources[guardrail_id] = source

        return parsed_guardrail

    def _create_callbacks(
        self,
        guardrail: Guardrail,
        guardrail_type: str,
        litellm_params: LitellmParams,
        config_file_path: str | None,
        llm_router: Optional["Router"],
    ) -> GuardrailCallbacks:
        initializer: Final = guardrail_initializer_registry.get(guardrail_type)
        if initializer:
            import inspect

            sig: Final = inspect.signature(initializer)
            if "llm_router" in sig.parameters:
                return _as_callback_tuple(initializer(litellm_params, guardrail, llm_router))
            return _as_callback_tuple(initializer(litellm_params, guardrail))
        if isinstance(guardrail_type, str) and "." in guardrail_type:
            return _as_callback_tuple(
                self.initialize_custom_guardrail(
                    guardrail=guardrail,
                    guardrail_type=guardrail_type,
                    litellm_params=litellm_params,
                    config_file_path=config_file_path,
                )
            )
        raise ValueError(f"Unsupported guardrail: {guardrail_type}")

    def _tracked_callbacks(self, guardrail_id: str) -> GuardrailCallbacks:
        primary: Final = self.guardrail_id_to_custom_guardrail.get(guardrail_id)
        siblings: Final = self.guardrail_id_to_sibling_callbacks.get(guardrail_id, ())
        return (() if primary is None else (primary,)) + siblings

    def _reject_invalid_logging_only_scope(self, guardrail_id: str, guardrail: Guardrail) -> None:
        """
        Strictly validate logging_only_scope on a row whose params are otherwise
        unchanged, without rebuilding the live callback.

        API write paths send the whole object, so an invalid scope must still be
        rejected even when the write changed nothing else. But an unchanged row
        must not force a teardown + re-append: initialize_guardrail appends the
        rebuilt callback at the END of litellm.callbacks, so a no-op PUT would
        reorder guardrails and change which one wins between a BLOCK and a MASK
        guardrail over the same content.
        """
        params: Final = guardrail.get("litellm_params")
        if not isinstance(params, (dict, LitellmParams)):
            return
        litellm_params: Final = LitellmParams(**params) if isinstance(params, dict) else params
        guardrail_name: Final = guardrail.get("guardrail_name", "Unknown")
        for custom_guardrail_callback in self._tracked_callbacks(guardrail_id):
            scope_error = _logging_only_scope_error(custom_guardrail_callback, guardrail_name, litellm_params)
            if scope_error is not None:
                raise ValueError(scope_error)

    def initialize_custom_guardrail(
        self,
        guardrail: Guardrail,
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

        _guardrail_class: Final[Callable[..., CustomGuardrail]] = get_instance_fn(
            guardrail_type, config_file_path=config_file_path
        )

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
        *,
        reject_invalid_logging_only_scope: bool = False,
    ) -> None:
        """
        Update a guardrail in memory: a changed name or litellm_params rebuilds the
        live callback from the new row (fail-closed: an invalid row keeps the
        previous instance and raises), anything else only refreshes the stored row
        """
        updated_guardrail: Final = cast(Guardrail, {**guardrail, "guardrail_id": guardrail_id})
        if self._has_guardrail_params_changed(guardrail_id, updated_guardrail):
            self.reinitialize_guardrail(
                guardrail=updated_guardrail,
                source=source,
                reject_invalid_logging_only_scope=reject_invalid_logging_only_scope,
            )
            return
        if reject_invalid_logging_only_scope:
            self._reject_invalid_logging_only_scope(guardrail_id, updated_guardrail)
        self.IN_MEMORY_GUARDRAILS[guardrail_id] = updated_guardrail
        self._sources[guardrail_id] = source

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

        tracked_callbacks: Final = self._tracked_callbacks(guardrail_id)
        self.guardrail_id_to_custom_guardrail.pop(guardrail_id, None)
        self.guardrail_id_to_sibling_callbacks.pop(guardrail_id, None)
        for custom_guardrail_callback in tracked_callbacks:
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
        params: LitellmParams | Mapping[str, object] | None,
        guardrail_name: str,
    ) -> Mapping[str, object] | None:
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
                return parse_tolerant_litellm_params(params, guardrail_name).model_dump()
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
        existing_dict: Final = self._normalize_litellm_params_for_comparison(
            existing.get("litellm_params"), existing.get("guardrail_name", "Unknown")
        )
        new_dict: Final = self._normalize_litellm_params_for_comparison(
            new_guardrail.get("litellm_params"), new_guardrail.get("guardrail_name", "Unknown")
        )

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
        *,
        reject_invalid_logging_only_scope: bool = False,
    ) -> Guardrail | None:
        """
        Force re-initialization of a guardrail even if it exists in memory.
        Removes old callback from litellm.callbacks and creates fresh instance.

        If the new config fails to initialize (e.g. an invalid on_flagged
        combination or an invalid regex), the previous instance is restored
        rather than left deleted, and the failure is re-raised as ValueError so
        every init failure reaches callers as one exception type: a caller
        reaching this point after already deleting the old instance would
        otherwise leave the guardrail providing no protection at all, not
        merely "still enforcing the old config."
        """
        guardrail_id: Final = guardrail.get("guardrail_id")
        if not guardrail_id:
            verbose_proxy_logger.error("Cannot reinitialize guardrail without guardrail_id")
            return None

        previous_guardrail: Final = self.IN_MEMORY_GUARDRAILS.get(guardrail_id)
        previous_source: Final = self._sources.get(guardrail_id, source)

        if guardrail_id in self.IN_MEMORY_GUARDRAILS:
            self.delete_in_memory_guardrail(guardrail_id)

        # Initialize fresh (will add new callback to litellm.callbacks). If the new
        # params are invalid (a raising guardrail __init__), restore the previous
        # instance instead of leaving the guardrail silently removed: a guardrail
        # that was enforcing must never fail open because an update was bad.
        try:
            return self.initialize_guardrail(
                guardrail=guardrail,
                config_file_path=config_file_path,
                source=source,
                reject_invalid_logging_only_scope=reject_invalid_logging_only_scope,
            )
        except Exception as init_error:
            if previous_guardrail is not None:
                verbose_proxy_logger.exception(
                    "Reinitializing guardrail %s with updated params failed; restoring the previous configuration",
                    guardrail_id,
                )
                try:
                    self.initialize_guardrail(
                        guardrail=previous_guardrail, config_file_path=config_file_path, source=previous_source
                    )
                except Exception:  # noqa: BLE001  # the original failure must propagate even if the restore breaks
                    verbose_proxy_logger.exception("Restoring previous guardrail %s also failed", guardrail_id)
            raise ValueError(f"Guardrail initialization failed: {init_error}") from init_error

    def _with_loaded_values_where_undecryptable(self, guardrail_id: str, guardrail: Guardrail) -> Guardrail:
        """Swap each DB litellm_params value that did not decrypt with the current key for the loaded guardrail's value,
        or keep the loaded guardrail whole when it has no value for one of them."""
        existing: Final = self.IN_MEMORY_GUARDRAILS.get(guardrail_id)
        stored_params: Final = guardrail.get("litellm_params")
        db_params: Final = _as_json_object(
            stored_params.model_dump() if isinstance(stored_params, BaseModel) else stored_params
        )
        if existing is None or db_params is None or not contains_encrypted_marker(db_params):
            return guardrail
        loaded_params: Final = self._normalize_litellm_params_for_comparison(
            existing.get("litellm_params"), guardrail.get("guardrail_name", "Unknown")
        )
        verbose_proxy_logger.warning(
            "Guardrail %s has litellm_params that do not decrypt with the current key; keeping the loaded values for "
            "them. Restart the proxy if the master key was rotated.",
            guardrail_id,
        )
        if loaded_params is None or any(
            contains_encrypted_marker(value) and loaded_params.get(key) is None for key, value in db_params.items()
        ):
            return existing
        return Guardrail(
            **{
                **guardrail,
                "litellm_params": {
                    key: loaded_params.get(key) if contains_encrypted_marker(value) else value
                    for key, value in db_params.items()
                },
            }
        )

    def sync_guardrail_from_db(
        self,
        guardrail: Guardrail,
        config_file_path: str | None = None,
        *,
        reject_invalid_logging_only_scope: bool = False,
    ) -> Guardrail | None:
        """
        Sync a guardrail from DB - initializes if new, re-initializes if changed.
        DB values that do not decrypt with the current key keep the loaded guardrail's values.
        This is the method to call during DB polling.
        """
        guardrail_id: Final = guardrail.get("guardrail_id")
        if not guardrail_id:
            verbose_proxy_logger.error("Cannot sync guardrail without guardrail_id")
            return None

        synced: Final = self._with_loaded_values_where_undecryptable(guardrail_id, guardrail)
        if self._has_guardrail_params_changed(guardrail_id, synced):
            guardrail_name: Final = synced.get("guardrail_name", "Unknown")
            verbose_proxy_logger.info(
                "Guardrail '%s' (ID: %s) params changed, re-initializing...", guardrail_name, guardrail_id
            )
            return self.reinitialize_guardrail(
                guardrail=synced,
                config_file_path=config_file_path,
                source="db",
                reject_invalid_logging_only_scope=reject_invalid_logging_only_scope,
            )

        if reject_invalid_logging_only_scope:
            self._reject_invalid_logging_only_scope(guardrail_id, synced)

        # Params unchanged but the entry is still DB-backed; make sure the
        # source marker reflects that even if it was previously set differently
        # (e.g. a config entry whose UUID later collided with a DB row).
        self._sources[guardrail_id] = "db"
        return self.IN_MEMORY_GUARDRAILS.get(guardrail_id)


########################################################
# In Memory Guardrail Handler for LiteLLM Proxy
########################################################
IN_MEMORY_GUARDRAIL_HANDLER: Final = InMemoryGuardrailHandler()

GUARDRAIL_RECONCILE_LOCK: Final = asyncio.Lock()
########################################################
