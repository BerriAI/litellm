"""
Loader for custom secret managers.

Handles dynamic loading of user-defined secret manager classes from Python files.
"""

import importlib.util
import os
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.types.secret_managers.main import KeyManagementSystem

_LOADED_ATTRIBUTE: Final = TypeAdapter(object)


def load_custom_secret_manager(config_file_path: str | None = None) -> None:
    """
    Load and initialize a custom secret manager from a python file.

    Similar to how custom guardrails are loaded - loads the class from
    the custom_secret_manager field in key_management_settings.

    Args:
        config_file_path: Path to the config.yaml file

    Raises:
        ValueError: If required configuration is missing or module path is invalid
        ImportError: If the custom secret manager module cannot be loaded
        AttributeError: If the class is not found in the module
        TypeError: If the class is not a subclass of CustomSecretManager
    """

    if not config_file_path:
        raise ValueError("CustomSecretManagerException - config_file_path is required to load custom secret manager")

    # Get the custom_secret_manager class path from settings
    if litellm._key_management_settings is None:
        raise ValueError(
            "CustomSecretManagerException - key_management_settings is required with custom_secret_manager field"
        )

    custom_secret_manager_path: Final[str | None] = getattr(
        litellm._key_management_settings, "custom_secret_manager", None
    )

    if not custom_secret_manager_path:
        raise ValueError(
            "CustomSecretManagerException - custom_secret_manager field is required in key_management_settings"
        )

    # Split into file_name and class_name (e.g., "my_secret_manager.InMemorySecretManager" or "pkg.submodule.MySecretManager")
    if "." not in custom_secret_manager_path:
        raise ValueError("CustomSecretManagerException - Invalid module path")

    _file_name, _class_name = custom_secret_manager_path.rsplit(".", 1)
    if not _file_name or not _class_name:
        raise ValueError("CustomSecretManagerException - Invalid module path")

    verbose_proxy_logger.debug(
        "Initializing custom secret manager: %s, file_name: %s, class_name: %s",
        custom_secret_manager_path,
        _file_name,
        _class_name,
    )

    # Load the module from the same directory as config.yaml
    directory: Final = os.path.dirname(config_file_path)
    module_file_path: Final = os.path.join(directory, _file_name) + ".py"

    # Validate against directory traversal
    if (
        os.path.isabs(_file_name)
        or _file_name.startswith(("/", "\\"))
        or any(segment == ".." for segment in _file_name.replace("\\", "/").split("/"))
        or os.path.relpath(os.path.realpath(module_file_path), os.path.realpath(directory)).startswith("..")
    ):
        raise ValueError("CustomSecretManagerException - Invalid module path")

    spec: Final = importlib.util.spec_from_file_location(_class_name, module_file_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not find a module specification for {module_file_path}")

    module: Final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, _class_name):
        raise AttributeError(f"CustomSecretManagerException - Class {_class_name} not found in {module_file_path}")

    _secret_manager_class: Final = _LOADED_ATTRIBUTE.validate_python(getattr(module, _class_name))

    # Validate that it's a CustomSecretManager subclass
    if not isinstance(_secret_manager_class, type) or not issubclass(
        _secret_manager_class,  # pyright: ignore[reportArgumentType]  # issubclass raises its own TypeError
        CustomSecretManager,
    ):
        raise TypeError(f"CustomSecretManagerException - {_class_name} must be a subclass of CustomSecretManager")

    # Instantiate the custom secret manager
    _secret_manager_instance: Final = _secret_manager_class()

    # Set it as the secret manager client
    litellm.secret_manager_client = _secret_manager_instance

    # Set the key management system to CUSTOM so get_secret knows to use it
    litellm._key_management_system = KeyManagementSystem.CUSTOM

    verbose_proxy_logger.info(
        "Successfully initialized custom secret manager: %s",
        custom_secret_manager_path,
    )
