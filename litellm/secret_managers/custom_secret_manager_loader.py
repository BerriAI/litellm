"""
Loader for custom secret managers.

Handles dynamic loading of user-defined secret manager classes from Python files.
"""

import importlib.util
import os
from typing import Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.types.secret_managers.main import KeyManagementSystem


def load_custom_secret_manager(config_file_path: Optional[str] = None) -> None:
    """
    Load and initialize a custom secret manager from a python file.
    
    Similar to how custom guardrails are loaded - loads the class from
    the custom_secret_manager field in key_management_settings.
    
    Args:
        config_file_path: Path to the config.yaml file
        
    Raises:
        ValueError: If required configuration is missing
        ImportError: If the custom secret manager module cannot be loaded
    """
    
    if not config_file_path:
        raise ValueError(
            "CustomSecretManagerException - config_file_path is required to load custom secret manager"
        )
    
    # Get the custom_secret_manager class path from settings
    if litellm._key_management_settings is None:
        raise ValueError(
            "CustomSecretManagerException - key_management_settings is required with custom_secret_manager field"
        )
    
    custom_secret_manager_path = getattr(
        litellm._key_management_settings, "custom_secret_manager", None
    )
    
    if not custom_secret_manager_path:
        raise ValueError(
            "CustomSecretManagerException - custom_secret_manager field is required in key_management_settings"
        )
    
    # Split into file_name and class_name (e.g., "my_secret_manager.InMemorySecretManager")
    _file_name, _class_name = custom_secret_manager_path.split(".")
    verbose_proxy_logger.debug(
        "Initializing custom secret manager: %s, file_name: %s, class_name: %s",
        custom_secret_manager_path,
        _file_name,
        _class_name,
    )
    
    # Load the module from the same directory as config.yaml
    directory = os.path.dirname(config_file_path)
    module_file_path = os.path.join(directory, _file_name) + ".py"
    
    spec = importlib.util.spec_from_file_location(_class_name, module_file_path)  # type: ignore
    if not spec:
        raise ImportError(
            f"Could not find a module specification for {module_file_path}"
        )
    
    module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    _secret_manager_class = getattr(module, _class_name)
    
    # Validate that it's a CustomSecretManager subclass
    if not issubclass(_secret_manager_class, CustomSecretManager):
        raise TypeError(
            f"CustomSecretManagerException - {_class_name} must be a subclass of CustomSecretManager"
        )
    
    # Instantiate the custom secret manager
    _secret_manager_instance = _secret_manager_class()
    
    # Set it as the secret manager client
    litellm.secret_manager_client = _secret_manager_instance
    
    # Set the key management system to CUSTOM so get_secret knows to use it
    litellm._key_management_system = KeyManagementSystem.CUSTOM
    
    verbose_proxy_logger.info(
        "Successfully initialized custom secret manager: %s",
        custom_secret_manager_path,
    )

