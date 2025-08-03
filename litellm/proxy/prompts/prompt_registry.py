import importlib
import os
from pathlib import Path
from typing import Callable, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.types.prompts.init_prompts import (
    PromptInfo,
    PromptLiteLLMParams,
    PromptSpec,
)

prompt_initializer_registry = {}


def get_prompt_initializer_from_integrations():
    """
    Get prompt initializers by discovering them from the prompt_integrations directory structure.

    Scans the integrations directory for subdirectories containing __init__.py files
    with either prompt_initializer_registry or initialize_prompt functions.

    Returns:
        Dict[str, Callable]: A dictionary mapping guardrail types to their initializer functions
    """
    discovered_initializers: Dict[str, Callable] = {}

    try:
        # Get the path to the prompt_integrations directory
        current_dir = Path(__file__).parent.parent.parent
        integrations_dir = os.path.join(current_dir, "integrations")

        if not os.path.exists(integrations_dir):
            verbose_proxy_logger.debug("integrations directory not found")
            return discovered_initializers

        # Scan each subdirectory in prompt_integrations
        for item in os.listdir(integrations_dir):
            item_path = os.path.join(integrations_dir, item)

            # Skip files and __pycache__ directories
            if not os.path.isdir(item_path) or item.startswith("__"):
                continue

            # Check if the directory has an __init__.py file
            init_file = os.path.join(item_path, "__init__.py")
            if not os.path.exists(init_file):
                continue

            module_path = f"litellm.integrations.{item}"
            try:
                # Import the module
                verbose_proxy_logger.debug(
                    f"Discovering prompt integrations in: {module_path}"
                )

                module = importlib.import_module(module_path)

                # Check for prompt_initializer_registry dictionary
                if hasattr(module, "prompt_initializer_registry"):
                    registry = getattr(module, "prompt_initializer_registry")
                    if isinstance(registry, dict):
                        discovered_initializers.update(registry)
                        verbose_proxy_logger.debug(
                            f"Found prompt_initializer_registry in {module_path}: {list(registry.keys())}"
                        )

            except ImportError as e:
                verbose_proxy_logger.error(f"Could not import {module_path}: {e}")
                continue
            except Exception as e:
                verbose_proxy_logger.error(f"Error processing {module_path}: {e}")
                continue

        verbose_proxy_logger.debug(
            f"Discovered {len(discovered_initializers)} prompt initializers: {list(discovered_initializers.keys())}"
        )

    except Exception as e:
        verbose_proxy_logger.error(f"Error discovering prompt initializers: {e}")

    return discovered_initializers


prompt_initializer_registry = get_prompt_initializer_from_integrations()


class InMemoryPromptRegistry:
    """
    Class that handles adding prompt callbacks to the CallbacksManager.
    """

    def __init__(self):
        self.IN_MEMORY_PROMPTS: Dict[str, PromptSpec] = {}
        """
        Prompt id to Prompt object mapping
        """

        self.prompt_id_to_custom_prompt: Dict[str, Optional[CustomPromptManagement]] = (
            {}
        )
        """
        Guardrail id to CustomGuardrail object mapping
        """

    def initialize_prompt(
        self,
        prompt: PromptSpec,
        config_file_path: Optional[str] = None,
    ) -> Optional[PromptSpec]:
        """
        Initialize a guardrail from a dictionary and add it to the litellm callback manager

        Returns a Guardrail object if the guardrail is initialized successfully
        """
        import litellm

        prompt_id = prompt.prompt_id
        if prompt_id in self.IN_MEMORY_PROMPTS:
            verbose_proxy_logger.debug("prompt_id already exists in IN_MEMORY_PROMPTS")
            return self.IN_MEMORY_PROMPTS[prompt_id]

        custom_prompt_callback: Optional[CustomPromptManagement] = None
        litellm_params_data = prompt.litellm_params
        verbose_proxy_logger.debug("litellm_params= %s", litellm_params_data)

        if isinstance(litellm_params_data, dict):
            litellm_params = PromptLiteLLMParams(**litellm_params_data)
        else:
            litellm_params = litellm_params_data

        prompt_integration = litellm_params.prompt_integration
        if prompt_integration is None:
            raise ValueError("prompt_integration is required")

        initializer = prompt_initializer_registry.get(prompt_integration)

        if initializer:
            custom_prompt_callback = initializer(litellm_params, prompt)
            if not isinstance(custom_prompt_callback, CustomPromptManagement):
                raise ValueError(
                    f"CustomPromptManagement is required, got {type(custom_prompt_callback)}"
                )
            litellm.logging_callback_manager.add_litellm_callback(custom_prompt_callback)  # type: ignore
        else:
            raise ValueError(f"Unsupported prompt: {prompt_integration}")

        parsed_prompt = PromptSpec(
            prompt_id=prompt_id,
            litellm_params=litellm_params,
            prompt_info=prompt.prompt_info or PromptInfo(prompt_type="config"),
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
        )

        # store references to the prompt in memory
        self.IN_MEMORY_PROMPTS[prompt_id] = parsed_prompt
        self.prompt_id_to_custom_prompt[prompt_id] = custom_prompt_callback

        return parsed_prompt

    def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptSpec]:
        """
        Get a prompt by its ID from memory
        """
        return self.IN_MEMORY_PROMPTS.get(prompt_id)

    def get_prompt_callback_by_id(
        self, prompt_id: str
    ) -> Optional[CustomPromptManagement]:
        """
        Get a prompt callback by its ID from memory
        """
        return self.prompt_id_to_custom_prompt.get(prompt_id)


IN_MEMORY_PROMPT_REGISTRY = InMemoryPromptRegistry()
