import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final

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
    discovered_initializers: Final[dict[str, Callable]] = {}

    try:
        # Get the path to the prompt_integrations directory
        current_dir: Final = Path(__file__).parent.parent.parent
        integrations_dir: Final = os.path.join(current_dir, "integrations")

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
                verbose_proxy_logger.debug("Discovering prompt integrations in: %s", module_path)

                module = importlib.import_module(module_path)

                # Check for prompt_initializer_registry dictionary
                if hasattr(module, "prompt_initializer_registry"):
                    registry = getattr(module, "prompt_initializer_registry")
                    if isinstance(registry, dict):
                        discovered_initializers.update(registry)
                        verbose_proxy_logger.debug(
                            "Found prompt_initializer_registry in %s: %s", module_path, list(registry.keys())
                        )

            except ImportError as e:
                verbose_proxy_logger.error("Could not import %s: %s", module_path, e)
                continue
            except Exception as e:
                verbose_proxy_logger.error("Error processing %s: %s", module_path, e)
                continue

        verbose_proxy_logger.debug(
            "Discovered %s prompt initializers: %s", len(discovered_initializers), list(discovered_initializers.keys())
        )

    except Exception as e:
        verbose_proxy_logger.error("Error discovering prompt initializers: %s", e)

    return discovered_initializers


prompt_initializer_registry = get_prompt_initializer_from_integrations()


class InMemoryPromptRegistry:
    """
    Class that handles adding prompt callbacks to the CallbacksManager.
    """

    def __init__(self):
        self.IN_MEMORY_PROMPTS: dict[str, PromptSpec] = {}
        """
        Prompt id to Prompt object mapping
        """

        self.prompt_id_to_custom_prompt: dict[str, CustomPromptManagement | None] = {}
        """
        Guardrail id to CustomGuardrail object mapping
        """

    def initialize_prompt(
        self,
        prompt: PromptSpec,
        config_file_path: str | None = None,
    ) -> PromptSpec | None:
        """
        Initialize a guardrail from a dictionary and add it to the litellm callback manager

        Returns a Guardrail object if the guardrail is initialized successfully
        """
        import litellm

        prompt_id: Final = prompt.prompt_id
        if prompt_id in self.IN_MEMORY_PROMPTS:
            verbose_proxy_logger.debug("prompt_id already exists in IN_MEMORY_PROMPTS")
            return self.IN_MEMORY_PROMPTS[prompt_id]

        parsed_prompt, custom_prompt_callback = self._build_prompt_callback(prompt=prompt)
        litellm.logging_callback_manager.add_litellm_callback(custom_prompt_callback)

        # store references to the prompt in memory
        self.IN_MEMORY_PROMPTS[prompt_id] = parsed_prompt
        self.prompt_id_to_custom_prompt[prompt_id] = custom_prompt_callback

        return parsed_prompt

    def _build_prompt_callback(self, prompt: PromptSpec) -> tuple[PromptSpec, CustomPromptManagement]:
        litellm_params_data: Final = prompt.litellm_params
        verbose_proxy_logger.debug("litellm_params= %s", litellm_params_data)

        if isinstance(litellm_params_data, dict):
            litellm_params = PromptLiteLLMParams(**litellm_params_data)
        else:
            litellm_params = litellm_params_data

        prompt_integration: Final = litellm_params.prompt_integration
        if prompt_integration is None:
            raise ValueError("prompt_integration is required")

        initializer: Final = prompt_initializer_registry.get(prompt_integration)
        if initializer is None:
            raise ValueError(f"Unsupported prompt: {prompt_integration}")

        custom_prompt_callback: Final = initializer(litellm_params, prompt)
        if not isinstance(custom_prompt_callback, CustomPromptManagement):
            raise ValueError(  # noqa: TRY004  # prompt endpoints map ValueError to HTTP 400; keep the existing contract
                f"CustomPromptManagement is required, got {type(custom_prompt_callback)}"
            )

        parsed_prompt: Final = PromptSpec(
            prompt_id=prompt.prompt_id,
            litellm_params=litellm_params,
            prompt_info=prompt.prompt_info or PromptInfo(prompt_type="config"),
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
            version=prompt.version,
            environment=prompt.environment,
            created_by=prompt.created_by,
        )
        return parsed_prompt, custom_prompt_callback

    def reload_prompt(self, prompt: PromptSpec) -> PromptSpec | None:
        import litellm

        parsed_prompt, new_callback = self._build_prompt_callback(prompt=prompt)
        stale_callback: Final = self.prompt_id_to_custom_prompt.pop(prompt.prompt_id, None)
        self.IN_MEMORY_PROMPTS.pop(prompt.prompt_id, None)
        if stale_callback is not None:
            litellm.logging_callback_manager.remove_callback_from_all_lists(stale_callback)
        litellm.logging_callback_manager.add_litellm_callback(new_callback)
        self.IN_MEMORY_PROMPTS[prompt.prompt_id] = parsed_prompt
        self.prompt_id_to_custom_prompt[prompt.prompt_id] = new_callback
        return parsed_prompt

    def sync_prompt_from_db(self, prompt: PromptSpec) -> PromptSpec | None:
        existing: Final = self.IN_MEMORY_PROMPTS.get(prompt.prompt_id)
        if existing is None:
            return self.initialize_prompt(prompt=prompt)
        if existing.litellm_params == prompt.litellm_params and existing.prompt_info == prompt.prompt_info:
            return existing
        return self.reload_prompt(prompt=prompt)

    def get_prompt_by_id(self, prompt_id: str) -> PromptSpec | None:
        """
        Get a prompt by its ID from memory
        """
        return self.IN_MEMORY_PROMPTS.get(prompt_id)

    def get_prompt_callback_by_id(self, prompt_id: str) -> CustomPromptManagement | None:
        """
        Get a prompt callback by its ID from memory
        """
        return self.prompt_id_to_custom_prompt.get(prompt_id)

    def remove_prompt(self, prompt_id: str) -> None:
        import litellm

        self.IN_MEMORY_PROMPTS.pop(prompt_id, None)
        stale_callback: Final = self.prompt_id_to_custom_prompt.pop(prompt_id, None)
        if stale_callback is not None:
            litellm.logging_callback_manager.remove_callback_from_all_lists(stale_callback)

    def delete_prompts_by_base_id(self, base_prompt_id: str, environment: str | None = None) -> list[str]:
        """
        Delete all prompts matching the given base prompt ID from memory, along with their
        registered callbacks; scoped to one environment when given.

        Args:
            base_prompt_id: The base prompt ID (without version suffix)
            environment: When set, only delete prompts deployed to this environment

        Returns:
            List of prompt IDs that were deleted
        """
        from litellm.proxy.prompts.prompt_endpoints import get_base_prompt_id

        prompts_to_delete: Final = [
            pid
            for pid, prompt in self.IN_MEMORY_PROMPTS.items()
            if get_base_prompt_id(prompt_id=pid) == base_prompt_id
            and (environment is None or prompt.environment == environment)
        ]

        for pid in prompts_to_delete:
            self.remove_prompt(prompt_id=pid)

        return prompts_to_delete


IN_MEMORY_PROMPT_REGISTRY: Final = InMemoryPromptRegistry()
