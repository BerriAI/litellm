import importlib
import os
from collections.abc import Callable, Sequence
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

DEFAULT_PROMPT_ENVIRONMENT: Final = "development"
PROMPT_ENVIRONMENT_SERVE_PRECEDENCE: Final = ("production", "staging", "development")


def get_base_prompt_id(prompt_id: str) -> str:
    """
    Extract the base prompt ID by stripping the version suffix if present.

    Examples:
        >>> get_base_prompt_id("jack_success.v1")
        "jack_success"
        >>> get_base_prompt_id("jack_success_v1")
        "jack_success"
        >>> get_base_prompt_id("jack_success")
        "jack_success"
    """
    if ".v" in prompt_id:
        return prompt_id.split(".v")[0]
    if "_v" in prompt_id:
        return prompt_id.split("_v")[0]
    return prompt_id


def get_version_number(prompt_id: str) -> int:
    """
    Extract the version number from a versioned prompt ID (defaults to 1).

    Examples:
        >>> get_version_number("jack_success.v2")
        2
        >>> get_version_number("jack_success_v2")
        2
        >>> get_version_number("jack_success")
        1
    """
    if ".v" in prompt_id:
        version_str = prompt_id.split(".v")[1]
        try:
            return int(version_str)
        except ValueError:
            pass

    if "_v" in prompt_id:
        version_str = prompt_id.split("_v")[1]
        try:
            return int(version_str)
        except ValueError:
            pass

    return 1


def prompt_environment_or_default(environment: str | None) -> str:
    return environment or DEFAULT_PROMPT_ENVIRONMENT


def registry_key_for_prompt(prompt: PromptSpec) -> str:
    return f"{prompt.prompt_id}::{prompt_environment_or_default(prompt.environment)}"


def _spec_version(prompt: PromptSpec) -> int:
    return prompt.version if prompt.version is not None else get_version_number(prompt_id=prompt.prompt_id)


def _default_serve_environment(prompts: Sequence[PromptSpec]) -> str:
    present: Final = frozenset(prompt_environment_or_default(prompt.environment) for prompt in prompts)
    ladder_pick: Final = next((env for env in PROMPT_ENVIRONMENT_SERVE_PRECEDENCE if env in present), None)
    if ladder_pick is not None:
        return ladder_pick
    return min(present) if present else DEFAULT_PROMPT_ENVIRONMENT


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

        registry_key: Final = registry_key_for_prompt(prompt)
        if registry_key in self.IN_MEMORY_PROMPTS:
            verbose_proxy_logger.debug("prompt already exists in IN_MEMORY_PROMPTS")
            return self.IN_MEMORY_PROMPTS[registry_key]

        parsed_prompt, custom_prompt_callback = self._build_prompt_callback(prompt=prompt)
        litellm.logging_callback_manager.add_litellm_callback(custom_prompt_callback)

        self.IN_MEMORY_PROMPTS[registry_key] = parsed_prompt
        self.prompt_id_to_custom_prompt[registry_key] = custom_prompt_callback

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
        registry_key: Final = registry_key_for_prompt(parsed_prompt)
        stale_callback: Final = self.prompt_id_to_custom_prompt.pop(registry_key, None)
        self.IN_MEMORY_PROMPTS.pop(registry_key, None)
        if stale_callback is not None:
            litellm.logging_callback_manager.remove_callback_from_all_lists(stale_callback)
        litellm.logging_callback_manager.add_litellm_callback(new_callback)
        self.IN_MEMORY_PROMPTS[registry_key] = parsed_prompt
        self.prompt_id_to_custom_prompt[registry_key] = new_callback
        return parsed_prompt

    def sync_prompt_from_db(self, prompt: PromptSpec) -> PromptSpec | None:
        existing: Final = self.IN_MEMORY_PROMPTS.get(registry_key_for_prompt(prompt))
        if existing is None:
            return self.initialize_prompt(prompt=prompt)
        if existing.litellm_params == prompt.litellm_params and existing.prompt_info == prompt.prompt_info:
            return existing
        return self.reload_prompt(prompt=prompt)

    def resolve_prompt_spec(
        self,
        prompt_id: str,
        version: int | None = None,
        environment: str | None = None,
    ) -> PromptSpec | None:
        """
        Resolve a prompt spec by base prompt id, optional version, and optional environment.

        With no environment, resolves within the default serve environment
        (production > staging > development > alphabetical first present).
        With no version, resolves to the highest version in the chosen environment.
        """
        base_prompt_id: Final = get_base_prompt_id(prompt_id=prompt_id)
        base_matches: Final = tuple(
            spec
            for spec in self.IN_MEMORY_PROMPTS.values()
            if get_base_prompt_id(prompt_id=spec.prompt_id) == base_prompt_id
        )
        if not base_matches:
            return None
        resolved_environment: Final = (
            environment if environment is not None else _default_serve_environment(base_matches)
        )
        env_matches: Final = tuple(
            spec for spec in base_matches if prompt_environment_or_default(spec.environment) == resolved_environment
        )
        if not env_matches:
            return None
        if version is not None:
            return next((spec for spec in env_matches if _spec_version(spec) == version), None)
        return max(env_matches, key=_spec_version)

    def get_prompt_callback_for_prompt(self, prompt: PromptSpec) -> CustomPromptManagement | None:
        return self.prompt_id_to_custom_prompt.get(registry_key_for_prompt(prompt))

    def has_config_prompt(self, base_prompt_id: str) -> bool:
        return any(
            spec.prompt_info.prompt_type == "config"
            for spec in self.IN_MEMORY_PROMPTS.values()
            if get_base_prompt_id(prompt_id=spec.prompt_id) == base_prompt_id
        )

    def delete_prompts_by_base_id(self, base_prompt_id: str, environment: str | None = None) -> list[str]:
        """
        Delete matching prompts from memory, scoped to one environment when given.

        Returns the registry keys that were deleted.
        """
        keys_to_delete: Final = [
            key
            for key, spec in self.IN_MEMORY_PROMPTS.items()
            if get_base_prompt_id(prompt_id=spec.prompt_id) == base_prompt_id
            and (environment is None or prompt_environment_or_default(spec.environment) == environment)
        ]

        for key in keys_to_delete:
            del self.IN_MEMORY_PROMPTS[key]
            self.prompt_id_to_custom_prompt.pop(key, None)

        return keys_to_delete


IN_MEMORY_PROMPT_REGISTRY: Final = InMemoryPromptRegistry()
