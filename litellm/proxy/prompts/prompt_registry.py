import importlib
import os
from pathlib import Path
from typing import Callable, Dict, Optional, List, Tuple, Iterable
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.gitlab import GitLabPromptCache, GitLabPromptManager
from collections import OrderedDict
from litellm.types.prompts.init_prompts import (
    PromptInfo,
    PromptLiteLLMParams,
    PromptSpec
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

    def load_all(self):
        verbose_proxy_logger.debug("Loading all prompts from InMemoryPromptRegistry")
        return self.IN_MEMORY_PROMPTS

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


class GitlabPromptRegistry:
    """
    Class that handles adding prompt callbacks to the CallbacksManager.
    """

    def __init__(self):

        self.gitlab_prompt_cache: Optional[GitLabPromptCache] = None
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

    def load_all(self):
        verbose_proxy_logger.debug("Loading all prompts from UnifiedPromptRegistry")
        if not self.gitlab_prompt_cache:
            import litellm
            self.gitlab_prompt_cache:GitLabPromptCache = GitLabPromptCache(
                litellm.global_gitlab_config
            )

            prompts_dict = self.gitlab_prompt_cache.load_all()
            for prompt_id, prompt_json in prompts_dict.items():
                verbose_proxy_logger.debug(
                    f"{prompt_id} --> {prompt_json}"
                )

                prompt_info: PromptInfo = PromptInfo(
                    prompt_type="config",
                    model_config={
                        'content': prompt_json.get('content'),
                        'metadata': prompt_json.get('metadata')
                    }
                )
                prompt_params: PromptLiteLLMParams = PromptLiteLLMParams(
                    prompt_id=prompt_id,
                    prompt_integration="gitlab",
                    model_config={
                        'content': prompt_json.get('content'),
                        'metadata': prompt_json.get('metadata')
                    }
                )
                prompt_spec: PromptSpec = PromptSpec(
                    prompt_id=prompt_id,
                    litellm_params=prompt_params,
                    prompt_info=prompt_info
                )
                self.IN_MEMORY_PROMPTS[prompt_id] = prompt_spec
                self.initialize_prompt(prompt_spec)
                verbose_proxy_logger.debug(
                    f"Loaded prompt {prompt_id} from GitLab into IN_MEMORY_PROMPTS. "
                    f"found prompt --> {self.prompt_id_to_custom_prompt.get(prompt_id, None)}"
                )

        verbose_proxy_logger.debug(
            f"found the gitlab prompts with these ids {list(set(self.IN_MEMORY_PROMPTS.keys()))}"
        )
        return self.IN_MEMORY_PROMPTS

    def gitlab_prompt_id_to_litellm_prompt_id(self):
        pass

    def initialize_prompt(
            self,
            prompt: PromptSpec,
            config_file_path: Optional[str] = None,
    ) -> Optional[PromptSpec]:
        """
        Initialize a guardrail from a dictionary and add it to the litellm callback manager

        Returns a Guardrail object if the guardrail is initialized successfully
        """
        verbose_proxy_logger.debug("Initializing prompt in GitlabPromptRegistry")
        import litellm

        prompt_id = prompt.prompt_id
        if prompt_id in self.IN_MEMORY_PROMPTS and prompt_id in self.prompt_id_to_custom_prompt:
            verbose_proxy_logger.debug(f"{prompt_id} already exists in IN_MEMORY_PROMPTS")
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
        initializer = GitLabPromptManager

        verbose_proxy_logger.debug(f"Using gitlab initializer for prompt_integration: {prompt_integration}-->{initializer}")
        if initializer:
            custom_prompt_callback = self.gitlab_prompt_cache.prompt_manager
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
        prompt_cb = self.prompt_id_to_custom_prompt.get(prompt_id, None)
        if prompt_cb is None and self.gitlab_prompt_cache:
            prompt_spec = self.IN_MEMORY_PROMPTS.get(prompt_id, None)
            if prompt_spec:
                self.initialize_prompt(prompt_spec)
                prompt_cb = self.prompt_id_to_custom_prompt.get(prompt_id, None)
        verbose_proxy_logger.debug(f"Found gitlab prompt_cb for {prompt_id} --> {prompt_cb}")
        return prompt_cb


GITLAB_PROMPT_REGISTRY = GitlabPromptRegistry()


class UnifiedPromptRegistry:
    """
    Aggregate multiple prompt registries behind one interface.

    Exposes:
      - IN_MEMORY_PROMPTS: aggregated view (precedence-aware) of PromptSpecs

    Registry contract (best-effort; we detect what exists):
      - .IN_MEMORY_PROMPTS: Dict[str, PromptSpec]     (preferred for fast indexing)
      - .get_prompt_by_id(prompt_id) -> Optional[PromptSpec]
      - .get_prompt_callback_by_id(prompt_id) -> Optional[CustomPromptManagement]
      - .initialize_prompt(prompt: PromptSpec, config_file_path: Optional[str] = None)
      - .load_all() -> None                           (optional)
    """

    def __init__(self) -> None:
        # Precedence-preserving container
        self._registries: "OrderedDict[str, object]" = OrderedDict()
        self._integration_to_registry: Dict[str, str] = {}
        # Aggregated, precedence-aware cache
        self.IN_MEMORY_PROMPTS: Dict[str, PromptSpec] = {}
        self.prompt_id_to_registry: "OrderedDict[str, object]" = OrderedDict()
        self.prompt_id_to_registry_name: "OrderedDict[str, object]" = OrderedDict()

    # -----------------------------
    # Wiring & setup
    # -----------------------------

    def register_registry(self, name: str, registry: object) -> None:
        """Add/replace a registry in search order, then rebuild the aggregate cache."""
        if name in self._registries:
            self._registries[name] = registry
        else:
            self._registries[name] = registry
        verbose_proxy_logger.debug("UnifiedPromptRegistry: registered %s", name)

        lname = name.lower()
        if "gitlab" in lname:
            self._integration_to_registry.setdefault("gitlab", name)
        if "memory" in lname or "in_memory" in lname:
            self._integration_to_registry.setdefault("in_memory", name)

        self._rebuild_cache()

    def set_integration_route(self, prompt_integration: str, registry_name: str) -> None:
        if registry_name not in self._registries:
            raise ValueError(f"Unknown registry '{registry_name}'")
        self._integration_to_registry[prompt_integration] = registry_name

    # -----------------------------
    # Bulk preload / listing
    # -----------------------------

    def load_all(self) -> None:
        """Ask each registry to preload if supported, then refresh the aggregate cache."""
        for name, reg in self._registries.items():
            if hasattr(reg, "load_all"):
                try:
                    reg.load_all()  # type: ignore[attr-defined]
                    verbose_proxy_logger.debug("UnifiedPromptRegistry: %s.load_all() OK", name)
                except Exception as e:
                    verbose_proxy_logger.debug("UnifiedPromptRegistry: %s.load_all() failed: %s", name, e)
        self._rebuild_cache()
        return self.IN_MEMORY_PROMPTS

    def list_prompt_ids(self) -> List[str]:
        """IDs from the aggregated cache (precedence already applied)."""
        return list(self.IN_MEMORY_PROMPTS.keys())

    def list_prompts(self) -> List[PromptSpec]:
        """PromptSpecs from the aggregated cache."""
        return list(self.IN_MEMORY_PROMPTS.values())

    # -----------------------------
    # Lookups
    # -----------------------------

    def get_prompt_by_id(self, prompt_id: str) -> Optional[PromptSpec]:
        """Fast path via aggregated cache; fallback to registries and update cache on hit."""
        spec = self.IN_MEMORY_PROMPTS.get(prompt_id)
        verbose_proxy_logger.debug(f"Found spec for {prompt_id} --> {spec}")
        if spec is not None:
            return spec

        # Fallback: search registries in order; on success, cache it
        for _, reg in self._registries.items():
            # Fast dict check
            im = getattr(reg, "IN_MEMORY_PROMPTS", None)
            if isinstance(im, dict) and prompt_id in im:
                self._cache_if_absent(prompt_id, im[prompt_id])
                return im[prompt_id]
            # Accessor
            if hasattr(reg, "get_prompt_by_id"):
                found = reg.get_prompt_by_id(prompt_id)  # type: ignore[attr-defined]
                if found is not None:
                    self._cache_if_absent(prompt_id, found)
                    return found
        return None

    def get_prompt_callback_by_id(self, prompt_id: str) -> Optional[CustomPromptManagement]:
        """Lookup callback by searching registries in precedence order."""
        registry = self.get_registry_from_prompt_id(prompt_id)
        registry_name = self.prompt_id_to_registry_name.get(prompt_id, "<unknown>")
        verbose_proxy_logger.debug(f"Found registry '{registry_name}' for prompt_id '{prompt_id}'")
        if registry and hasattr(registry, "get_prompt_callback_by_id"):

            cb = registry.get_prompt_callback_by_id(prompt_id)  # type: ignore[attr-defined]
            return cb

        for _, reg in self._registries.items():
            if hasattr(reg, "get_prompt_callback_by_id"):
                cb = reg.get_prompt_callback_by_id(prompt_id)  # type: ignore[attr-defined]
                if cb is not None:
                    return cb
        return None

    def find_with_origin(self, prompt_id: str) -> Optional[Tuple[str, PromptSpec]]:
        """Return (registry_name, PromptSpec) for the first match in precedence order."""
        # Try aggregate cache first; if present, identify origin with a pass through registries
        spec = self.IN_MEMORY_PROMPTS.get(prompt_id)
        if spec is not None:
            for name, reg in self._registries.items():
                im = getattr(reg, "IN_MEMORY_PROMPTS", None)
                if isinstance(im, dict) and prompt_id in im:
                    return name, im[prompt_id]
                if hasattr(reg, "get_prompt_by_id"):
                    found = reg.get_prompt_by_id(prompt_id)  # type: ignore[attr-defined]
                    if found is spec:
                        return name, found
            # Fallback: unknown origin but we do have the spec
            return "<aggregated>", spec

        # Not in cache: search registries
        for name, reg in self._registries.items():
            im = getattr(reg, "IN_MEMORY_PROMPTS", None)
            if isinstance(im, dict) and prompt_id in im:
                self._cache_if_absent(prompt_id, im[prompt_id])
                return name, im[prompt_id]
            if hasattr(reg, "get_prompt_by_id"):
                found = reg.get_prompt_by_id(prompt_id)  # type: ignore[attr-defined]
                if found is not None:
                    self._cache_if_absent(prompt_id, found)
                    return name, found
        return None

    # -----------------------------
    # Initialization routing
    # -----------------------------

    def initialize_prompt(self, prompt: PromptSpec, config_file_path: Optional[str] = None) -> Optional[PromptSpec]:
        """
        Route initialization to the correct underlying registry based on
        prompt.litellm_params.prompt_integration, then refresh cache for that id.
        """
        integration = None
        try:
            lp = prompt.litellm_params
            integration = getattr(lp, "prompt_integration", None)
            if integration is None and isinstance(lp, dict):
                integration = lp.get("prompt_integration")
        except Exception:
            pass

        target_name = self._integration_to_registry.get(str(integration).lower()) if integration else None
        target_registry = self._registries.get(target_name) if target_name else next(iter(self._registries.values()), None)

        if target_registry is None:
            raise RuntimeError("UnifiedPromptRegistry has no registered registries to initialize the prompt.")
        if not hasattr(target_registry, "initialize_prompt"):
            raise RuntimeError(f"Registry '{target_name or 'UNKNOWN'}' does not support initialize_prompt().")

        initialized = target_registry.initialize_prompt(prompt, config_file_path)  # type: ignore[attr-defined]

        # Update aggregate cache just for this id (avoid full rebuild)
        if initialized is not None:
            self._cache_replace(prompt.prompt_id, initialized)

        return initialized

    # -----------------------------
    # Cache management
    # -----------------------------

    def refresh(self) -> None:
        """Public method to rebuild the aggregate cache on demand."""
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """Rebuild aggregated IN_MEMORY_PROMPTS respecting registry precedence."""
        agg: Dict[str, PromptSpec] = {}
        prompt_id_to_registry = OrderedDict()
        prompt_id_to_registry_name = OrderedDict()
        seen: set[str] = set()

        for reg_name, reg in self._registries.items():
            # Prefer direct dict for speed
            im = getattr(reg, "IN_MEMORY_PROMPTS", None)
            if isinstance(im, dict):
                for pid, spec in im.items():
                    if pid not in seen:
                        agg[pid] = spec
                        seen.add(pid)
                continue

            # Fallback: if no dict, try to iterate via list of ids from accessor
            ids: Iterable[str] = []
            if hasattr(reg, "list_prompt_ids"):
                try:
                    ids = reg.list_prompt_ids()  # type: ignore[attr-defined]
                except Exception:
                    ids = []
            for pid in ids:
                if pid in seen:
                    continue
                spec = None
                if hasattr(reg, "get_prompt_by_id"):
                    spec = reg.get_prompt_by_id(pid)  # type: ignore[attr-defined]
                if spec is not None:
                    agg[pid] = spec
                    seen.add(pid)
                    prompt_id_to_registry[pid] = reg
                    prompt_id_to_registry_name[pid] = reg_name
                    self.pr = reg.get_prompt_callback_by_id(pid)  # type: ignore[attr-defined]


        self.IN_MEMORY_PROMPTS = agg
        self.prompt_id_to_registry = prompt_id_to_registry
        self.prompt_id_to_registry_name = prompt_id_to_registry_name

    def get_registry_from_prompt_id(self, prompt_id):
        return self.prompt_id_to_registry.get(prompt_id, None)

    def _cache_if_absent(self, prompt_id: str, spec: PromptSpec) -> None:
        """Add to aggregate cache if not present (without breaking precedence)."""
        if prompt_id not in self.IN_MEMORY_PROMPTS:
            self.IN_MEMORY_PROMPTS[prompt_id] = spec

    def _cache_replace(self, prompt_id: str, spec: PromptSpec) -> None:
        """
        Replace/insert a single PromptSpec in the aggregate cache.
        Safe after initializing via the correct backend.
        """
        self.IN_MEMORY_PROMPTS[prompt_id] = spec

PROMPT_HUB = UnifiedPromptRegistry()
PROMPT_HUB.register_registry("in_memory", IN_MEMORY_PROMPT_REGISTRY)
PROMPT_HUB.register_registry("gitlab", GITLAB_PROMPT_REGISTRY)
