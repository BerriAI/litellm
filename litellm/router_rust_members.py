"""
``RustRegistryMembers``: the Rust core's public registry members outside ``registry()`` and
``apply``: ``add_deployment``, ``set_model_list``, the ``model_list`` attribute, the environment
gate, the deprecated id hash and ``print_deployment``.

Each one works on the core's host registry (``litellm/router_rust_registry.py``), which hands every
change to the Rust core. The rules are the Python router core's, ported (never imported).
"""

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.router_api import deployment_id
from litellm.router_rust_registry import HostRegistry, model_info_is_active_for_environment
from litellm.types.router import Deployment

if TYPE_CHECKING:
    from litellm.router_python_host import PythonHost


class RustRegistryMembers:
    """Mixed into ``RustCore``, which supplies its host registry and host."""

    @property
    def _registry_state(self) -> HostRegistry:
        raise NotImplementedError

    @property
    def _python_host(self) -> "PythonHost":
        raise NotImplementedError

    def add_deployment(self, deployment: Deployment) -> Deployment | None:
        """Add a deployment unless its id exists (``add_deployment``)."""
        return self._registry_state.add_deployment(deployment)

    def set_model_list(self, model_list: Sequence[object]) -> None:
        """Replace every deployment (``set_model_list``) with its wildcard routes and names; aliases stay."""
        self._registry_state.set_model_list(model_list)

    @property
    def model_list(self) -> Sequence[object]:
        """The live model list (the deprecated ``model_list`` attribute)."""
        return self._registry_state._core_model_list  # pyright: ignore[reportPrivateUsage]  # the core reads its own host registry

    def _assign_model_list(self, model_list: Sequence[object]) -> None:
        """``router.model_list = [...]``: the list is swapped as given and reindexed, and the
        registry version moves once."""
        self._registry_state.assign_model_list(model_list)

    @staticmethod
    def deployment_is_active_for_environment(deployment: Deployment) -> bool:
        """Whether LITELLM_ENVIRONMENT admits the deployment (``deployment_is_active_for_environment``)."""
        return model_info_is_active_for_environment(model_info=deployment.model_info)  # pyright: ignore[reportArgumentType]  # ModelInfo reads like a mapping

    @staticmethod
    def generate_model_id(model_group: str, litellm_params: Mapping[str, object]) -> str:
        """The deprecated id hash (``generate_model_id``)."""
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the router's deprecation notice
        )

        _note("generate_model_id", "litellm.router_api.deployment_id(model_group, litellm_params)")
        return deployment_id(model_group, litellm_params)

    async def set_response_headers(
        self, response: object, model_group: str | None = None, request_kwargs: Mapping[str, object] | None = None
    ) -> object:
        """Stamp the group's remaining usage and the model id on a response (``set_response_headers``)."""
        headed: Final[object] = await self._registry_state.set_response_headers(  # pyright: ignore[reportAny]  # the response as the core hands it back
            response, model_group, None if request_kwargs is None else dict(request_kwargs)
        )
        return headed

    def replay_model_cost_registrations(self) -> None:
        """Re-register the served deployments' cost-map entries (``replay_model_cost_registrations``)."""
        self._registry_state.replay_model_cost_registrations()

    def print_deployment(self, deployment: Mapping[str, object]) -> Mapping[str, object]:
        """A copy of the deployment with its api key masked (``print_deployment``)."""
        try:
            printed: Final = dict(copy.deepcopy(deployment))
            litellm_params: Final = printed["litellm_params"]
            if self._python_host.dynamic().redact_user_api_key_info:
                masker: Final = SensitiveDataMasker(visible_prefix=2, visible_suffix=0)
                printed["litellm_params"] = masker.mask_dict(litellm_params)  # pyright: ignore[reportArgumentType]  # the deployment's params dict
            elif isinstance(litellm_params, dict) and "api_key" in litellm_params:
                litellm_params["api_key"] = litellm_params["api_key"][:2] + "*" * 10
        except Exception as e:
            verbose_router_logger.debug("Error occurred while printing deployment - %s", e)
            raise
        else:
            return printed
