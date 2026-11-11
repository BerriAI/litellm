"""
``RustRegistryMembers``: the Rust core's public registry members outside ``registry()`` and
``apply``: ``add_deployment``, ``set_model_list``, the ``model_list`` attribute, the environment
gate, the deprecated id hash, ``print_deployment`` and the pre-call checks the plugin context reads.

Each one works on the core's host registry (``litellm/router_rust/surface/registry.py``), which hands every
change to the Rust core. The rules are the Python router core's, ported (never imported).
"""

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.router_api import deployment_id
from litellm.router_rust.steps.precall import count_tokens, pre_call_checks
from litellm.router_rust.surface.registry import HostRegistry, model_info_is_active_for_environment
from litellm.types.router import Deployment

if TYPE_CHECKING:
    from litellm.router_python_host import PythonHost
    from litellm.router_rust.core import RustCore


class _CallCounter:
    """``total_calls``, ``success_calls`` or ``fail_calls``: plain attributes on the Python core,
    which a write replaces and a ``del`` removes; every call then reads what is there."""

    def __init__(self, name: str) -> None:
        self._name: Final = name

    def __get__(self, core: "RustCore | None", owner: type | None = None) -> object:
        if core is None:
            return self
        try:
            return getattr(core._endpoints.counters, self._name)  # pyright: ignore[reportPrivateUsage]  # the core's own counters
        except AttributeError:
            raise core._no_attribute(self._name) from None  # pyright: ignore[reportPrivateUsage]  # see above

    def __set__(self, core: "RustCore", value: object) -> None:
        setattr(core._endpoints.counters, self._name, value)  # pyright: ignore[reportPrivateUsage]  # see __get__

    def __delete__(self, core: "RustCore") -> None:
        try:
            delattr(core._endpoints.counters, self._name)  # pyright: ignore[reportPrivateUsage]  # see __get__
        except AttributeError:
            raise core._no_attribute(self._name) from None  # pyright: ignore[reportPrivateUsage]  # see __get__


class RustRegistryMembers:
    """Mixed into ``RustCore``, which supplies its host registry and host."""

    total_calls: Final = _CallCounter("total_calls")
    success_calls: Final = _CallCounter("success_calls")
    fail_calls: Final = _CallCounter("fail_calls")

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
        """Replace every deployment (``set_model_list``); aliases and wildcard routes stay."""
        self._registry_state.set_model_list(model_list)

    @property
    def model_list(self) -> Sequence[object]:
        """The live model list (the deprecated ``model_list`` attribute)."""
        return self._registry_state._core_model_list  # pyright: ignore[reportPrivateUsage]  # the core reads its own host registry

    def _assign_model_list(self, model_list: Sequence[object]) -> None:
        """``router.model_list = [...]``: the list is swapped as given, without reindexing, and the
        registry version moves once."""
        self._registry_state.replace_model_list(model_list)  # pyright: ignore[reportArgumentType]  # the list as assigned, as the core keeps it

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

    def _count_pre_call_check_tokens(
        self,
        messages: Sequence[Mapping[str, object]] | None,
        input: str | list[object] | None,  # mutable-ok: the request field, as the Python core takes it
        request_kwargs: Mapping[str, object] | None = None,
        tokenizer_model: str = "",
        deployments: Sequence[Mapping[str, object]] = (),
    ) -> int:
        """``_count_pre_call_check_tokens``, which context compaction counts with."""
        return count_tokens(self._python_host, messages, input, request_kwargs, tokenizer_model, deployments)

    def _pre_call_checks(
        self,
        model: str,
        healthy_deployments: Sequence[Mapping[str, object]],
        messages: object = None,
        input: object = None,
        request_kwargs: Mapping[str, object] | None = None,
        input_token_count: int | None = None,
        skip_inline_token_count: bool = False,
        request_deployments: Sequence[Mapping[str, object]] | None = None,
    ) -> Sequence[Mapping[str, object]]:
        """``_pre_call_checks``, which context compaction admits a client-managed request with."""
        return pre_call_checks(
            self._python_host,
            self._registry_state,  # pyright: ignore[reportArgumentType]  # the host registry serves the reads
            model,
            healthy_deployments,
            messages,
            input,
            request_kwargs,
            input_token_count,
            skip_inline_token_count,
            request_deployments,
        )

    def print_deployment(self, deployment: Mapping[str, object]) -> Mapping[str, object]:
        """A copy of the deployment, of the type it was given, with its api key masked
        (``print_deployment``). A ``Deployment`` and its params read and write like dicts."""
        try:
            printed: Final = copy.deepcopy(deployment)
            litellm_params: Final = printed["litellm_params"]
            if self._python_host.dynamic().redact_user_api_key_info:
                masker: Final = SensitiveDataMasker(visible_prefix=2, visible_suffix=0)
                printed["litellm_params"] = masker.mask_dict(litellm_params)  # pyright: ignore[reportArgumentType, reportIndexIssue]  # the deployment's params, written as the Python core writes them
            elif "api_key" in litellm_params:  # pyright: ignore[reportOperatorIssue]  # see the docstring
                litellm_params["api_key"] = litellm_params["api_key"][:2] + "*" * 10  # pyright: ignore[reportIndexIssue]  # see the docstring
        except Exception as e:
            verbose_router_logger.debug("Error occurred while printing deployment - %s", e)
            raise
        else:
            return printed
