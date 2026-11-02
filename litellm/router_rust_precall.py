"""
Host-side facts for the Rust core's ``enable_pre_call_checks`` filter.

The Rust core decides which deployments the pre-call checks drop and which error a group with
nothing left raises (``router-core/src/precall.rs``). What needs live Python objects is answered
here, on a ``pre_call`` select step: the prompt's token count (messages, Responses input and
instructions, tools and an Anthropic ``system`` block), each deployment's context window from
its router model info, and which requested params each deployment takes. Ported from the Python
router core's ``_pre_call_checks`` helpers; it never reaches a Python router core.
"""

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
from litellm.litellm_core_utils.token_counter import (
    offload_token_count,
    served_openai_models,
    single_encoding_openai_model,
)
from litellm.router_python_host import python_host
from litellm.router_strategy.complexity_router.context_compaction import compaction_pending
from litellm.types.router import LiteLLM_Params

if TYPE_CHECKING:
    from litellm.router_python_host import PythonHost

# The chat params a deployment must take when the request sends them (`_PRE_CALL_CHECK_FILTERED_PARAMS`).
_FILTERED_PARAMS: Final = frozenset({"response_format"})


class PreCallRegistry(Protocol):
    """The host registry reads the facts need."""

    def _core_get_model_info(self, id: str) -> Mapping[str, object] | None: ...  # the registry's name

    def _core_deployments_named(self, model_name: str) -> Sequence[Mapping[str, object]] | None: ...

    def _core_get_router_model_info(
        self, deployment: Mapping[str, object], received_model_name: str
    ) -> Mapping[str, object]: ...


def count_tokens(
    host: "PythonHost",
    messages: object,
    input: object,  # the request field's name
    request_kwargs: Mapping[str, object] | None,
    tokenizer_model: str,
) -> int:
    """``_count_pre_call_check_tokens``: messages, or Responses input normalized to messages, plus
    the tools and Anthropic ``system`` block the message list never carries."""
    from litellm.llms.anthropic.messages.native.utils import anthropic_system_to_openai_message

    extras: Final[Mapping[str, object]] = request_kwargs if request_kwargs is not None else MappingProxyType({})
    raw_instructions: Final = extras.get("instructions")
    instructions: Final = raw_instructions if isinstance(raw_instructions, str) else None
    raw_tools: Final = extras.get("tools")
    tools: Final = raw_tools if _is_list(raw_tools) and raw_tools else None
    system_message: Final = anthropic_system_to_openai_message(extras.get("system"))
    counter: Final = host.invoke("token_counter")  # pyright: ignore[reportUnknownMemberType]  # the host's overloads name litellm's untyped returns
    if messages is not None:
        counted_messages: Final = (system_message, *_items(messages)) if system_message is not None else messages
        return counter(model=tokenizer_model, messages=counted_messages, tools=tools)
    if input is not None:
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        input_messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # its input type is untyped
            input=input,  # pyright: ignore[reportArgumentType]  # str | list, as the request carries it
            responses_api_request={"instructions": instructions} if instructions is not None else {},
            only_since_compaction=True,
        )
        return counter(model=tokenizer_model, messages=input_messages, tools=tools)
    raise ValueError("Either messages or input must be provided to count tokens")


def _max_input_tokens(registry: PreCallRegistry, model: str, deployment: Mapping[str, object]) -> int | None:
    """``_deployment_max_input_tokens``: the declared context window, or None when there is none or
    the deployment cannot be resolved."""
    try:
        model_info: Final = registry._core_get_router_model_info(  # pyright: ignore[reportPrivateUsage]  # the host registry's lookup
            deployment=deployment, received_model_name=model
        )
    except Exception as e:  # noqa: BLE001  # best-effort: an unmappable deployment must not hide the others
        verbose_router_logger.debug("litellm.router.py::_deployment_max_input_tokens: skipping deployment. Got - %s", e)
        return None
    max_input_tokens: Final = model_info.get("max_input_tokens")
    return max_input_tokens if isinstance(max_input_tokens, int) else None


def _items(value: object) -> tuple[object, ...]:
    """The items of a list, tuple or set; nothing for any other value."""
    return tuple(value) if isinstance(value, list | tuple | set | frozenset) else ()  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # the items are read as objects


def _is_list(value: object) -> bool:
    return isinstance(value, list)


def _declared_param_allowlist(params: Mapping[str, object]) -> frozenset[str]:
    declared: Final = params.get("allowed_openai_params")
    return frozenset(entry for entry in _items(declared) if isinstance(entry, str))


def _capability_model_and_provider(deployment_params: Mapping[str, object], group: str) -> tuple[str, str]:
    model_label: Final = str(deployment_params.get("model") or group)
    custom_llm_provider: Final = deployment_params.get("custom_llm_provider")
    authenticating_provider: Final = declared_authenticating_provider(
        model_label,
        custom_llm_provider,  # pyright: ignore[reportArgumentType]  # passed as configured, as the Python core does
    )
    if authenticating_provider is not None:
        return model_label.removeprefix(f"{authenticating_provider}/"), authenticating_provider
    model, provider, _, _ = python_host.invoke("get_llm_provider")(  # pyright: ignore[reportUnknownMemberType]  # see count_tokens
        model=model_label,
        custom_llm_provider=custom_llm_provider,
        litellm_params=LiteLLM_Params.model_validate(dict(deployment_params)),
    )
    return model, provider


def _supports_param(deployment: Mapping[str, object], group: str, param: str) -> bool:
    """``_deployment_supports_param``."""
    deployment_params: Final = as_str_mapping(deployment.get("litellm_params"))
    if not deployment_params:
        return True
    deployment_model_info: Final = as_str_mapping(deployment.get("model_info"))
    base_model: Final = (
        deployment_model_info.get("base_model") if deployment_model_info else None
    ) or deployment_params.get("base_model")
    try:
        model, custom_llm_provider = _capability_model_and_provider(deployment_params, group)
        supported: Final = python_host.invoke("get_supported_openai_params")(  # pyright: ignore[reportUnknownMemberType]  # see count_tokens
            model=model,
            custom_llm_provider=custom_llm_provider,
            base_model=base_model if isinstance(base_model, str) else None,
        )
    except Exception as e:  # noqa: BLE001  # best-effort filter: an unresolvable provider must not narrow the request
        verbose_router_logger.debug(
            "litellm.router.py::_deployment_supports_param: keeping %s for model=%s. Got - %s", param, group, e
        )
        return True
    return supported is None or param in supported


def _lacks_params(deployment: Mapping[str, object], group: str, params: frozenset[str]) -> bool:
    return any(not _supports_param(deployment, group, param) for param in params)


def _rejects_params(
    deployment: Mapping[str, object], group: str, params: frozenset[str], request_kwargs: Mapping[str, object]
) -> bool:
    """``_deployment_rejects_params``: lacks a param neither the deployment nor the request drops or allows."""
    call_settings: Final[Mapping[str, object]] = {
        **(as_str_mapping(deployment.get("litellm_params")) or {}),
        **request_kwargs,
    }
    if call_settings.get("drop_params") is True:
        return False
    dropped: Final = call_settings.get("additional_drop_params")
    dropped_names: Final = frozenset(
        entry for entry in (_items(dropped) if _is_list(dropped) else ()) if isinstance(entry, str)
    )
    return _lacks_params(deployment, group, params - _declared_param_allowlist(call_settings) - dropped_names)


def _requested_params(request_kwargs: Mapping[str, object]) -> frozenset[str]:
    """``_pre_call_check_requested_chat_params``."""
    from litellm import utils as litellm_utils

    return _filtered_params(
        litellm_utils.get_non_default_params(passed_params=dict(request_kwargs))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # an untyped helper
    )


def _filtered_params(requested: Mapping[object, object]) -> frozenset[str]:
    return frozenset(
        param
        for param, value in requested.items()
        if isinstance(param, str) and param in _FILTERED_PARAMS and not isinstance(value, str)
    )


class PreCallFacts:
    """The facts of one ``pre_call`` step."""

    def __init__(
        self,
        host: "PythonHost",
        registry: PreCallRegistry,
        step: Mapping[str, object],
        records: Callable[[str], Mapping[str, object]],
        request: Mapping[str, object] | None,
        messages: object,
        input: object,  # the request field's name
    ) -> None:
        self._host: Final = host
        self._registry: Final = registry
        self._model: Final = str(step.get("model_group"))
        self._name: Final = str(step.get("model"))
        self._candidates: Final = [str(id) for id in _items(step.get("candidates"))]
        self._pool: Final = [str(id) for id in _items(step.get("pool"))]
        self._records: Final = records
        self._request: Final = request
        self._messages: Final = messages
        self._input: Final = input
        self._has_input: Final = messages is not None or input is not None
        self._windows: Final = {id: _max_input_tokens(registry, self._model, records(id)) for id in self._candidates}

    def _record(self, id: str) -> Mapping[str, object]:  # a deployment id
        return self._records(id)

    def _needs_count(self) -> bool:
        return self._has_input and any(limit is not None for limit in self._windows.values())

    def _tokenizer_model(self) -> str:
        named: Final = self._registry._core_deployments_named(self._model) or ()  # pyright: ignore[reportPrivateUsage]  # the host registry's index
        return (
            single_encoding_openai_model(
                served_openai_models(named) or served_openai_models(self._record(id) for id in self._candidates)
            )
            or ""
        )

    def _count(self) -> int | None:
        try:
            return count_tokens(self._host, self._messages, self._input, self._request, self._tokenizer_model())
        except Exception as e:  # noqa: BLE001  # best-effort: an uncountable prompt must not fail the request
            verbose_router_logger.error(
                "litellm.router.py::_acount_pre_call_check_tokens: failed to count tokens. Got - %s", e
            )
            return None

    async def _count_async(self) -> int | None:
        try:
            return await offload_token_count(count_tokens)(
                self._host, self._messages, self._input, self._request, self._tokenizer_model()
            )
        except Exception as e:  # noqa: BLE001  # best-effort: an uncountable prompt must not fail the request
            verbose_router_logger.error(
                "litellm.router.py::_acount_pre_call_check_tokens: failed to count tokens. Got - %s", e
            )
            return None

    def _answer(self, input_tokens: int | None) -> Mapping[str, object]:
        if not self._has_input:
            return {"has_input": False}
        request: Final[Mapping[str, object]] = self._request if self._request is not None else {}
        params: Final = (
            _requested_params(request)
            if self._request is not None and self._host.dynamic().drop_params is False
            else frozenset[str]()
        )
        deployments: Final[dict[str, dict[str, object]]] = {}  # mutable-ok: built here, then returned
        for id in dict.fromkeys([*self._candidates, *self._pool]):
            record = self._record(id)
            deployments[id] = {
                "max_input_tokens": self._windows.get(id),
                "lacks_params": bool(params) and id in self._candidates and _lacks_params(record, self._model, params),
                "rejects_params": bool(params) and _rejects_params(record, self._model, params, request),
            }
        allowed_region: Final = request.get("allowed_model_region")
        return {
            "has_input": True,
            "countable": not compaction_pending(self._request),
            "input_tokens": input_tokens,
            "allowed_model_region": allowed_region,
            "requested_params": sorted(params),
            "deployments": deployments,
        }

    def answer(self) -> Mapping[str, object]:
        """The facts on the sync path: the prompt is counted inline."""
        return self._answer(self._count() if self._needs_count() else None)

    async def answer_async(self) -> Mapping[str, object]:
        """The facts on the async path: the prompt is counted off the event loop."""
        return self._answer(await self._count_async() if self._needs_count() else None)


def request_carries_image(messages: object, request: Mapping[str, object]) -> bool:
    """``_request_carries_image``: the messages (or the request's own messages or input) hold an image."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import request_contains_image_content

    return request_contains_image_content(messages or request.get("messages") or request.get("input"))


def declares_no_vision(deployment: Mapping[str, object], group: str) -> bool:
    """``_deployment_declares_no_vision``: ``model_info.supports_vision`` is ``False``, or, when it
    is unset, the cost map says the model has vision explicitly disabled."""
    from litellm import utils as litellm_utils

    model_info: Final = as_str_mapping(deployment.get("model_info")) or {}
    declared: Final = model_info.get("supports_vision")
    if declared is not None:
        return declared is False
    params: Final = as_str_mapping(deployment.get("litellm_params")) or {}
    if not params:
        return False
    base_model: Final = model_info.get("base_model") or params.get("base_model")
    try:
        model, custom_llm_provider = _capability_model_and_provider(params, group)
        return litellm_utils.is_vision_explicitly_disabled(
            base_model if isinstance(base_model, str) else model, custom_llm_provider
        )
    except Exception as e:  # noqa: BLE001  # best-effort filter: an unresolvable provider must not narrow the request
        verbose_router_logger.debug(
            "litellm.router.py::_deployment_declares_no_vision: keeping deployment for model=%s. Got - %s", group, e
        )
        return False


__all__ = ["PreCallFacts", "PreCallRegistry", "count_tokens", "declares_no_vision", "request_carries_image"]
