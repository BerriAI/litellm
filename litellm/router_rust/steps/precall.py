"""
Host-side facts for the Rust core's ``enable_pre_call_checks`` filter.

The Rust core decides which deployments the pre-call checks drop and which error a group with
nothing left raises (``router-core/src/precall.rs``). What needs live Python objects is answered
here, on a ``pre_call`` select step: the prompt's token count (messages, Responses input and
instructions, tools and an Anthropic ``system`` block), each deployment's context window from
its router model info, and which requested params each deployment takes. Ported from the Python
router core's ``_pre_call_checks`` helpers; it never reaches a Python router core.

``pre_call_checks`` is the one decision made here: ``RoutingContext.pre_call_checks`` hands over
deployment records rather than a step, and those need not be in the Rust registry.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.deployment_pricing import deployment_backend_models
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
from litellm.litellm_core_utils.prompt_templates.tpm_media_utils import is_plain_embedding_text
from litellm.litellm_core_utils.token_counter import (
    offload_token_count,
    served_openai_models,
    single_encoding_openai_model,
    split_embedding_input_tokens,
)
from litellm.router_attempt import for_requested_model
from litellm.router_python_host import python_host
from litellm.router_strategy.complexity_router.context_compaction import compaction_pending
from litellm.types.router import LiteLLM_Params

if TYPE_CHECKING:
    from litellm.router_python_host import PythonHost
    from litellm.types.utils import LlmProviders

# The chat params a deployment must take when the request sends them (`_PRE_CALL_CHECK_FILTERED_PARAMS`).
_FILTERED_PARAMS: Final = frozenset({"response_format"})
_CHARS_PER_OFFLOADED_COUNT: Final = 1 << 16


@dataclass(frozen=True, slots=True)
class _TextBatch:
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TokenIdBatch:
    longest: int


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
    deployments: Sequence[Mapping[str, object]] = (),
    fits_within: int = 0,
) -> int:
    """``_count_pre_call_check_tokens``: messages, or Responses input normalized to messages, plus
    the tools and Anthropic ``system`` block the message list never carries. Embedding media an
    item of ``deployments`` reads as media is left out of the input: it never counts as text."""
    from litellm.llms.anthropic.messages.native.utils import anthropic_system_to_openai_message

    if messages is None and input is not None:
        return _input_tokens(host, _text_input(input, deployments), request_kwargs, tokenizer_model, fits_within)
    if messages is None:
        raise ValueError("Either messages or input must be provided to count tokens")
    extras: Final = _extras(request_kwargs)
    system_message: Final = anthropic_system_to_openai_message(extras.get("system"))
    counter: Final = host.invoke("token_counter")  # pyright: ignore[reportUnknownMemberType]  # the host's overloads name litellm's untyped returns
    counted_messages: Final = (system_message, *_items(messages)) if system_message is not None else messages
    return counter(model=tokenizer_model, messages=counted_messages, tools=_tools(extras))


def _input_tokens(
    host: "PythonHost",
    text_input: object,
    request_kwargs: Mapping[str, object] | None,
    tokenizer_model: str,
    fits_within: int,
) -> int:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    counter: Final = host.invoke("token_counter")  # pyright: ignore[reportUnknownMemberType]  # see count_tokens
    match _embedding_batch(text_input):
        case _TextBatch(texts):
            return _longest_text_tokens(counter, tokenizer_model, texts, fits_within)
        case _TokenIdBatch(longest):
            return _token_id_count(host, longest)
        case None:
            pass
    extras: Final = _extras(request_kwargs)
    raw_instructions: Final = extras.get("instructions")
    instructions: Final = raw_instructions if isinstance(raw_instructions, str) else None
    input_messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # its input type is untyped
        input=text_input,  # pyright: ignore[reportArgumentType]  # str | list, as the request carries it
        responses_api_request={"instructions": instructions} if instructions is not None else {},
        only_since_compaction=True,
    )
    return counter(model=tokenizer_model, messages=input_messages, tools=_tools(extras))


def _extras(request_kwargs: Mapping[str, object] | None) -> Mapping[str, object]:
    return request_kwargs if request_kwargs is not None else MappingProxyType({})


def _tools(extras: Mapping[str, object]) -> object:
    raw_tools: Final = extras.get("tools")
    return raw_tools if _is_list(raw_tools) and raw_tools else None


def _text_input(input: object, deployments: Sequence[Mapping[str, object]]) -> object:  # the request field's name
    text_input, _ = split_embedding_input_tokens(input, partial(deployment_backend_models, deployments))
    return text_input


def _embedding_batch(text_input: object) -> _TextBatch | _TokenIdBatch | None:
    items: Final = _items(text_input) if _is_list(text_input) else ()
    if not items:
        return None
    texts: Final = tuple(item for item in items if isinstance(item, str))
    if len(texts) == len(items):
        return _TextBatch(texts)
    if all(_is_token_id(item) for item in items):
        return _TokenIdBatch(len(items))
    if all(_is_token_id_list(item) for item in items):
        return _TokenIdBatch(max(len(_items(item)) for item in items))
    return None


def _longest_text_tokens(
    counter: Callable[..., int], tokenizer_model: str, texts: Sequence[str], fits_within: int
) -> int:
    return max(_text_tokens(counter, tokenizer_model, text, fits_within) for text in texts)


def _offload_slices(texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    ends: Final = accumulate(len(text) for text in texts)
    slices: Final = groupby(
        zip(ends, texts, strict=True), key=lambda end_and_text: (end_and_text[0] - 1) // _CHARS_PER_OFFLOADED_COUNT
    )
    return tuple(tuple(text for _, text in texts_in_slice) for _, texts_in_slice in slices)


def _text_tokens(counter: Callable[..., int], tokenizer_model: str, text: str, fits_within: int) -> int:
    size: Final = len(text.encode(errors="surrogatepass"))
    return size if size <= fits_within else counter(model=tokenizer_model, text=text)


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_token_id_list(value: object) -> bool:
    return _is_list(value) and all(_is_token_id(token) for token in _items(value))


def _token_id_count(host: "PythonHost", count: int) -> int:
    return 0 if host.dynamic().disable_token_counter else count


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
        # A wildcard deployment serves the requested model (`PatternMatchRouter.route`).
        named: Final = tuple(
            for_requested_model(record, self._name)
            for record in self._registry._core_deployments_named(self._model) or ()  # pyright: ignore[reportPrivateUsage]  # the host registry's index
        )
        return (
            single_encoding_openai_model(
                served_openai_models(named)
                or served_openai_models(for_requested_model(self._record(id), self._name) for id in self._candidates)
            )
            or ""
        )

    def _candidate_records(self) -> Sequence[Mapping[str, object]]:
        return tuple(self._record(id) for id in self._candidates)

    def _fits_within(self) -> int:
        return min((limit for limit in self._windows.values() if limit is not None), default=0)

    def _input_is_a_list(self) -> bool:
        return self._messages is None and _is_list(self._input)

    def _count(self) -> int | None:
        try:
            tokenizer_model: Final = self._tokenizer_model()
            if not self._input_is_a_list():
                return count_tokens(
                    self._host,
                    self._messages,
                    self._input,
                    self._request,
                    tokenizer_model,
                    self._candidate_records(),
                    self._fits_within(),
                )
            text_input: Final = _text_input(self._input, self._candidate_records())
            match _embedding_batch(text_input):
                case _TextBatch() if not tokenizer_model:
                    return 0
                case _:
                    return _input_tokens(self._host, text_input, self._request, tokenizer_model, self._fits_within())
        except Exception as e:  # noqa: BLE001  # best-effort: an uncountable prompt must not fail the request
            verbose_router_logger.error(
                "litellm.router.py::_acount_pre_call_check_tokens: failed to count tokens. Got - %s", e
            )
            return None

    async def _count_async(self) -> int | None:
        try:
            tokenizer_model: Final = self._tokenizer_model()
            if not self._input_is_a_list():
                return await offload_token_count(count_tokens)(
                    self._host,
                    self._messages,
                    self._input,
                    self._request,
                    tokenizer_model,
                    self._candidate_records(),
                    self._fits_within(),
                )
            text_input: Final = (
                self._input
                if all(map(is_plain_embedding_text, _items(self._input)))
                else await offload_token_count(_text_input)(self._input, self._candidate_records())
            )
            match _embedding_batch(text_input):
                case _TextBatch() if not tokenizer_model:
                    return 0
                case _TextBatch(texts):
                    counter: Final = self._host.invoke("token_counter")  # pyright: ignore[reportUnknownMemberType]  # see count_tokens
                    fits_within: Final = self._fits_within()
                    count_slice: Final = offload_token_count(_longest_text_tokens)
                    return max(
                        [
                            await count_slice(counter, tokenizer_model, texts_in_slice, fits_within)
                            for texts_in_slice in _offload_slices(texts)
                        ]
                    )
                case _:
                    return await offload_token_count(_input_tokens)(
                        self._host, text_input, self._request, tokenizer_model, self._fits_within()
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


def pre_call_checks(
    host: "PythonHost",
    registry: PreCallRegistry,
    model: str,
    deployments: Sequence[Mapping[str, object]],
    messages: object,
    input: object,  # the request field's name
    request_kwargs: Mapping[str, object] | None,
    input_token_count: int | None = None,
    skip_inline_token_count: bool = False,
    request_deployments: Sequence[Mapping[str, object]] | None = None,
) -> Sequence[Mapping[str, object]]:
    """``_pre_call_checks`` over the given deployments, as ``router-core/src/precall.rs`` decides it."""
    import litellm.exceptions as litellm_exceptions

    pool: Final = (
        tuple(request_deployments)
        if request_deployments is not None
        else (tuple(registry._core_deployments_named(model) or ()) or tuple(deployments))  # pyright: ignore[reportPrivateUsage]  # the host registry's index
    )
    candidates: Final = {f"candidate-{index}": record for index, record in enumerate(deployments)}
    group: Final = {f"pool-{index}": record for index, record in enumerate(pool)}
    records: Final = {**candidates, **group}
    step: Final = {"model_group": model, "model": model, "candidates": list(candidates), "pool": list(group)}
    facts: Final = PreCallFacts(host, registry, step, records.__getitem__, request_kwargs, messages, input)
    counted: Final = (
        input_token_count
        if input_token_count is not None or skip_inline_token_count or not facts._needs_count()  # pyright: ignore[reportPrivateUsage]  # this module's own class
        else facts._count()  # pyright: ignore[reportPrivateUsage]  # see above
    )
    answer: Final = facts._answer(counted)  # pyright: ignore[reportPrivateUsage]  # see above
    rows: Final = as_str_mapping(answer.get("deployments")) or {}
    allowed_region: Final = answer.get("allowed_model_region")

    def row(id: str) -> Mapping[str, object]:  # a candidate or pool key
        return as_str_mapping(rows.get(id)) or {}

    def over_window(id: str) -> bool:
        limit: Final = row(id).get("max_input_tokens")
        return isinstance(limit, int) and answer.get("countable") is True and counted is not None and counted > limit

    def out_of_region(id: str) -> bool:
        params: Final = as_str_mapping(records[id].get("litellm_params")) or {}
        return allowed_region is not None and params.get("region_name") != allowed_region

    too_small: Final = tuple(id for id in candidates if over_window(id))
    invalid: Final = frozenset(too_small) | frozenset(id for id in candidates if out_of_region(id))
    param_candidates: Final = tuple(id for id in candidates if id not in invalid)
    lacking: Final = tuple(row(id).get("lacks_params") is True for id in param_candidates)
    rejecting: Final = tuple(row(id).get("rejects_params") is True for id in param_candidates)
    failing: Final = rejecting if lacking and all(lacking) else lacking
    excluded: Final = invalid | frozenset(id for id, fails in zip(param_candidates, failing, strict=True) if fails)
    if len(excluded) == len(candidates):
        requested: Final = [str(param) for param in _items(answer.get("requested_params"))]
        if not too_small and group and all(row(id).get("rejects_params") is True for id in group):
            raise litellm_exceptions.UnsupportedParamsError(
                message=(
                    f"no deployment of model={model} supports {requested}. "
                    "To drop these, set `litellm.drop_params=True` or for proxy `litellm_settings: drop_params: true`, "
                    f"or send allowed_openai_params={requested} to pass them through"
                ),
                model=model,
            )
        if too_small:
            detail: Final = "".join(
                f"Model={_error_model(records[id])}, Max Input Tokens={row(id).get('max_input_tokens')}, Got={counted}"
                for id in too_small
            )
            raise litellm_exceptions.ContextWindowExceededError(
                message="litellm._pre_call_checks: Context Window exceeded for given call. No models have context "
                f"window large enough for this call.\n{detail}",
                model=model,
                llm_provider="",
            )
    return tuple(record for id, record in candidates.items() if id not in excluded)


def _error_model(deployment: Mapping[str, object]) -> object:
    """The name a context-window error gives a deployment: its base model, else its model."""
    params: Final = as_str_mapping(deployment.get("litellm_params")) or {}
    model_info: Final = as_str_mapping(deployment.get("model_info")) or {}
    return model_info.get("base_model") or params.get("base_model") or params.get("model")


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


__all__ = [
    "PreCallFacts",
    "PreCallRegistry",
    "count_tokens",
    "declares_no_vision",
    "pre_call_checks",
    "request_carries_image",
]


def _route_capability_resolvers() -> Mapping[str, Callable[[str, "LlmProviders"], object | None]]:
    """``_ROUTE_CAPABILITY_RESOLVERS``: the calls only some providers serve, by handler name."""
    from litellm.utils import ProviderConfigManager

    return {
        "adecisions": ProviderConfigManager.get_provider_decisions_config,  # pyright: ignore[reportUnknownMemberType]  # litellm's config lookup
        "decisions": ProviderConfigManager.get_provider_decisions_config,  # pyright: ignore[reportUnknownMemberType]  # see above
    }


# ``_DECISIONS_MEDIA_CALL_TYPES``: the calls whose audio or videos some deployments refuse.
DECISIONS_MEDIA_CALL_TYPES: Final = frozenset({"adecisions", "decisions", "asystemone", "systemone"})


def serves_route_capability(handler: str | None) -> bool:
    """Whether only some providers serve the call ``handler`` names."""
    return handler is not None and handler in _route_capability_resolvers()


def cannot_serve_route(deployment: Mapping[str, object], group: str, handler: str) -> bool:
    """Not ``_deployment_serves_route``: the deployment's provider has no config for the call."""
    from litellm.types.utils import LlmProviders

    resolver: Final = _route_capability_resolvers().get(handler)
    params: Final = as_str_mapping(deployment.get("litellm_params"))
    if resolver is None or params is None:
        return False
    try:
        model, provider_name = _capability_model_and_provider(params, group)
        provider: Final = next((member for member in LlmProviders if member.value == provider_name), None)
        return provider is None or resolver(model, provider) is None
    except Exception as e:  # noqa: BLE001  # best-effort capability filter, as the Python core keeps it
        verbose_router_logger.debug(
            "litellm.router.py::_deployment_serves_route: keeping deployment for model=%s. Got - %s", group, e
        )
        return False


def refuses_decisions_media(
    deployment: Mapping[str, object], group: str, sent_kwargs: Mapping[str, object], *, audio: bool, videos: bool
) -> bool:
    """``_decisions_deployment_refuses_media``."""
    from litellm.decisions.call import deployment_refuses_media

    model_info: Final = as_str_mapping(deployment.get("model_info")) or {}
    params: Final = as_str_mapping(deployment.get("litellm_params")) or {}
    try:
        model, custom_llm_provider = _capability_model_and_provider(params, group)
        return deployment_refuses_media(
            model, custom_llm_provider, model_info, {**params, **sent_kwargs}, audio=audio, videos=videos
        )
    except Exception as e:  # noqa: BLE001  # best-effort filter: a deployment it cannot resolve answers for itself
        verbose_router_logger.debug(
            "litellm.router.py::_decisions_deployment_refuses_media: keeping a deployment of model=%s. Got - %s",
            group,
            e,
        )
        return False
