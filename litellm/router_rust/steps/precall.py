"""
Host-side facts for the Rust core's ``enable_pre_call_checks`` filter.

The Rust core decides which deployments the pre-call checks drop and which error a group with
nothing left raises (``router-core/src/precall/mod.rs``). What needs live Python objects is answered
here, as read and with no rule applied. A ``pre_call`` select step asks for the prompt's token count
(messages, Responses input and instructions, tools and an Anthropic ``system`` block), each
deployment's context window from its router model info, the live ``litellm.drop_params`` and how the
request sends the params the step names. A ``param_support`` step asks which of its params each
deployment does not support. Ported from the Python router core's ``_pre_call_checks`` helpers; it
never reaches a Python router core.

``pre_call_checks`` serves ``RoutingContext.pre_call_checks``, which hands over deployment records
rather than a step. Those need not be in the Rust registry, so it runs the same Rust filter over the
records and answers its two questions with the same facts.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, chain, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.deployment_pricing import deployment_backend_models
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import parts_of
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
from litellm.types.decisions import (
    DecisionsIRMessages,
    DecisionsIRState,
    OpenAIDecisionInputJSON,
    OpenAIDecisionInputMessage,
)
from litellm.types.router import LiteLLM_Params

if TYPE_CHECKING:
    from litellm.router_python_host import PythonHost
    from litellm.types.utils import LlmProviders

_CALL_SETTINGS: Final = ("drop_params", "allowed_openai_params", "additional_drop_params")
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
        responses_api_request={"instructions": instructions, "tools": extras.get("tools")},
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


def _sent_as(value: object) -> str:
    """How a request sends a param, as ``router-core/src/precall/params.rs`` names it."""
    if value is None:
        return "null"
    return "text" if isinstance(value, str) else "structured"


def unsupported_params(
    step: Mapping[str, object], records: Callable[[str], Mapping[str, object]]
) -> Mapping[str, Sequence[str]]:
    """The answer of a ``param_support`` step: which of its params each candidate and pool deployment does not support."""
    group: Final = str(step.get("model_group"))
    params: Final = tuple(str(param) for param in _items(step.get("params")))
    ids: Final = dict.fromkeys(str(id) for id in (*_items(step.get("candidates")), *_items(step.get("pool"))))
    return {id: [param for param in params if not _supports_param(records(id), group, param)] for id in ids}


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
        self._params: Final = tuple(str(param) for param in _items(step.get("params")))
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
        return {
            "has_input": True,
            "countable": not compaction_pending(self._request),
            "input_tokens": input_tokens,
            "allowed_model_region": request.get("allowed_model_region"),
            "drop_params": self._host.dynamic().drop_params,
            "sent_params": {param: _sent_as(request[param]) for param in self._params if param in request},
            "call_settings": {key: request[key] for key in _CALL_SETTINGS if key in request},
            "deployments": {
                id: {"max_input_tokens": self._windows.get(id)}
                for id in dict.fromkeys([*self._candidates, *self._pool])
            },
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
    """``_pre_call_checks`` over the given deployments, decided by ``router-core/src/precall/mod.rs``."""
    from litellm.rust_bridge._native import RouterCoreHandle

    pool: Final = (
        tuple(request_deployments)
        if request_deployments is not None
        else (tuple(registry._core_deployments_named(model) or ()) or tuple(deployments))  # pyright: ignore[reportPrivateUsage]  # the host registry's index
    )
    candidates: Final = {f"candidate-{index}": record for index, record in enumerate(deployments)}
    group: Final = {f"pool-{index}": record for index, record in enumerate(pool)}
    records: Final = {**candidates, **group}
    step: Final = {"model_group": model, "model": model, "candidates": list(candidates), "pool": list(group)}

    def facts(params: Sequence[str]) -> Mapping[str, object]:
        found: Final = PreCallFacts(
            host, registry, {**step, "params": list(params)}, records.__getitem__, request_kwargs, messages, input
        )
        counted: Final = (
            input_token_count
            if input_token_count is not None or skip_inline_token_count or not found._needs_count()  # pyright: ignore[reportPrivateUsage]  # this module's own class
            else found._count()  # pyright: ignore[reportPrivateUsage]  # see above
        )
        return found._answer(counted)  # pyright: ignore[reportPrivateUsage]  # see above

    def unsupported(params: Sequence[str]) -> Mapping[str, Sequence[str]]:
        return unsupported_params({**step, "params": list(params)}, records.__getitem__)

    outcome: Final = RouterCoreHandle.pre_call_filter(
        model, list(candidates.items()), list(group.items()), facts, unsupported
    )
    error: Final = pre_call_error(outcome, model)
    if error is not None:
        raise error
    kept: Final = frozenset(_items(outcome.get("kept")))
    return tuple(record for id, record in candidates.items() if id in kept)


def pre_call_error(failed: Mapping[str, object], model: str) -> BaseException | None:
    """The error the pre-call checks raise for a group with nothing left, when ``failed`` names one."""
    import litellm.exceptions as litellm_exceptions

    kind: Final = failed.get("error_kind")
    error_model: Final = str(failed.get("error_model") or model)
    if kind == "context_window_exceeded":
        return litellm_exceptions.ContextWindowExceededError(
            message="litellm._pre_call_checks: Context Window exceeded for given call. No models have context "
            f"window large enough for this call.\n{failed.get('error_detail')}",
            model=error_model,
            llm_provider="",
        )
    if kind == "unsupported_params":
        params: Final = sorted(str(param) for param in _items(failed.get("error_params")))
        return litellm_exceptions.UnsupportedParamsError(
            message=(
                f"no deployment of model={error_model} supports {params}. "
                "To drop these, set `litellm.drop_params=True` or for proxy `litellm_settings: drop_params: true`, "
                f"or send allowed_openai_params={params} to pass them through"
            ),
            model=error_model,
        )
    return None


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


# ``_DECISIONS_MEDIA_CALL_TYPES``: the calls whose audio, videos or JSON input some deployments refuse.
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


def _is_json_input_part(part: object) -> bool:
    match part:
        case {"type": "input_json"} | OpenAIDecisionInputJSON():
            return True
        case _:
            return False


def _content_parts(message: object) -> Sequence[object]:
    if isinstance(message, OpenAIDecisionInputMessage):
        return () if isinstance(message.content, str) else message.content
    return parts_of((as_str_mapping(message) or {}).get("content"))


def request_carries_json_input(request_input: object) -> bool:
    """A Decisions ``input`` holding an ``input_json`` part, as sent or as validated messages."""
    messages: Final = parts_of(request_input)
    return any(map(_is_json_input_part, chain.from_iterable(map(_content_parts, messages))))


def refuses_decisions_media(
    deployment: Mapping[str, object],
    group: str,
    sent_kwargs: Mapping[str, object],
    *,
    audio: bool,
    videos: bool,
    decision_input: DecisionsIRState | DecisionsIRMessages | None,
) -> bool:
    """``_decisions_deployment_refuses_media``, JSON input parts included."""
    from litellm.decisions.call import deployment_refuses_media

    model_info: Final = as_str_mapping(deployment.get("model_info")) or {}
    params: Final = as_str_mapping(deployment.get("litellm_params")) or {}
    try:
        model, custom_llm_provider = _capability_model_and_provider(params, group)
        return deployment_refuses_media(
            model,
            custom_llm_provider,
            model_info,
            {**params, **sent_kwargs},
            audio=audio,
            videos=videos,
            decision_input=decision_input,
        )
    except Exception as e:  # noqa: BLE001  # best-effort filter: a deployment it cannot resolve answers for itself
        verbose_router_logger.debug(
            "litellm.router.py::_decisions_deployment_refuses_media: keeping a deployment of model=%s. Got - %s",
            group,
            e,
        )
        return False
