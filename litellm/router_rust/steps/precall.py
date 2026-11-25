"""
Host-side facts for the Rust core's ``enable_pre_call_checks`` filter.

The Rust core decides which deployments the pre-call checks drop and which error a group with
nothing left raises (``router-core/src/precall/mod.rs``). What needs live Python objects is answered
here, on a ``pre_call`` select step: the prompt's token count (messages, Responses input and
instructions, tools and an Anthropic ``system`` block), each deployment's context window from
its router model info, and which requested params each deployment takes. Ported from the Python
router core's ``_pre_call_checks`` helpers; it never reaches a Python router core.

``pre_call_checks`` is the one decision made here: ``RoutingContext.pre_call_checks`` hands over
deployment records rather than a step, and those need not be in the Rust registry.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import accumulate, chain, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_router_logger
from litellm.constants import PDF_DATA_URL_PREFIX
from litellm.litellm_core_utils.core_helpers import allowed_openai_param_names, as_str_mapping
from litellm.litellm_core_utils.deployment_pricing import deployment_backend_models
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import parts_of
from litellm.litellm_core_utils.prompt_templates.tpm_media_utils import is_plain_embedding_text
from litellm.litellm_core_utils.token_counter import (
    MessageCountFraming,
    deployment_openai_model,
    inline_pdf_token_deltas,
    message_count_framing,
    offload_token_count,
    openai_tokenizer_encoding_name,
    split_embedding_input_tokens,
    token_counter,
)
from litellm.router_attempt import for_requested_model
from litellm.router_python_host import python_host
from litellm.router_strategy.complexity_router.context_compaction import compaction_pending
from litellm.rust_bridge.tokenizer import get_encoding
from litellm.types.decisions import (
    DecisionsIRMessages,
    DecisionsIRState,
    OpenAIDecisionInputJSON,
    OpenAIDecisionInputMessage,
)
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import SelectTokenizerResponse

if TYPE_CHECKING:
    from litellm.router_python_host import PythonHost
    from litellm.types.utils import LlmProviders

# The chat params a deployment must take when the request sends them (`_PRE_CALL_CHECK_FILTERED_PARAMS`).
_FILTERED_PARAMS: Final = frozenset({"response_format"})
_CHARS_PER_OFFLOADED_COUNT: Final = 1 << 16
_STAND_IN_ENCODING: Final = "o200k_base"
STAND_IN_OVERCOUNT_MARGIN: Final = 2


@dataclass(frozen=True, slots=True)
class _Tokenizer:
    model: str
    custom: SelectTokenizerResponse | None


@dataclass(frozen=True, slots=True)
class _TokenCountKey:
    encoding: str
    framing: MessageCountFraming


@dataclass(frozen=True, slots=True)
class _DeploymentCount:
    tokens: int
    estimated: bool


def stand_in_tokenizer() -> SelectTokenizerResponse:
    return {"type": "openai_tokenizer", "tokenizer": get_encoding(_STAND_IN_ENCODING)}


def _tokenizer(openai_model: str | None) -> _Tokenizer:
    if openai_model is not None:
        return _Tokenizer(openai_model, None)
    return _Tokenizer("", stand_in_tokenizer())


def _token_count_key(openai_model: str | None) -> _TokenCountKey:
    return _TokenCountKey(
        _STAND_IN_ENCODING if openai_model is None else openai_tokenizer_encoding_name(openai_model),
        message_count_framing(openai_model or ""),
    )


def _counter(host: "PythonHost", tokenizer: _Tokenizer) -> Callable[..., int]:
    return partial(
        host.invoke("token_counter"),  # pyright: ignore[reportUnknownMemberType]  # the host's overloads name litellm's untyped returns
        model=tokenizer.model,
        custom_tokenizer=tokenizer.custom,
    )


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
    return _count_tokens(
        host,
        _counter(host, _Tokenizer(tokenizer_model, None)),
        messages,
        input,
        request_kwargs,
        deployments,
        fits_within,
    )


def _count_tokens(
    host: "PythonHost",
    counter: Callable[..., int],
    messages: object,
    input: object,  # the request field's name
    request_kwargs: Mapping[str, object] | None,
    deployments: Sequence[Mapping[str, object]],
    fits_within: int,
) -> int:
    from litellm.llms.anthropic.messages.native.utils import anthropic_system_to_openai_message

    if messages is None and input is not None:
        return _input_tokens(host, counter, _text_input(input, deployments), request_kwargs, fits_within)
    if messages is None:
        raise ValueError("Either messages or input must be provided to count tokens")
    extras: Final = _extras(request_kwargs)
    system_message: Final = anthropic_system_to_openai_message(extras.get("system"))
    counted_messages: Final = (system_message, *_items(messages)) if system_message is not None else messages
    return counter(messages=counted_messages, tools=_tools(extras))


def _input_tokens(
    host: "PythonHost",
    counter: Callable[..., int],
    text_input: object,
    request_kwargs: Mapping[str, object] | None,
    fits_within: int,
) -> int:
    match _embedding_batch(text_input):
        case _TextBatch(texts):
            return _longest_text_tokens(counter, texts, fits_within)
        case _TokenIdBatch(longest):
            return _token_id_count(host, longest)
        case None:
            pass
    return counter(messages=_responses_messages(text_input, request_kwargs), tools=_tools(_extras(request_kwargs)))


def _responses_messages(text_input: object, request_kwargs: Mapping[str, object] | None) -> object:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    extras: Final = _extras(request_kwargs)
    raw_instructions: Final = extras.get("instructions")
    instructions: Final = raw_instructions if isinstance(raw_instructions, str) else None
    return LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # its input type is untyped
        input=text_input,  # pyright: ignore[reportArgumentType]  # str | list, as the request carries it
        responses_api_request={"instructions": instructions, "tools": extras.get("tools")},
        only_since_compaction=True,
    )


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


def _longest_text_tokens(counter: Callable[..., int], texts: Sequence[str], fits_within: int) -> int:
    return max(_text_tokens(counter, text, fits_within) for text in texts)


def _offload_slices(texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    ends: Final = accumulate(len(text) for text in texts)
    slices: Final = groupby(
        zip(ends, texts, strict=True), key=lambda end_and_text: (end_and_text[0] - 1) // _CHARS_PER_OFFLOADED_COUNT
    )
    return tuple(tuple(text for _, text in texts_in_slice) for _, texts_in_slice in slices)


def _text_tokens(counter: Callable[..., int], text: str, fits_within: int) -> int:
    size: Final = len(text.encode(errors="surrogatepass"))
    return size if size <= fits_within else counter(text=text)


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_token_id_list(value: object) -> bool:
    return _is_list(value) and all(_is_token_id(token) for token in _items(value))


def _token_id_count(host: "PythonHost", count: int) -> int:
    return 0 if host.dynamic().disable_token_counter else count


def _router_model_info(registry: PreCallRegistry, model: str, deployment: Mapping[str, object]) -> Mapping[str, object]:
    """``_deployment_max_input_tokens``'s lookup: the deployment's router model info, or none when
    the deployment cannot be resolved."""
    try:
        return registry._core_get_router_model_info(  # pyright: ignore[reportPrivateUsage]  # the host registry's lookup
            deployment=deployment, received_model_name=model
        )
    except Exception as e:  # noqa: BLE001  # best-effort: an unmappable deployment must not hide the others
        verbose_router_logger.debug("litellm.router.py::_deployment_max_input_tokens: skipping deployment. Got - %s", e)
        return MappingProxyType({})


def _max_input_tokens(model_info: Mapping[str, object]) -> int | None:
    max_input_tokens: Final = model_info.get("max_input_tokens")
    return max_input_tokens if isinstance(max_input_tokens, int) else None


def _caps_below_catalog(
    registry: PreCallRegistry, model: str, deployment: Mapping[str, object], window: int | None
) -> bool:
    model_info: Final = as_str_mapping(deployment.get("model_info")) or {}
    if window is None or "max_input_tokens" not in model_info:
        return False
    uncapped: Final = {
        **deployment,
        "model_info": {key: value for key, value in model_info.items() if key != "max_input_tokens"},
    }
    catalog_window: Final = _max_input_tokens(_router_model_info(registry, model, uncapped))
    return catalog_window is not None and window < catalog_window


_SILENTLY_TRUNCATING_PROVIDERS: Final = frozenset({"ollama", "ollama_chat"})


def _truncates_silently(deployment: Mapping[str, object]) -> bool:
    params: Final = as_str_mapping(deployment.get("litellm_params")) or {}
    provider: Final = params.get("custom_llm_provider") or str(params.get("model") or "").partition("/")[0]
    return provider in _SILENTLY_TRUNCATING_PROVIDERS


def _items(value: object) -> tuple[object, ...]:
    """The items of a list, tuple or set; nothing for any other value."""
    return tuple(value) if isinstance(value, list | tuple | set | frozenset) else ()  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # the items are read as objects


def _is_list(value: object) -> bool:
    return isinstance(value, list)


def _declared_param_allowlist(params: Mapping[str, object]) -> frozenset[str]:
    return frozenset(allowed_openai_param_names(params.get("allowed_openai_params")))


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
        self._model: Final = str(step.get("model_group"))
        self._name: Final = str(step.get("model"))
        self._candidates: Final = [str(id) for id in _items(step.get("candidates"))]
        self._pool: Final = [str(id) for id in _items(step.get("pool"))]
        self._records: Final = records
        self._request: Final = request
        self._messages: Final = messages
        self._input: Final = input
        self._has_input: Final = messages is not None or input is not None
        self._registry: Final = registry
        self._model_infos: Final = {
            id: _router_model_info(registry, self._model, records(id)) for id in self._candidates
        }
        self._windows: Final = {id: _max_input_tokens(info) for id, info in self._model_infos.items()}

    def _record(self, id: str) -> Mapping[str, object]:  # a deployment id
        return self._records(id)

    def _window(self, deployment_id: str) -> int | None:
        if deployment_id in self._windows:
            return self._windows[deployment_id]
        return _max_input_tokens(_router_model_info(self._registry, self._model, self._record(deployment_id)))

    def _needs_count(self) -> bool:
        return self._has_input and any(limit is not None for limit in self._windows.values())

    def _openai_models(self) -> Mapping[str, str | None]:
        # A wildcard deployment serves the requested model (`PatternMatchRouter.route`).
        return {
            id: deployment_openai_model(for_requested_model(self._record(id), self._name))
            for id, limit in self._windows.items()
            if limit is not None
        }

    def _window_is_exact(self, id: str) -> bool:  # a deployment id
        record: Final = self._record(id)
        return (
            self._model_infos[id].get("mode") == "embedding"
            or _truncates_silently(record)
            or _caps_below_catalog(self._registry, self._model, record, self._windows[id])
        )

    def _estimated(self, models: Mapping[str, str | None]) -> frozenset[str]:
        stand_ins: Final = (id for id, model in models.items() if model is None)
        return frozenset(id for id in stand_ins if not self._window_is_exact(id))

    def _candidate_records(self) -> Sequence[Mapping[str, object]]:
        return tuple(self._record(id) for id in self._candidates)

    def _fits_within(self) -> int:
        return min((limit for limit in self._windows.values() if limit is not None), default=0)

    def _input_is_a_list(self) -> bool:
        return self._messages is None and _is_list(self._input)

    def _count(self, openai_model: str | None) -> int | None:
        try:
            counter: Final = _counter(self._host, _tokenizer(openai_model))
            if not self._input_is_a_list():
                return _count_tokens(
                    self._host,
                    counter,
                    self._messages,
                    self._input,
                    self._request,
                    self._candidate_records(),
                    self._fits_within(),
                )
            text_input: Final = _text_input(self._input, self._candidate_records())
            match _embedding_batch(text_input):
                case _TextBatch() if openai_model is None:
                    return 0
                case _:
                    return _input_tokens(self._host, counter, text_input, self._request, self._fits_within())
        except Exception as e:  # noqa: BLE001  # best-effort: an uncountable prompt must not fail the request
            verbose_router_logger.error(
                "litellm.router.py::_acount_pre_call_check_tokens: failed to count tokens. Got - %s", e
            )
            return None

    async def _count_async(self, openai_model: str | None) -> int | None:
        try:
            counter: Final = _counter(self._host, _tokenizer(openai_model))
            if not self._input_is_a_list():
                return await offload_token_count(_count_tokens)(
                    self._host,
                    counter,
                    self._messages,
                    self._input,
                    self._request,
                    self._candidate_records(),
                    self._fits_within(),
                )
            text_input: Final = (
                self._input
                if all(map(is_plain_embedding_text, _items(self._input)))
                else await offload_token_count(_text_input)(self._input, self._candidate_records())
            )
            match _embedding_batch(text_input):
                case _TextBatch() if openai_model is None:
                    return 0
                case _TextBatch(texts):
                    fits_within: Final = self._fits_within()
                    count_slice: Final = offload_token_count(_longest_text_tokens)
                    return max(
                        [
                            await count_slice(counter, texts_in_slice, fits_within)
                            for texts_in_slice in _offload_slices(texts)
                        ]
                    )
                case _:
                    return await offload_token_count(_input_tokens)(
                        self._host, counter, text_input, self._request, self._fits_within()
                    )
        except Exception as e:  # noqa: BLE001  # best-effort: an uncountable prompt must not fail the request
            verbose_router_logger.error(
                "litellm.router.py::_acount_pre_call_check_tokens: failed to count tokens. Got - %s", e
            )
            return None

    def _pdf_repriced(self, models: Mapping[str, str | None], counts: Mapping[str, _DeploymentCount]) -> frozenset[str]:
        """The stand-in counts over their window for a deployment that bills an inline PDF like one image. Only
        litellm's own counter prices a PDF the way the repricing assumes, so a custom one keeps its count."""
        if not counts or compaction_pending(self._request) or not _counts_with_litellm(self._host):
            return frozenset()
        if not _may_carry_an_inline_pdf(self._messages, self._input):
            return frozenset()
        overflowing: Final = (
            id for id, count in counts.items() if models[id] is None and _overflows(self._windows[id], count.tokens)
        )
        return frozenset(
            id
            for id in overflowing
            if _bills_pdfs_like_gemini(for_requested_model(self._record(id), self._name), self._model)
        )

    def _counted_messages(self) -> object:
        """The messages ``_count`` priced: the request's own, or its Responses input normalized the same way."""
        if self._messages is not None:
            return self._messages
        if self._input is None:
            return ()
        text_input: Final = _text_input(self._input, self._candidate_records())
        return () if _embedding_batch(text_input) is not None else _responses_messages(text_input, self._request)

    def _stand_in_single_image_pdf_delta(self) -> int | None:
        try:
            deltas: Final = inline_pdf_token_deltas(
                "", self._counted_messages(), ("single_image",), custom_tokenizer=stand_in_tokenizer()
            )
            return deltas["single_image"]
        except Exception as e:  # noqa: BLE001  # best-effort: an unpriceable PDF keeps the stand-in count
            verbose_router_logger.debug("litellm.router.py::_pre_call_checks: PDF repricing failed. Got - %s", e)
            return None

    def _counts(self) -> Mapping[str, _DeploymentCount]:
        models: Final = self._openai_models()
        keys: Final = {id: _token_count_key(model) for id, model in models.items()}
        counted: Final = {key: self._count(model) for key, model in _model_by_key(models, keys).items()}
        counts: Final = _deployment_counts(keys, counted, self._estimated(models))
        repriced: Final = self._pdf_repriced(models, counts)
        return _repriced(counts, repriced, self._stand_in_single_image_pdf_delta() if repriced else None)

    async def _counts_async(self) -> Mapping[str, _DeploymentCount]:
        models: Final = self._openai_models()
        keys: Final = {id: _token_count_key(model) for id, model in models.items()}
        counted: Final = {key: await self._count_async(model) for key, model in _model_by_key(models, keys).items()}
        counts: Final = _deployment_counts(keys, counted, self._estimated(models))
        repriced: Final = self._pdf_repriced(models, counts)
        delta: Final = await offload_token_count(self._stand_in_single_image_pdf_delta)() if repriced else None
        return _repriced(counts, repriced, delta)

    def _answer(self, counts: Mapping[str, _DeploymentCount]) -> Mapping[str, object]:
        if not self._has_input:
            return {"has_input": False}
        request: Final[Mapping[str, object]] = self._request if self._request is not None else {}
        params: Final = (
            _requested_params(request)
            if self._request is not None and self._host.dynamic().drop_params is False
            else frozenset[str]()
        )
        deployments: Final = {
            id: self._deployment_facts(id, counts.get(id), params, request)
            for id in dict.fromkeys([*self._candidates, *self._pool])
        }
        allowed_region: Final = request.get("allowed_model_region")
        return {
            "has_input": True,
            "countable": not compaction_pending(self._request),
            "allowed_model_region": allowed_region,
            "requested_params": sorted(params),
            "deployments": deployments,
        }

    def _deployment_facts(
        self, id: str, count: _DeploymentCount | None, params: frozenset[str], request: Mapping[str, object]
    ) -> Mapping[str, object]:
        record: Final = self._record(id)
        return {
            "max_input_tokens": self._window(id),
            "input_tokens": count.tokens if count is not None else None,
            "estimated": count is not None and count.estimated,
            "lacks_params": bool(params) and id in self._candidates and _lacks_params(record, self._model, params),
            "rejects_params": bool(params) and _rejects_params(record, self._model, params, request),
        }

    def answer(self) -> Mapping[str, object]:
        """The facts on the sync path: the prompt is counted inline."""
        return self._answer(self._counts() if self._needs_count() else {})

    async def answer_async(self) -> Mapping[str, object]:
        """The facts on the async path: the prompt is counted off the event loop."""
        return self._answer(await self._counts_async() if self._needs_count() else {})


def _model_by_key(
    models: Mapping[str, str | None], keys: Mapping[str, _TokenCountKey]
) -> Mapping[_TokenCountKey, str | None]:
    return {keys[id]: model for id, model in models.items()}


def _deployment_counts(
    keys: Mapping[str, _TokenCountKey], counted: Mapping[_TokenCountKey, int | None], estimated: frozenset[str]
) -> Mapping[str, _DeploymentCount]:
    return {
        id: _DeploymentCount(tokens, estimated=id in estimated)
        for id, key in keys.items()
        if (tokens := counted[key]) is not None
    }


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
    """``_pre_call_checks`` over the given deployments, as ``router-core/src/precall/mod.rs`` decides it."""
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
    counts: Final[Mapping[str, _DeploymentCount]] = (
        {id: _DeploymentCount(input_token_count, estimated=False) for id in candidates}
        if input_token_count is not None
        else {}
        if skip_inline_token_count or not facts._needs_count()  # pyright: ignore[reportPrivateUsage]  # this module's own class
        else facts._counts()  # pyright: ignore[reportPrivateUsage]  # see above
    )
    answer: Final = facts._answer(counts)  # pyright: ignore[reportPrivateUsage]  # see above
    rows: Final = as_str_mapping(answer.get("deployments")) or {}
    allowed_region: Final = answer.get("allowed_model_region")

    def row(id: str) -> Mapping[str, object]:  # a candidate or pool key
        return as_str_mapping(rows.get(id)) or {}

    def fit(id: str) -> _Fit:
        if answer.get("countable") is not True:
            return _Fit.FITS
        return _fit(row(id).get("max_input_tokens"), counts.get(id))

    def out_of_region(id: str) -> bool:
        params: Final = as_str_mapping(records[id].get("litellm_params")) or {}
        return allowed_region is not None and params.get("region_name") != allowed_region

    fits: Final = {id: fit(id) for id in candidates}
    too_small: Final = tuple(id for id in candidates if fits[id] is _Fit.EXCEEDS)
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
                f"Model={_error_model(records[id])}, Max Input Tokens={row(id).get('max_input_tokens')}, "
                f"Got={counts[id].tokens}"
                for id in too_small
            )
            raise litellm_exceptions.ContextWindowExceededError(
                message="litellm._pre_call_checks: Context Window exceeded for given call. No models have context "
                f"window large enough for this call.\n{detail}",
                model=model,
                llm_provider="",
            )
    kept: Final = tuple(id for id in candidates if id not in excluded)
    fitting: Final = tuple(id for id in kept if fits[id] is not _Fit.UNCERTAIN)
    return tuple(candidates[id] for id in fitting or kept)


class _Fit(Enum):
    FITS = "fits"
    UNCERTAIN = "uncertain"
    EXCEEDS = "exceeds"


def _fit(limit: object, count: _DeploymentCount | None) -> _Fit:
    if not isinstance(limit, int) or count is None or count.tokens <= limit:
        return _Fit.FITS
    if count.estimated and count.tokens <= limit * STAND_IN_OVERCOUNT_MARGIN:
        return _Fit.UNCERTAIN
    return _Fit.EXCEEDS


def _overflows(limit: int | None, input_tokens: int) -> bool:
    return limit is not None and input_tokens > limit


def _counts_with_litellm(host: "PythonHost") -> bool:
    """The host counts with litellm's own counter, the public one and the core function it calls, so repricing the
    inline PDFs it read changes the count exactly by the PDFs. A core function replaced before this module was
    imported cannot be told apart and is assumed to price PDFs as litellm's does."""
    import litellm
    from litellm import utils as litellm_utils
    from litellm._lazy_imports import get_token_counter_new
    from litellm.router_python_host import PythonHost

    public: Final = (as_str_mapping(vars(litellm)) or {}).get("token_counter")
    original: Final = (as_str_mapping(vars(litellm_utils)) or {}).get("token_counter")
    return type(host).invoke is PythonHost.invoke and public is original and get_token_counter_new() is token_counter


def _is_inline_pdf_part(value: object) -> bool:
    match value:
        case (
            {"type": "file", "file": {"file_data": str() as data}}
            | {"type": "input_file", "file_data": str() as data}
            | {"type": "document", "source": {"type": "url", "url": str() as data}}
        ):
            return data.startswith(PDF_DATA_URL_PREFIX)
        case {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": str() as data}}:
            return bool(data)
        case _:
            return False


def _pdf_content_children(value: object) -> tuple[object, ...]:
    fields: Final = as_str_mapping(value)
    if fields is None:
        return _items(value)
    output: Final = (
        fields.get("output") if fields.get("type") in ("function_call_output", "custom_tool_call_output") else None
    )
    source: Final = as_str_mapping(fields.get("source")) or {}
    nested: Final = (
        source.get("content") if fields.get("type") == "document" and source.get("type") == "content" else None
    )
    content: Final = fields.get("content")
    contents: Final = (content,) if as_str_mapping(content) is not None else _items(content)
    return (*contents, *_items(output), *_items(nested))


def _may_carry_an_inline_pdf(*values: object) -> bool:
    pending: Final[list[object]] = list(values)  # mutable-ok: content traversal without recursion
    visited: Final[set[int]] = set()  # mutable-ok: prevent cycles in caller-owned content containers
    while pending:
        (value,) = (pending.pop(),)
        if id(value) in visited:
            continue
        visited.add(id(value))
        if _is_inline_pdf_part(value):
            return True
        pending.extend(_pdf_content_children(value))
    return False


def _repriced(
    counts: Mapping[str, _DeploymentCount], repriced: frozenset[str], delta: int | None
) -> Mapping[str, _DeploymentCount]:
    if delta is None or delta >= 0:
        return counts
    return {
        id: _DeploymentCount(count.tokens + delta, count.estimated) if id in repriced else count
        for id, count in counts.items()
    }


def _bills_pdfs_like_gemini(deployment: Mapping[str, object], group: str) -> bool:
    from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute, get_vertex_ai_model_route

    params: Final = as_str_mapping(deployment.get("litellm_params"))
    if not params:
        return False
    try:
        model, provider = _capability_model_and_provider(params, group)
    except Exception as e:  # noqa: BLE001  # best-effort: an unresolvable deployment keeps its count
        verbose_router_logger.debug("litellm.router.py::_pre_call_checks: no provider for deployment. Got - %s", e)
        return False
    match provider:
        case "gemini":
            return True
        case "vertex_ai" | "vertex_ai_beta":
            return get_vertex_ai_model_route(model, params) is VertexAIModelRoute.GEMINI
        case _:
            return False


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
    "STAND_IN_OVERCOUNT_MARGIN",
    "PreCallFacts",
    "PreCallRegistry",
    "count_tokens",
    "declares_no_vision",
    "pre_call_checks",
    "request_carries_image",
    "stand_in_tokenizer",
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
