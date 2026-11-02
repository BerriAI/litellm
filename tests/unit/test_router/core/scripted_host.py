"""
A host for driving ``PythonCore`` in isolation.

``PythonCore`` reaches outside itself only through its host (``litellm/router_python_host.py``):
inference calls, backoff sleeps, the process settings, callback registration and the process-wide
cache. ``ScriptedHost`` serves all of those from the test. Inference ops answer from a per-model
script and record every call; sleeps are recorded and return at once; the process settings are a
fixed ``RouterDynamicConfig`` the test can replace; callbacks live in the host's own lists, and an
answered call runs the success or failure callbacks the core registered, as litellm's logging does.
Ops that are not inference (``get_model_info``, ``should_retry``, ...) are litellm's own functions.
"""

import dataclasses
import itertools
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from types import MethodType
from typing import Final

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager
from litellm.router_config import RouterDynamicConfig, build_router_config
from litellm.router_core import RouterCore
from litellm.router_python_core import PythonCore
from litellm.router_python_host import CallbackList, Endpoint, PythonHost
from litellm.types.utils import EmbeddingResponse, ModelResponse

INFERENCE_OPS: Final = frozenset(
    {
        "acompletion",
        "completion",
        "atext_completion",
        "text_completion",
        "aembedding",
        "embedding",
        "aimage_generation",
        "image_generation",
        "aspeech",
        "atranscription",
        "arerank",
        "rerank",
        "amoderation",
        "aresponses",
        "responses",
        "anthropic_messages",
        "agenerate_content",
        "aadapter_generate_content",
        "aocr",
        "asearch",
        "avideo_generation",
        "acreate_batch",
        "acreate_file",
        "afile_content",
        "afile_delete",
        "acreate_fine_tuning_job",
        "aget_responses",
        "adelete_responses",
        "acancel_responses",
        "alist_input_items",
    }
)

Outcome = object
"""What a scripted call answers: a ``BaseException`` instance is raised, a callable is called with
the request's keyword arguments (and its result used in turn), anything else is returned as is.
``None`` builds a default response for the op."""


@dataclasses.dataclass(frozen=True, slots=True)
class Call:
    op: str
    model: str
    kwargs: Mapping[str, object]

    @property
    def deployment_id(self) -> str | None:
        model_info: Final = self.kwargs.get("model_info")
        return str(model_info["id"]) if isinstance(model_info, Mapping) and "id" in model_info else None

    @property
    def model_group(self) -> str | None:
        metadata: Final = self.kwargs.get("metadata") or self.kwargs.get("litellm_metadata")
        return metadata.get("model_group") if isinstance(metadata, Mapping) else None


def completion_response(content: str = "ok", total_tokens: int = 10, model: str = "gpt-4o") -> ModelResponse:
    return ModelResponse(
        model=model,
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        usage={
            "prompt_tokens": total_tokens // 2,
            "completion_tokens": total_tokens - total_tokens // 2,
            "total_tokens": total_tokens,
        },
    )


def embedding_response(model: str = "text-embedding-3-small") -> EmbeddingResponse:
    return EmbeddingResponse(
        model=model,
        data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        usage={"prompt_tokens": 2, "total_tokens": 2},
    )


def _fresh_error(template: BaseException) -> BaseException:
    """A copy of ``template`` with the same attributes (``copy.copy`` rebuilds litellm errors from
    their args and drops ``status_code`` and the rest)."""
    error: Final = BaseException.__new__(type(template))
    error.args = template.args
    error.__dict__.update(template.__dict__)
    return error


def _default_response(op: str, model: str) -> object:
    if op in {"aembedding", "embedding"}:
        return embedding_response(model)
    return completion_response(model=model)


def rate_limit(model: str = "gpt-4o", message: str = "slow down") -> litellm.RateLimitError:
    return litellm.RateLimitError(message=message, llm_provider="openai", model=model)


def bad_request(model: str = "gpt-4o", message: str = "bad input") -> litellm.BadRequestError:
    return litellm.BadRequestError(message=message, llm_provider="openai", model=model)


def context_window(model: str = "gpt-4o") -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(message="too long", llm_provider="openai", model=model)


def content_policy(model: str = "gpt-4o") -> litellm.ContentPolicyViolationError:
    return litellm.ContentPolicyViolationError(message="blocked", llm_provider="openai", model=model)


def server_error(model: str = "gpt-4o") -> litellm.InternalServerError:
    return litellm.InternalServerError(message="boom", llm_provider="openai", model=model)


def timeout(model: str = "gpt-4o") -> litellm.Timeout:
    return litellm.Timeout(message="timed out", llm_provider="openai", model=model)


def auth_error(model: str = "gpt-4o") -> litellm.AuthenticationError:
    return litellm.AuthenticationError(message="bad key", llm_provider="openai", model=model)


def dynamic_config(**changes: object) -> RouterDynamicConfig:  # kwargs-ok: RouterDynamicConfig fields
    """Process settings as a fresh litellm process has them, with no callbacks, cache or fallbacks."""
    base: Final = RouterDynamicConfig(
        version=1,
        num_retries=None,
        max_fallbacks=None,
        router_max_fallbacks=5,
        allowed_fails=3,
        request_timeout=6000.0,
        fallbacks=None,
        default_fallbacks=None,
        context_window_fallbacks=None,
        content_policy_fallbacks=None,
        drop_params=None,
        provider_list=litellm.provider_list,
        expose_router_debug_in_errors=False,
        redact_user_api_key_info=None,
        model_alias_map={},
        model_cost=litellm.model_cost,
        known_custom_logger_compatible_callbacks=litellm._known_custom_logger_compatible_callbacks,
        exception_types=litellm.LITELLM_EXCEPTION_TYPES,
        callbacks=[],
        success_callback=[],
        failure_callback=[],
        input_callback=[],
        callback_manager=litellm.logging_callback_manager,
        cache=None,
    )
    return dataclasses.replace(base, **changes)


class HostCallbacks:
    """The callback registration the core asks its host for, over the host's own lists."""

    def __init__(self, host: "ScriptedHost") -> None:
        self._host: Final = host
        self._keys: Final = LoggingCallbackManager()

    def _add(self, name: CallbackList, callback: object) -> None:
        target: Final = self._host.lists[name]
        if callback not in target:
            target.append(callback)

    def add_litellm_callback(self, callback: object) -> None:
        self._add_to("callbacks", callback)

    def _add_to(self, name: str, callback: object) -> None:
        target: Final = self._host.lists.setdefault(name, [])
        if callback not in target:
            target.append(callback)

    def add_litellm_success_callback(self, callback: object) -> None:
        self._add("success_callback", callback)

    def add_litellm_async_success_callback(self, callback: object) -> None:
        self._add("_async_success_callback", callback)

    def add_litellm_failure_callback(self, callback: object) -> None:
        self._add("failure_callback", callback)

    def add_litellm_async_failure_callback(self, callback: object) -> None:
        self._add("_async_failure_callback", callback)

    def add_litellm_input_callback(self, callback: object) -> None:
        self._add("input_callback", callback)

    def get_custom_logger_key(self, custom_logger: CustomLogger) -> str:
        return self._keys.get_custom_logger_key(custom_logger)

    def with_custom_logger_keys(
        self, candidates: Iterable[CustomLogger], keys: frozenset[str]
    ) -> tuple[CustomLogger, ...]:
        return self._keys.with_custom_logger_keys(candidates, keys)

    def without_custom_logger_keys(self, callbacks: Iterable[object], keys: frozenset[str]) -> Iterator[object]:
        return self._keys.without_custom_logger_keys(callbacks, keys)  # pyright: ignore[reportArgumentType]  # any callback list

    def get_all_callbacks(self) -> list[object]:  # mutable-ok: matches LoggingCallbackManager
        return [callback for callbacks in self._host.lists.values() for callback in callbacks]

    def get_custom_loggers_for_type(
        self, callback_type: type[CustomLogger]
    ) -> list[CustomLogger]:  # mutable-ok: matches LoggingCallbackManager
        found: Final[list[CustomLogger]] = []
        for callback in self.get_all_callbacks():
            if isinstance(callback, callback_type) and callback not in found:
                found.append(callback)
        return found

    def drop_callbacks(self, callbacks: Iterable[object], method_owners: Iterable[object] = ()) -> frozenset[str]:
        dropped_ids: Final = frozenset(map(id, callbacks))
        owner_ids: Final = frozenset(map(id, method_owners))

        def keep(callback: object) -> bool:
            if id(callback) in dropped_ids:
                return False
            return not (isinstance(callback, MethodType) and id(callback.__self__) in owner_ids)

        vacated: Final = frozenset(
            self.get_custom_logger_key(callback)
            for callback in self._host.lists.get("callbacks", [])
            if isinstance(callback, CustomLogger) and id(callback) in dropped_ids
        )
        for name, callbacks_in_list in self._host.lists.items():
            self._host.lists[name] = [callback for callback in callbacks_in_list if keep(callback)]
        return vacated


class ScriptedHost(PythonHost):
    """
    ``script`` maps a deployment's ``litellm_params.model`` (or ``"*"`` for any) to the outcomes its
    calls answer in turn; the last outcome repeats. A model with no script answers a default response.
    """

    __slots__ = ("_dynamic", "_scripts", "_turns", "cache", "calls", "lists", "raised", "sleeps", "suppressed")

    def __init__(
        self, script: Mapping[str, Iterable[Outcome]] | None = None, **dynamic: object
    ) -> None:  # kwargs-ok: RouterDynamicConfig fields
        self._scripts: dict[str, tuple[Outcome, ...]] = {k: tuple(v) for k, v in (script or {}).items()}
        self._turns: dict[str, Iterator[int]] = {}
        self._dynamic: RouterDynamicConfig = dynamic_config(**dynamic)
        self.calls: Final[list[Call]] = []
        self.sleeps: Final[list[float]] = []
        # The errors the scripted calls raised, in order: each is a fresh copy of its scripted template.
        self.raised: Final[list[BaseException]] = []
        self.lists: Final[dict[str, list[object]]] = {
            "callbacks": [],
            "success_callback": [],
            "_async_success_callback": [],
            "failure_callback": [],
            "_async_failure_callback": [],
            "input_callback": [],
        }
        self.cache: object = None
        self.suppressed = False

    # --- what the test controls -------------------------------------------------------------

    def answer(self, model: str, *outcomes: Outcome) -> None:
        """Script the calls to ``model`` from now on."""
        self._scripts[model] = outcomes
        self._turns.pop(model, None)

    def set_dynamic(self, **changes: object) -> None:  # kwargs-ok: RouterDynamicConfig fields
        self._dynamic = dataclasses.replace(self._dynamic, **changes, version=self._dynamic.version + 1)

    def calls_to(self, op: str | None = None) -> list[Call]:  # mutable-ok: a fresh list
        return [call for call in self.calls if op is None or call.op == op]

    def deployments_called(self, op: str | None = None) -> list[str | None]:  # mutable-ok: a fresh list
        return [call.deployment_id for call in self.calls_to(op)]

    # --- the host operations ----------------------------------------------------------------

    def invoke(self, op: str) -> Endpoint:
        if op not in INFERENCE_OPS:
            return super().invoke(op)
        if op.startswith("a") or op in {"anthropic_messages", "_arealtime"}:

            async def endpoint(*args: object, **kwargs: object) -> object:  # kwargs-ok: the endpoint's request
                return await self._answer_async(op, args, kwargs)

            return endpoint

        def sync_endpoint(*args: object, **kwargs: object) -> object:  # kwargs-ok: the endpoint's request
            return self._answer_sync(op, args, kwargs)

        return sync_endpoint

    def sleep(self, seconds: float) -> Awaitable[None]:
        self.sleeps.append(seconds)

        async def done() -> None:
            return None

        return done()

    def dynamic(self) -> RouterDynamicConfig:
        return self._dynamic

    @property
    def callbacks(self) -> HostCallbacks:  # pyright: ignore[reportIncompatibleVariableOverride]  # the core uses the same methods
        return HostCallbacks(self)

    def callback_list(self, name: CallbackList) -> list[object]:  # mutable-ok: the host's own list
        return self.lists.setdefault(name, [])

    def set_callback_list(self, name: CallbackList, value: list[object]) -> None:  # mutable-ok: assigned as given
        self.lists[name] = value

    def global_cache(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # any cache object
        return self.cache

    def set_global_cache(self, cache: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # any cache object
        self.cache = cache

    def suppress_debug_info(self) -> None:
        self.suppressed = True

    # --- answering ------------------------------------------------------------------------

    def _outcome(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> tuple[Call, object]:
        model: Final = str(kwargs.get("model") or (args[0] if args else ""))
        call: Final = Call(op=op, model=model, kwargs=dict(kwargs))
        self.calls.append(call)
        key: Final = model if model in self._scripts else "*" if "*" in self._scripts else None
        if key is None:
            return call, None
        outcomes: Final = self._scripts[key]
        turn: Final = next(self._turns.setdefault(key, itertools.count()))
        outcome: object = outcomes[min(turn, len(outcomes) - 1)] if outcomes else None
        while callable(outcome) and not isinstance(outcome, BaseException | type):
            outcome = outcome(**kwargs)  # rebind-ok: a callable outcome answers with what it returns
        if isinstance(outcome, BaseException):
            # A fresh error per call: the core stamps attributes (num_retries, ...) on the one it raises.
            outcome = _fresh_error(outcome)  # rebind-ok: the scripted error is a template
            self.raised.append(outcome)
        return call, outcome

    def _logging_kwargs(
        self, call: Call, error: BaseException | None, response: object
    ) -> dict[str, object]:  # mutable-ok: litellm's logging kwargs
        kwargs: Final = call.kwargs
        usage: Final = getattr(response, "usage", None)
        metadata: Final = kwargs.get("metadata") or kwargs.get("litellm_metadata") or {}
        model_info: Final = kwargs.get("model_info") or {}
        return {
            "call_type": call.op,
            "model": call.model,
            "exception": error,
            "litellm_params": {
                "metadata": metadata,
                "model_info": model_info,
                "api_base": kwargs.get("api_base"),
                **({"cooldown_time": kwargs["cooldown_time"]} if "cooldown_time" in kwargs else {}),
            },
            "standard_logging_object": {
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                "model_group": metadata.get("model_group"),
                "model_id": model_info.get("id"),
                "hidden_params": {"litellm_model_name": call.model},
            },
        }

    def _finish(self, op: str, call: Call, outcome: object) -> tuple[BaseException | None, object]:
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            outcome = outcome("scripted failure")  # rebind-ok: an exception class raises an instance
        if isinstance(outcome, BaseException):
            return outcome, None
        return None, _default_response(op, call.model) if outcome is None else outcome

    async def _answer_async(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        call, outcome = self._outcome(op, args, kwargs)
        error, response = self._finish(op, call, outcome)
        logged: Final = self._logging_kwargs(call, error, response)
        if error is not None:
            for callback in list(self.lists.get("failure_callback", ())):
                if callable(callback):
                    callback(logged, None, 0.0, 0.0)
            for callback in list(self.lists.get("_async_failure_callback", ())):
                if callable(callback):
                    await callback(logged, None, 0.0, 0.0)
            raise error
        for callback in list(self.lists.get("_async_success_callback", ())):
            if callable(callback):
                await callback(logged, response, 0.0, 0.0)
        return response

    def _answer_sync(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        call, outcome = self._outcome(op, args, kwargs)
        error, response = self._finish(op, call, outcome)
        logged: Final = self._logging_kwargs(call, error, response)
        if error is not None:
            for callback in list(self.lists.get("failure_callback", ())):
                if callable(callback):
                    callback(logged, None, 0.0, 0.0)
            raise error
        for callback in list(self.lists.get("success_callback", ())):
            if callable(callback):
                callback(logged, response, 0.0, 0.0)
        return response


def deployment(
    model_name: str,
    deployment_id: str,
    model: str | None = None,
    **params: object,  # kwargs-ok: extra litellm_params
) -> dict[str, object]:  # mutable-ok: a model_list entry
    """One ``model_list`` entry; ``model`` defaults to ``openai/<deployment_id>`` so each deployment
    answers its own script."""
    model_info: Final = params.pop("model_info", None)
    return {
        "model_name": model_name,
        "litellm_params": {"model": model or f"openai/{deployment_id}", "api_key": "sk-test", **params},
        "model_info": {"id": deployment_id, **(model_info if isinstance(model_info, dict) else {})},
    }


def python_core(host: ScriptedHost, **arguments: object) -> PythonCore:  # kwargs-ok: the router's constructor arguments
    """A ``PythonCore`` built the way ``Router`` builds one, on ``host``."""
    config: Final = build_router_config(arguments)
    return PythonCore(**config.arguments, _host=host)  # pyright: ignore[reportArgumentType]  # the same arguments Router binds


CoreBuilder = Callable[..., RouterCore]
"""``builder(host, **router_arguments)``: a router core on ``host``."""

CORE_BUILDERS: Final[dict[str, CoreBuilder]] = {"python": python_core}
"""Every core the suite runs on. Another core adds its builder here; a builder that cannot take a
configuration raises ``CoreDeclined``, and the test is skipped for that core."""


class CoreDeclined(Exception):
    """The core does not serve this configuration (the router would stay on the Python core)."""


__all__ = [
    "Call",
    "HostCallbacks",
    "INFERENCE_OPS",
    "CORE_BUILDERS",
    "CoreBuilder",
    "CoreDeclined",
    "ScriptedHost",
    "auth_error",
    "bad_request",
    "completion_response",
    "content_policy",
    "context_window",
    "deployment",
    "dynamic_config",
    "embedding_response",
    "python_core",
    "rate_limit",
    "server_error",
    "timeout",
]
