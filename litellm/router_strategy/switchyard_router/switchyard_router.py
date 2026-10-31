from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, TypeAlias
from uuid import uuid4

from litellm.exceptions import (
    APIConnectionError,
    APIError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.router import PreRoutingHookResponse
from litellm.types.utils import ModelResponse

from .config import SwitchyardRouterConfig
from .native import Aggregate, NativeApi, build_algorithm, load_native
from .translation import chat_request, normalize_request, normalize_response

Executor: TypeAlias = Callable[[str, Mapping[str, object]], Awaitable[ModelResponse | CustomStreamWrapper]]


def eligible_fallback(error: Exception) -> bool:
    return isinstance(
        error,
        (
            APIConnectionError,
            Timeout,
            RateLimitError,
            ContextWindowExceededError,
            InternalServerError,
            ServiceUnavailableError,
        ),
    ) or (isinstance(error, APIError) and error.status_code >= 500)


class _Execution:
    def __init__(
        self, config: SwitchyardRouterConfig, api: NativeApi, request: Mapping[str, object], executor: Executor
    ) -> None:
        self.config = config
        self.api = api
        self.request = request
        self.executor = executor
        self.responses: Mapping[str, ModelResponse] = {}
        self.provider_error: Exception | None = None

    async def call_candidates(
        self,
        candidates: Sequence[str],
        request: Mapping[str, object],
        *,
        stream: bool,
        fallback_request: Mapping[str, object] | None = None,
        intermediate: bool = False,
    ) -> ModelResponse | CustomStreamWrapper:
        if not candidates:
            raise ValueError("Switchyard returned no model candidates")
        target: Final = candidates[0]
        if not any(target in models for models in self.config.models.values()):
            raise ValueError("Switchyard selected an unconfigured model")
        payload: Final = chat_request(
            request,
            self.request,
            model=target,
            stream=stream,
            judge=intermediate and target in self.config.models.get("judge", ()) and bool(request.get("instructions")),
        )
        try:
            return await self.executor(target, payload)
        except Exception as error:
            if not self.config.fallback_enabled or len(candidates) == 1 or not eligible_fallback(error):
                raise
            return await self.call_candidates(
                candidates[1:], fallback_request or request, stream=stream, intermediate=intermediate
            )

    async def respond(self, candidates: Sequence[str], request: Mapping[str, object]) -> Aggregate:
        try:
            response: Final = await self.call_candidates(candidates, request, stream=False, intermediate=True)
        except Exception as error:
            self.provider_error = error
            if isinstance(error, ContextWindowExceededError):
                raise self.api.ContextWindowExceededError(str(error)) from error
            raise
        self.provider_error = None
        if not isinstance(response, ModelResponse):
            raise TypeError("Switchyard intermediate calls must return aggregate chat responses")
        response_id: Final = str(uuid4())
        self.responses = {**self.responses, response_id: response}
        return self.api.LlmResponse.Agg(normalize_response(response, response_id))

    def completed_response(self, aggregate: Aggregate) -> ModelResponse:
        response_id: Final = aggregate.response.get("id")
        if not isinstance(response_id, str) or response_id not in self.responses:
            raise ValueError("Switchyard returned an unknown completed response")
        return self.responses[response_id]


class SwitchyardAutoRouter:
    def __init__(self, config: SwitchyardRouterConfig, *, native_api: NativeApi | None = None) -> None:
        self.config = config
        self.session_namespace: Final[str] = str(uuid4())
        self._api = native_api if native_api is not None else load_native()
        self._algorithm = build_algorithm(self._api, config)

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Mapping[str, object],
        messages: Sequence[Mapping[str, object]] | None = None,
        input: str | Sequence[object] | None = None,
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse | None:
        raise ValueError("Switchyard supports async chat completions only")

    async def execute(
        self, request: Mapping[str, object], executor: Executor, headers: Mapping[str, str] | None = None
    ) -> ModelResponse | CustomStreamWrapper:
        normalized: Final = normalize_request(request)
        execution: Final = _Execution(self.config, self._api, request, executor)
        try:
            return await self._run(execution, normalized, request.get("stream") is True, headers)
        except self._api.LibsyError:
            if execution.provider_error is not None:
                raise execution.provider_error
            raise

    async def _run(
        self,
        execution: _Execution,
        normalized: Mapping[str, object],
        stream: bool,
        headers: Mapping[str, str] | None,
    ) -> ModelResponse | CustomStreamWrapper:
        async for step in self._algorithm.run_stream(normalized, self.config.models, headers=headers):
            if isinstance(step, self._api.Step.CallModel):
                await step.call.respond(execution.respond(step.call.models, step.call.request))
            elif isinstance(step, self._api.Step.Done):
                if step.outcome.response is not None:
                    return execution.completed_response(step.outcome.response)
                return await execution.call_candidates(
                    step.outcome.selected_model_ids,
                    step.outcome.request,
                    stream=stream,
                    fallback_request=step.outcome.fallback_request,
                )
            else:
                raise TypeError("Switchyard returned a step unsupported by the configured chat routing algorithms")
        raise ValueError("Switchyard ended without a routing outcome")
