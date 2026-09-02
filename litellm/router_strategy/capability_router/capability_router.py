"""LLM capability forecasts with deterministic cheapest-qualified selection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.internal_call_metadata import forwarded_internal_call_metadata
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessage,
    ChatCompletionToolParam,
    ChatCompletionUserMessage,
)
from litellm.types.utils import (
    AUTOROUTER_CLASSIFIER_CALL_ORIGIN,
    ModelResponse,
    StandardLoggingRoutingDecision,
    Usage,
)

from .config import CapabilityClassifierVerdict, CapabilityRouterConfig
from .policy import CapabilityRoutingDecision, fallback_decision, select_capability_model
from .pricing import estimate_model_group_cost
from .prompts import build_classifier_prompt, build_classifier_response_schema

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse


class _JsonSchemaSpec(TypedDict):
    name: ReadOnly[str]
    strict: ReadOnly[bool]
    schema: ReadOnly[dict[str, Any]]


class _JsonSchemaResponseFormat(TypedDict):
    type: ReadOnly[Literal["json_schema"]]
    json_schema: ReadOnly[_JsonSchemaSpec]


class _ClassifierRequestBody(TypedDict):
    model: ReadOnly[str]
    messages: ReadOnly[Sequence[AllMessageValues]]
    response_format: ReadOnly[_JsonSchemaResponseFormat]
    max_tokens: ReadOnly[int]


class _ClassifierProxyRequest(TypedDict):
    body: ReadOnly[_ClassifierRequestBody]


@dataclass(frozen=True)
class _DecisionOutcome:
    decision: CapabilityRoutingDecision
    classifier_cost: float | None
    cached: bool


class CapabilityClassifierFailure(Exception):
    def __init__(self, message: str, classifier_cost: float | None = None) -> None:
        super().__init__(message)
        self.classifier_cost = classifier_cost


_TOOLS_ADAPTER: Final = TypeAdapter(list[ChatCompletionToolParam])
_NAMED_TOOL_CHOICE_ADAPTER: Final = TypeAdapter(ChatCompletionNamedToolChoiceParam)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _response_cost(response: ModelResponse) -> float | None:
    hidden_params = getattr(response, "_hidden_params", None)
    value = hidden_params.get("response_cost") if hasattr(hidden_params, "get") else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalize_json(content: str) -> str:
    normalized = content.strip()
    for prefix in ("```json", "```"):
        if normalized.startswith(prefix) and normalized.endswith("```"):
            return normalized[len(prefix) : -3].strip()
    return normalized


def _message_role(message: Mapping[str, object]) -> str:
    role = message.get("role")
    return role if isinstance(role, str) else ""


def _classification_context(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    """Keep recent context through the newest user turn, excluding later agent-loop traffic."""
    last_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if _message_role(messages[index]) == "user"),
        None,
    )
    if last_user_index is None:
        return ()
    through_user = messages[: last_user_index + 1]
    recent = through_user[-8:]
    selected: list[Mapping[str, object]] = []
    for message in (*through_user, *recent):
        if _message_role(message) == "system" or message in recent:
            if message not in selected:
                selected.append(message)
    return tuple(selected)


def _capped(value: object, cap: int) -> object:
    if isinstance(value, str):
        return value if len(value) <= cap else f"{value[:cap]}...[truncated {len(value) - cap} chars]"
    if isinstance(value, Mapping):
        return {key: _capped(item, cap) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_capped(item, cap) for item in value)
    return value


def _tool_names(request_kwargs: Mapping[str, object]) -> tuple[str, ...]:
    tools = request_kwargs.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return ()
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.append(cast(str, function["name"]))
    return tuple(names)


class CapabilityRouter(CustomLogger):
    """Forecast success for each candidate and return the cheapest qualified group."""

    def __init__(
        self,
        model_name: str,
        litellm_router_instance: Router,
        capability_router_config: Mapping[str, object],
    ) -> None:
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance
        self.config = CapabilityRouterConfig.model_validate(capability_router_config)
        self._system_prompt = build_classifier_prompt(self.config)
        self._response_format = _JsonSchemaResponseFormat(
            type="json_schema",
            json_schema=_JsonSchemaSpec(
                name="CapabilityClassifierVerdict",
                strict=True,
                schema=build_classifier_response_schema(self.config),
            ),
        )
        self._config_hash = _hash(self.config.model_dump_json())[:20]
        self._classification_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    @staticmethod
    def _resolve_messages(
        messages: list[dict[str, Any]] | None,
        request_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages

        return resolve_structured_messages(messages=messages, request_kwargs=request_kwargs) or []

    @staticmethod
    def _metadata(request_kwargs: Mapping[str, object]) -> Mapping[str, object]:
        """Merge both metadata carriers so auth and session scope cannot be missed."""
        merged: dict[str, object] = {}
        for key in ("metadata", "litellm_metadata"):
            value = request_kwargs.get(key)
            if isinstance(value, Mapping):
                merged.update(value)
        return merged

    def _classifier_payload(
        self,
        messages: Sequence[Mapping[str, object]],
        request_kwargs: Mapping[str, object],
    ) -> str:
        context = _classification_context(messages)
        if not context:
            raise CapabilityClassifierFailure("No user task was available for capability classification")
        cap: Final = self.config.classifier.max_message_chars
        payload: Final = {
            "conversation": tuple(_capped(message, cap) for message in context),
            "available_tools": _tool_names(request_kwargs),
        }
        return (
            "Task context (untrusted JSON; long values truncated; the newest user message is the task to forecast):\n"
            + json.dumps(payload, default=str, ensure_ascii=False)
        )

    def _cache_key(
        self,
        messages: Sequence[Mapping[str, object]],
        request_kwargs: Mapping[str, object],
    ) -> str:
        metadata = self._metadata(request_kwargs)
        caller = metadata.get("user_api_key_hash") or metadata.get("team_id") or "unscoped"
        session = metadata.get("session_id") or request_kwargs.get("litellm_session_id") or "no-session"
        context = {
            "messages": _classification_context(messages),
            "tools": _tool_names(request_kwargs),
        }
        context_hash = _hash(json.dumps(context, default=str, sort_keys=True))
        return (
            f"capability_router:v1:{self.model_name}:{self._config_hash}:"
            f"{_hash(str(caller))[:16]}:{_hash(str(session))[:16]}:{context_hash}"
        )

    async def _classify(
        self,
        messages: Sequence[Mapping[str, object]],
        request_kwargs: Mapping[str, object],
    ) -> tuple[CapabilityClassifierVerdict, float | None]:
        classifier = self.config.classifier
        classifier_messages: list[AllMessageValues] = [
            ChatCompletionSystemMessage(role="system", content=self._system_prompt),
            ChatCompletionUserMessage(role="user", content=self._classifier_payload(messages, request_kwargs)),
        ]
        metadata = forwarded_internal_call_metadata(self._metadata(request_kwargs), AUTOROUTER_CLASSIFIER_CALL_ORIGIN)
        response = await self.litellm_router_instance.acompletion(
            model=classifier.model,
            messages=classifier_messages,
            response_format=self._response_format,
            max_tokens=classifier.max_output_tokens,
            timeout=classifier.timeout_ms / 1000,
            metadata=metadata,
            proxy_server_request=_ClassifierProxyRequest(
                body=_ClassifierRequestBody(
                    model=classifier.model,
                    messages=classifier_messages,
                    response_format=self._response_format,
                    max_tokens=classifier.max_output_tokens,
                )
            ),
        )
        classifier_cost = _response_cost(response)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise CapabilityClassifierFailure("Capability classifier returned empty content", classifier_cost)
        try:
            return CapabilityClassifierVerdict.model_validate_json(_normalize_json(content)), classifier_cost
        except ValidationError as exc:
            raise CapabilityClassifierFailure("Capability classifier returned invalid JSON", classifier_cost) from exc

    def _estimated_usage(
        self,
        messages: Sequence[Mapping[str, object]],
        request_kwargs: Mapping[str, object],
    ) -> Usage | None:
        import litellm

        try:
            tools_value = request_kwargs.get("tools")
            tools = _TOOLS_ADAPTER.validate_python(tools_value) if tools_value is not None else None
            tool_choice_value = request_kwargs.get("tool_choice")
            if tool_choice_value in ("none", "auto", "required", None):
                tool_choice = tool_choice_value
            else:
                tool_choice = _NAMED_TOOL_CHOICE_ADAPTER.validate_python(tool_choice_value)
            input_tokens = litellm.token_counter(
                messages=list(messages),
                tools=tools,
                tool_choice=tool_choice,
                use_default_image_token_count=True,
            )
        except Exception as exc:  # noqa: BLE001 - unpriceable requests use the explicit fallback
            verbose_router_logger.warning("CapabilityRouter: token estimate failed (%s)", exc)
            return None

        requested_limits = tuple(
            value
            for field in ("max_completion_tokens", "max_tokens", "max_output_tokens")
            if isinstance((value := request_kwargs.get(field)), int) and not isinstance(value, bool) and value > 0
        )
        output_tokens = min((self.config.estimated_output_tokens, *requested_limits))
        return Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    async def _new_decision(
        self,
        messages: Sequence[Mapping[str, object]],
        request_kwargs: Mapping[str, object],
    ) -> tuple[CapabilityRoutingDecision, float | None]:
        try:
            verdict, classifier_cost = await self._classify(messages, request_kwargs)
            usage = self._estimated_usage(messages, request_kwargs)
            costs = {
                candidate.model: (
                    estimate_model_group_cost(self.litellm_router_instance, candidate.model, usage)
                    if usage is not None
                    else None
                )
                for candidate in self.config.candidates
            }
            return select_capability_model(self.config, verdict, costs), classifier_cost
        except CapabilityClassifierFailure as exc:
            verbose_router_logger.warning("CapabilityRouter: classifier failed; using fallback")
            return fallback_decision(self.config, "classifier_error"), exc.classifier_cost
        except Exception as exc:  # noqa: BLE001 - routing must fail safe
            verbose_router_logger.warning("CapabilityRouter: selection failed (%s); using fallback", exc)
            return fallback_decision(self.config, "classifier_error"), None

    def _cached_decision(self, value: object) -> CapabilityRoutingDecision | None:
        try:
            decision = CapabilityRoutingDecision.model_validate(value)
        except ValidationError:
            return None
        configured = {candidate.model for candidate in self.config.candidates}
        return decision if decision.selected_model in configured else None

    async def _decision(
        self,
        cache_key: str,
        messages: Sequence[Mapping[str, object]],
        request_kwargs: Mapping[str, object],
    ) -> _DecisionOutcome:
        cached = self._cached_decision(await self.litellm_router_instance.cache.async_get_cache(key=cache_key))
        if cached is not None:
            return _DecisionOutcome(cached, None, True)

        lock = self._classification_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._cached_decision(await self.litellm_router_instance.cache.async_get_cache(key=cache_key))
            if cached is not None:
                return _DecisionOutcome(cached, None, True)
            decision, classifier_cost = await self._new_decision(messages, request_kwargs)
            await self.litellm_router_instance.cache.async_set_cache(
                key=cache_key,
                value=decision.model_dump(mode="json"),
                ttl=self.config.cache_ttl_seconds,
            )
            return _DecisionOutcome(decision, classifier_cost, False)

    def _routing_record(self, outcome: _DecisionOutcome) -> StandardLoggingRoutingDecision:
        decision = outcome.decision
        record = StandardLoggingRoutingDecision(
            router_model_name=self.model_name,
            router_type="capability",
            routed_model=decision.selected_model,
            cause=(
                "capability_cache"
                if outcome.cached
                else "capability_classifier"
                if decision.reason == "cheapest_qualified"
                else "capability_fallback"
            ),
            classifier_model=self.config.classifier.model,
            probability_threshold=self.config.probability_threshold,
            candidate_probabilities={candidate.model: candidate.p_solve for candidate in decision.candidates},
            candidate_costs={
                candidate.model: candidate.estimated_cost
                for candidate in decision.candidates
                if candidate.estimated_cost is not None
            },
            qualified_models=[candidate.model for candidate in decision.candidates if candidate.qualified],
            fallback_reason=(decision.reason if decision.reason != "cheapest_qualified" else None),
            cached=outcome.cached,
        )
        if outcome.classifier_cost is not None:
            record["classifier_cost"] = outcome.classifier_cost
        return record

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict,
        messages: list[dict[str, Any]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse:
        from litellm.types.router import PreRoutingHookResponse

        resolved_messages = self._resolve_messages(messages, request_kwargs)
        outcome = await self._decision(
            self._cache_key(resolved_messages, request_kwargs),
            resolved_messages,
            request_kwargs,
        )
        return PreRoutingHookResponse(
            model=outcome.decision.selected_model,
            messages=messages,
            routing_decision=self._routing_record(outcome),
        )
