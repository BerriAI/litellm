"""
Implements logging integration with Datadog's LLM Observability Service


API Reference: https://docs.datadoghq.com/llm_observability/setup/api/?tab=example#api-standards

"""

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final, Literal

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.constants import REDACTED_BY_LITELLM
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.datadog.datadog_handler import (
    get_datadog_base_url_from_env,
    get_datadog_service,
    get_datadog_tags,
    normalize_datadog_tag_value,
)
from litellm.integrations.datadog.datadog_mock_client import (
    create_mock_datadog_client,
    should_use_datadog_mock,
)
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
    handle_any_messages_to_chat_completion_str_messages_conversion,
)
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens, extract_cache_read_tokens
from litellm.types.integrations.datadog_llm_obs import *
from litellm.types.utils import (
    PROMPT_QUOTING_ROUTING_DECISION_FIELDS,
    CallTypes,
    StandardLoggingGuardrailInformation,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
)

_EMPTY_MAPPING: Final[Mapping[str, Any]] = MappingProxyType({})
_EMPTY_MESSAGE: Final[Message] = {"role": "", "content": ""}
_MAX_PARSED_TOOL_ARGUMENT_CHARS: Final = 256 * 1024

_PROMPT_CARRYING_METADATA_FIELDS: Final = frozenset(
    {
        "routing_decision",
        "requester_metadata",
        "prompt_management_metadata",
        "mcp_tool_call_metadata",
        "vector_store_request_metadata",
    }
)

_ROUTER_SPAN_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tier": "router_tier",
        "cause": "router_cause",
        "score": "router_score",
        "escalated": "router_escalated",
        "signals": "router_signals",
        "routed_model": "routed_model",
    }
)
_ROUTER_DIMENSIONS: Final[tuple[str, ...]] = ("router_tier", "router_cause", "router_escalated", "routed_model")
_COST_DIMENSIONS: Final[tuple[str, ...]] = ("team", "user", "key_alias", "model_group", *_ROUTER_DIMENSIONS)


def _metadata_of(standard_logging_payload: StandardLoggingPayload) -> Mapping[str, Any]:
    metadata: Final = standard_logging_payload.get("metadata")
    return metadata or _EMPTY_MAPPING


def _router_span_fields(
    standard_logging_payload: StandardLoggingPayload, redact_prompt_text: bool
) -> Mapping[str, object]:
    """Flatten the auto-router decision, omitting prompt-quoting fields when redaction is enabled."""
    routing_decision: Final = _mapping_field(_metadata_of(standard_logging_payload), "routing_decision")
    if not routing_decision:
        return _EMPTY_MAPPING
    escalated: Final = bool(routing_decision.get("escalated") or routing_decision.get("context_escalated"))
    return MappingProxyType(
        {
            _ROUTER_SPAN_FIELDS[record_field]: value
            for record_field, value in (*routing_decision.items(), ("escalated", escalated))
            if record_field in _ROUTER_SPAN_FIELDS
            and value is not None
            and not (redact_prompt_text and record_field in PROMPT_QUOTING_ROUTING_DECISION_FIELDS)
        }
    )


def _metadata_without_prompt_carriers(standard_logging_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """The metadata minus the records that quote prompts, tool arguments, tool results, or retrieved text."""
    return MappingProxyType(
        {
            field: value
            for field, value in standard_logging_metadata.items()
            if field not in _PROMPT_CARRYING_METADATA_FIELDS
        }
    )


def _redact_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Each message's shape with its content replaced and tool payloads dropped; no message is invented."""
    return tuple({"role": message.get("role", ""), "content": REDACTED_BY_LITELLM} for message in messages)


def _cost_dimension_tags(
    standard_logging_payload: StandardLoggingPayload, router_fields: Mapping[str, object]
) -> tuple[str, ...]:
    """The dimensions LLM Obs breaks token and cost metrics down by, as span tags."""
    metadata: Final = _metadata_of(standard_logging_payload)
    dimensions: Final = (
        ("user", metadata.get("user_api_key_user_id")),
        ("key_alias", metadata.get("user_api_key_alias")),
        ("model_group", standard_logging_payload.get("model_group")),
        *((dimension, router_fields.get(dimension)) for dimension in _ROUTER_DIMENSIONS),
    )
    return tuple(
        f"{key}:{normalized}"
        for key, value in dimensions
        if value is not None and (normalized := normalize_datadog_tag_value(value)) != ""
    )


def _declared_cost_tags(span_tags: Sequence[str]) -> tuple[str, ...]:
    """Declare only cost dimensions carrying a value on this span."""
    present: Final = frozenset(key for tag in span_tags if (key := tag.partition(":")[0]) and tag.partition(":")[2])
    return tuple(dimension for dimension in _COST_DIMENSIONS if dimension in present)


def _reasoning_output_tokens(usage_object: Mapping[str, Any] | None) -> float:
    """The provider's reasoning-token count, from either the chat or the responses spelling."""
    if usage_object is None:
        return 0.0
    return next(
        (
            float(reasoning_tokens)
            for details_field in ("completion_tokens_details", "output_tokens_details")
            if isinstance(
                reasoning_tokens := _mapping_field(usage_object, details_field).get("reasoning_tokens"), (int, float)
            )
            and not isinstance(reasoning_tokens, bool)
        ),
        0.0,
    )


def _mapping_field(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """The value at `key` when it is a mapping, else an empty one."""
    value: Final = source.get(key)
    return value if isinstance(value, dict) else _EMPTY_MAPPING


def _content_blocks(message: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    content: Final = message.get("content")
    if not isinstance(content, list):
        return ()
    return tuple(block for block in content if isinstance(block, dict))


def _to_dd_arguments(raw_arguments: object) -> dict[str, Any] | str:
    """
    Arguments as the object LLM Obs types them as, or the raw string when they are not one.

    Strings past the size bound ship unparsed: decoding multiplies memory on hostile compact
    JSON, and the raw string is what the intake receives either way.
    """
    if not isinstance(raw_arguments, str):
        return raw_arguments if isinstance(raw_arguments, dict) else str(raw_arguments)
    if len(raw_arguments) > _MAX_PARSED_TOOL_ARGUMENT_CHARS:
        return raw_arguments
    parsed: Final = safe_json_loads(raw_arguments)
    return parsed if isinstance(parsed, dict) else raw_arguments


def _to_dd_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """
    The tool calls a message carries, in LLM Obs' ToolCall schema, from either dialect.

    OpenAI puts them in `tool_calls` with the callee nested under `function` and `arguments`
    serialized; Anthropic puts them in `content` as `tool_use` blocks with `input` already an
    object. LLM Obs reads `name` / `arguments` / `tool_id` either way.
    """
    raw_tool_calls: Final = message.get("tool_calls")
    openai_calls: Final = tuple(
        ToolCall(
            name=function.get("name", ""),
            arguments=_to_dd_arguments(function.get("arguments", "")),
            tool_id=tool_call.get("id", ""),
            type=tool_call.get("type", "function"),
        )
        for tool_call in (raw_tool_calls if isinstance(raw_tool_calls, list) else ())
        if isinstance(tool_call, dict)
        for function in [_mapping_field(tool_call, "function")]
    )
    anthropic_calls: Final = tuple(
        ToolCall(
            name=block.get("name", ""),
            arguments=_to_dd_arguments(block.get("input") or {}),
            tool_id=block.get("id", ""),
            type="tool_use",
        )
        for block in _content_blocks(message)
        if block.get("type") == "tool_use"
    )
    return openai_calls + anthropic_calls


def _to_dd_tool_results(message: Mapping[str, Any], tool_call_names: Mapping[str, str]) -> tuple[ToolResult, ...]:
    """
    The tool results a message carries, linked back to the call each answers.

    OpenAI models a result as a whole `role: "tool"` message keyed by `tool_call_id`;
    Anthropic nests `tool_result` blocks inside a user message, keyed by `tool_use_id`.
    """

    def to_result(tool_id: str, result: object) -> ToolResult:
        return ToolResult(
            name=tool_call_names.get(tool_id, ""),
            result=result if isinstance(result, str) else safe_dumps(result),
            tool_id=tool_id,
            type="function",
        )

    if message.get("role") == "tool":
        return (to_result(str(message.get("tool_call_id", "")), message.get("content") or ""),)
    return tuple(
        to_result(str(block.get("tool_use_id", "")), block.get("content") or "")
        for block in _content_blocks(message)
        if block.get("type") == "tool_result"
    )


def _tool_call_names_by_id(messages: Sequence[object]) -> Mapping[str, str]:
    """Ids to tool names for result linking; reads names structurally and parses nothing."""
    openai_pairs: Final = tuple(
        (tool_call.get("id"), function.get("name", ""))
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list)
        for tool_call in message["tool_calls"]
        if isinstance(tool_call, dict)
        for function in [_mapping_field(tool_call, "function")]
    )
    anthropic_pairs: Final = tuple(
        (block.get("id"), block.get("name", ""))
        for message in messages
        if isinstance(message, dict)
        for block in _content_blocks(message)
        if block.get("type") == "tool_use"
    )
    return MappingProxyType({str(tool_id): str(name) for tool_id, name in openai_pairs + anthropic_pairs if tool_id})


def _to_dd_message(message: object, tool_call_names: Mapping[str, str]) -> Message:
    """
    Map one chat message onto LLM Obs' Message schema, adding fields and never destroying content.

    Content collapses to its text only when it has text; a content list with none (tool blocks,
    images) rides along unchanged so nothing the caller logged is lost. Tool calls and results
    move into the fields the LLM Obs Tools panel reads, from both the OpenAI and Anthropic shapes.
    """
    if not isinstance(message, dict):
        converted: Final = handle_any_messages_to_chat_completion_str_messages_conversion(message)
        return converted[0] if converted else _EMPTY_MESSAGE

    text: Final = convert_content_list_to_str(message)  # pyright: ignore[reportArgumentType]  # caller-supplied dict
    original_content: Final = message.get("content")
    content: Final = (
        text if text or not isinstance(original_content, list) or not original_content else original_content
    )
    reasoning: Final = message.get("reasoning_content")
    tool_calls: Final = _to_dd_tool_calls(message)
    tool_results: Final = _to_dd_tool_results(message, tool_call_names)
    dd_message: Final[Message] = {
        "role": message.get("role", ""),
        "content": content,
        **({"reasoning_content": reasoning} if reasoning is not None else {}),
        **({"tool_calls": tool_calls} if tool_calls else {}),
        **({"tool_results": tool_results} if tool_results else {}),
    }
    return dd_message


def _to_dd_messages(messages: object) -> tuple[Message, ...]:
    """Map a whole conversation, resolving each tool result against the calls that precede it."""
    if messages is None:
        return ()
    if not isinstance(messages, list):
        return tuple(handle_any_messages_to_chat_completion_str_messages_conversion(messages))
    tool_call_names: Final = _tool_call_names_by_id(messages)
    return tuple(_to_dd_message(message, tool_call_names) for message in messages)


def _to_dd_tool_definition(entry: Mapping[str, Any]) -> ToolDefinition | None:
    function: Final = entry.get("function")
    declared: Final[Mapping[str, Any]] = function if isinstance(function, dict) else entry
    name: Final = declared.get("name")
    if not name:
        return None
    schema: Final = declared.get("parameters") or declared.get("input_schema")
    description: Final = declared.get("description", "")
    if not isinstance(schema, dict):
        return ToolDefinition(name=name, description=description)
    return ToolDefinition(name=name, description=description, schema=schema)


def _to_dd_tool_definitions(model_parameters: object) -> tuple[ToolDefinition, ...]:
    """
    Map the request's declared tools onto LLM Obs' ToolDefinition schema.

    Handles the wrapped chat-completions shape and the bare shape the Anthropic and
    Responses surfaces use, since both reach this logger through `model_parameters`.
    """
    if not isinstance(model_parameters, dict):
        return ()
    raw_tools: Final = model_parameters.get("tools") or model_parameters.get("functions")
    if not isinstance(raw_tools, list):
        return ()
    return tuple(
        definition
        for entry in raw_tools
        if isinstance(entry, dict)
        if (definition := _to_dd_tool_definition(entry)) is not None
    )


class DataDogLLMObsLogger(CustomBatchLogger):
    def __init__(self, **kwargs):
        try:
            verbose_logger.debug("DataDogLLMObs: Initializing logger")

            self.is_mock_mode = should_use_datadog_mock()

            if self.is_mock_mode:
                create_mock_datadog_client()
                verbose_logger.debug("[DATADOG MOCK] DataDogLLMObs logger initialized in mock mode")

            # Configure DataDog endpoint (Agent or Direct API)
            # Use LITELLM_DD_AGENT_HOST to avoid conflicts with ddtrace's DD_AGENT_HOST
            # Check for agent mode FIRST - agent mode doesn't require DD_API_KEY or DD_SITE
            dd_agent_host: Final = os.getenv("LITELLM_DD_AGENT_HOST")

            self.async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
            self.DD_API_KEY = os.getenv("DD_API_KEY")

            if dd_agent_host:
                self._configure_dd_agent(dd_agent_host=dd_agent_host)
            else:
                # Only require DD_API_KEY and DD_SITE for direct API mode
                if os.getenv("DD_API_KEY", None) is None:
                    raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>'")
                if os.getenv("DD_SITE", None) is None:
                    raise Exception("DD_SITE is not set, set 'DD_SITE=<>', example sit = `us5.datadoghq.com`")
                self._configure_dd_direct_api()

            # Optional override for testing
            dd_base_url: Final = get_datadog_base_url_from_env()
            if dd_base_url:
                self.intake_url = f"{dd_base_url}/api/intake/llm-obs/v1/trace/spans"

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            self.log_queue: list[LLMObsPayload] = []

            #########################################################
            # Handle datadog_llm_observability_params set as litellm.datadog_llm_observability_params
            #########################################################
            dict_datadog_llm_obs_params: Final = self._get_datadog_llm_obs_params()
            kwargs.update(dict_datadog_llm_obs_params)
            CustomBatchLogger.__init__(self, **kwargs, flush_lock=self.flush_lock)
        except Exception as e:
            verbose_logger.exception("DataDogLLMObs: Error initializing - %s", e)
            raise e

    def _configure_dd_agent(self, dd_agent_host: str):
        """
        Configure the Datadog logger to send traces to the Agent.
        """
        # When using the Agent, LLM Observability Intake does NOT require the API Key
        # Reference: https://docs.datadoghq.com/llm_observability/setup/sdk/#agent-setup

        # Use specific port for LLM Obs (Trace Agent) to avoid conflict with Logs Agent (10518)
        agent_port: Final = os.getenv("LITELLM_DD_LLM_OBS_PORT", "8126")
        self.DD_SITE = "localhost"  # Not used for URL construction in agent mode
        self.intake_url = f"http://{dd_agent_host}:{agent_port}/api/intake/llm-obs/v1/trace/spans"
        verbose_logger.debug("DataDogLLMObs: Using DD Agent at %s", self.intake_url)

    def _configure_dd_direct_api(self):
        """
        Configure the Datadog logger to send traces directly to the Datadog API.
        """
        if not self.DD_API_KEY:
            raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>'")

        self.DD_SITE = os.getenv("DD_SITE")
        if not self.DD_SITE:
            raise Exception("DD_SITE is not set, set 'DD_SITE=<>', example site = `us5.datadoghq.com`")

        self.intake_url = f"https://api.{self.DD_SITE}/api/intake/llm-obs/v1/trace/spans"

    def _get_datadog_llm_obs_params(self) -> dict:
        """
        Get the datadog_llm_observability_params from litellm.datadog_llm_observability_params

        These are params specific to initializing the DataDogLLMObsLogger e.g. turn_off_message_logging
        """
        dict_datadog_llm_obs_params: dict = {}
        if litellm.datadog_llm_observability_params is not None:
            if isinstance(litellm.datadog_llm_observability_params, DatadogLLMObsInitParams):
                dict_datadog_llm_obs_params = litellm.datadog_llm_observability_params.model_dump(exclude_unset=True)
            elif isinstance(litellm.datadog_llm_observability_params, dict):
                # only allow params that are of DatadogLLMObsInitParams
                dict_datadog_llm_obs_params = DatadogLLMObsInitParams(
                    **litellm.datadog_llm_observability_params
                ).model_dump(exclude_unset=True)
        return dict_datadog_llm_obs_params

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug("DataDogLLMObs: Logging success event for model %s", kwargs.get("model", "unknown"))
            payload: Final = self.create_llm_obs_payload(kwargs, start_time, end_time)
            verbose_logger.debug("DataDogLLMObs: Payload: %s", payload)
            self.log_queue.append(payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()
        except Exception as e:
            verbose_logger.exception("DataDogLLMObs: Error logging success event - %s", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug("DataDogLLMObs: Logging failure event for model %s", kwargs.get("model", "unknown"))
            payload: Final = self.create_llm_obs_payload(kwargs, start_time, end_time)
            verbose_logger.debug("DataDogLLMObs: Payload: %s", payload)
            self.log_queue.append(payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()
        except Exception as e:
            verbose_logger.exception("DataDogLLMObs: Error logging failure event - %s", e)

    async def async_send_batch(self):
        try:
            if not self.log_queue:
                return

            verbose_logger.debug("DataDogLLMObs: Flushing %s events", len(self.log_queue))

            if self.is_mock_mode:
                verbose_logger.debug("[DATADOG MOCK] Mock mode enabled - API calls will be intercepted")

            # Prepare the payload
            payload: Final = {
                "data": DDIntakePayload(
                    type="span",
                    attributes=DDSpanAttributes(
                        ml_app=get_datadog_service(),
                        tags=get_datadog_tags(),
                        spans=self.log_queue,
                    ),
                ),
            }

            # serialize datetime objects - for budget reset time in spend metrics
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            try:
                verbose_logger.debug("payload %s", safe_dumps(payload))
            except Exception as debug_error:
                verbose_logger.debug("payload serialization failed: %s", str(debug_error))

            json_payload: Final = safe_dumps(payload)

            headers: Final = {"Content-Type": "application/json"}
            if self.DD_API_KEY:
                headers["DD-API-KEY"] = self.DD_API_KEY

            response: Final = await self.async_client.post(
                url=self.intake_url,
                content=json_payload,
                headers=headers,
            )

            if response.status_code != 202:
                raise Exception(
                    f"DataDogLLMObs: Unexpected response - status_code: {response.status_code}, text: {response.text}"
                )

            if self.is_mock_mode:
                verbose_logger.debug("[DATADOG MOCK] Batch of %s events successfully mocked", len(self.log_queue))
            else:
                verbose_logger.debug("DataDogLLMObs: Successfully sent batch - status_code: %s", response.status_code)
            self.log_queue.clear()
        except httpx.HTTPStatusError as e:
            verbose_logger.exception("DataDogLLMObs: Error sending batch - %s", e.response.text)
        except Exception as e:
            verbose_logger.exception("DataDogLLMObs: Error sending batch - %s", e)

    def create_llm_obs_payload(self, kwargs: dict, start_time: datetime, end_time: datetime) -> LLMObsPayload:
        standard_logging_payload: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object")
        if standard_logging_payload is None:
            raise Exception("DataDogLLMObs: standard_logging_object is not set")

        raw_metadata: Final = kwargs.get("litellm_params", {}).get("metadata", {})
        metadata: Final = raw_metadata if isinstance(raw_metadata, dict) else {}
        redact_payload: Final = self._payload_logging_is_off(kwargs)

        input_messages: Final = _to_dd_messages(standard_logging_payload.get("messages"))
        output_messages: Final = self._get_response_messages(
            standard_logging_payload=standard_logging_payload,
            call_type=standard_logging_payload.get("call_type"),
        )
        input_meta: Final = InputMeta(messages=_redact_messages(input_messages) if redact_payload else input_messages)
        output_meta: Final = OutputMeta(
            messages=_redact_messages(output_messages) if redact_payload else output_messages
        )

        error_info: Final = self._assemble_error_info(standard_logging_payload)

        raw_parent_id: Final = metadata.get("parent_id")
        metadata_parent_id: Final[str | None] = raw_parent_id if isinstance(raw_parent_id, str) else None

        tool_definitions: Final = (
            () if redact_payload else _to_dd_tool_definitions(standard_logging_payload.get("model_parameters"))
        )
        span_kind: Final = self._get_datadog_span_kind(standard_logging_payload.get("call_type"), metadata_parent_id)
        router_fields: Final = _router_span_fields(standard_logging_payload, redact_prompt_text=redact_payload)
        span_tags: Final = [
            *get_datadog_tags(standard_logging_object=standard_logging_payload),
            *_cost_dimension_tags(standard_logging_payload, router_fields),
        ]
        payload_metadata: Final = self._get_dd_llm_obs_payload_metadata(
            standard_logging_payload,
            router_fields=router_fields,
            cost_tags=_declared_cost_tags(span_tags),
            redact_prompt_text=redact_payload,
        )

        meta: Final[Meta] = {
            "kind": span_kind,
            "input": input_meta,
            "output": output_meta,
            "metadata": payload_metadata,
            "error": error_info,
            **({"tool_definitions": tool_definitions} if tool_definitions else {}),
        }

        metrics: Final = self._assemble_metrics(standard_logging_payload)

        payload: Final[LLMObsPayload] = LLMObsPayload(
            parent_id=metadata_parent_id if metadata_parent_id else "undefined",
            trace_id=standard_logging_payload.get("trace_id", str(uuid.uuid4())),
            span_id=metadata.get("span_id", str(uuid.uuid4())),
            name=metadata.get("name", "litellm_llm_call"),
            meta=meta,
            start_ns=int(start_time.timestamp() * 1e9),
            duration=int((end_time - start_time).total_seconds() * 1e9),
            metrics=metrics,
            status="error" if error_info else "ok",
            tags=span_tags,
        )

        apm_trace_id: Final = self._get_apm_trace_id()
        if apm_trace_id is not None:
            payload["apm_id"] = apm_trace_id

        return payload

    def _get_apm_trace_id(self) -> str | None:
        """Retrieve the current APM trace ID if available."""
        try:
            current_span_fn: Final = getattr(tracer, "current_span", None)
            if callable(current_span_fn):
                current_span: Final = current_span_fn()
                if current_span is not None:
                    trace_id: Final = getattr(current_span, "trace_id", None)
                    if trace_id is not None:
                        return str(trace_id)
        except Exception:
            pass
        return None

    def _assemble_error_info(self, standard_logging_payload: StandardLoggingPayload) -> DDLLMObsError | None:
        """
        Assemble error information for failure cases according to DD LLM Obs API spec
        """
        # Handle error information for failure cases according to DD LLM Obs API spec
        error_info: DDLLMObsError | None = None

        if standard_logging_payload.get("status") == "failure":
            # Try to get structured error information first
            error_information: Final[StandardLoggingPayloadErrorInformation | None] = standard_logging_payload.get(
                "error_information"
            )

            if error_information:
                error_info = DDLLMObsError(
                    message=error_information.get("error_message")
                    or standard_logging_payload.get("error_str")
                    or "Unknown error",
                    type=error_information.get("error_class"),
                    stack=error_information.get("traceback"),
                )
        return error_info

    def _payload_logging_is_off(self, kwargs: Mapping[str, Any]) -> bool:
        return (
            bool(self.turn_off_message_logging)
            or self.message_logging is not True
            or should_redact_message_logging(dict(kwargs))
        )

    def _assemble_metrics(self, standard_logging_payload: StandardLoggingPayload) -> LLMMetrics:
        """
        Build the span metrics, including the prompt-cache counts LLM Obs charts cache savings from.

        Cache counts resolve through the same owners the savings dashboard uses, so every provider
        spelling is covered, and `non_cached_input_tokens` subtracts BOTH cache categories because
        litellm's normalized prompt count includes both (the invariant the cost calculator's custom
        pricing helper documents). A zero residual on a fully cached request is real data and is
        emitted; a zero read or write count is absence and is not.
        """
        prompt_tokens: Final = float(standard_logging_payload.get("prompt_tokens", 0))
        completion_tokens: Final = float(standard_logging_payload.get("completion_tokens", 0))
        total_tokens: Final = float(standard_logging_payload.get("total_tokens", 0))
        total_cost: Final = float(standard_logging_payload.get("response_cost", 0))
        time_to_first_token: Final = self._get_time_to_first_token_seconds(standard_logging_payload)

        raw_usage: Final = _metadata_of(standard_logging_payload).get("usage_object")
        usage_object: Final = raw_usage if isinstance(raw_usage, dict) else None
        cache_read: Final = float(extract_cache_read_tokens(usage_object))
        cache_write: Final = float(extract_cache_creation_tokens(usage_object))
        reasoning_output_tokens: Final = _reasoning_output_tokens(usage_object)

        metrics: Final[LLMMetrics] = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "time_to_first_token": time_to_first_token,
            **(
                {
                    **({"cache_read_input_tokens": cache_read} if cache_read else {}),
                    **({"cache_write_input_tokens": cache_write} if cache_write else {}),
                    "non_cached_input_tokens": max(prompt_tokens - cache_read - cache_write, 0.0),
                }
                if cache_read or cache_write
                else {}
            ),
            **({"reasoning_output_tokens": reasoning_output_tokens} if reasoning_output_tokens else {}),
        }
        return metrics

    def _get_time_to_first_token_seconds(self, standard_logging_payload: StandardLoggingPayload) -> float:
        """
        Get the time to first token in seconds

        CompletionStartTime - StartTime = Time to first token

        For non streaming calls, CompletionStartTime is time we get the response back
        """
        start_time: Final[float | None] = standard_logging_payload.get("startTime")
        completion_start_time: Final[float | None] = standard_logging_payload.get("completionStartTime")
        end_time: Final[float | None] = standard_logging_payload.get("endTime")

        if completion_start_time is not None and start_time is not None:
            return completion_start_time - start_time
        elif end_time is not None and start_time is not None:
            return end_time - start_time
        else:
            return 0.0

    def _get_response_messages(
        self, standard_logging_payload: StandardLoggingPayload, call_type: str | None
    ) -> tuple[Message, ...]:
        """
        Get the messages from the response object

        for now this handles logging /chat/completions responses
        """

        response_obj = standard_logging_payload.get("response")
        if response_obj is None:
            return ()

        # edge case: handle response_obj is a string representation of a dict
        if isinstance(response_obj, str):
            try:
                import ast

                response_obj = ast.literal_eval(response_obj)
            except (ValueError, SyntaxError):
                try:
                    # fallback to json parsing
                    response_obj = json.loads(str(response_obj))
                except json.JSONDecodeError:
                    return ()

        if call_type in [
            CallTypes.completion.value,
            CallTypes.acompletion.value,
            CallTypes.text_completion.value,
            CallTypes.atext_completion.value,
            CallTypes.generate_content.value,
            CallTypes.agenerate_content.value,
            CallTypes.generate_content_stream.value,
            CallTypes.agenerate_content_stream.value,
            CallTypes.anthropic_messages.value,
        ]:
            try:
                # Safely extract message from response_obj, handle failure cases
                if isinstance(response_obj, dict) and "choices" in response_obj:
                    choices: Final = response_obj["choices"]
                    if choices and len(choices) > 0 and "message" in choices[0]:
                        return _to_dd_messages([choices[0]["message"]])
                return ()
            except (KeyError, IndexError, TypeError):
                # In case of any error accessing the response structure, return empty list
                return ()
        return ()

    def _get_datadog_span_kind(
        self, call_type: str | None, parent_id: str | None = None
    ) -> Literal["llm", "tool", "task", "embedding", "retrieval"]:
        """
        Map liteLLM call_type to appropriate DataDog LLM Observability span kind.

        Available DataDog span kinds: "llm", "tool", "task", "embedding", "retrieval"
        see: https://docs.datadoghq.com/ja/llm_observability/terms/
        """
        # Non llm/workflow/agent kinds cannot be root spans, so fallback to "llm" when parent metadata is missing
        if call_type is None or parent_id is None:
            return "llm"

        # Embedding operations
        if call_type in [CallTypes.embedding.value, CallTypes.aembedding.value]:
            return "embedding"

        # LLM completion operations
        if call_type in [
            CallTypes.completion.value,
            CallTypes.acompletion.value,
            CallTypes.text_completion.value,
            CallTypes.atext_completion.value,
            CallTypes.generate_content.value,
            CallTypes.agenerate_content.value,
            CallTypes.generate_content_stream.value,
            CallTypes.agenerate_content_stream.value,
            CallTypes.anthropic_messages.value,
            CallTypes.responses.value,
            CallTypes.aresponses.value,
        ]:
            return "llm"

        # Tool operations
        if call_type in [CallTypes.call_mcp_tool.value]:
            return "tool"

        # Retrieval operations
        if call_type in [
            CallTypes.get_assistants.value,
            CallTypes.aget_assistants.value,
            CallTypes.get_thread.value,
            CallTypes.aget_thread.value,
            CallTypes.get_messages.value,
            CallTypes.aget_messages.value,
            CallTypes.afile_retrieve.value,
            CallTypes.file_retrieve.value,
            CallTypes.afile_list.value,
            CallTypes.file_list.value,
            CallTypes.afile_content.value,
            CallTypes.file_content.value,
            CallTypes.retrieve_batch.value,
            CallTypes.aretrieve_batch.value,
            CallTypes.retrieve_fine_tuning_job.value,
            CallTypes.aretrieve_fine_tuning_job.value,
            CallTypes.alist_input_items.value,
        ]:
            return "retrieval"

        # Task operations (batch, fine-tuning, file operations, etc.)
        if call_type in [
            CallTypes.create_batch.value,
            CallTypes.acreate_batch.value,
            CallTypes.create_fine_tuning_job.value,
            CallTypes.acreate_fine_tuning_job.value,
            CallTypes.cancel_fine_tuning_job.value,
            CallTypes.acancel_fine_tuning_job.value,
            CallTypes.list_fine_tuning_jobs.value,
            CallTypes.alist_fine_tuning_jobs.value,
            CallTypes.create_assistants.value,
            CallTypes.acreate_assistants.value,
            CallTypes.delete_assistant.value,
            CallTypes.adelete_assistant.value,
            CallTypes.create_thread.value,
            CallTypes.acreate_thread.value,
            CallTypes.add_message.value,
            CallTypes.a_add_message.value,
            CallTypes.run_thread.value,
            CallTypes.arun_thread.value,
            CallTypes.run_thread_stream.value,
            CallTypes.arun_thread_stream.value,
            CallTypes.file_delete.value,
            CallTypes.afile_delete.value,
            CallTypes.create_file.value,
            CallTypes.acreate_file.value,
            CallTypes.image_generation.value,
            CallTypes.aimage_generation.value,
            CallTypes.image_edit.value,
            CallTypes.aimage_edit.value,
            CallTypes.moderation.value,
            CallTypes.amoderation.value,
            CallTypes.transcription.value,
            CallTypes.atranscription.value,
            CallTypes.speech.value,
            CallTypes.aspeech.value,
            CallTypes.rerank.value,
            CallTypes.arerank.value,
        ]:
            return "task"

        # Default fallback for unknown or passthrough operations
        return "llm"

    def _get_dd_llm_obs_payload_metadata(
        self,
        standard_logging_payload: StandardLoggingPayload,
        router_fields: Mapping[str, object] | None = None,
        cost_tags: Sequence[str] = (),
        redact_prompt_text: bool = False,
    ) -> dict[str, object]:
        """
        Fields to track in DD LLM Observability metadata from litellm standard logging payload
        """
        raw_metadata: Final = _metadata_of(standard_logging_payload)
        standard_logging_metadata: Final = (
            _metadata_without_prompt_carriers(raw_metadata) if redact_prompt_text else raw_metadata
        )
        return {
            "model_name": standard_logging_payload.get("model", "unknown"),
            "model_provider": standard_logging_payload.get("custom_llm_provider", "unknown"),
            "id": standard_logging_payload.get("id", "unknown"),
            "trace_id": standard_logging_payload.get("trace_id", "unknown"),
            "cache_hit": standard_logging_payload.get("cache_hit", "unknown"),
            "cache_key": standard_logging_payload.get("cache_key", "unknown"),
            "saved_cache_cost": standard_logging_payload.get("saved_cache_cost", 0),
            "guardrail_information": (
                None if redact_prompt_text else standard_logging_payload.get("guardrail_information", None)
            ),
            "is_streamed_request": self._get_stream_value_from_payload(standard_logging_payload),
            "latency_metrics": dict(self._get_latency_metrics(standard_logging_payload)),
            "spend_metrics": dict(self._get_spend_metrics(standard_logging_payload)),
            **standard_logging_metadata,
            **(router_fields or _EMPTY_MAPPING),
            **(
                {"_dd": {**_mapping_field(standard_logging_metadata, "_dd"), "cost_tags": list(cost_tags)}}
                if cost_tags
                else _EMPTY_MAPPING
            ),
        }

    def _get_latency_metrics(self, standard_logging_payload: StandardLoggingPayload) -> DDLLMObsLatencyMetrics:
        """
        Get the latency metrics from the standard logging payload
        """
        latency_metrics: Final[DDLLMObsLatencyMetrics] = DDLLMObsLatencyMetrics()
        # Add latency metrics to metadata
        # Time to first token (convert from seconds to milliseconds for consistency)
        time_to_first_token_seconds: Final = self._get_time_to_first_token_seconds(standard_logging_payload)
        if time_to_first_token_seconds > 0:
            latency_metrics["time_to_first_token_ms"] = time_to_first_token_seconds * 1000

        # LiteLLM overhead time
        hidden_params: Final = standard_logging_payload.get("hidden_params", {})
        litellm_overhead_ms: Final = hidden_params.get("litellm_overhead_time_ms")
        if litellm_overhead_ms is not None:
            latency_metrics["litellm_overhead_time_ms"] = litellm_overhead_ms

        # Guardrail overhead latency
        guardrail_info: Final[list[StandardLoggingGuardrailInformation] | None] = standard_logging_payload.get(
            "guardrail_information"
        )
        if guardrail_info is not None:
            total_duration = 0.0
            for info in guardrail_info:
                _guardrail_duration_seconds: float | None = info.get("duration")
                if _guardrail_duration_seconds is not None:
                    total_duration += float(_guardrail_duration_seconds)

            if total_duration > 0:
                # Convert from seconds to milliseconds for consistency
                latency_metrics["guardrail_overhead_time_ms"] = total_duration * 1000

        return latency_metrics

    def _get_stream_value_from_payload(self, standard_logging_payload: StandardLoggingPayload) -> bool:
        """
        Extract the stream value from standard logging payload.

        The stream field in StandardLoggingPayload is only set to True for completed streaming responses.
        For non-streaming requests, it's None. The original stream parameter is in model_parameters.

        Returns:
            bool: True if this was a streaming request, False otherwise
        """
        # Check top-level stream field first (only True for completed streaming)
        stream_value = standard_logging_payload.get("stream")
        if stream_value is True:
            return True

        # Fallback to model_parameters.stream for original request parameters
        model_params: Final = standard_logging_payload.get("model_parameters", {})
        if isinstance(model_params, dict):
            stream_value = model_params.get("stream")
            if stream_value is True:
                return True

        # Default to False for non-streaming requests
        return False

    def _get_spend_metrics(self, standard_logging_payload: StandardLoggingPayload) -> DDLLMObsSpendMetrics:
        """
        Get the spend metrics from the standard logging payload
        """
        spend_metrics: Final[DDLLMObsSpendMetrics] = DDLLMObsSpendMetrics()

        # send response cost
        spend_metrics["response_cost"] = standard_logging_payload.get("response_cost", 0.0)

        # Get budget information from metadata
        metadata: Final = _metadata_of(standard_logging_payload)

        # API key max budget
        user_api_key_max_budget: Final = metadata.get("user_api_key_max_budget")
        if user_api_key_max_budget is not None:
            spend_metrics["user_api_key_max_budget"] = float(user_api_key_max_budget)

        # API key spend
        user_api_key_spend: Final = metadata.get("user_api_key_spend")
        if user_api_key_spend is not None:
            try:
                spend_metrics["user_api_key_spend"] = float(user_api_key_spend)
            except (ValueError, TypeError):
                verbose_logger.debug("Invalid user_api_key_spend value: %s", user_api_key_spend)

        # API key budget reset datetime
        user_api_key_budget_reset_at: Final = metadata.get("user_api_key_budget_reset_at")
        if user_api_key_budget_reset_at is not None:
            try:
                from datetime import datetime, timezone

                budget_reset_at = None
                if isinstance(user_api_key_budget_reset_at, str):
                    # Handle ISO format strings that might have 'Z' suffix
                    iso_string = user_api_key_budget_reset_at.replace("Z", "+00:00")
                    budget_reset_at = datetime.fromisoformat(iso_string)
                elif isinstance(user_api_key_budget_reset_at, datetime):
                    budget_reset_at = user_api_key_budget_reset_at

                if budget_reset_at is not None:
                    # Preserve timezone info if already present
                    if budget_reset_at.tzinfo is None:
                        budget_reset_at = budget_reset_at.replace(tzinfo=timezone.utc)

                    # Convert to ISO string format for JSON serialization
                    # This prevents circular reference issues and ensures proper timezone representation
                    iso_string = budget_reset_at.isoformat()
                    spend_metrics["user_api_key_budget_reset_at"] = iso_string

                    # Debug logging to verify the conversion
                    verbose_logger.debug("Converted budget_reset_at to ISO format: %s", iso_string)
            except Exception as e:
                verbose_logger.debug("Error processing budget reset datetime: %s", e)
                verbose_logger.debug("Original value: %s", user_api_key_budget_reset_at)

        return spend_metrics
