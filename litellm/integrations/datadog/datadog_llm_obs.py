"""
Implements logging integration with Datadog's LLM Observability Service


API Reference: https://docs.datadoghq.com/llm_observability/setup/api/?tab=example#api-standards

"""

import asyncio
import json
import os
from litellm._uuid import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_any_messages_to_chat_completion_str_messages_conversion,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.datadog_llm_obs import *
from litellm.types.utils import (
    CallTypes,
    StandardLoggingGuardrailInformation,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
)


class DataDogLLMObsLogger(DataDogLogger, CustomBatchLogger):
    def __init__(self, **kwargs):
        try:
            verbose_logger.debug("DataDogLLMObs: Initializing logger")
            if os.getenv("DD_API_KEY", None) is None:
                raise Exception("DD_API_KEY is not set, set 'DD_API_KEY=<>'")
            if os.getenv("DD_SITE", None) is None:
                raise Exception(
                    "DD_SITE is not set, set 'DD_SITE=<>', example sit = `us5.datadoghq.com`"
                )

            self.async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            self.DD_API_KEY = os.getenv("DD_API_KEY")
            self.DD_SITE = os.getenv("DD_SITE")
            self.intake_url = (
                f"https://api.{self.DD_SITE}/api/intake/llm-obs/v1/trace/spans"
            )

            # testing base url
            dd_base_url = os.getenv("DD_BASE_URL")
            if dd_base_url:
                self.intake_url = f"{dd_base_url}/api/intake/llm-obs/v1/trace/spans"

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()
            self.log_queue: List[LLMObsPayload] = []

            #########################################################
            # Handle datadog_llm_observability_params set as litellm.datadog_llm_observability_params
            #########################################################
            dict_datadog_llm_obs_params = self._get_datadog_llm_obs_params()
            kwargs.update(dict_datadog_llm_obs_params)
            CustomBatchLogger.__init__(self, **kwargs, flush_lock=self.flush_lock)
        except Exception as e:
            verbose_logger.exception(f"DataDogLLMObs: Error initializing - {str(e)}")
            raise e

    def _get_datadog_llm_obs_params(self) -> Dict:
        """
        Get the datadog_llm_observability_params from litellm.datadog_llm_observability_params

        These are params specific to initializing the DataDogLLMObsLogger e.g. turn_off_message_logging
        """
        dict_datadog_llm_obs_params: Dict = {}
        if litellm.datadog_llm_observability_params is not None:
            if isinstance(
                litellm.datadog_llm_observability_params, DatadogLLMObsInitParams
            ):
                dict_datadog_llm_obs_params = (
                    litellm.datadog_llm_observability_params.model_dump()
                )
            elif isinstance(litellm.datadog_llm_observability_params, Dict):
                # only allow params that are of DatadogLLMObsInitParams
                dict_datadog_llm_obs_params = DatadogLLMObsInitParams(
                    **litellm.datadog_llm_observability_params
                ).model_dump()
        return dict_datadog_llm_obs_params

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                f"DataDogLLMObs: Logging success event for model {kwargs.get('model', 'unknown')}"
            )
            payload = self.create_llm_obs_payload(kwargs, start_time, end_time)
            verbose_logger.debug(f"DataDogLLMObs: Payload: {payload}")
            self.log_queue.append(payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()
        except Exception as e:
            verbose_logger.exception(
                f"DataDogLLMObs: Error logging success event - {str(e)}"
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                f"DataDogLLMObs: Logging failure event for model {kwargs.get('model', 'unknown')}"
            )
            payload = self.create_llm_obs_payload(kwargs, start_time, end_time)
            verbose_logger.debug(f"DataDogLLMObs: Payload: {payload}")
            self.log_queue.append(payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()
        except Exception as e:
            verbose_logger.exception(
                f"DataDogLLMObs: Error logging failure event - {str(e)}"
            )

    async def async_send_batch(self):
        try:
            if not self.log_queue:
                return

            verbose_logger.debug(
                f"DataDogLLMObs: Flushing {len(self.log_queue)} events"
            )

            # Prepare the payload
            payload = {
                "data": DDIntakePayload(
                    type="span",
                    attributes=DDSpanAttributes(
                        ml_app=self._get_datadog_service(),
                        tags=[self._get_datadog_tags()],
                        spans=self.log_queue,
                    ),
                ),
            }

            # serialize datetime objects - for budget reset time in spend metrics
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            try:
                verbose_logger.debug("payload %s", safe_dumps(payload))
            except Exception as debug_error:
                verbose_logger.debug(
                    "payload serialization failed: %s", str(debug_error)
                )

            json_payload = safe_dumps(payload)

            response = await self.async_client.post(
                url=self.intake_url,
                content=json_payload,
                headers={
                    "DD-API-KEY": self.DD_API_KEY,
                    "Content-Type": "application/json",
                },
            )

            if response.status_code != 202:
                raise Exception(
                    f"DataDogLLMObs: Unexpected response - status_code: {response.status_code}, text: {response.text}"
                )

            verbose_logger.debug(
                f"DataDogLLMObs: Successfully sent batch - status_code: {response.status_code}"
            )
            self.log_queue.clear()
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                f"DataDogLLMObs: Error sending batch - {e.response.text}"
            )
        except Exception as e:
            verbose_logger.exception(f"DataDogLLMObs: Error sending batch - {str(e)}")

    def create_llm_obs_payload(
        self, kwargs: Dict, start_time: datetime, end_time: datetime
    ) -> LLMObsPayload:
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload is None:
            raise Exception("DataDogLLMObs: standard_logging_object is not set")

        messages = standard_logging_payload["messages"]
        messages = self._ensure_string_content(messages=messages)

        metadata = kwargs.get("litellm_params", {}).get("metadata", {})

        input_meta = InputMeta(
            messages=handle_any_messages_to_chat_completion_str_messages_conversion(
                messages
            )
        )
        output_meta = OutputMeta(
            messages=self._get_response_messages(
                standard_logging_payload=standard_logging_payload,
                call_type=standard_logging_payload.get("call_type"),
            )
        )

        error_info = self._assemble_error_info(standard_logging_payload)

        meta = Meta(
            kind=self._get_datadog_span_kind(standard_logging_payload.get("call_type")),
            input=input_meta,
            output=output_meta,
            metadata=self._get_dd_llm_obs_payload_metadata(standard_logging_payload),
            error=error_info,
        )

        # Calculate metrics (you may need to adjust these based on available data)
        metrics = LLMMetrics(
            input_tokens=float(standard_logging_payload.get("prompt_tokens", 0)),
            output_tokens=float(standard_logging_payload.get("completion_tokens", 0)),
            total_tokens=float(standard_logging_payload.get("total_tokens", 0)),
            total_cost=float(standard_logging_payload.get("response_cost", 0)),
            time_to_first_token=self._get_time_to_first_token_seconds(
                standard_logging_payload
            ),
        )

        payload: LLMObsPayload = LLMObsPayload(
            parent_id=metadata.get("parent_id", "undefined"),
            trace_id=standard_logging_payload.get("trace_id", str(uuid.uuid4())),
            span_id=metadata.get("span_id", str(uuid.uuid4())),
            name=metadata.get("name", "litellm_llm_call"),
            meta=meta,
            start_ns=int(start_time.timestamp() * 1e9),
            duration=int((end_time - start_time).total_seconds() * 1e9),
            metrics=metrics,
            status="error" if error_info else "ok",
            tags=[
                self._get_datadog_tags(standard_logging_object=standard_logging_payload)
            ],
        )

        apm_trace_id = self._get_apm_trace_id()
        if apm_trace_id is not None:
            payload["apm_id"] = apm_trace_id

        return payload

    def _get_apm_trace_id(self) -> Optional[str]:
        """Retrieve the current APM trace ID if available."""
        try:
            current_span_fn = getattr(tracer, "current_span", None)
            if callable(current_span_fn):
                current_span = current_span_fn()
                if current_span is not None:
                    trace_id = getattr(current_span, "trace_id", None)
                    if trace_id is not None:
                        return str(trace_id)
        except Exception:
            pass
        return None

    def _assemble_error_info(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> Optional[DDLLMObsError]:
        """
        Assemble error information for failure cases according to DD LLM Obs API spec
        """
        # Handle error information for failure cases according to DD LLM Obs API spec
        error_info: Optional[DDLLMObsError] = None

        if standard_logging_payload.get("status") == "failure":
            # Try to get structured error information first
            error_information: Optional[
                StandardLoggingPayloadErrorInformation
            ] = standard_logging_payload.get("error_information")

            if error_information:
                error_info = DDLLMObsError(
                    message=error_information.get("error_message")
                    or standard_logging_payload.get("error_str")
                    or "Unknown error",
                    type=error_information.get("error_class"),
                    stack=error_information.get("traceback"),
                )
        return error_info

    def _get_time_to_first_token_seconds(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> float:
        """
        Get the time to first token in seconds

        CompletionStartTime - StartTime = Time to first token

        For non streaming calls, CompletionStartTime is time we get the response back
        """
        start_time: Optional[float] = standard_logging_payload.get("startTime")
        completion_start_time: Optional[float] = standard_logging_payload.get(
            "completionStartTime"
        )
        end_time: Optional[float] = standard_logging_payload.get("endTime")

        if completion_start_time is not None and start_time is not None:
            return completion_start_time - start_time
        elif end_time is not None and start_time is not None:
            return end_time - start_time
        else:
            return 0.0

    def _get_response_messages(
        self, standard_logging_payload: StandardLoggingPayload, call_type: Optional[str]
    ) -> List[Any]:
        """
        Get the messages from the response object

        for now this handles logging /chat/completions responses
        """

        response_obj = standard_logging_payload.get("response")
        if response_obj is None:
            return []

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
                    return []

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
                    choices = response_obj["choices"]
                    if choices and len(choices) > 0 and "message" in choices[0]:
                        return [choices[0]["message"]]
                return []
            except (KeyError, IndexError, TypeError):
                # In case of any error accessing the response structure, return empty list
                return []
        return []

    def _get_datadog_span_kind(
        self, call_type: Optional[str]
    ) -> Literal["llm", "tool", "task", "embedding", "retrieval"]:
        """
        Map liteLLM call_type to appropriate DataDog LLM Observability span kind.

        Available DataDog span kinds: "llm", "tool", "task", "embedding", "retrieval"
        """
        if call_type is None:
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
            CallTypes.responses.value,
            CallTypes.aresponses.value,
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

    def _ensure_string_content(
        self, messages: Optional[Union[str, List[Any], Dict[Any, Any]]]
    ) -> List[Any]:
        if messages is None:
            return []
        if isinstance(messages, str):
            return [messages]
        elif isinstance(messages, list):
            return [message for message in messages]
        elif isinstance(messages, dict):
            return [str(messages.get("content", ""))]
        return []

    def _get_dd_llm_obs_payload_metadata(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> Dict[str, Any]:
        """
        Fields to track in DD LLM Observability metadata from litellm standard logging payload
        """
        _metadata: Dict[str, Any] = {
            "model_name": standard_logging_payload.get("model", "unknown"),
            "model_provider": standard_logging_payload.get(
                "custom_llm_provider", "unknown"
            ),
            "id": standard_logging_payload.get("id", "unknown"),
            "trace_id": standard_logging_payload.get("trace_id", "unknown"),
            "cache_hit": standard_logging_payload.get("cache_hit", "unknown"),
            "cache_key": standard_logging_payload.get("cache_key", "unknown"),
            "saved_cache_cost": standard_logging_payload.get("saved_cache_cost", 0),
            "guardrail_information": standard_logging_payload.get(
                "guardrail_information", None
            ),
            "is_streamed_request": self._get_stream_value_from_payload(
                standard_logging_payload
            ),
        }

        #########################################################
        # Add latency metrics to metadata
        #########################################################
        latency_metrics = self._get_latency_metrics(standard_logging_payload)
        _metadata.update({"latency_metrics": dict(latency_metrics)})

        #########################################################
        # Add spend metrics to metadata
        #########################################################
        spend_metrics = self._get_spend_metrics(standard_logging_payload)
        _metadata.update({"spend_metrics": dict(spend_metrics)})

        ## extract tool calls and add to metadata
        tool_call_metadata = self._extract_tool_call_metadata(standard_logging_payload)
        _metadata.update(tool_call_metadata)

        _standard_logging_metadata: dict = (
            dict(standard_logging_payload.get("metadata", {})) or {}
        )
        _metadata.update(_standard_logging_metadata)
        return _metadata

    def _get_latency_metrics(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> DDLLMObsLatencyMetrics:
        """
        Get the latency metrics from the standard logging payload
        """
        latency_metrics: DDLLMObsLatencyMetrics = DDLLMObsLatencyMetrics()
        # Add latency metrics to metadata
        # Time to first token (convert from seconds to milliseconds for consistency)
        time_to_first_token_seconds = self._get_time_to_first_token_seconds(
            standard_logging_payload
        )
        if time_to_first_token_seconds > 0:
            latency_metrics["time_to_first_token_ms"] = (
                time_to_first_token_seconds * 1000
            )

        # LiteLLM overhead time
        hidden_params = standard_logging_payload.get("hidden_params", {})
        litellm_overhead_ms = hidden_params.get("litellm_overhead_time_ms")
        if litellm_overhead_ms is not None:
            latency_metrics["litellm_overhead_time_ms"] = litellm_overhead_ms

        # Guardrail overhead latency
        guardrail_info: Optional[
            list[StandardLoggingGuardrailInformation]
        ] = standard_logging_payload.get("guardrail_information")
        if guardrail_info is not None:
            total_duration = 0.0
            for info in guardrail_info:
                _guardrail_duration_seconds: Optional[float] = info.get("duration")
                if _guardrail_duration_seconds is not None:
                    total_duration += float(_guardrail_duration_seconds)

            if total_duration > 0:
                # Convert from seconds to milliseconds for consistency
                latency_metrics["guardrail_overhead_time_ms"] = total_duration * 1000

        return latency_metrics

    def _get_stream_value_from_payload(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> bool:
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
        model_params = standard_logging_payload.get("model_parameters", {})
        if isinstance(model_params, dict):
            stream_value = model_params.get("stream")
            if stream_value is True:
                return True

        # Default to False for non-streaming requests
        return False

    def _get_spend_metrics(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> DDLLMObsSpendMetrics:
        """
        Get the spend metrics from the standard logging payload
        """
        spend_metrics: DDLLMObsSpendMetrics = DDLLMObsSpendMetrics()

        # send response cost
        spend_metrics["response_cost"] = standard_logging_payload.get(
            "response_cost", 0.0
        )

        # Get budget information from metadata
        metadata = standard_logging_payload.get("metadata", {})

        # API key max budget
        user_api_key_max_budget = metadata.get("user_api_key_max_budget")
        if user_api_key_max_budget is not None:
            spend_metrics["user_api_key_max_budget"] = float(user_api_key_max_budget)

        # API key spend
        user_api_key_spend = metadata.get("user_api_key_spend")
        if user_api_key_spend is not None:
            try:
                spend_metrics["user_api_key_spend"] = float(user_api_key_spend)
            except (ValueError, TypeError):
                verbose_logger.debug(
                    f"Invalid user_api_key_spend value: {user_api_key_spend}"
                )

        # API key budget reset datetime
        user_api_key_budget_reset_at = metadata.get("user_api_key_budget_reset_at")
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
                    verbose_logger.debug(
                        f"Converted budget_reset_at to ISO format: {iso_string}"
                    )
            except Exception as e:
                verbose_logger.debug(f"Error processing budget reset datetime: {e}")
                verbose_logger.debug(f"Original value: {user_api_key_budget_reset_at}")

        return spend_metrics

    def _process_input_messages_preserving_tool_calls(
        self, messages: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Process input messages while preserving tool_calls and tool message types.

        This bypasses the lossy string conversion when tool calls are present,
        allowing complex nested tool_calls objects to be preserved for Datadog.
        """
        processed = []
        for msg in messages:
            if isinstance(msg, dict):
                # Preserve messages with tool_calls or tool role as-is
                if "tool_calls" in msg or msg.get("role") == "tool":
                    processed.append(msg)
                else:
                    # For regular messages, still apply string conversion
                    converted = (
                        handle_any_messages_to_chat_completion_str_messages_conversion(
                            [msg]
                        )
                    )
                    processed.extend(converted)
            else:
                # For non-dict messages, apply string conversion
                converted = (
                    handle_any_messages_to_chat_completion_str_messages_conversion(
                        [msg]
                    )
                )
                processed.extend(converted)
        return processed

    @staticmethod
    def _tool_calls_kv_pair(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract tool call information into key-value pairs for Datadog metadata.

        Similar to OpenTelemetry's implementation but adapted for Datadog's format.
        """
        kv_pairs: Dict[str, Any] = {}
        for idx, tool_call in enumerate(tool_calls):
            try:
                # Extract tool call ID
                tool_id = tool_call.get("id")
                if tool_id:
                    kv_pairs[f"tool_calls.{idx}.id"] = tool_id

                # Extract tool call type
                tool_type = tool_call.get("type")
                if tool_type:
                    kv_pairs[f"tool_calls.{idx}.type"] = tool_type

                # Extract function information
                function = tool_call.get("function")
                if function:
                    function_name = function.get("name")
                    if function_name:
                        kv_pairs[f"tool_calls.{idx}.function.name"] = function_name

                    function_arguments = function.get("arguments")
                    if function_arguments:
                        # Store arguments as JSON string for Datadog
                        if isinstance(function_arguments, str):
                            kv_pairs[
                                f"tool_calls.{idx}.function.arguments"
                            ] = function_arguments
                        else:
                            import json

                            kv_pairs[
                                f"tool_calls.{idx}.function.arguments"
                            ] = json.dumps(function_arguments)
            except (KeyError, TypeError, ValueError) as e:
                verbose_logger.debug(
                    f"DataDogLLMObs: Error processing tool call {idx}: {str(e)}"
                )
                continue

        return kv_pairs

    def _extract_tool_call_metadata(
        self, standard_logging_payload: StandardLoggingPayload
    ) -> Dict[str, Any]:
        """
        Extract tool call information from both input messages and response for Datadog metadata.
        """
        tool_call_metadata: Dict[str, Any] = {}

        try:
            # Extract tool calls from input messages
            messages = standard_logging_payload.get("messages", [])
            if messages and isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and "tool_calls" in message:
                        tool_calls = message.get("tool_calls")
                        if tool_calls:
                            input_tool_calls_kv = self._tool_calls_kv_pair(tool_calls)
                            # Prefix with "input_" to distinguish from response tool calls
                            for key, value in input_tool_calls_kv.items():
                                tool_call_metadata[f"input_{key}"] = value

            # Extract tool calls from response
            response_obj = standard_logging_payload.get("response")
            if response_obj and isinstance(response_obj, dict):
                choices = response_obj.get("choices", [])
                for choice in choices:
                    if isinstance(choice, dict):
                        message = choice.get("message")
                        if message and isinstance(message, dict):
                            tool_calls = message.get("tool_calls")
                            if tool_calls:
                                response_tool_calls_kv = self._tool_calls_kv_pair(
                                    tool_calls
                                )
                                # Prefix with "output_" to distinguish from input tool calls
                                for key, value in response_tool_calls_kv.items():
                                    tool_call_metadata[f"output_{key}"] = value

        except Exception as e:
            verbose_logger.debug(
                f"DataDogLLMObs: Error extracting tool call metadata: {str(e)}"
            )

        return tool_call_metadata
