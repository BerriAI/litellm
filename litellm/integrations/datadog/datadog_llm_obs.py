"""
Implements logging integration with Datadog's LLM Observability Service


API Reference: https://docs.datadoghq.com/llm_observability/setup/api/?tab=example#api-standards

"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_any_messages_to_chat_completion_str_messages_conversion,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.datadog_llm_obs import *
from litellm.types.utils import CallTypes, StandardLoggingPayload


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
            CustomBatchLogger.__init__(self, **kwargs, flush_lock=self.flush_lock)
        except Exception as e:
            verbose_logger.exception(f"DataDogLLMObs: Error initializing - {str(e)}")
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug(
                f"DataDogLLMObs: Logging success event for model {kwargs.get('model', 'unknown')}"
            )
            payload = self.create_llm_obs_payload(
                kwargs, response_obj, start_time, end_time
            )
            verbose_logger.debug(f"DataDogLLMObs: Payload: {payload}")
            self.log_queue.append(payload)

            if len(self.log_queue) >= self.batch_size:
                await self.async_send_batch()
        except Exception as e:
            verbose_logger.exception(
                f"DataDogLLMObs: Error logging success event - {str(e)}"
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
            verbose_logger.debug("payload %s", json.dumps(payload, indent=4))
            response = await self.async_client.post(
                url=self.intake_url,
                json=payload,
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
        self, kwargs: Dict, response_obj: Any, start_time: datetime, end_time: datetime
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
        output_meta = OutputMeta(messages=self._get_response_messages(response_obj))

        meta = Meta(
            kind=self._get_datadog_span_kind(standard_logging_payload.get("call_type")),
            input=input_meta,
            output=output_meta,
            metadata=self._get_dd_llm_obs_payload_metadata(standard_logging_payload),
        )

        # Calculate metrics (you may need to adjust these based on available data)
        metrics = LLMMetrics(
            input_tokens=float(standard_logging_payload.get("prompt_tokens", 0)),
            output_tokens=float(standard_logging_payload.get("completion_tokens", 0)),
            total_tokens=float(standard_logging_payload.get("total_tokens", 0)),
            total_cost=float(standard_logging_payload.get("response_cost", 0)),
            time_to_first_token=self._get_time_to_first_token_seconds(standard_logging_payload),
        )

        return LLMObsPayload(
            parent_id=metadata.get("parent_id", "undefined"),
            trace_id=standard_logging_payload.get("trace_id", str(uuid.uuid4())),
            span_id=metadata.get("span_id", str(uuid.uuid4())),
            name=metadata.get("name", "litellm_llm_call"),
            meta=meta,
            start_ns=int(start_time.timestamp() * 1e9),
            duration=int((end_time - start_time).total_seconds() * 1e9),
            metrics=metrics,
            tags=[
                self._get_datadog_tags(standard_logging_object=standard_logging_payload)
            ],
        )
    
    def _get_time_to_first_token_seconds(self, standard_logging_payload: StandardLoggingPayload) -> float:
        """
        Get the time to first token in seconds

        CompletionStartTime - StartTime = Time to first token

        For non streaming calls, CompletionStartTime is time we get the response back
        """
        start_time: Optional[float] = standard_logging_payload.get("startTime")
        completion_start_time: Optional[float] = standard_logging_payload.get("completionStartTime")
        end_time: Optional[float] = standard_logging_payload.get("endTime")

        if completion_start_time is not None and start_time is not None:
            return completion_start_time - start_time
        elif end_time is not None and start_time is not None:
            return end_time - start_time
        else:
            return 0.0


    def _get_response_messages(self, response_obj: Any) -> List[Any]:
        """
        Get the messages from the response object

        for now this handles logging /chat/completions responses
        """
        if isinstance(response_obj, litellm.ModelResponse):
            return [response_obj["choices"][0]["message"].json()]
        return []

    def _get_datadog_span_kind(self, call_type: Optional[str]) -> Literal["llm", "tool", "task", "embedding", "retrieval"]:
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
            CallTypes.anthropic_messages.value
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
            CallTypes.alist_input_items.value
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
            CallTypes.arerank.value
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
    ) -> Dict:
        """
        Fields to track in DD LLM Observability metadata from litellm standard logging payload
        """
        _metadata = {
            "model_name": standard_logging_payload.get("model", "unknown"),
            "model_provider": standard_logging_payload.get(
                "custom_llm_provider", "unknown"
            ),
            "id": standard_logging_payload.get("id", "unknown"),
            "trace_id": standard_logging_payload.get("trace_id", "unknown"),
            "cache_hit": standard_logging_payload.get("cache_hit", "unknown"),
            "cache_key": standard_logging_payload.get("cache_key", "unknown"),
            "saved_cache_cost": standard_logging_payload.get("saved_cache_cost", 0),
        }
        _standard_logging_metadata: dict = (
            dict(standard_logging_payload.get("metadata", {})) or {}
        )
        _metadata.update(_standard_logging_metadata)
        return _metadata
