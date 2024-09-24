import datetime
import json
from typing import Optional
from litellm.integrations.custom_logger import CustomLogger
from litellm.utils import verbose_logger

import mlflow
from mlflow.entities import Span, SpanType, SpanStatusCode
from mlflow.tracking import MlflowClient


_UNKNOWN_CALL_ID = "UNKNOWN"
class MlflowLogger(CustomLogger):
    def __init__(self):
        self._client = MlflowClient()
        self._call_id_to_span = {}

    def log_pre_api_call(self, model, messages, kwargs):
        try:
            verbose_logger.debug(f"MLflow logging start")

            litellm_call_id = kwargs.get("litellm_call_id", _UNKNOWN_CALL_ID)
            span_name = f"litellm-{kwargs.get('call_type', 'completion')}"

            start_time_dt = kwargs.get("api_call_start_time", datetime.datetime.now())
            start_time_ns = int(start_time_dt.timestamp() * 1e9)

            # Construct inputs with optional parameters
            inputs = {"messages": messages}
            for key in ["functions", "tools", "stream", "tool_choice", "user"]:
                if value := kwargs.get("optional_params", {}).pop(key, None):
                    inputs[key] = value

            # Extract metadata about the call
            litellm_params = kwargs.get("litellm_params", {})
            attributes = {
                "model": model or kwargs.get("model", None),
                "call_type": kwargs.get("call_type", "completion"),
                "litellm_call_id": litellm_call_id,
                "custom_llm_provider": kwargs.get("custom_llm_provider", None),
                "api_base": litellm_params.get("api_base", None),
                "metadata": litellm_params.get("metadata", {})
            }

            span = self._start_span_or_trace(
                name=span_name,
                # Tentative - determined more precisely in the post_api_call
                span_type=self._get_span_type(kwargs.get("call_type")),
                inputs=inputs,
                attributes=attributes,
                start_time_ns=start_time_ns,
            )
            self._call_id_to_span[litellm_call_id] = span
        except Exception as e:
            verbose_logger.debug(f"MLflow Logging Error: {e}", stack_info=True)
            pass

    def _get_span_type(self, call_type: Optional[str]) -> str:
        if call_type in ["completion", "acompletion"]:
            return SpanType.LLM
        elif call_type == "embeddings":
            return SpanType.EMBEDDING
        else:
            return SpanType.LLM


    def log_post_api_call(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
    ):
        """

        NB: We use log_post_api_call as this is only available synchronous hook in the current
            LiteLLM logger.
        """
        try:
            litellm_call_id = kwargs.get("litellm_call_id", _UNKNOWN_CALL_ID)
            span = self._call_id_to_span.pop(litellm_call_id, None)
            end_time_ns = int(end_time.timestamp() * 1e9)

            if span is None:
                verbose_logger.debug(f"MLflow Logging Error - span not found for call_id: {litellm_call_id}")
                return

            try:
                outputs = json.loads(response_obj)
            except:
                outputs = response_obj

            self._end_span_or_trace(
                span=span,
                outputs=outputs,
                # NB: The response object passed to the log_post_api_call hook is serialized and does not
                #  contain information about whether the call was successful or not. Therefore, we always
                #  set the status to OK.
                status=SpanStatusCode.OK,
                end_time_ns=end_time_ns,
            )

            verbose_logger.debug("MLflow logging end")
        except Exception as e:
            verbose_logger.debug(f"MLflow logging error: {e}", stack_info=True)
            pass

    def _start_span_or_trace(self, name, span_type, inputs, attributes, start_time_ns) -> Span:
        if active_span := mlflow.get_current_active_span():
            return self._client.start_span(
                name=name,
                request_id=active_span.request_id,
                parent_id=active_span.span_id,
                span_type=span_type,
                inputs=inputs,
                attributes=attributes,
                start_time_ns=start_time_ns,
            )
        else:
            return self._client.start_trace(
                name=name,
                span_type=span_type,
                inputs=inputs,
                attributes=attributes,
                start_time_ns=start_time_ns,
            )

    def _end_span_or_trace(self, span, outputs, end_time_ns, status):
        if span.parent_id is None:
            self._client.end_trace(
                request_id=span.request_id,
                outputs=outputs,
                status=status,
                end_time_ns=end_time_ns,
            )
        else:
            self._client.end_span(
                request_id=span.request_id,
                span_id=span.span_id,
                outputs=outputs,
                status=status,
                end_time_ns=end_time_ns,
            )
