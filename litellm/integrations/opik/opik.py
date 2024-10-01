"""
Opik Logger that logs LLM events to an Opik server
"""

from typing import Any, Dict, Callable
import datetime

import litellm
from litellm.types.utils import ModelResponse
from litellm.integrations.custom_logger import CustomLogger

from .utils import (
    pformat,
    get_current_span_id,
    get_current_trace_id,
    model_response_to_dict,
    redact_secrets,
)

class OpikLogger(CustomLogger):
    """
    Opik Logger for logging events to an Opik Server
    """
    def log_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: ModelResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        print_verbose: Callable,
    ) -> None:
        """
        Args:
            kwargs: the request dictionary
            response_obj: ModelResponse from LLM model
            start_time: datetime
            end_time: datetime
            print_verbose: function used for printing
        """
        if self.opik is None:
            print_verbose(pformat("opik is not installed", "error"))
            print_verbose(pformat("pip install opik"))
            return

        if kwargs.get("stream", False):
            print_verbose("opik stream logging")
            if kwargs.get("complete_streaming_response"):
                response_obj = kwargs["complete_streaming_response"]
            elif kwargs.get("async_complete_streaming_response"):
                response_obj = kwargs["async_complete_streaming_response"]
            else:
                print_verbose("opik skipping chunk; waiting for end...")
                return
        else:
            print_verbose("opik non-stream logging")

        # These can be set in the metadata, or in environment:
        workspace = None
        project_name = None
        host = None
        # litellm metadata:
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        # -----
        litellm_opik_metadata = metadata.get("opik", {})
        # Opik specific:
        workspace = litellm_opik_metadata.get("workspace", None)
        project_name = litellm_opik_metadata.get("project_name", None)
        host = litellm_opik_metadata.get("host", None)
        current_span_id = get_current_span_id(litellm_opik_metadata)
        current_trace_id = get_current_trace_id(litellm_opik_metadata)
        opik_metadata = litellm_opik_metadata.get("metadata", None)
        opik_tags = litellm_opik_metadata.get("tags", [])

        client = self.opik.Opik(
            workspace=workspace,
            project_name=project_name,
            host=host,
        )

        span_name = "%s_%s_%s" % (
            response_obj.get("model", "unknown-model"),
            response_obj.get("object", "unknown-object"),
            response_obj.get("created", 0),
        )
        trace_name = response_obj.get("object", "unknown type")

        input_data = redact_secrets(kwargs)
        output_data = model_response_to_dict(response_obj)
        metadata = opik_metadata or {}
        metadata["created_from"] = "litellm"
        if kwargs.get("custom_llm_provider"):
            opik_tags.append(kwargs["custom_llm_provider"])
        if "object" in response_obj:
            metadata["type"] = response_obj["object"]
        if "model" in response_obj:
            metadata["model"] = response_obj["model"]
        if "response_cost" in kwargs:
            metadata["cost"] = {
                "total_tokens": kwargs["response_cost"],
                "currency": "USD"
            }

        if current_trace_id is not None:
            print_verbose(pformat("opik trace found!"))
        else:
            print_verbose(pformat("new opik trace created!"))
            trace = client.trace(
                name=trace_name,
                input=input_data,
                output=output_data,
                metadata=metadata,
                start_time=start_time,
                end_time=end_time,
                tags=opik_tags,
            )
            current_trace_id = trace.id

        span = client.span(
            trace_id=current_trace_id,
            parent_span_id=current_span_id,
            name=span_name,
            type="llm",
            input=input_data,
            output=output_data,
            metadata=metadata,
            usage=output_data.get("usage"),
            start_time=start_time,
            end_time=end_time,
            tags=opik_tags,
        )
        client.flush()
