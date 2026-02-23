"""
Transformation for Bedrock Invoke Agent

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeAgent.html
"""

import base64
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.bedrock_invoke_agents import (
    InvokeAgentChunkPayload,
    InvokeAgentEvent,
    InvokeAgentEventHeaders,
    InvokeAgentEventList,
    InvokeAgentMetadata,
    InvokeAgentModelInvocationInput,
    InvokeAgentModelInvocationOutput,
    InvokeAgentOrchestrationTrace,
    InvokeAgentPreProcessingTrace,
    InvokeAgentTrace,
    InvokeAgentTracePayload,
    InvokeAgentUsage,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonInvokeAgentConfig(BaseConfig, BaseAWSLLM):
    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        This is a base invoke agent model mapping. For Invoke Agent - define a bedrock provider specific config that extends this class.

        Bedrock Invoke Agents has 0 OpenAI compatible params

        As of May 29th, 2025 - they don't support streaming.
        """
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        This is a base invoke agent model mapping. For Invoke Agent - define a bedrock provider specific config that extends this class.
        """
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request
        """
        ### SET RUNTIME ENDPOINT ###
        aws_bedrock_runtime_endpoint = optional_params.get(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
        endpoint_url, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=self._get_aws_region_name(
                optional_params=optional_params, model=model
            ),
            endpoint_type="agent",
        )

        agent_id, agent_alias_id = self._get_agent_id_and_alias_id(model)
        session_id = self._get_session_id(optional_params)

        endpoint_url = f"{endpoint_url}/agents/{agent_id}/agentAliases/{agent_alias_id}/sessions/{session_id}/text"

        return endpoint_url

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
            api_key=api_key,
        )

    def _get_agent_id_and_alias_id(self, model: str) -> tuple[str, str]:
        """
        model = "agent/L1RT58GYRW/MFPSBCXYTW"
        agent_id = "L1RT58GYRW"
        agent_alias_id = "MFPSBCXYTW"
        """
        # Split the model string by '/' and extract components
        parts = model.split("/")
        if len(parts) != 3 or parts[0] != "agent":
            raise ValueError(
                "Invalid model format. Expected format: 'model=agent/AGENT_ID/ALIAS_ID'"
            )

        return parts[1], parts[2]  # Return (agent_id, agent_alias_id)

    def _get_session_id(self, optional_params: dict) -> str:
        """ """
        return optional_params.get("sessionID", None) or str(uuid.uuid4())

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # use the last message content as the query
        query: str = convert_content_list_to_str(messages[-1])
        return {
            "inputText": query,
            "enableTrace": True,
            **optional_params,
        }

    def _parse_aws_event_stream(self, raw_content: bytes) -> InvokeAgentEventList:
        """
        Parse AWS event stream format using boto3/botocore's built-in parser.
        This is the same approach used in the existing AWSEventStreamDecoder.
        """
        try:
            from botocore.eventstream import EventStreamBuffer
            from botocore.parsers import EventStreamJSONParser
        except ImportError:
            raise ImportError("boto3/botocore is required for AWS event stream parsing")

        events: InvokeAgentEventList = []
        parser = EventStreamJSONParser()
        event_stream_buffer = EventStreamBuffer()

        # Add the entire response to the buffer
        event_stream_buffer.add_data(raw_content)

        # Process all events in the buffer
        for event in event_stream_buffer:
            try:
                headers = self._extract_headers_from_event(event)

                event_type = headers.get("event_type", "")

                if event_type == "chunk":
                    # Handle chunk events specially - they contain decoded content, not JSON
                    message = self._parse_message_from_event(event, parser)
                    parsed_event: InvokeAgentEvent = InvokeAgentEvent()
                    if message:
                        # For chunk events, create a payload with the decoded content
                        parsed_event = {
                            "headers": headers,
                            "payload": {
                                "bytes": base64.b64encode(
                                    message.encode("utf-8")
                                ).decode("utf-8")
                            },  # Re-encode for consistency
                        }
                        events.append(parsed_event)

                elif event_type == "trace":
                    # Handle trace events normally - they contain JSON
                    message = self._parse_message_from_event(event, parser)

                    if message:
                        try:
                            event_data = json.loads(message)
                            parsed_event = {
                                "headers": headers,
                                "payload": event_data,
                            }
                            events.append(parsed_event)
                        except json.JSONDecodeError as e:
                            verbose_logger.warning(
                                f"Failed to parse trace event JSON: {e}"
                            )
                else:
                    verbose_logger.debug(f"Unknown event type: {event_type}")

            except Exception as e:
                verbose_logger.error(f"Error processing event: {e}")
                continue

        return events

    def _parse_message_from_event(self, event, parser) -> Optional[str]:
        """Extract message content from an AWS event, adapted from AWSEventStreamDecoder."""
        try:
            response_dict = event.to_response_dict()
            verbose_logger.debug(f"Response dict: {response_dict}")

            # Use the same response shape parsing as the existing decoder
            parsed_response = parser.parse(
                response_dict, self._get_response_stream_shape()
            )
            verbose_logger.debug(f"Parsed response: {parsed_response}")

            if response_dict["status_code"] != 200:
                decoded_body = response_dict["body"].decode()
                if isinstance(decoded_body, dict):
                    error_message = decoded_body.get("message")
                elif isinstance(decoded_body, str):
                    error_message = decoded_body
                else:
                    error_message = ""
                exception_status = response_dict["headers"].get(":exception-type")
                error_message = exception_status + " " + error_message
                raise BedrockError(
                    status_code=response_dict["status_code"],
                    message=(
                        json.dumps(error_message)
                        if isinstance(error_message, dict)
                        else error_message
                    ),
                )

            if "chunk" in parsed_response:
                chunk = parsed_response.get("chunk")
                if not chunk:
                    return None
                return chunk.get("bytes").decode()
            else:
                chunk = response_dict.get("body")
                if not chunk:
                    return None
                return chunk.decode()

        except Exception as e:
            verbose_logger.debug(f"Error parsing message from event: {e}")
            return None

    def _extract_headers_from_event(self, event) -> InvokeAgentEventHeaders:
        """Extract headers from an AWS event for categorization."""
        try:
            response_dict = event.to_response_dict()
            headers = response_dict.get("headers", {})

            # Extract the event-type and content-type headers that we care about
            return InvokeAgentEventHeaders(
                event_type=headers.get(":event-type", ""),
                content_type=headers.get(":content-type", ""),
                message_type=headers.get(":message-type", ""),
            )
        except Exception as e:
            verbose_logger.debug(f"Error extracting headers: {e}")
            return InvokeAgentEventHeaders(
                event_type="", content_type="", message_type=""
            )

    def _get_response_stream_shape(self):
        """Get the response stream shape for parsing, reusing existing logic."""
        try:
            # Try to reuse the cached shape from the existing decoder
            from litellm.llms.bedrock.chat.invoke_handler import (
                get_response_stream_shape,
            )

            return get_response_stream_shape()
        except ImportError:
            # Fallback: create our own shape
            try:
                from botocore.loaders import Loader
                from botocore.model import ServiceModel

                loader = Loader()
                bedrock_service_dict = loader.load_service_model(
                    "bedrock-runtime", "service-2"
                )
                bedrock_service_model = ServiceModel(bedrock_service_dict)
                return bedrock_service_model.shape_for("ResponseStream")
            except Exception as e:
                verbose_logger.warning(f"Could not load response stream shape: {e}")
                return None

    def _extract_response_content(self, events: InvokeAgentEventList) -> str:
        """Extract the final response content from parsed events."""
        response_parts = []

        for event in events:
            headers = event.get("headers", {})
            payload = event.get("payload")

            event_type = headers.get(
                "event_type"
            )  # Note: using event_type not event-type

            if event_type == "chunk" and payload:
                # Extract base64 encoded content from chunk events
                chunk_payload: InvokeAgentChunkPayload = payload  # type: ignore
                encoded_bytes = chunk_payload.get("bytes", "")
                if encoded_bytes:
                    try:
                        decoded_content = base64.b64decode(encoded_bytes).decode(
                            "utf-8"
                        )
                        response_parts.append(decoded_content)
                    except Exception as e:
                        verbose_logger.warning(f"Failed to decode chunk content: {e}")

        return "".join(response_parts)

    def _extract_usage_info(self, events: InvokeAgentEventList) -> InvokeAgentUsage:
        """Extract token usage information from trace events."""
        usage_info = InvokeAgentUsage(
            inputTokens=0,
            outputTokens=0,
            model=None,
        )

        response_model: Optional[str] = None

        for event in events:
            if not self._is_trace_event(event):
                continue

            trace_data = self._get_trace_data(event)
            if not trace_data:
                continue

            verbose_logger.debug(f"Trace event: {trace_data}")

            # Extract usage from pre-processing trace
            self._extract_and_update_preprocessing_usage(
                trace_data=trace_data,
                usage_info=usage_info,
            )

            # Extract model from orchestration trace
            if response_model is None:
                response_model = self._extract_orchestration_model(trace_data)

        usage_info["model"] = response_model
        return usage_info

    def _is_trace_event(self, event: InvokeAgentEvent) -> bool:
        """Check if the event is a trace event."""
        headers = event.get("headers", {})
        event_type = headers.get("event_type")
        payload = event.get("payload")
        return event_type == "trace" and payload is not None

    def _get_trace_data(self, event: InvokeAgentEvent) -> Optional[InvokeAgentTrace]:
        """Extract trace data from a trace event."""
        payload = event.get("payload")
        if not payload:
            return None

        trace_payload: InvokeAgentTracePayload = payload  # type: ignore
        return trace_payload.get("trace", {})

    def _extract_and_update_preprocessing_usage(
        self, trace_data: InvokeAgentTrace, usage_info: InvokeAgentUsage
    ) -> None:
        """Extract usage information from preprocessing trace."""
        pre_processing: Optional[InvokeAgentPreProcessingTrace] = trace_data.get(
            "preProcessingTrace"
        )
        if not pre_processing:
            return

        model_output: Optional[InvokeAgentModelInvocationOutput] = (
            pre_processing.get("modelInvocationOutput")
            or InvokeAgentModelInvocationOutput()
        )
        if not model_output:
            return

        metadata: Optional[InvokeAgentMetadata] = (
            model_output.get("metadata") or InvokeAgentMetadata()
        )
        if not metadata:
            return

        usage: Optional[Union[InvokeAgentUsage, Dict]] = metadata.get("usage", {})
        if not usage:
            return

        usage_info["inputTokens"] += usage.get("inputTokens", 0)
        usage_info["outputTokens"] += usage.get("outputTokens", 0)

    def _extract_orchestration_model(
        self, trace_data: InvokeAgentTrace
    ) -> Optional[str]:
        """Extract model information from orchestration trace."""
        orchestration_trace: Optional[InvokeAgentOrchestrationTrace] = trace_data.get(
            "orchestrationTrace"
        )
        if not orchestration_trace:
            return None

        model_invocation: Optional[InvokeAgentModelInvocationInput] = (
            orchestration_trace.get("modelInvocationInput")
            or InvokeAgentModelInvocationInput()
        )
        if not model_invocation:
            return None

        return model_invocation.get("foundationModel")

    def _build_model_response(
        self,
        content: str,
        model: str,
        usage_info: InvokeAgentUsage,
        model_response: ModelResponse,
    ) -> ModelResponse:
        """Build the final ModelResponse object."""

        # Create the message content
        message = Message(content=content, role="assistant")

        # Create choices
        choice = Choices(finish_reason="stop", index=0, message=message)

        # Update model response
        model_response.choices = [choice]
        model_response.model = usage_info.get("model", model)

        # Add usage information if available
        if usage_info:
            from litellm.types.utils import Usage

            usage = Usage(
                prompt_tokens=usage_info.get("inputTokens", 0),
                completion_tokens=usage_info.get("outputTokens", 0),
                total_tokens=usage_info.get("inputTokens", 0)
                + usage_info.get("outputTokens", 0),
            )
            setattr(model_response, "usage", usage)

        return model_response

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            # Get the raw binary content
            raw_content = raw_response.content
            verbose_logger.debug(
                f"Processing {len(raw_content)} bytes of AWS event stream data"
            )

            # Parse the AWS event stream format
            events = self._parse_aws_event_stream(raw_content)
            verbose_logger.debug(f"Parsed {len(events)} events from stream")

            # Extract response content from chunk events
            content = self._extract_response_content(events)

            # Extract usage information from trace events
            usage_info = self._extract_usage_info(events)

            # Build and return the model response
            return self._build_model_response(
                content=content,
                model=model,
                usage_info=usage_info,
                model_response=model_response,
            )

        except Exception as e:
            verbose_logger.error(
                f"Error processing Bedrock Invoke Agent response: {str(e)}"
            )
            raise BedrockError(
                message=f"Error processing response: {str(e)}",
                status_code=raw_response.status_code,
            )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message)

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        return True
