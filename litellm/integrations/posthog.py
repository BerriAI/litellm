#### What this does ####
#    On success/failure, logs events to PostHog
import os
import threading
import traceback
from typing import Any, Dict

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse
)


class PosthogLogger(CustomLogger):
    # Class variables
    def __init__(self):
        # Instance variables
        try:
            import posthog
            self.posthog = posthog
            self.posthog_api_key = os.getenv("POSTHOG_API_KEY")
            self.posthog_host = os.getenv("POSTHOG_API_URL", "https://us.i.posthog.com")
            
            # Initialize PostHog client
            if self.posthog_api_key:
                self.posthog.api_key = self.posthog_api_key
                if self.posthog_host:
                    self.posthog.host = self.posthog_host
            else:
                verbose_logger.warning("PostHog API Key not found in environment variables")
                
            # Setup stream tracking
            self._stream_id_to_data = {}
            self._lock = threading.Lock()  # lock for stream data
                
        except ImportError:
            verbose_logger.warning("PostHog Python SDK not installed. Run `pip install posthog`")
            raise

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success(kwargs, response_obj, start_time, end_time)

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        """
        Log the success event to PostHog.
        """
        try:
            verbose_logger.debug("PostHog logging start for success event")

            if kwargs.get("stream"):
                self._handle_stream_event(kwargs, response_obj, start_time, end_time)
            else:
                litellm_call_id = kwargs.get("litellm_call_id")
                model = kwargs.get("model", "")
                call_type = kwargs.get("call_type", "completion")
                
                # Create the event properties based on call type
                if call_type == "embeddings":
                    self._log_embedding_event(model, kwargs, response_obj, start_time, end_time, litellm_call_id)
                else:
                    self._log_completion_event(model, kwargs, response_obj, start_time, end_time, litellm_call_id)
                    
        except Exception:
            verbose_logger.debug("PostHog Logging Error", stack_info=True)
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        """
        Log the failure event to PostHog.
        """
        try:
            verbose_logger.debug("PostHog logging start for failure event")
            
            litellm_call_id = kwargs.get("litellm_call_id")
            model = kwargs.get("model", "")
            call_type = kwargs.get("call_type", "completion")
            
            # Calculate latency
            latency = (end_time - start_time).total_seconds()
            
            # Get user_id and metadata
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or {}
            user_id = metadata.get("user_id", litellm_call_id)
            
            # Prepare event properties
            event_properties = {
                "$ai_trace_id": litellm_call_id,
                "$ai_provider": self._get_provider_from_model(model),
                "$ai_model": model,
                "$ai_latency": latency,
                "$ai_http_status": getattr(response_obj, "status_code", 500),
                "$ai_is_error": True,
                "$ai_error": str(response_obj),
            }
            
            # Add input based on call type
            if call_type == "embeddings":
                event_properties["$ai_input"] = self._prepare_embedding_input(kwargs)
            else:
                event_properties["$ai_input"] = self._prepare_completion_input(kwargs)
                # Add tools separately
                tools = kwargs.get("tools") or kwargs.get("optional_params", {}).get("tools")
                if tools:
                    event_properties["$ai_tools"] = tools
            
            # Add model parameters and metadata
            self._add_model_params_and_metadata(event_properties, kwargs)
            
            # Capture the event
            self.posthog.capture(
                distinct_id=user_id,
                event="$ai_generation",
                properties=event_properties
            )
            
            verbose_logger.debug("PostHog Logging - Failure logged!")
            
        except Exception:
            verbose_logger.debug(f"PostHog Logging Error - {traceback.format_exc()}")
            pass

    def _handle_stream_event(self, kwargs, response_obj, start_time, end_time):
        """
        Handle streaming events from LiteLLM.
        For streaming calls, we:
        1. Track the first chunk and store initial data
        2. For the final chunk, gather everything and send to PostHog
        """
        litellm_call_id = kwargs.get("litellm_call_id")
        
        # If this is the first chunk
        if litellm_call_id not in self._stream_id_to_data:
            with self._lock:
                if litellm_call_id not in self._stream_id_to_data:
                    # Initialize stream data with first chunk info
                    self._stream_id_to_data[litellm_call_id] = {
                        "start_time": start_time,
                        "chunks": [],
                        "model": kwargs.get("model", ""),
                        "input": self._prepare_completion_input(kwargs),
                        "metadata": self._extract_metadata(kwargs),
                    }
        
        # Track this chunk if we have stream data
        if litellm_call_id in self._stream_id_to_data:
            # Add this chunk to our tracking
            if hasattr(response_obj, "choices") and len(response_obj.choices) > 0:
                with self._lock:
                    for choice in response_obj.choices:
                        if hasattr(choice, "delta"):
                            delta = choice.delta.model_dump() if hasattr(choice.delta, "model_dump") else choice.delta
                            self._stream_id_to_data[litellm_call_id]["chunks"].append(delta)
        
        # If this is the final chunk, send the complete event
        if kwargs.get("complete_streaming_response"):
            final_response = kwargs.get("complete_streaming_response")
            with self._lock:
                stream_data = self._stream_id_to_data.pop(litellm_call_id, None)
                
            if stream_data:
                self._log_completion_event(
                    stream_data["model"],
                    kwargs,
                    final_response,
                    stream_data["start_time"],
                    end_time,
                    litellm_call_id,
                    is_streaming=True
                )

    def _log_completion_event(self, model, kwargs, response_obj, start_time, end_time, litellm_call_id, is_streaming=False):
        """Log a completion (chat or text) event to PostHog"""
        # Calculate latency
        latency = (end_time - start_time).total_seconds()
        
        # Get user_id and metadata
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or {}
        user_id = metadata.get("user_id", None)
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response_obj, "usage"):
            input_tokens = getattr(response_obj.usage, "prompt_tokens", 0)
            output_tokens = getattr(response_obj.usage, "completion_tokens", 0)
        
        # Prepare event properties
        event_properties = {
            "$ai_trace_id": litellm_call_id,
            "$ai_provider": self._get_provider_from_model(model),
            "$ai_model": model,
            "$ai_input": self._prepare_completion_input(kwargs),
            "$ai_latency": latency,
            "$ai_input_tokens": input_tokens,
            "$ai_output_tokens": output_tokens,
            "$ai_http_status": 200,
            "$ai_is_error": False,
        }
        
        # Handle optional tool calls
        tools = kwargs.get("tools") or kwargs.get("optional_params", {}).get("tools")
        if tools:
            event_properties["$ai_tools"] = tools

        if user_id is None:
            event_properties["$process_person_profile"] = False
            user_id = litellm_call_id
        
        # Add model parameters and metadata
        self._add_model_params_and_metadata(event_properties, kwargs)
        
        # Add response outputs (message content or text)
        if isinstance(response_obj, (ModelResponse, TextCompletionResponse)):
            if hasattr(response_obj, "choices") and len(response_obj.choices) > 0:
                outputs = []
                for choice in response_obj.choices:
                    if hasattr(choice, "message"):
                        outputs.append({
                            "role": "assistant",
                            "content": choice.message.content,
                            "tool_calls": getattr(choice.message, "tool_calls", None),
                        })
                    elif hasattr(choice, "text"):
                        outputs.append({"role": "assistant", "content": choice.text})
                event_properties["$ai_output_choices"] = outputs
        
        # Capture the event
        self.posthog.capture(
            distinct_id=user_id,
            event="$ai_generation",
            properties=event_properties
        )
        
        verbose_logger.debug("PostHog Logging - Completion success logged!")

    def _log_embedding_event(self, model, kwargs, response_obj, start_time, end_time, litellm_call_id):
        """Log an embedding event to PostHog"""
        # Calculate latency
        latency = (end_time - start_time).total_seconds()
        
        # Get user_id and metadata
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or {}
        user_id = metadata.get("user_id", litellm_call_id)
        
        # Extract token usage
        input_tokens = 0
        if hasattr(response_obj, "usage"):
            input_tokens = getattr(response_obj.usage, "prompt_tokens", 0)
        
        # Prepare event properties
        event_properties = {
            "$ai_trace_id": litellm_call_id,
            "$ai_provider": self._get_provider_from_model(model),
            "$ai_model": model,
            "$ai_input": self._prepare_embedding_input(kwargs),
            "$ai_latency": latency,
            "$ai_input_tokens": input_tokens,
            "$ai_http_status": 200,
            "$ai_is_error": False,
            "$ai_embedding_dimensions": len(response_obj.data[0].embedding) if hasattr(response_obj, "data") and len(response_obj.data) > 0 else 0,
        }

        if user_id is None:
            event_properties["$process_person_profile"] = False
            user_id = litellm_call_id
        
        # Add model parameters and metadata
        self._add_model_params_and_metadata(event_properties, kwargs)
        
        # Capture the event
        self.posthog.capture(
            distinct_id=user_id,
            event="$ai_embedding",
            properties=event_properties
        )
        
        verbose_logger.debug("PostHog Logging - Embedding success logged!")

    def _prepare_completion_input(self, kwargs):
        """Prepare the input for a completion event"""
        messages = kwargs.get("messages", [])
        # For text completions, create a single message
        if prompt := kwargs.get("prompt"):
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = [{"role": "user", "content": "\n".join(prompt)}]
        
        # Return only messages as the input
        return messages

    def _prepare_embedding_input(self, kwargs):
        """Prepare the input for an embedding event"""
        input_data = kwargs.get("input", [])
        if isinstance(input_data, str):
            input_data = [input_data]
        return input_data

    def _add_model_params_and_metadata(self, event_properties, kwargs):
        """Add model parameters and metadata to event properties"""
        # Add model parameters
        optional_params = kwargs.get("optional_params", {})
        if optional_params:
            event_properties["$ai_model_parameters"] = self._extract_model_params(optional_params)
        
        # Add metadata from litellm_params
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) or {}
        for key, value in metadata.items():
            if key not in event_properties:
                event_properties[key] = value

    def _extract_metadata(self, kwargs):
        """Extract metadata from kwargs"""
        litellm_params = kwargs.get("litellm_params", {})
        return litellm_params.get("metadata", {}) or {}

    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string"""
        if model.startswith("gpt-"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        elif "bedrock" in model:
            return "bedrock"
        elif "palm" in model or "gemini" in model:
            return "google"
        elif "llama" in model.lower():
            return "meta"
        else:
            # Try to extract provider from model string
            for provider in ["azure", "anthropic", "cohere", "replicate", "huggingface", "ollama"]:
                if provider in model.lower():
                    return provider
            return "unknown"
    
    def _extract_model_params(self, optional_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant model parameters for logging"""
        relevant_params = {}
        param_keys = [
            "temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", 
            "presence_penalty", "stop", "stream", "n", "seed"
        ]
        
        for key in param_keys:
            if key in optional_params:
                relevant_params[key] = optional_params[key]
        
        return relevant_params