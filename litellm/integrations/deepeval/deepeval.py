import os
import deepeval
from deepeval.tracing.tracing import ToolAttributes
from litellm.integrations.custom_logger import CustomLogger
from deepeval.tracing import update_current_span_attributes, LlmAttributes, observe
from litellm.integrations.deepeval.utils import _prepare_input_str

class DeepEvalLogger(CustomLogger):
    """Logs litellm traces to DeepEval's platform."""
    # Class variables or attributes
    def __init__(self) -> None:
        self._validate_environment()
        super().__init__()

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure_event(kwargs, response_obj, start_time, end_time)       

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure_event(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success_event(kwargs, response_obj, start_time, end_time)

    def _handle_failure_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_obj = kwargs.get('standard_logging_object', {})
        error_str = standard_logging_obj.get('error_str', "")
        @observe(type="llm", model=kwargs["model"], name="litellm_message_failure", error_str=error_str)
        def send_litellm_message_failure_trace(
            input,
            output
        ):
            # rasing exception after updating attributes to ensure trace is logged
            update_current_span_attributes(
                LlmAttributes(
                    input=input,
                    output=output
                )
            )
            #TODO: subject to change with next update in tracing
            raise Exception(error_str)
        
        send_litellm_message_failure_trace(
            input=_prepare_input_str(kwargs["input"]),
            output=error_str
        )

    def _validate_environment(self):
        """
        Validate the DeepEval environment.
        """
        try:
            import deepeval
        except ImportError:
            raise ImportError(
                "DeepEval is not installed. Please install it using:\n"
                "pip install -U deepeval\n\n"
                "For more information, visit: https://documentation.confident-ai.com/"
            )

        if os.getenv("CONFIDENT_API_KEY", None) is None:
            raise ValueError(
                "CONFIDENT_API_KEY is not set. Please set it using:\n"
                "export CONFIDENT_API_KEY=<your_api_key>\n\n"
                "For more information, visit: https://documentation.confident-ai.com/getting-started/create-account"
            )
        
        deepeval.login_with_confident_api_key(os.getenv("CONFIDENT_API_KEY"))

    def _handle_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log the success event as a DeepEval. This method is asynchronous by default.
        """
        _cost_per_input_token = None
        _cost_per_output_token = None
        input_token_count = None
        output_token_count = None
        metrics = []

        # Get cost per token
        try:
            _cost_per_input_token, _cost_per_output_token = self._get_cost_per_token(kwargs)
        except Exception as e:
            #TODO: see if there is any verbose print method
            print(f"Error getting cost per token: {e}")

        # Get token count
        try:
            input_token_count, output_token_count = self._get_token_count(response_obj)
        except Exception as e:
            print(f"Error getting token count: {e}")

        # Get metrics
        try:
            if kwargs.get("litellm_params") is not None:
                metrics = kwargs.get("litellm_params").get("metadata")
                if metrics is not None:
                    metrics = metrics.get("deepeval_metrics")
        except Exception as e:
            print(f"Error getting metrics: {e}")
        
        # message success Trace
        # PS: execution time is logged as 0 (spanned on a hook rather than a llm call)
        @observe(type="llm", model=kwargs["model"], name="litellm_message_success", cost_per_output_token=_cost_per_output_token, cost_per_input_token=_cost_per_input_token, metrics=metrics)
        def send_litellm_message_success_trace(
            input, 
            output,
            input_token_count=None,
            output_token_count=None
        ):
            update_current_span_attributes(
                LlmAttributes(
                    input=input,    
                    output=output,
                    # TODO: Ask if we need to add prompt
                    #prompt=?
                    input_token_count=input_token_count,
                    output_token_count=output_token_count
                )
            )

        # tool success trace
        @observe(type="tool", cost_per_output_token=_cost_per_output_token, cost_per_input_token=_cost_per_input_token, metrics=metrics)
        def send_litellm_tool_success_trace(
            input_parameters,
            output=None
        ):
            update_current_span_attributes(
                ToolAttributes(
                    # TODO: input and output subject to change with next update in tracing failure
                    input_parameters=input_parameters,
                    output=output           
                )
            )
            
        # response_object will decide the span
        for choice in response_obj.choices:
            if hasattr(choice, 'message'):
                if hasattr(choice.message, 'content') and choice.message.content is not None:
                    send_litellm_message_success_trace(
                        # TODO: input and output subject to change with next update in tracing
                        input=_prepare_input_str(kwargs["input"]), 
                        output=choice.message.content,
                        input_token_count=input_token_count,
                        output_token_count=output_token_count
                    )
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls is not None:
                    for tool_call in choice.message.tool_calls:
                        if hasattr(tool_call, 'function') and tool_call.function is not None:
                            send_litellm_tool_success_trace(
                                input_parameters=tool_call.function.to_dict()
                            )

    def _get_cost_per_token(self, kwargs):
        """
        Get the cost per token for the input and output of the LLM call.
        """
        # Safely get litellm_params
        litellm_params = kwargs.get('litellm_params', {})
        standard_logging_obj = kwargs.get('standard_logging_object', {})
        model_map_info = standard_logging_obj.get('model_map_information', {})
        model_map_value = model_map_info.get('model_map_value', {})

        # Safely get input cost
        if litellm_params and litellm_params.get('input_cost_per_token') is not None:
            _cost_per_input_token = litellm_params['input_cost_per_token']
        elif model_map_value and model_map_value.get('input_cost_per_token') is not None:
            _cost_per_input_token = model_map_value['input_cost_per_token']

        # Safely get output cost
        if litellm_params and litellm_params.get('output_cost_per_token') is not None:
            _cost_per_output_token = litellm_params['output_cost_per_token']
        elif model_map_value and model_map_value.get('output_cost_per_token') is not None:
            _cost_per_output_token = model_map_value['output_cost_per_token']
        
        return _cost_per_input_token, _cost_per_output_token
    
    def _get_token_count(self, response_obj):
        """
        Get the token count from the response object.
        """
        if response_obj is not None:
            input_token_count = response_obj.usage.prompt_tokens if response_obj.usage is not None else None
            output_token_count = response_obj.usage.completion_tokens if response_obj.usage is not None else None
        
        return input_token_count, output_token_count