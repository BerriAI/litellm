"""Cohere Chat V2 API Integration Module.

This module provides the necessary classes and functions to interact with Cohere's V2 Chat API.
It handles the transformation of requests and responses between LiteLLM's standard format and
Cohere's specific API requirements.
"""

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import cohere_messages_pt_v3
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

# Use absolute imports instead of relative imports
from litellm.llms.cohere.common_utils import ModelResponseIterator as CohereModelResponseIterator
from litellm.llms.cohere.common_utils import validate_environment as cohere_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CohereErrorV2(BaseLLMException):
    """
    Exception class for Cohere v2 API errors.
    
    This class handles errors returned by the Cohere v2 API and formats them
    in a way that is consistent with the LiteLLM error handling system.
    """
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.cohere.com/v2/chat")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class CohereChatConfigV2(BaseConfig):
    """
    Configuration class for Cohere's V2 API interface.

    Args:
        preamble (str, optional): When specified, the default Cohere preamble will be replaced 
            with the provided one.
        generation_id (str, optional): Unique identifier for the generated reply.
        conversation_id (str, optional): Creates or resumes a persisted conversation.
        prompt_truncation (str, optional): Dictates how the prompt will be constructed. 
            Options: 'AUTO', 'AUTO_PRESERVE_ORDER', 'OFF'.
        connectors (List[Dict[str, str]], optional): List of connectors (e.g., web-search) 
            to enrich the model's reply.
        search_queries_only (bool, optional): When true, the response will only contain a list 
            of generated search queries.
        documents (List[Dict[str, str]] or List[str], optional): A list of relevant documents 
            that the model can cite.
        temperature (float, optional): A non-negative float that tunes the degree of randomness 
            in generation.
        max_tokens (int, optional): The maximum number of tokens the model will generate as part 
            of the response.
        k (int, optional): Ensures only the top k most likely tokens are considered for generation 
            at each step.
        p (float, optional): Ensures that only the most likely tokens, with total probability mass 
            of p, are considered for generation.
        frequency_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        presence_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        tools (List[Dict[str, str]], optional): A list of available tools (functions) that the model 
            may suggest invoking.
        tool_results (List[Dict[str, Any]], optional): A list of results from invoking tools.
        seed (int, optional): A seed to assist reproducibility of the model's response.
    """

    preamble: Optional[str] = None
    generation_id: Optional[str] = None
    conversation_id: Optional[str] = None
    prompt_truncation: Optional[str] = None
    connectors: Optional[list] = None
    search_queries_only: Optional[bool] = None
    documents: Optional[list] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    k: Optional[int] = None
    p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[list] = None
    tool_results: Optional[list] = None
    seed: Optional[int] = None

    def __init__(self, **kwargs) -> None:
        """
        Initialize the CohereChatConfigV2 with parameters matching Cohere v2 API specification.
        
        All parameters are passed as keyword arguments and set as class attributes
        if they have a non-None value. This approach allows for future API changes
        without requiring code modifications.
        
        Args:
            **kwargs: Arbitrary keyword arguments matching Cohere v2 API parameters.
                      See class docstring for details on supported parameters.
        """
        # Process all keyword arguments and set as class attributes if not None
        for key, value in kwargs.items():
            if value is not None:
                setattr(self.__class__, key, value)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        **kwargs,
    ) -> dict:
        # Extract api_key from kwargs if present
        api_key = kwargs.get('api_key')
        return cohere_validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_version="v2",  # Specify v2 API version
        )

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "tools",
            "tool_choice",
            "seed",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "n":
                optional_params["num_generations"] = value
            if param == "top_p":
                optional_params["p"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "seed":
                optional_params["seed"] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        **kwargs,
    ) -> dict:
        # Extract necessary parameters from kwargs
        # These variables are used by the parent class implementation
        _ = kwargs.get('litellm_params', {})
        _ = kwargs.get('headers', {})
        ## Load Config
        for k, v in litellm.CohereChatConfigV2.get_config().items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > cohere_config(top_k=3)
                # Allows for dynamic variables to be passed in
                optional_params[k] = v

        # In v2, messages are combined in a single array
        cohere_messages = cohere_messages_pt_v3(
            messages=messages, model=model, llm_provider="cohere_chat"
        )
        optional_params["messages"] = cohere_messages
        optional_params["model"] = model.split("/")[-1]  # Extract model name from model string

        ## Handle Tool Calling
        if "tools" in optional_params:
            cohere_tools = self._construct_cohere_tool(tools=optional_params["tools"])
            optional_params["tools"] = cohere_tools

        # Handle tool results if present
        if "tool_results" in optional_params and isinstance(optional_params["tool_results"], list):
            # Convert tool results to v2 format if needed
            tool_results = []
            for result in optional_params["tool_results"]:
                if isinstance(result, dict) and "content" in result:
                    # Format from v1 to v2
                    tool_result = {
                        "tool_call_id": result.get("tool_call_id", ""),
                        "output": result.get("content", ""),
                    }
                    tool_results.append(tool_result)
                else:
                    # Already in v2 format
                    tool_results.append(result)
            optional_params["tool_results"] = tool_results

        return optional_params

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        **_  # Unused kwargs
    ) -> ModelResponse:
        try:
            raw_response_json = raw_response.json()
            # Get the text content from the response
            # Set the text content from the response
            model_response.choices[0].message.content = raw_response_json.get("text", "")
        except Exception as exc:
            raise CohereErrorV2(
                message=raw_response.text, status_code=raw_response.status_code
            ) from exc

        ## ADD CITATIONS
        # Add citation information to the model response if available
        if "citations" in raw_response_json:
            citations = raw_response_json["citations"]
            setattr(model_response, "citations", citations)

        ## Tool calling response
        cohere_tools_response = raw_response_json.get("tool_calls", None)
        if cohere_tools_response is not None and cohere_tools_response != []:
            # convert cohere_tools_response to OpenAI response format
            tool_calls = []
            for tool in cohere_tools_response:
                function_name = tool.get("name", "")
                tool_call_id = tool.get("id", "")
                parameters = tool.get("parameters", {})
                tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(parameters),
                    },
                }
                tool_calls.append(tool_call)
            _message = litellm.Message(
                tool_calls=tool_calls,
                content=None,
            )
            model_response.choices[0].message = _message  # type: ignore

        ## CALCULATING USAGE - use cohere `billed_units` for returning usage
        billed_units = raw_response_json.get("usage", {})

        prompt_tokens = billed_units.get("input_tokens", 0)
        completion_tokens = billed_units.get("output_tokens", 0)

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def _construct_cohere_tool(
        self,
        tools: Optional[list] = None,
    ):
        if tools is None:
            tools = []
        cohere_tools = []
        for tool in tools:
            cohere_tool = self._translate_openai_tool_to_cohere(tool)
            cohere_tools.append(cohere_tool)
        return cohere_tools

    def _translate_openai_tool_to_cohere(
        self,
        openai_tool: dict,
    ):
        """
        Translates OpenAI tool format to Cohere v2 tool format
        
        Cohere v2 tools look like this:
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
        """

        cohere_tool = {
            "name": openai_tool["function"]["name"],
            "description": openai_tool["function"]["description"],
            "input_schema": openai_tool["function"]["parameters"],
        }

        return cohere_tool

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return CohereModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CohereErrorV2(status_code=status_code, message=error_message)
