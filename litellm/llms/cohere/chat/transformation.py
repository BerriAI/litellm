import json
import time
from typing import TYPE_CHECKING, Any, List, Optional

import httpx

import litellm
from litellm.llms.base_llm.transformation import BaseConfig
from litellm.llms.prompt_templates.cohere import cohere_messages_pt_v2
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CohereError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.cohere.ai/v1/chat")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(self.message)  # Ca


class CohereChatConfig(BaseConfig):
    def validate_environment(
        self,
        api_key: str,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
    ) -> dict:
        """
        Return headers to use for cohere chat completion request

        Cohere API Ref: https://docs.cohere.com/reference/chat
        Expected headers:
        {
            "Request-Source": "unspecified:litellm",
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "bearer $CO_API_KEY"
        }
        """
        headers.update(
            {
                "Request-Source": "unspecified:litellm",
                "accept": "application/json",
                "content-type": "application/json",
            }
        )
        if api_key:
            headers["Authorization"] = f"bearer {api_key}"
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:

        ## Load Config
        # for k, v in self.items():
        #     if (
        #         k not in optional_params
        #     ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
        #         optional_params[k] = v

        most_recent_message, chat_history = cohere_messages_pt_v2(
            messages=messages, model=model, llm_provider="cohere_chat"
        )

        ## Handle Tool Calling
        if "tools" in optional_params:
            _is_function_call = True
            cohere_tools = self._construct_cohere_tool(tools=optional_params["tools"])
            optional_params["tools"] = cohere_tools
        if isinstance(most_recent_message, dict):
            optional_params["tool_results"] = [most_recent_message]
        elif isinstance(most_recent_message, str):
            optional_params["message"] = most_recent_message

        ## check if chat history message is 'user' and 'tool_results' is given -> force_single_step=True, else cohere api fails
        if len(chat_history) > 0 and chat_history[-1]["role"] == "USER":
            optional_params["force_single_step"] = True

        return optional_params

    def transform_response(
        self,
        model: str,
        httpx_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        encoding: str,
    ) -> ModelResponse:

        try:
            raw_response = httpx_response.json()
            model_response.choices[0].message.content = raw_response["text"]  # type: ignore
        except Exception:
            raise CohereError(
                message=httpx_response.text, status_code=httpx_response.status_code
            )

        ## ADD CITATIONS
        if "citations" in raw_response:
            setattr(model_response, "citations", raw_response["citations"])

        ## Tool calling response
        cohere_tools_response = raw_response.get("tool_calls", None)
        if cohere_tools_response is not None and cohere_tools_response != []:
            # convert cohere_tools_response to OpenAI response format
            tool_calls = []
            for tool in cohere_tools_response:
                function_name = tool.get("name", "")
                generation_id = tool.get("generation_id", "")
                parameters = tool.get("parameters", {})
                tool_call = {
                    "id": f"call_{generation_id}",
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
        billed_units = raw_response.get("meta", {}).get("billed_units", {})

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
        # cohere tools look like this
        """
        {
        "name": "query_daily_sales_report",
        "description": "Connects to a database to retrieve overall sales volumes and sales information for a given day.",
        "parameter_definitions": {
            "day": {
                "description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.",
                "type": "str",
                "required": True
            }
        }
        }
        """

        # OpenAI tools look like this
        """
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
        """
        cohere_tool = {
            "name": openai_tool["function"]["name"],
            "description": openai_tool["function"]["description"],
            "parameter_definitions": {},
        }

        for param_name, param_def in openai_tool["function"]["parameters"][
            "properties"
        ].items():
            required_params = (
                openai_tool.get("function", {})
                .get("parameters", {})
                .get("required", [])
            )
            cohere_param_def = {
                "description": param_def.get("description", ""),
                "type": param_def.get("type", ""),
                "required": param_name in required_params,
            }
            cohere_tool["parameter_definitions"][param_name] = cohere_param_def

        return cohere_tool
