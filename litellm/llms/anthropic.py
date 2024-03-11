import os, types
import json
from enum import Enum
import requests, copy
import time, uuid
from typing import Callable, Optional
from litellm.utils import ModelResponse, Usage, map_finish_reason
import litellm
from .prompt_templates.factory import (
    prompt_factory,
    custom_prompt,
    construct_tool_use_system_prompt,
    extract_between_tags,
    parse_xml_params,
)
import httpx


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "


class AnthropicError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.anthropic.com/v1/messages"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class AnthropicConfig:
    """
    Reference: https://docs.anthropic.com/claude/reference/complete_post

    to pass metadata to anthropic, it's {"user_id": "any-relevant-information"}
    """

    max_tokens: Optional[int] = litellm.max_tokens  # anthropic requires a default
    stop_sequences: Optional[list] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None
    system: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = 256,  # anthropic requires a default
        stop_sequences: Optional[list] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
        system: Optional[str] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }


# makes headers for API call
def validate_environment(api_key, user_headers):
    if api_key is None:
        raise ValueError(
            "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
        )
    headers = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    if user_headers is not None and isinstance(user_headers, dict):
        headers = {**headers, **user_headers}
    return headers


def completion(
    model: str,
    messages: list,
    api_base: str,
    custom_prompt_dict: dict,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    headers={},
):
    headers = validate_environment(api_key, headers)
    _is_function_call = False
    messages = copy.deepcopy(messages)
    if model in custom_prompt_dict:
        # check if the model has a registered custom prompt
        model_prompt_details = custom_prompt_dict[model]
        prompt = custom_prompt(
            role_dict=model_prompt_details["roles"],
            initial_prompt_value=model_prompt_details["initial_prompt_value"],
            final_prompt_value=model_prompt_details["final_prompt_value"],
            messages=messages,
        )
    else:
        # Separate system prompt from rest of message
        system_prompt_idx: Optional[int] = None
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                optional_params["system"] = message["content"]
                system_prompt_idx = idx
                break
        if system_prompt_idx is not None:
            messages.pop(system_prompt_idx)
        # Format rest of message according to anthropic guidelines
        messages = prompt_factory(
            model=model, messages=messages, custom_llm_provider="anthropic"
        )

    ## Load Config
    config = litellm.AnthropicConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    ## Handle Tool Calling
    if "tools" in optional_params:
        _is_function_call = True
        tool_calling_system_prompt = construct_tool_use_system_prompt(
            tools=optional_params["tools"]
        )
        optional_params["system"] = (
            optional_params.get("system", "\n") + tool_calling_system_prompt
        )  # add the anthropic tool calling prompt to the system prompt
        optional_params.pop("tools")

    data = {
        "model": model,
        "messages": messages,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "api_base": api_base,
            "headers": headers,
        },
    )

    ## COMPLETION CALL
    if "stream" in optional_params and optional_params["stream"] == True:
        response = requests.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=optional_params["stream"],
        )

        if response.status_code != 200:
            raise AnthropicError(
                status_code=response.status_code, message=response.text
            )

        return response.iter_lines()
    else:
        response = requests.post(api_base, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise AnthropicError(
                status_code=response.status_code, message=response.text
            )

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        try:
            completion_response = response.json()
        except:
            raise AnthropicError(
                message=response.text, status_code=response.status_code
            )
        if "error" in completion_response:
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
            )
        elif len(completion_response["content"]) == 0:
            raise AnthropicError(
                message="No content in response",
                status_code=response.status_code,
            )
        else:
            text_content = completion_response["content"][0].get("text", None)
            ## TOOL CALLING - OUTPUT PARSE
            if text_content is not None and "invoke" in text_content:
                function_name = extract_between_tags("tool_name", text_content)[0]
                function_arguments_str = extract_between_tags("invoke", text_content)[
                    0
                ].strip()
                function_arguments_str = f"<invoke>{function_arguments_str}</invoke>"
                function_arguments = parse_xml_params(function_arguments_str)
                _message = litellm.Message(
                    tool_calls=[
                        {
                            "id": f"call_{uuid.uuid4()}",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(function_arguments),
                            },
                        }
                    ],
                    content=None,
                )
                model_response.choices[0].message = _message  # type: ignore
            else:
                model_response.choices[0].message.content = text_content  # type: ignore
            model_response.choices[0].finish_reason = map_finish_reason(
                completion_response["stop_reason"]
            )

        ## CALCULATING USAGE
        prompt_tokens = completion_response["usage"]["input_tokens"]
        completion_tokens = completion_response["usage"]["output_tokens"]
        total_tokens = prompt_tokens + completion_tokens

        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        model_response.usage = usage
        return model_response


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
